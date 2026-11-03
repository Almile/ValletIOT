import cv2
import easyocr
import re
import threading
from collections import Counter, deque
from api_client import enviar_placa_para_api
from ultralytics import YOLO
import numpy as np
from config import (CONFIRMAR_OCUPADA, CONFIRMAR_LIVRE, YOLO_CLASSES, YOLO_CONF,
                    YOLO_AREA_MIN, OCR_CONF_MIN, LEITURAS_PARA_CONFIRMAR, CONF_VOTO_MIN,
                    PLACA_CONFIRMA_OCUPACAO, USAR_YOLO, OCR_ESCALA, DIFF_MAX, DEBUG_OCR)


print("🚀 Carregando YOLOv8 oficial...")
model_yolo = YOLO("yolov8n.pt")

print("🔤 Inicializando EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

ocr_lock = threading.Lock()
placas_em_uso = {} 

def verificar_veiculo_yolo(frame_vaga):
    """True só se houver um veículo de tamanho relevante dentro da vaga."""
    if frame_vaga is None or frame_vaga.size == 0:
        return False
    h, w = frame_vaga.shape[:2]
    results = model_yolo.predict(source=frame_vaga, verbose=False,
                                 conf=YOLO_CONF, classes=YOLO_CLASSES)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if (x2 - x1) * (y2 - y1) >= YOLO_AREA_MIN * w * h:
                return True
    return False

def validar_e_corrigir_placa(texto):
    """Higieniza e valida o formato Mercosul e Antigo."""
    texto = ''.join(c for c in texto.upper() if c.isalnum())
    if len(texto) != 7:
        return None
    
    char_para_num = {'O': '0', 'Q': '0', 'Z': '2', 'S': '5', 'B': '8', 'I': '1', 'G': '6'}
    num_para_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}
    lista = list(texto)
    
    for i in range(3):
        if lista[i].isdigit() and lista[i] in num_para_char:
            lista[i] = num_para_char[lista[i]]
            
    if lista[3].isalpha() and lista[3] in char_para_num:
        lista[3] = char_para_num[lista[3]]
        
    for i in (5, 6):
        if lista[i].isalpha() and lista[i] in char_para_num:
            lista[i] = char_para_num[lista[i]]
            
    placa_corrigida = "".join(lista)
    
    if re.match(r'^[A-Z]{3}[0-9]{4}$', placa_corrigida) or re.match(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', placa_corrigida):
        return placa_corrigida
    return None

ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
from config import DEBUG_OCR

def _janelas_de_7(texto):
    t = ''.join(c for c in texto.upper() if c.isalnum())
    return [t[i:i+7] for i in range(len(t) - 6)] if len(t) >= 7 else []

def _ocr(img):
    res = reader.readtext(img, detail=1, allowlist=ALLOWLIST, paragraph=False,
                          text_threshold=0.5, low_text=0.3, width_ths=0.9)
    res.sort(key=lambda r: r[0][0][0])
    cands = [(t, float(c)) for _, t, c in res]
    if len(res) > 1:
        cands.append((''.join(r[1] for r in res), float(min(r[2] for r in res))))
    return cands

def _melhor_placa(cands):
    for texto, conf in sorted(cands, key=lambda x: -x[1]):
        if conf < OCR_CONF_MIN:
            continue
        for janela in _janelas_de_7(texto):
            placa = validar_e_corrigir_placa(janela)
            if placa:
                return placa, conf
    return None, 0.0

def ler_placa(crop, tag=""):
    """Uma passada em cor; só tenta o CLAHE se a primeira falhar."""
    big = cv2.resize(crop, None, fx=OCR_ESCALA, fy=OCR_ESCALA, interpolation=cv2.INTER_CUBIC)
    with ocr_lock:
        c1 = _ocr(big)
        placa, conf = _melhor_placa(c1)
        c2 = []
        if not placa:
            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            c2 = _ocr(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray))
            placa, conf = _melhor_placa(c2)
    if DEBUG_OCR and (c1 or c2):
        print(f"[{tag}] cor={[(t, round(c, 2)) for t, c in c1]} clahe={[(t, round(c, 2)) for t, c in c2]}")
    return placa, conf

def _miniatura(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (32, 16)).astype("float32")

def _liberar(vaga_info):
    if vaga_info["placa_atual"]:
        placas_em_uso.pop(vaga_info["placa_atual"], None)
        print(f"🚪 Vaga {vaga_info['vaga']} liberada.")
    vaga_info.update({"ocupada": False, "placa_atual": None, "ref_placa": None,
                      "ausencias": 0, "deteccoes": 0})
    vaga_info["candidatas"].clear()


def processar_ocr_e_enviar(crop_vaga, crop_placa, vaga_info):
    def tarefa_pipeline():
        try:
            vaga = vaga_info["vaga"]
            vaga_info.setdefault("candidatas", deque(maxlen=8))
            placa_atual = vaga_info["placa_atual"]

            # --- Vaga com placa confirmada: checagem rápida, sem OCR ---
            if placa_atual:
                ref = vaga_info.get("ref_placa")
                if ref is not None and np.abs(_miniatura(crop_placa) - ref).mean() < DIFF_MAX:
                    vaga_info["ausencias"] = 0
                    return
                placa_lida, _ = ler_placa(crop_placa, tag=vaga)
                if placa_lida == placa_atual:
                    vaga_info["ausencias"] = 0
                    vaga_info["ref_placa"] = _miniatura(crop_placa)
                else:
                    vaga_info["ausencias"] += 1
                    if vaga_info["ausencias"] >= CONFIRMAR_LIVRE:
                        _liberar(vaga_info)
                return

            # --- Vaga sem placa confirmada ---
            tem_veiculo = verificar_veiculo_yolo(crop_vaga) if USAR_YOLO else False
            placa_lida, conf = (None, 0.0)
            if tem_veiculo or PLACA_CONFIRMA_OCUPACAO:
                placa_lida, conf = ler_placa(crop_placa, tag=vaga)

            if tem_veiculo or placa_lida:
                vaga_info["ausencias"] = 0
                vaga_info["deteccoes"] += 1
                if placa_lida or vaga_info["deteccoes"] >= CONFIRMAR_OCUPADA:
                    vaga_info["ocupada"] = True
            else:
                vaga_info["deteccoes"] = 0
                vaga_info["ausencias"] += 1
                if vaga_info["ausencias"] >= CONFIRMAR_LIVRE:
                    _liberar(vaga_info)
                return

            if not placa_lida or conf < CONF_VOTO_MIN:
                return

            vaga_info["candidatas"].append(placa_lida)
            placa, votos = Counter(vaga_info["candidatas"]).most_common(1)[0]
            if votos < LEITURAS_PARA_CONFIRMAR:
                return

            dona = placas_em_uso.get(placa)
            if dona and dona != vaga:
                return

            placas_em_uso[placa] = vaga
            vaga_info["placa_atual"] = placa
            vaga_info["ref_placa"] = _miniatura(crop_placa)
            _, buffer = cv2.imencode('.jpg', crop_placa)
            payload = {
                "id_estacionamento": vaga_info["id_estacionamento"],
                "id_vaga": vaga_info["id_vaga_firestore"],
                "numero_vaga": vaga,
                "placa": placa,
                "confianca": f"{votos / len(vaga_info['candidatas']):.2f}"
            }
            enviar_placa_para_api(payload, buffer.tobytes())
            print(f"✅ Placa {placa} confirmada na vaga {vaga} ({votos} leituras iguais).")
        except Exception as e:
            print(f"🚨 ERRO na thread de processamento: {e}")
        finally:
            vaga_info["processando_ocr"] = False

    vaga_info["processando_ocr"] = True
    threading.Thread(target=tarefa_pipeline, daemon=True).start()