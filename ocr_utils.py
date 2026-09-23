import cv2
import easyocr
import re
import threading
from api_client import enviar_placa_para_api
from ultralytics import YOLO

print("🚀 Carregando YOLOv8 oficial...")
model_yolo = YOLO("yolov8n.pt")

print("🔤 Inicializando EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

def verificar_veiculo_yolo(frame_vaga):
    """
    Retorna True se detectar QUALQUER OBJETO dentro da vaga.
    Ideal para testes com celular, papel ou objetos de bancada.
    """
    if frame_vaga is None or frame_vaga.size == 0:
        print("⚠️ Erro: Recorte da vaga está vazio!")
        return False

    # 🛑 MUDANÇA AQUI: Tiramos o "classes=[...]" para ele aceitar Celular, Pessoa, TV, etc.
    results = model_yolo.predict(
        source=frame_vaga, 
        verbose=False, 
        conf=0.15 # Confiança em 15% para não perder nada no teste
    )

    for result in results:
        if len(result.boxes) > 0:
            # Pega o nome do que ele achou só para te mostrar no terminal
            nome_objeto = model_yolo.names[int(result.boxes[0].cls[0])]
            print(f"👁️ YOLO detectou algo: {nome_objeto}")
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

def processar_ocr_e_enviar(crop_vaga, crop_placa, vaga_info):
    """Verifica se há objeto com YOLO e depois extrai a placa via OCR."""
    def tarefa_pipeline():
        try:
            # 1. Verifica no recorte GRANDE se tem algo
            tem_veiculo = verificar_veiculo_yolo(crop_vaga)
            vaga_info["ocupada"] = tem_veiculo
            
            # SISTEMA ANTI-FLICKER DO YOLO
            if not tem_veiculo:
                ausencias = vaga_info.get("ausencia_consecutiva", 0) + 1
                vaga_info["ausencia_consecutiva"] = ausencias
                print(f"👀 Vaga {vaga_info['vaga']}: Nada detectado pelo YOLO. (Ausências: {ausencias})")
                
                # Só reseta a placa se a vaga estiver realmente vazia por 3 leituras
                if ausencias >= 3:
                    vaga_info["placa_atual"] = None
                return

            vaga_info["ausencia_consecutiva"] = 0

            # 2. Se já leu essa placa antes, não precisa ler de novo
            placa_salva = vaga_info.get("placa_atual")
            if placa_salva is not None:
                print(f"🔒 Vaga {vaga_info['vaga']}: Objeto detectado. Placa ({placa_salva}) já enviada antes. Pula OCR.")
                return

            print(f"🔍 Vaga {vaga_info['vaga']}: NOVO Objeto detectado! Ligando o EasyOCR para ler a placa...")
            
            # 3. Roda o OCR na area_placa
            resultados = reader.readtext(
                crop_placa, 
                detail=1, 
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                paragraph=False
            )
            
            placa_encontrada = False
            for (bbox, texto, confianca) in resultados:
                placa_validada = validar_e_corrigir_placa(texto)
                
                if placa_validada:
                    placa_encontrada = True
                    vaga_info["placa_atual"] = placa_validada
                    
                    _, buffer = cv2.imencode('.jpg', crop_placa)
                    imagem_bytes = buffer.tobytes()

                    payload = {
                        "id_estacionamento": 'perS63z9MTjKZWRASKt3', # vaga_info["id_estacionamento"],
                        "id_vaga": vaga_info["id_vaga_firestore"],
                        "numero_vaga": vaga_info["vaga"],
                        "placa": placa_validada,
                        "confianca": str(confianca)
                    }
                    
                    enviar_placa_para_api(payload, imagem_bytes)
                    print(f"✅ SUCESSO! Placa {placa_validada} enviada para a API. Vaga trancada.")
                    break 
            
            if not placa_encontrada:
                print(f"❌ Vaga {vaga_info['vaga']}: Objeto está lá, mas o OCR não identificou nenhuma placa válida neste frame.")

        except Exception as e:
            print(f"🚨 ERRO na thread de processamento: {e}")
        finally:
            vaga_info["processando_ocr"] = False

    vaga_info["processando_ocr"] = True
    threading.Thread(target=tarefa_pipeline, daemon=True).start()