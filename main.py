import cv2
import numpy as np
from collections import deque

from config import CAMERA_WIDTH, CAMERA_HEIGHT, VAGAS_RELATIVAS, map_coords
from api_client import iniciar_worker_api
from ocr_utils import processar_ocr_e_enviar

# Inicializa as vagas mapeadas com seus estados locais
vagas_map = []
for vaga_rel in VAGAS_RELATIVAS:
    vaga_abs = vaga_rel.copy()
    vaga_abs["coords"] = map_coords(vaga_rel["coords_rel"], CAMERA_WIDTH, CAMERA_HEIGHT)
    vaga_abs["area_placa"] = map_coords(vaga_rel["area_placa_rel"], CAMERA_WIDTH, CAMERA_HEIGHT)
    vaga_abs.update({
        "placa_atual": None,
        "ocupada": False,
        "processando_ocr": False,
        "historico_posicoes": deque(maxlen=10)
    })
    vagas_map.append(vaga_abs)

def obter_camera():
    """Tenta abrir webcams no Windows usando DirectShow."""
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"🎥 Câmera conectada com sucesso no índice {index}!")
                return cap
            cap.release()
    return None

def main():
    iniciar_worker_api()
    cap = obter_camera()
    usando_camera_virtual = False

    if cap is None:
        print("⚠️ Nenhuma webcam física encontrada ou acessível.")
        print("💡 Entrando no MODO DE TESTE VIRTUAL...")
        usando_camera_virtual = True
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    print("👉 O sistema está buscando veículos e placas automaticamente.")
    print("👉 Pressione 'q' para encerrar.\n")

    frame_count = 0
    CHECK_EVERY_N_FRAMES = 30 

    while True:
        if usando_camera_virtual:
            frame = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 40, dtype=np.uint8)
            cv2.putText(frame, "MODO DE TESTE (SEM CAMERA)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Falha ao ler frame da câmera.")
                break
            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

        frame_count += 1
        frame_original = frame.copy()

        for vaga in vagas_map:
            # 1. GATILHO AUTOMÁTICO: Dispara a cada 30 frames
            if frame_count % CHECK_EVERY_N_FRAMES == 0 and not vaga["processando_ocr"]:
                vx1, vy1, vx2, vy2 = vaga["coords"]
                ax1, ay1, ax2, ay2 = vaga["area_placa"]
                
                # Recorta a VAGA para achar o veículo e a PLACA para o OCR ler
                crop_vaga = frame_original[vy1:vy2, vx1:vx2].copy()
                crop_placa = frame_original[ay1:ay2, ax1:ax2].copy()
                
                processar_ocr_e_enviar(crop_vaga, crop_placa, vaga)

            # 2. DESENHA NA TELA
            vx1, vy1, vx2, vy2 = vaga["coords"]
            ax1, ay1, ax2, ay2 = vaga["area_placa"]

            cor = (130, 95, 95) if vaga["ocupada"] else (59, 80, 66)
            
            # Ajuste inteligente do texto da interface
            if vaga["placa_atual"]:
                status_texto = vaga["placa_atual"]
            elif vaga["ocupada"] and vaga["processando_ocr"]:
                status_texto = "LENDO PLACA..."
            elif vaga["ocupada"]:
                status_texto = "OCUPADA"
            elif vaga["processando_ocr"]:
                status_texto = "ANALISANDO..."
            else:
                status_texto = "LIVRE"

            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), cor, 2)
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (255, 100, 0), 1)
            cv2.putText(frame, f"{vaga['vaga']}: {status_texto}", (vx1, vy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        cv2.imshow("Vallet IoT - Leitor de Placas", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()