import cv2
import numpy as np
from collections import deque

from config import CAMERA_WIDTH, CAMERA_HEIGHT, VAGAS_RELATIVAS, map_coords
from ocr_utils import processar_ocr_e_enviar
from ui import desenhar_interface
import time
from api_client import (iniciar_worker_api, iniciar_heartbeat, enviar_alerta_para_api, aguardar_envios)
from monitor import MonitorObstrucao

vagas_map = []
for vaga_rel in VAGAS_RELATIVAS:
    vaga_abs = vaga_rel.copy()
    vaga_abs["coords"] = map_coords(vaga_rel["coords_rel"], CAMERA_WIDTH, CAMERA_HEIGHT)
    vaga_abs["area_placa"] = map_coords(vaga_rel["area_placa_rel"], CAMERA_WIDTH, CAMERA_HEIGHT)
    vaga_abs.update({
        "placa_atual": None,
        "ref_placa": None,
        "ocupada": False,
        "processando_ocr": False,
        "deteccoes": 0,
        "ausencias": 0,
        "historico_posicoes": deque(maxlen=10),
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
    estado = {"ultimo_frame": time.time()}
    iniciar_heartbeat(lambda: estado["ultimo_frame"])
    monitor = MonitorObstrucao()
    cap = obter_camera()
    usando_camera_virtual = False

    if cap is None:
        print("⚠️ Nenhuma webcam física encontrada ou acessível.")
        print("💡 Entrando no MODO DE TESTE VIRTUAL...")
        usando_camera_virtual = True
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        print("Resolução nativa:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("👉 O sistema está buscando veículos e placas automaticamente.")
    print("👉 Pressione 'q' para encerrar.\n")

    JANELA = "Vallet IoT - Leitor de Placas"
    cv2.namedWindow(JANELA, cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    CHECK_EVERY_N_FRAMES = 15

    while True:
        if usando_camera_virtual:
            frame = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 40, dtype=np.uint8)
            cv2.putText(frame, "MODO DE TESTE (SEM CAMERA)", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Falha ao ler frame da câmera.")
                enviar_alerta_para_api("sistema_erro", "O leitor perdeu a conexão com a câmera.")
                break
            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

        estado["ultimo_frame"] = time.time()
        frame_count += 1
        if frame_count % 5 == 0 and not usando_camera_virtual:
            monitor.verificar(frame)
        frame_original = frame.copy()

        for vaga in vagas_map:
            if frame_count % CHECK_EVERY_N_FRAMES == 0 and not vaga["processando_ocr"]:
                vx1, vy1, vx2, vy2 = vaga["coords"]
                ax1, ay1, ax2, ay2 = vaga["area_placa"]
                crop_vaga = frame_original[vy1:vy2, vx1:vx2].copy()
                crop_placa = frame_original[ay1:ay2, ax1:ax2].copy()
                processar_ocr_e_enviar(crop_vaga, crop_placa, vaga)

        desenhar_interface(frame, vagas_map)
        cv2.imshow(JANELA, frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    aguardar_envios()
    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()