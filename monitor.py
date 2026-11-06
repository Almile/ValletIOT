import time
import cv2
from api_client import enviar_alerta_para_api
from config import (OBSTRUCAO_SEGUNDOS, OBSTRUCAO_COOLDOWN_S,
                    OBSTRUCAO_BRILHO_MIN, OBSTRUCAO_NITIDEZ_MIN)


class MonitorObstrucao:
    """Câmera tampada, suja ou apagada: imagem escura ou sem detalhes por alguns segundos."""

    def __init__(self):
        self.desde = None
        self.alertado = False
        self.ultimo_alerta = 0.0

    def verificar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brilho = float(gray.mean())
        nitidez = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        obstruida = brilho < OBSTRUCAO_BRILHO_MIN or nitidez < OBSTRUCAO_NITIDEZ_MIN
        agora = time.time()

        if not obstruida:
            self.desde, self.alertado = None, False
            return False

        if self.desde is None:
            self.desde = agora
        elif (not self.alertado and agora - self.desde >= OBSTRUCAO_SEGUNDOS
              and agora - self.ultimo_alerta >= OBSTRUCAO_COOLDOWN_S):
            enviar_alerta_para_api(
                "obstrucao_camera",
                f"A câmera está com a visão obstruída há {OBSTRUCAO_SEGUNDOS}s "
                f"(brilho {brilho:.0f}, nitidez {nitidez:.0f})."
            )
            self.alertado = True
            self.ultimo_alerta = agora
        return True