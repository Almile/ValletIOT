import time
import cv2
import numpy as np
from config import PALETA, MOSTRAR_AREA_PLACA

FONTE = cv2.FONT_HERSHEY_SIMPLEX


def _fill(frame, p1, p2, cor, alpha):
    """Retângulo translúcido (só processa a região, é barato)."""
    x1, y1 = p1
    x2, y2 = p2
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, cor), alpha, 0)


def _cantos(frame, p1, p2, cor, esp=3, tam=22):
    """Cantoneiras em vez de caixa fechada: visual de câmera de monitoramento."""
    x1, y1 = p1
    x2, y2 = p2
    for x, y, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (x, y), (x + dx * tam, y), cor, esp, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + dy * tam), cor, esp, cv2.LINE_AA)


def _texto(frame, txt, org, escala, cor, esp=1):
    cv2.putText(frame, txt, org, FONTE, escala, cor, esp, cv2.LINE_AA)


def _largura(txt, escala, esp=1):
    return cv2.getTextSize(txt, FONTE, escala, esp)[0][0]


def _desenhar_vaga(frame, vaga):
    x1, y1, x2, y2 = vaga["coords"]

    if vaga["placa_atual"]:
        cor, status = PALETA["placa_ok"], vaga["placa_atual"]
    elif vaga["ocupada"]:
        cor, status = PALETA["ocupada"], "OCUPADA"
    else:
        cor, status = PALETA["livre"], "LIVRE"

    _fill(frame, (x1, y1), (x2, y2), cor, 0.18)            # véu de cor na vaga
    _cantos(frame, (x1, y1), (x2, y2), cor)

    _fill(frame, (x1, y1), (x2, y1 + 30), cor, 0.88)       # barra de título dentro da vaga
    _texto(frame, vaga["vaga"], (x1 + 10, y1 + 21), 0.6, PALETA["texto"], 2)
    _texto(frame, status, (x2 - _largura(status, 0.55) - 10, y1 + 21), 0.55, PALETA["texto"])

    if MOSTRAR_AREA_PLACA:
        ax1, ay1, ax2, ay2 = vaga["area_placa"]
        cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), PALETA["area_placa"], 1)


def _cabecalho(frame, vagas):
    h, w = frame.shape[:2]
    _fill(frame, (0, 0), (w, 44), PALETA["fundo_texto"], 0.78)
    _texto(frame, "VALLET", (16, 30), 0.85, PALETA["texto"], 2)

    data = time.strftime("%d/%m/%Y  %H:%M:%S")
    _texto(frame, data, ((w - _largura(data, 0.5)) // 2, 29), 0.5, PALETA["texto"])

    livres = sum(1 for v in vagas if not v["ocupada"])
    resumo = f"{livres} livres  |  {len(vagas) - livres} ocupadas"
    _texto(frame, resumo, (w - _largura(resumo, 0.55) - 16, 29), 0.55, PALETA["texto"])


def desenhar_interface(frame, vagas):
    for vaga in vagas:
        _desenhar_vaga(frame, vaga)
    _cabecalho(frame, vagas)