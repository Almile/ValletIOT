import json, os

API_BASE = "http://localhost:8000/camera/api/v1/eventos"
API_URL = f"{API_BASE}/placa-detectada"
API_URL_SAIDA = f"{API_BASE}/saida-detectada"
API_URL_ALERTA = f"{API_BASE}/alerta-sistema"
API_URL_HEARTBEAT = f"{API_BASE}/heartbeat"

HEARTBEAT_INTERVALO_S = 30
OBSTRUCAO_SEGUNDOS = 10
OBSTRUCAO_COOLDOWN_S = 600
OBSTRUCAO_BRILHO_MIN = 25
OBSTRUCAO_NITIDEZ_MIN = 15

# Mesma proporção da webcam (4:3) para não distorcer o texto.
# Se a sua câmera for 16:9, use 1280 x 720.
CAMERA_WIDTH = 848
CAMERA_HEIGHT = 480

USAR_YOLO = False
OCR_ESCALA = 2

LEITURAS_PARA_CONFIRMAR = 2
CONF_VOTO_MIN = 0.40
CONFIRMAR_LIVRE = 4
DIFF_MAX = 18


def hex_para_bgr(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


PALETA = {
    "livre":     hex_para_bgr("#42503B"),
    "ocupada":   hex_para_bgr("#5F5F82"),
    "placa_ok":  hex_para_bgr("#81AA83"),
    "area_placa":hex_para_bgr("#2E7D32"),
    "texto":     hex_para_bgr("#FFFFFF"),
    "fundo_texto": hex_para_bgr("#1E1E1E"),
}

# YOLO: 2=carro, 3=moto, 5=ônibus, 7=caminhão
YOLO_CLASSES = [2, 3, 5, 7]
YOLO_CONF = 0.40
YOLO_AREA_MIN = 0.10

DEBUG_OCR = False
MOSTRAR_AREA_PLACA = True

OCR_CONF_MIN = 0.30

PLACA_CONFIRMA_OCUPACAO = True
CONFIRMAR_OCUPADA = 2

VAGAS_RELATIVAS = [
    {"vaga": "A-1", "coords_rel": (40, 250, 255, 920),  "area_placa_rel": (40, 680, 255, 920),
     "id_estacionamento": "AepP9oc0sK1ZOeboKUVM", "id_vaga_firestore": "NlnwGiPmSbW4Jd6GrhdG"},
    {"vaga": "A-2", "coords_rel": (275, 250, 490, 920), "area_placa_rel": (275, 680, 490, 920),
     "id_estacionamento": "AepP9oc0sK1ZOeboKUVM", "id_vaga_firestore": "S68iDXJ6YEOHecx7TIjA"},
    {"vaga": "A-3", "coords_rel": (510, 250, 725, 920), "area_placa_rel": (510, 680, 725, 920),
     "id_estacionamento": "AepP9oc0sK1ZOeboKUVM", "id_vaga_firestore": "idRFtsgV1kQG2WCirMR8"},
    {"vaga": "A-4", "coords_rel": (745, 250, 960, 920), "area_placa_rel": (745, 680, 960, 920),
     "id_estacionamento": "AepP9oc0sK1ZOeboKUVM", "id_vaga_firestore": "in5QEeGVmN89FlFzNn28"},
]

# _ARQ_CALIBRACAO = os.path.join(os.path.dirname(__file__), "vagas.json")
# if os.path.exists(_ARQ_CALIBRACAO):
#     with open(_ARQ_CALIBRACAO) as _f:
#         _cal = json.load(_f)
#     for _v in VAGAS_RELATIVAS:
#         if _v["vaga"] in _cal:
#             _v["coords_rel"] = tuple(_cal[_v["vaga"]]["coords_rel"])
#             _v["area_placa_rel"] = tuple(_cal[_v["vaga"]]["area_placa_rel"])

ID_ESTACIONAMENTO = VAGAS_RELATIVAS[0]["id_estacionamento"]

def map_coords(coords_rel, width, height):
    """Converte coordenadas relativas (0-1000) em pixels da tela."""
    x1_rel, y1_rel, x2_rel, y2_rel = coords_rel
    return (
        int(x1_rel * width / 1000),
        int(y1_rel * height / 1000),
        int(x2_rel * width / 1000),
        int(y2_rel * height / 1000)
    )