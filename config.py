# --- CONFIGURAÇÕES DO SISTEMA ---
API_URL = "http://localhost:8000/camera/api/v1/eventos/placa-detectada"
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 450

# Coordenadas relativas das vagas (0-1000)
VAGAS_RELATIVAS = [
    {
        "vaga": "A1", 
        "coords_rel": (50, 300, 270, 950),
        "area_placa_rel": (50, 700, 270, 950),
        "id_estacionamento": "perS63z9MTjKZWRASKt3", 
        "id_vaga_firestore": "zQ5eg0oG72MGDa3BUwT2"
    },
    {
        "vaga": "B1", 
        "coords_rel": (320, 300, 540, 950),
        "area_placa_rel": (320, 700, 540, 950),
        "id_estacionamento": "perS63z9MTjKZWRASKt3",
        "id_vaga_firestore": "g0rVHsuSaWrpbkQGGwM1"
    },
    {
        "vaga": "C1", 
        "coords_rel": (590, 300, 810, 950),
        "area_placa_rel": (590, 700, 810, 950),
        "id_estacionamento": "perS63z9MTjKZWRASKt3",
        "id_vaga_firestore": "G63wnmu77VTYm4tJq9i2"
    },
    {
        "vaga": "D1", 
        "coords_rel": (860, 300, 950, 950),
        "area_placa_rel": (860, 700, 950, 950),
        "id_estacionamento": "perS63z9MTjKZWRASKt3",
        "id_vaga_firestore": "QenNjq5bEZKloWEqNIdD"
    }
]

def map_coords(coords_rel, width, height):
    """Converte coordenadas relativas (0-1000) em pixels da tela."""
    x1_rel, y1_rel, x2_rel, y2_rel = coords_rel
    return (
        int(x1_rel * width / 1000),
        int(y1_rel * height / 1000),
        int(x2_rel * width / 1000),
        int(y2_rel * height / 1000)
    )