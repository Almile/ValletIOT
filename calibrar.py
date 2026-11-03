import json
import cv2
from config import CAMERA_WIDTH, CAMERA_HEIGHT, VAGAS_RELATIVAS

def abrir_camera():
    for i in (0, 1, 2):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened() and cap.read()[0]:
            return cap
        cap.release()
    return None

def rel(x, y, w, h):
    return (round(x * 1000 / CAMERA_WIDTH), round(y * 1000 / CAMERA_HEIGHT),
            round((x + w) * 1000 / CAMERA_WIDTH), round((y + h) * 1000 / CAMERA_HEIGHT))

def main():
    cap = abrir_camera()
    if cap is None:
        print("Câmera não encontrada.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    for _ in range(10):          # descarta frames iniciais (exposição)
        cap.read()
    _, frame = cap.read()
    cap.release()
    frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

    resultado = {}
    for v in VAGAS_RELATIVAS:
        nome = v["vaga"]
        print(f"[{nome}] desenhe a VAGA e aperte ENTER (c = pular)")
        x, y, w, h = cv2.selectROI("Calibracao", frame, showCrosshair=False)
        if w == 0:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f"[{nome}] desenhe a AREA DA PLACA dentro da vaga e aperte ENTER")
        px, py, pw, ph = cv2.selectROI("Calibracao", frame, showCrosshair=False)
        if pw == 0:              # sem placa: usa o terço inferior da vaga
            px, py, pw, ph = x, y + int(h * 0.65), w, int(h * 0.35)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 100, 0), 1)
        resultado[nome] = {"coords_rel": rel(x, y, w, h), "area_placa_rel": rel(px, py, pw, ph)}

    with open("vagas.json", "w") as f:
        json.dump(resultado, f, indent=2)
    cv2.destroyAllWindows()
    print("Salvo em vagas.json")

if __name__ == "__main__":
    main()