import cv2
import easyocr
import numpy as np
from collections import deque
import time
import firebase_admin
from firebase_admin import credentials, firestore, messaging
from google.cloud.firestore_v1.base_query import FieldFilter
import torch

# --- CONFIGURAÇÃO FIREBASE ---
try:
    # Tenta carregar as credenciais do Firebase
    cred = credentials.Certificate("./firebase_key.json")
    firebase_admin.initialize_app(cred)
except ValueError:
    # Ignora se o app já foi inicializado (comum em ambientes de notebook)
    pass
except Exception as e:
    print(f"⚠️ Erro ao inicializar Firebase Admin SDK: {e}")
    print("Certifique-se de que as credenciais estão configuradas corretamente.")

db = firestore.client()

# --- CONFIGURAÇÃO OCR E VAGAS ---
reader = easyocr.Reader(['en', 'pt']) # Adicionando português para placas Mercosul

# Resolução desejada para a câmera (REDUZIDA para 800x450)
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 450

# Coordenadas relativas das vagas (AJUSTADAS com espaçamento de 50 entre vagas e aumento de 20 em cada vaga)
VAGAS_RELATIVAS = [
    # Vaga 1: Aumentada em 20 e com espaçamento
    {
        "vaga": "A1", 
        "coords_rel": (50, 300, 270, 950),  # Aumentada de 200 para 220 (200+20) na largura
        "area_placa_rel": (50, 700, 270, 950),
        "id_estacionamento": "OzBB12ZThIzzsD0JWZmI", 
        "id_vaga_firestore": "Ls6mqldsHpQ6Gdb2hZ5x"
    },
    # Vaga 2: Aumentada em 20 e com espaçamento de 50 da anterior
    {
        "vaga": "B1", 
        "coords_rel": (320, 300, 540, 950),  # 270 + 50 = 320, aumentada para 220 de largura
        "area_placa_rel": (320, 700, 540, 950),
        "id_estacionamento": "OzBB12ZThIzzsD0JWZmI",
        "id_vaga_firestore": "gv3ZxbC6OCldonAi6nhn"
    },
    # Vaga 3: Aumentada em 20 e com espaçamento de 50 da anterior
    {
        "vaga": "C1", 
        "coords_rel": (590, 300, 810, 950),  # 540 + 50 = 590, aumentada para 220 de largura
        "area_placa_rel": (590, 700, 810, 950),
        "id_estacionamento": "OzBB12ZThIzzsD0JWZmI",
        "id_vaga_firestore": "VJZpS5OHwTv3nRjZbF7r"
    },
    # Vaga 4: Aumentada em 20 e com espaçamento de 50 da anterior
    {
        "vaga": "D1", 
        "coords_rel": (860, 300, 950, 950),  # 810 + 50 = 860, ajustada para caber na tela
        "area_placa_rel": (860, 700, 950, 950),
        "id_estacionamento": "OzBB12ZThIzzsD0JWZmI",
        "id_vaga_firestore": "D1"
    }
]

# --- CONFIGURAÇÃO YOLO ---
def setup_yolo():
    """Configura o modelo YOLO para detecção de carros"""
    try:
        print("🚀 Carregando modelo YOLOv5...")
        # Carrega YOLOv5 pré-treinado
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # Filtra apenas para veículos: car (2), bus (5), truck (7)
        model.classes = [2, 5, 7]
        model.conf = 0.3  # Confiança mínima ajustada para ambiente de estacionamento
        model.iou = 0.4   # Intersection Over Union threshold
        print("✅ Modelo YOLOv5 carregado com sucesso!")
        return model
    except Exception as e:
        print(f"⚠️ Erro ao carregar YOLO: {e}")
        print("Instale as dependências: pip install torch torchvision")
        return None

# Inicializa o modelo YOLO
yolo_model = setup_yolo()

def map_coords(coords_rel, width, height):
    x1_rel, y1_rel, x2_rel, y2_rel = coords_rel
    
    x1 = int(x1_rel * width / 1000)
    y1 = int(y1_rel * height / 1000)
    x2 = int(x2_rel * width / 1000)
    y2 = int(y2_rel * height / 1000)
    
    return (x1, y1, x2, y2)

vagas_map = []
for vaga_rel in VAGAS_RELATIVAS:
    vaga_abs = vaga_rel.copy()
    vaga_abs["coords"] = map_coords(vaga_rel["coords_rel"], CAMERA_WIDTH, CAMERA_HEIGHT)
    vaga_abs["area_placa"] = map_coords(vaga_rel["area_placa_rel"], CAMERA_WIDTH, CAMERA_HEIGHT)
    vagas_map.append(vaga_abs)

for vaga in vagas_map:
    vaga.update({
        "placa_atual": None,
        "ocupada": False,
        "carro_parado": False,
        "historico_posicoes": deque(maxlen=10),
        "tempo_estacionado": 0,
        "ultima_leitura": 0,
        "frames_sem_movimento": 0,
        "carro_detectado_atual": None,
        "ultima_placa_lida": None,
        "notificacao_enviada": False,
        "confianca_placa_atual": 0.0,
        "ultima_deteccao_yolo": None,
        "frames_sem_deteccao": 0
    })

cap = cv2.VideoCapture(0)
# Tenta definir a resolução da câmera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# --- FUNÇÕES DE DETECÇÃO COM YOLO ---
def detectar_carros_yolo(frame):
    """
    Detecta carros usando YOLO
    Retorna lista de bounding boxes [x1, y1, x2, y2, confiança]
    """
    if yolo_model is None:
        return []
    
    try:
        # Realiza a detecção
        results = yolo_model(frame)
        
        # Processa os resultados
        detections = []
        detections_data = results.pandas().xyxy[0]
        
        for _, det in detections_data.iterrows():
            if det['confidence'] > 0.3:  # Confiança mínima
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                conf = det['confidence']
                detections.append((x1, y1, x2, y2, conf))
        
        return detections
    except Exception as e:
        print(f"Erro na detecção YOLO: {e}")
        return []

def non_max_suppression(boxes, threshold=0.5):
    """
    Remove bounding boxes sobrepostos (Non-Maximum Suppression)
    """
    if len(boxes) == 0:
        return []
    
    # Converte para numpy array
    boxes_array = np.array([b[:4] for b in boxes])  # Apenas coordenadas
    confidences = np.array([b[4] for b in boxes]) if len(boxes[0]) > 4 else np.ones(len(boxes))
    
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = confidences.argsort()[::-1]  # Ordena por confiança
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / areas[order[1:]]
        
        inds = np.where(overlap <= threshold)[0]
        order = order[inds + 1]
    
    # Retorna apenas as detecções mantidas
    return [boxes[i] for i in keep]

def mapear_carros_para_vagas(carros_detectados, vagas):
    """
    Mapeia cada carro detectado para sua vaga correspondente
    """
    carros_por_vaga = [None] * len(vagas)
    
    for carro in carros_detectados:
        x1, y1, x2, y2, conf = carro
        melhor_vaga_idx = -1
        melhor_overlap = 0.4  # Overlap mínimo
        
        for i, vaga in enumerate(vagas):
            if carro_na_vaga((x1, y1, x2, y2), vaga):
                # Calcula overlap específico
                overlap = calcular_overlap((x1, y1, x2, y2), vaga["coords"])
                if overlap > melhor_overlap:
                    melhor_overlap = overlap
                    melhor_vaga_idx = i
        
        if melhor_vaga_idx != -1:
            # Se já tem um carro na vaga, escolhe o com maior confiança
            if carros_por_vaga[melhor_vaga_idx] is None or conf > carros_por_vaga[melhor_vaga_idx][4]:
                carros_por_vaga[melhor_vaga_idx] = carro
    
    return carros_por_vaga

def calcular_overlap(carro, vaga_coords):
    """Calcula a porcentagem de overlap entre carro e vaga"""
    cx1, cy1, cx2, cy2 = carro
    vx1, vy1, vx2, vy2 = vaga_coords
    
    inter_x1 = max(cx1, vx1)
    inter_y1 = max(cy1, vy1)
    inter_x2 = min(cx2, vx2)
    inter_y2 = min(cy2, vy2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    carro_area = (cx2 - cx1) * (cy2 - cy1)
    
    if carro_area == 0:
        return 0
    
    return inter_area / carro_area

# --- FUNÇÕES DE UTILIDADE (MANTIDAS) ---
def carro_na_vaga(carro, vaga):
    x1, y1, x2, y2 = carro[:4]  # Pega apenas coordenadas
    vx1, vy1, vx2, vy2 = vaga["coords"]
    
    inter_x1 = max(x1, vx1)
    inter_y1 = max(y1, vy1)
    inter_x2 = min(x2, vx2)
    inter_y2 = min(y2, vy2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return False
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    carro_area = (x2 - x1) * (y2 - y1)
    
    if carro_area == 0:
        return False
    
    return (inter_area / carro_area) > 0.4

def detectar_se_carro_parou(vaga, posicao_atual):
    if posicao_atual is None:
        vaga["frames_sem_movimento"] = 0
        return False
    
    vaga["historico_posicoes"].append(posicao_atual)
    
    if len(vaga["historico_posicoes"]) < 5:
        return False
    
    ultimas_posicoes = list(vaga["historico_posicoes"])[-5:]
    deslocamentos = []
    
    for i in range(1, len(ultimas_posicoes)):
        x1, y1, x2, y2 = ultimas_posicoes[i-1]
        x1_curr, y1_curr, x2_curr, y2_curr = ultimas_posicoes[i]
        
        centro_anterior = ((x1 + x2) / 2, (y1 + y2) / 2)
        centro_atual = ((x1_curr + x2_curr) / 2, (y1_curr + y2_curr) / 2)
        
        deslocamento = np.sqrt((centro_atual[0] - centro_anterior[0])**2 + 
                              (centro_atual[1] - centro_anterior[1])**2)
        deslocamentos.append(deslocamento)
    
    deslocamento_medio = np.mean(deslocamentos) if deslocamentos else 100
    
    if deslocamento_medio < 2.0:
        vaga["frames_sem_movimento"] += 1
    else:
        vaga["frames_sem_movimento"] = 0
    
    return vaga["frames_sem_movimento"] >= 10

def preprocessar_placa(imagem_placa):
    if imagem_placa.size == 0:
        return None
    
    if len(imagem_placa.shape) == 3:
        cinza = cv2.cvtColor(imagem_placa, cv2.COLOR_BGR2GRAY)
    else:
        cinza = imagem_placa
    
    altura, largura = cinza.shape
    if altura < 20 or largura < 40:
        return None
    
    nova_altura = 60
    fator = nova_altura / altura
    nova_largura = int(largura * fator)
    
    if nova_largura < 40:
        return None
        
    redimensionada = cv2.resize(cinza, (nova_largura, nova_altura))
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrastada = clahe.apply(redimensionada)
    
    return contrastada

def ler_placa_da_vaga(vaga, frame):
    """
    Realiza a leitura da placa na área definida da vaga.
    """
    ax1, ay1, ax2, ay2 = vaga["area_placa"]
    area_placa = frame[ay1:ay2, ax1:ax2]
    
    if area_placa.size == 0:
        return None, 0.0
    
    try:
        resultados = reader.readtext(
            preprocessar_placa(area_placa),
            detail=1,
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            text_threshold=0.5,
            low_text=0.4
        )
        
        placas_candidatas = []
        
        for (bbox, texto, confianca) in resultados:
            texto_limpo = ''.join(c for c in texto.upper() if c.isalnum())
            
            if 6 <= len(texto_limpo) <= 8:
                tem_letras = any(c.isalpha() for c in texto_limpo)
                tem_numeros = any(c.isdigit() for c in texto_limpo)
                
                if tem_letras and tem_numeros:
                    placas_candidatas.append((texto_limpo, confianca))
        
        if placas_candidatas:
            melhor_placa, maior_confianca = max(placas_candidatas, key=lambda item: item[1])
            return melhor_placa, maior_confianca
                    
    except Exception as e:
        pass
    
    return None, 0.0

def disparar_evento_estacionamento(token_fcm_usuario: str, titulo: str, corpo: str, dados: dict = None):
    """
    Dispara o evento de estacionamento enviando uma mensagem de dados via FCM
    """
    if not token_fcm_usuario:
        print("❌ Token de dispositivo vazio/inválido")
        return False

    print(f"🚀 Disparando evento de estacionamento para token: {token_fcm_usuario[:10]}...")
    
    if dados:
        print(f"📦 Dados que serão enviados:")
        for key, value in dados.items():
            print(f"   {key}: {value}")

    try:
        sucesso = enviar_notificacao(
            token_dispositivo=token_fcm_usuario,
            titulo=titulo,
            corpo=corpo,
            dados=dados
        )
    except NameError:
        print("⚠️  Função 'enviar_notificacao' não definida. Pulando envio de notificação.")
        sucesso = False

    if sucesso:
        print("✅ Evento de estacionamento disparado com sucesso via FCM.")
    else:
        print("❌ Falha ao disparar evento de estacionamento via FCM.")
    
    return sucesso

def enviar_notificacao(token_dispositivo, titulo, corpo, dados=None):
    try:
        if not token_dispositivo:
            print("❌ Token de dispositivo vazio/inválido")
            return False

        print(f"📤 Enviando notificação para token: {token_dispositivo[:10]}...")

        safe_dados = {k: str(v) for k, v in (dados or {}).items()}

        message = messaging.Message(
            data={
                "title": titulo,
                "body": corpo,
                **safe_dados
            },
            token=token_dispositivo,
        )

        response = messaging.send(message)
        print(f"✅ Notificação enviada com sucesso: {response}")
        return True

    except messaging.UnregisteredError as e:
        print(f"❌ Token FCM inválido ou expirado: {e}")
        return False
    except ValueError as e:
        print(f"❌ Token FCM mal formatado: {e}")
        return False
    except firebase_admin.exceptions.FirebaseError as e:
        print(f"❌ Erro do Firebase: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro geral ao enviar notificação: {e}")
        return False

def remover_token_invalido(placa):
    try:
        veiculo_doc = db.collection('veiculo').document(placa).get()
        if not veiculo_doc.exists:
            return
            
        usuario_email = veiculo_doc.get('usuarioEmail')
        if not usuario_email:
            return
            
        usuarios_ref = db.collection('usuario').where(filter=FieldFilter('email', '==', usuario_email)).limit(1).get()
        
        if usuarios_ref:
            usuario_doc = usuarios_ref[0]
            usuario_ref = db.collection('usuario').document(usuario_doc.id)
            usuario_ref.update({'fcm_token': firestore.DELETE_FIELD})
            print(f"🗑️  Token FCM inválido removido para {usuario_email}")
            
    except Exception as e:
        print(f"Erro ao remover token inválido: {e}")

def buscar_token_motorista(placa):
    """Busca o token FCM do motorista pela placa do carro"""
    try:
        print(f"🔍 Buscando token para placa: {placa}")
        
        veiculo_doc = db.collection('veiculo').document(placa).get()
        if not veiculo_doc.exists:
            print(f"❌ Veículo com placa {placa} não encontrado")
            return None, None
        
        usuario_email = veiculo_doc.get('usuarioEmail')
        if not usuario_email:
            print(f"❌ Email do usuário não encontrado para placa {placa}")
            return None, None
            
        print(f"📧 Email do usuário: {usuario_email}")

        usuarios_ref = db.collection('usuario').where(filter=FieldFilter('email', '==', usuario_email)).limit(1).get()
        
        if not usuarios_ref:
            print(f"❌ Usuário com email {usuario_email} não encontrado")
            return None, None
            
        usuario_doc = usuarios_ref[0]
        usuario_data = usuario_doc.to_dict()
        
        if 'fcm_token' not in usuario_data:
            print(f"⚠️  Campo 'fcm_token' não encontrado no documento do usuário")
            return None, usuario_doc.id
            
        token = usuario_data.get('fcm_token')
        uid_motorista = usuario_doc.id
        
        if not token:
            print(f"❌ Token FCM vazio para usuário {usuario_email}")
            return None, uid_motorista
            
        print(f"✅ Token encontrado: {token[:10]}...")
        return token, uid_motorista
        
    except Exception as e:
        print(f"❌ Erro ao buscar token do motorista: {e}")
        return None, None

def buscar_token_admin(id_estacionamento):
    """Busca o token FCM do administrador pelo ID do estacionamento."""
    try:
        print(f"🔍 Buscando token admin para estacionamento: {id_estacionamento}")
        
        estacionamento_doc = db.collection('estacionamento').document(id_estacionamento).get()
        if not estacionamento_doc.exists:
            print(f"❌ Estacionamento {id_estacionamento} não encontrado")
            return None
            
        estacionamento_data = estacionamento_doc.to_dict()
        uid_admin = estacionamento_data.get('adminUid')
        
        if not uid_admin:
            print(f"❌ adminUid não encontrado no estacionamento")
            return None
            
        print(f"👤 Admin UID: {uid_admin}")
        
        usuario_doc = db.collection('usuario').document(uid_admin).get()
        if not usuario_doc.exists:
            print(f"❌ Usuário admin {uid_admin} não encontrado")
            return None
            
        usuario_data = usuario_doc.to_dict()
        token = usuario_data.get('fcm_token')
        
        if not token:
            print(f"⚠️  Admin {uid_admin} não possui token FCM")
            return None
            
        print(f"✅ Token admin encontrado: {token[:10]}...")
        return token
        
    except Exception as e:
        print(f"❌ Erro ao buscar token do administrador: {e}")
        return None

def atualizar_vaga_firestore(vaga, placa=None, ocupada=None, status=None):
    """
    Atualiza o estado da vaga no Firestore
    """
    try:
        vaga_ref = db.collection('vaga').document(vaga['id_vaga_firestore'])
        
        update_data = {}
        if placa is not None:
            update_data['placa_atual'] = placa
        if ocupada is not None:
            update_data['ocupada'] = ocupada
        if status is not None:
            update_data['status'] = status
        
        if not vaga_ref.get().exists:
            update_data['numero'] = vaga['vaga']
            update_data['id_estacionamento'] = vaga['id_estacionamento']
            update_data['ocupada'] = ocupada if ocupada is not None else False
            update_data['status'] = status if status is not None else 'LIVRE'
            
            vaga_ref.set(update_data, merge=True)
        
        if update_data:
            vaga_ref.set(update_data, merge=True)
            print(f"✅ Firestore: Vaga {vaga['vaga']} ATUALIZADA com sucesso.")
            return True
        return False
    except Exception as e:
        print(f"❌ Erro ao atualizar Firestore para Vaga {vaga['vaga']}: {e}")
        return False

def verificar_reserva(uid_motorista, id_vaga_firestore):
    """
    Verifica se o motorista tem reserva ativa para esta vaga
    """
    try:
        reserva_ref = db.collection('reserva')\
            .where(filter=FieldFilter('usuarioId', '==', uid_motorista))\
            .where(filter=FieldFilter('vagaId', '==', id_vaga_firestore))\
            .where(filter=FieldFilter('status', '==', 'ativa'))\
            .limit(1).get()
        return len(reserva_ref) > 0
    except Exception as e:
        print(f"Erro ao verificar reserva: {e}")
        return False

def tratar_placa(placa_bruta):
    """
    Trata a placa para garantir que tenha 7 caracteres
    """
    placa_limpa = ''.join(c for c in placa_bruta.upper() if c.isalnum())
    
    # 1. Tenta o formato Mercosul (LLLNLNN) ou antigo (LLLNNNN)
    if len(placa_limpa) == 7:
        letras = sum(1 for c in placa_limpa if c.isalpha())
        if letras == 3:
            return placa_limpa
            
    # 2. Tenta truncar para 7 caracteres se for maior
    if len(placa_limpa) > 7:
        placa_truncada = placa_limpa[:7]
        letras = sum(1 for c in placa_truncada if c.isalpha())
        if letras == 3:
            return placa_truncada
            
    return placa_bruta

def processar_vaga(vaga, carro_atual, frame_original):
    """Lógica central de detecção, leitura e comunicação com o Firebase."""
    
    # Extrai apenas coordenadas do carro (ignora confiança)
    carro_coords = carro_atual[:4] if carro_atual else None
    vaga_esta_ocupada = carro_atual is not None
    
    # Se houver carro detectado, registra o tempo da última detecção
    if carro_atual:
        vaga["ultima_deteccao_yolo"] = time.time()
        vaga["frames_sem_deteccao"] = 0
    else:
        vaga["frames_sem_deteccao"] += 1
    
    # 1. CARRO ENTROU NA VAGA
    if vaga_esta_ocupada and not vaga["ocupada"]:
        vaga["ocupada"] = True
        vaga["carro_parado"] = False
        vaga["placa_atual"] = None
        vaga["tempo_estacionado"] = time.time()
        vaga["historico_posicoes"].clear()
        vaga["frames_sem_movimento"] = 0
        vaga["notificacao_enviada"] = False
        vaga["confianca_placa_atual"] = 0.0
        vaga["ultima_deteccao_yolo"] = time.time()
        print(f"🚗 CARRO ENTROU na Vaga {vaga['vaga']}")
        
        atualizar_vaga_firestore(vaga, ocupada=True, status='OCUPADA')
        
    # 2. CARRO SAIU DA VAGA (considera se não detectou carro por 5 frames consecutivos)
    elif not vaga_esta_ocupada and vaga["ocupada"] and vaga["frames_sem_deteccao"] >= 5:
        placa_antiga = vaga["placa_atual"]
        vaga["ocupada"] = False
        vaga["carro_parado"] = False
        vaga["placa_atual"] = None
        vaga["historico_posicoes"].clear()
        vaga["frames_sem_movimento"] = 0
        vaga["ultima_leitura"] = 0
        vaga["notificacao_enviada"] = False
        vaga["confianca_placa_atual"] = 0.0
        vaga["frames_sem_deteccao"] = 0
        
        if placa_antiga:
            print(f"🚗 CARRO SAIU - Placa {placa_antiga} saiu da Vaga {vaga['vaga']}")
        else:
            print(f"🚗 CARRO SAIU da Vaga {vaga['vaga']}")
            
        atualizar_vaga_firestore(vaga, ocupada=False, status='LIVRE', placa=None)
        
    # 3. CARRO ESTÁ NA VAGA
    elif vaga_esta_ocupada and vaga["ocupada"]:
        carro_parado = detectar_se_carro_parou(vaga, carro_coords)
        
        if carro_parado and not vaga["carro_parado"]:
            vaga["carro_parado"] = True
            vaga["confianca_placa_atual"] = 0.0
            print(f"🅿️  CARRO PAROU na Vaga {vaga['vaga']} - Iniciando leitura...")
        
        elif not carro_parado and vaga["carro_parado"]:
            vaga["carro_parado"] = False
            print(f"🚗 CARRO EM MOVIMENTO na Vaga {vaga['vaga']}")
            
        if vaga["carro_parado"]:
            placa_lida, confianca = ler_placa_da_vaga(vaga, frame_original)
            
            if placa_lida and confianca > vaga.get("confianca_placa_atual", 0.0):
                placa_tratada = tratar_placa(placa_lida)
                
                vaga["placa_atual"] = placa_tratada
                vaga["confianca_placa_atual"] = confianca
                vaga["ultima_placa_lida"] = placa_lida
                
                print(f"✅ PLACA ATUALIZADA - Vaga {vaga['vaga']}: {placa_tratada} (Confiança: {confianca:.2f})")
                
                atualizar_vaga_firestore(vaga, placa=placa_tratada)
                
                if not vaga["notificacao_enviada"]:
                    print(f"🔔 Disparando notificações para placa {placa_tratada}")
                    enviar_notificacoes(vaga, placa_tratada)

def enviar_notificacoes(vaga, placa_tratada):
    """Função separada para envio de notificações"""
    try:
        token_motorista, uid_motorista = buscar_token_motorista(placa_tratada)
        token_admin = buscar_token_admin(vaga['id_estacionamento'])
        
        print(f"🔔 Tentando enviar notificações para placa {placa_tratada}")
        
        # Notificação para o Motorista
        notificacao_enviada = False
        
        if token_motorista and uid_motorista:
            titulo = "Seu veículo foi estacionado!"
            corpo = f"Sua placa {placa_tratada} foi identificada na Vaga {vaga['vaga']}."
            
            sucesso = disparar_evento_estacionamento(
                token_fcm_usuario=token_motorista,
                titulo=titulo,
                corpo=corpo,
                dados={
                    'vagaId': vaga['id_vaga_firestore'],
                    'placa': placa_tratada,
                    'tipo': 'estacionamento',
                    'usuarioId': uid_motorista
                }
            )
            
            if sucesso:
                notificacao_enviada = True
                vaga["notificacao_enviada"] = True
                print(f"✅ Notificação enviada com sucesso para o motorista")
            else:
                print(f"🔄 Token inválido detectado, removendo...")
                remover_token_invalido(placa_tratada)
        else:
            print(f"⚠️  Token do motorista não encontrado para {placa_tratada}")

        # Notificação para o Administrador
        if token_admin:
            print(f"👨‍💼 Enviando notificação para admin...")
            
            # Verifica se é uma reserva
            tem_reserva = False
            if uid_motorista:
                tem_reserva = verificar_reserva(uid_motorista, vaga['id_vaga_firestore'])
            
            if tem_reserva:
                titulo = "Reserva Confirmada"
                corpo = f"O veículo reservado {placa_tratada} entrou na Vaga {vaga['vaga']}."
                tipo_notificacao = 'reserva'
            else:
                titulo = "Veículo Estacionado"
                corpo = f"Veículo {placa_tratada} estacionou na Vaga {vaga['vaga']}."
                tipo_notificacao = 'estacionamento'
            
            sucesso_admin = disparar_evento_estacionamento(
                token_fcm_usuario=token_admin,
                titulo=titulo,
                corpo=corpo,
                dados={
                    'vagaId': vaga['id_vaga_firestore'],
                    'placa': placa_tratada,
                    'tipo': tipo_notificacao,
                    'usuarioId': uid_motorista or 'desconhecido',
                    'vagaNumero': vaga['vaga']
                }
            )
            
            if sucesso_admin:
                print(f"✅ Notificação enviada para o administrador")
        
        return notificacao_enviada
        
    except Exception as e:
        print(f"❌ Erro em enviar_notificacoes: {e}")
        return False

# Cores em BGR
COR_LIVRE = (0, 100, 0) # Verde Escuro
COR_OCUPADA_PARADA_PLACA = (0, 0, 150) # Vermelho Escuro
COR_OCUPADA_PARADA_LENDO = (0, 150, 255) # Laranja Escuro
COR_OCUPADA_MOVIMENTO = (0, 200, 200) # Amarelo Escuro
COR_TEXTO_CLARO = (255, 255, 255)
COR_TEXTO_ESCURO = (0, 0, 0)
COR_AREA_PLACA = (255, 100, 0) # Azul
COR_DETECCAO_YOLO = (255, 255, 0) # Ciano para detecções YOLO

print("🚗 Sistema de Estacionamento - Versão com YOLO Automático")
print(f"🎥 Resolução da Câmera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
print("📍 Aguardando carros estacionarem...")

# Verifica se YOLO foi carregado
if yolo_model is None:
    print("⚠️  YOLO não carregado. O sistema usará detecção manual.")
    print("💡 Para habilitar YOLO, instale: pip install torch torchvision")
    usar_yolo = False
else:
    usar_yolo = True
    print("✅ YOLO está ativo e pronto para detecção automática")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler frame da câmera.")
        break
    
    frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
    frame_original = frame.copy()
    
    # *** DETECÇÃO AUTOMÁTICA COM YOLO ***
    if usar_yolo:
        # Detecta carros usando YOLO
        carros_detectados = detectar_carros_yolo(frame)
        
        # Aplica Non-Maximum Suppression para remover duplicatas
        carros_detectados = non_max_suppression(carros_detectados, threshold=0.3)
        
        # Mapeia carros para vagas
        carros_por_vaga = mapear_carros_para_vagas(carros_detectados, vagas_map)
        
        # Desenha detecções YOLO no frame (apenas para debug/visualização)
        for carro in carros_detectados:
            x1, y1, x2, y2, conf = carro
            cv2.rectangle(frame, (x1, y1), (x2, y2), COR_DETECCAO_YOLO, 2)
            cv2.putText(frame, f"Carro: {conf:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_DETECCAO_YOLO, 1)
    else:
        # Modo manual (para debug/fallback)
        carros_por_vaga = [None] * len(vagas_map)
        # Aqui você pode adicionar lógica manual se necessário
    
    # Processa cada vaga
    for i, vaga in enumerate(vagas_map):
        processar_vaga(vaga, carros_por_vaga[i], frame_original)
    
    # Desenha as vagas e status
    for vaga in vagas_map:
        vx1, vy1, vx2, vy2 = vaga["coords"]
        ax1, ay1, ax2, ay2 = vaga["area_placa"]
        
        # Determina cor baseada no status
        if not vaga["ocupada"]:
            cor_vaga = COR_LIVRE
            status = "LIVRE"
            cor_texto = COR_TEXTO_CLARO
        elif vaga["carro_parado"] and vaga["placa_atual"]:
            cor_vaga = COR_OCUPADA_PARADA_PLACA
            status = f"{vaga['placa_atual']} ({vaga['confianca_placa_atual']:.2f})"
            cor_texto = COR_TEXTO_CLARO
        elif vaga["carro_parado"]:
            cor_vaga = COR_OCUPADA_PARADA_LENDO
            status = "LENDO PLACA"
            cor_texto = COR_TEXTO_CLARO
        else:
            cor_vaga = COR_OCUPADA_MOVIMENTO
            status = "EM MOVIMENTO"
            cor_texto = COR_TEXTO_ESCURO
        
        # Desenha o retângulo da vaga
        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), cor_vaga, 3)
        
        # Desenha o retângulo da área da placa
        cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), COR_AREA_PLACA, 2)
        cv2.putText(frame, "AREA PLACA", (ax1, ay1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COR_AREA_PLACA, 1)
        
        # Desenha o status da vaga
        cv2.rectangle(frame, (vx1, vy1-30), (vx1+220, vy1-5), (0,0,0), -1)
        cv2.putText(frame, f"Vaga {vaga['vaga']}: {status}", 
                   (vx1+5, vy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_texto, 1)
    
    # Textos de cabeçalho
    cv2.rectangle(frame, (0, 0), (CAMERA_WIDTH, 90), (0,0,0), -1)
    
    status_yolo = "ATIVO" if usar_yolo else "DESATIVADO"
    cv2.putText(frame, f"ESTACIONAMENTO - YOLO {status_yolo}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COR_TEXTO_CLARO, 2)
    cv2.putText(frame, "Sistema automático - Detecção em tempo real", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_TEXTO_CLARO, 1)
    cv2.putText(frame, "Pressione 'q' para sair, 'r' para resetar", 
               (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_TEXTO_CLARO, 1)
    
    # Mostra contador de FPS/detecções
    frame_count += 1
    if frame_count % 30 == 0:  # A cada 30 frames
        deteccoes_ativas = sum(1 for vaga in vagas_map if vaga["ocupada"])
        cv2.putText(frame, f"Vagas ocupadas: {deteccoes_ativas}/{len(vagas_map)}", 
                   (CAMERA_WIDTH - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_TEXTO_CLARO, 1)
    
    cv2.imshow("Sistema de Estacionamento - YOLO", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        for vaga in vagas_map:
            vaga["placa_atual"] = None
            vaga["confianca_placa_atual"] = 0.0
            vaga["notificacao_enviada"] = False
            vaga["carro_parado"] = False
            vaga["ocupada"] = False
            vaga["frames_sem_deteccao"] = 0
            atualizar_vaga_firestore(vaga, placa=None, ocupada=False, status='LIVRE')
        print("🔄 Estado das vagas resetado.")
    elif key == ord('d'):  # Toggle para mostrar/ocultar detecções YOLO
        # Esta funcionalidade pode ser expandida se necessário
        pass

cap.release()
cv2.destroyAllWindows()

print("👋 Sistema finalizado.")