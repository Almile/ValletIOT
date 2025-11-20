import cv2
import easyocr
import numpy as np
from collections import deque
import time
import firebase_admin
from firebase_admin import credentials, firestore, messaging
from google.cloud.firestore_v1.base_query import FieldFilter

# --- CONFIGURAÇÃO FIREBASE ---
try:
    cred = credentials.Certificate("./firebase_key.json")
    firebase_admin.initialize_app(cred)
except ValueError:
    pass
except Exception as e:
    print(f"⚠️ Erro ao inicializar Firebase Admin SDK: {e}")
    print("Certifique-se de que as credenciais estão configuradas corretamente.")

db = firestore.client()

# --- CONFIGURAÇÃO OCR E VAGAS ---
reader = easyocr.Reader(['en'])

# Resolução desejada para a câmera
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

VAGAS_RELATIVAS = [
    {
        "vaga": 1, 
        "coords_rel": (100, 200, 350, 600),
        "area_placa_rel": (100, 400, 350, 600),
         "id_estacionamento": "OzBB12ZThIzzsD0JWZmI", 
        "id_vaga_firestore": "A1"
    },
    {
        "vaga": 2, 
        "coords_rel": (400, 200, 650, 600),
        "area_placa_rel": (400, 400, 650, 600),
        "id_estacionamento": "OzBB12ZThIzzsD0JWZmI",
        "id_vaga_firestore": "B1"
    },
    {
        "vaga": 3, 
        "coords_rel": (700, 200, 950, 600),
        "area_placa_rel": (700, 400, 950, 600),
        "id_estacionamento": "OzBB12ZThIzzsD0JWZmI",
        "id_vaga_firestore": "C1"
    }
]

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
        "notificacao_enviada": False
    })

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# --- FUNÇÕES DE UTILIDADE (MANTIDAS) ---
def carro_na_vaga(carro, vaga):
    x1, y1, x2, y2 = carro
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
    tempo_atual = time.time()
    
    if tempo_atual - vaga["ultima_leitura"] < 3.0:
        return None
    
    ax1, ay1, ax2, ay2 = vaga["area_placa"]
    area_placa = frame[ay1:ay2, ax1:ax2]
    
    if area_placa.size == 0:
        return None
    
    try:
        resultados = reader.readtext(
            preprocessar_placa(area_placa),
            detail=0,
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            batch_size=1,
            text_threshold=0.4,
            low_text=0.3
        )
        
        placas_validas = []
        for texto in resultados:
            texto_limpo = ''.join(c for c in texto.upper() if c.isalnum())
            
            if 6 <= len(texto_limpo) <= 8:
                tem_letras = any(c.isalpha() for c in texto_limpo)
                tem_numeros = any(c.isdigit() for c in texto_limpo)
                
                if tem_letras and tem_numeros:
                    placas_validas.append(texto_limpo)
        
        if placas_validas:
            from collections import Counter
            contador = Counter(placas_validas)
            melhor_placa, count = contador.most_common(1)[0]
            
            vaga["ultima_leitura"] = tempo_atual
            return melhor_placa
                    
    except Exception as e:
        print(f"Erro OCR na vaga {vaga['vaga']}: {e}")
    
    vaga["ultima_leitura"] = tempo_atual
    return None

# --- FUNÇÕES FIREBASE E NOTIFICAÇÃO (ADAPTADAS) ---

def enviar_notificacao(token_dispositivo, titulo, corpo, dados=None):
    """Envia notificação via FCM para um token específico."""
    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title=titulo,
                body=corpo,
            ),
            data=dados,
            token=token_dispositivo,
        )
        response = messaging.send(message)
        print(f"✅ Notificação enviada com sucesso: {response}")
    except Exception as e:
        print(f"❌ Erro ao enviar notificação: {e}")

def buscar_token_motorista(placa):
    """Busca o token FCM do motorista pela placa do carro (Coleção 'veiculo')."""
    try:
        veiculo_doc = db.collection('veiculo').document(placa).get()
        if not veiculo_doc.exists:
            return None, None
        
        usuario_email = veiculo_doc.get('usuarioEmail')
        
        usuario_ref = db.collection('usuario').where('email', '==', usuario_email).limit(1).get()
        if not usuario_ref:
            return None, None
            
        usuario_doc = usuario_ref[0]
        uid_motorista = usuario_doc.id
        
        return usuario_doc.get('fcm_token'), uid_motorista
        
    except Exception as e:
        print(f"Erro ao buscar token do motorista: {e}")
        return None, None

def buscar_token_admin(id_estacionamento):
    """Busca o token FCM do administrador pelo ID do estacionamento."""
    try:
        estacionamento_doc = db.collection('estacionamento').document(id_estacionamento).get()
        if estacionamento_doc.exists:
            uid_admin = estacionamento_doc.get('adminUid')
            
            usuario_doc = db.collection('usuario').document(uid_admin).get()
            if usuario_doc.exists:
                return usuario_doc.get('fcm_token')
        
        return None
    except Exception as e:
        print(f"Erro ao buscar token do administrador: {e}")
        return None

def atualizar_vaga_firestore(vaga, placa=None, ocupada=None, status=None):
    """
    Atualiza o estado da vaga no Firestore (Coleção 'vaga' no nível raiz).
    Usa set(merge=True) para criar o documento se não existir (evita erro 404).
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
            print(f"✅ Firestore: Vaga {vaga['vaga']} CRIADA/ATUALIZADA com sucesso.")
            return True
        
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
    Verifica se o motorista (UID) tem reserva ativa para esta vaga (id_vaga_firestore).
    CORREÇÃO: Busca por 'usuarioId' e usa len() para verificar o resultado.
    """
    try:
        # Assumindo que a reserva está na coleção 'reserva'
        # Filtra por usuarioId e vagaId
        reserva_ref = db.collection('reserva')\
            .where('usuarioId', '==', uid_motorista)\
            .where('vagaId', '==', id_vaga_firestore)\
            .where('status', '==', 'ativa')\
            .limit(1).get()
        # CORREÇÃO: Usa len() para verificar se a lista de resultados não está vazia
        return len(reserva_ref) > 0
    except Exception as e:
        print(f"Erro ao verificar reserva: {e}")
        return False

# --- FUNÇÃO PRINCIPAL DE PROCESSAMENTO (MANTIDA) ---

def processar_vaga(vaga, carro_atual, frame_original):
    """Lógica central de detecção, leitura e comunicação com o Firebase."""
    
    vaga_esta_ocupada = carro_atual is not None
    
    # 1. CARRO ENTROU NA VAGA
    if vaga_esta_ocupada and not vaga["ocupada"]:
        vaga["ocupada"] = True
        vaga["carro_parado"] = False
        vaga["placa_atual"] = None
        vaga["tempo_estacionado"] = time.time()
        vaga["historico_posicoes"].clear()
        vaga["frames_sem_movimento"] = 0
        vaga["notificacao_enviada"] = False
        print(f"🚗 CARRO ENTROU na Vaga {vaga['vaga']}")
        
        # Atualiza Firestore: Ocupada
        atualizar_vaga_firestore(vaga, ocupada=True, status='OCUPADA')
        
    # 2. CARRO SAIU DA VAGA
    elif not vaga_esta_ocupada and vaga["ocupada"]:
        placa_antiga = vaga["placa_atual"]
        vaga["ocupada"] = False
        vaga["carro_parado"] = False
        vaga["placa_atual"] = None
        vaga["historico_posicoes"].clear()
        vaga["frames_sem_movimento"] = 0
        vaga["ultima_leitura"] = 0
        vaga["notificacao_enviada"] = False
        
        if placa_antiga:
            print(f"🚗 CARRO SAIU - Placa {placa_antiga} saiu da Vaga {vaga['vaga']}")
        else:
            print(f"🚗 CARRO SAIU da Vaga {vaga['vaga']}")
            
        # Atualiza Firestore: Livre
        atualizar_vaga_firestore(vaga, placa=None, ocupada=False, status='LIVRE')
        
    # 3. CARRO ESTÁ NA VAGA
    elif vaga_esta_ocupada and vaga["ocupada"]:
        carro_parado = detectar_se_carro_parou(vaga, carro_atual)
        
        if carro_parado and not vaga["carro_parado"]:
            vaga["carro_parado"] = True
            vaga["ultima_leitura"] = 0
            print(f"🅿️  CARRO PAROU na Vaga {vaga['vaga']} - Lendo placa...")
        
        elif not carro_parado and vaga["carro_parado"]:
            vaga["carro_parado"] = False
            print(f"🚗 CARRO EM MOVIMENTO na Vaga {vaga['vaga']}")
            
        if vaga["carro_parado"]:
            placa = ler_placa_da_vaga(vaga, frame_original)
            
            if placa and vaga["placa_atual"] != placa:
                vaga["placa_atual"] = placa
                vaga["ultima_placa_lida"] = placa
                print(f"✅ PLACA IDENTIFICADA - Vaga {vaga['vaga']}: {placa}")
                
                atualizar_vaga_firestore(vaga, placa=placa)
                
                if not vaga["notificacao_enviada"]:
                    token_motorista, uid_motorista = buscar_token_motorista(placa)
                    token_admin = buscar_token_admin(vaga['id_estacionamento'])
                    
                    # Notificação para o Motorista
                    if token_motorista:
                        titulo = "Seu veículo foi estacionado!"
                        corpo = f"Sua placa {placa} foi identificada na Vaga {vaga['vaga']}."
                        enviar_notificacao(token_motorista, titulo, corpo, {'vaga': str(vaga['vaga']), 'placa': placa})
                    
                    # Notificação para o Administrador (se for reserva)
                    if token_admin and uid_motorista and verificar_reserva(uid_motorista, vaga['id_vaga_firestore']):
                        titulo = "Reserva Confirmada"
                        corpo = f"O veículo reservado entrou na Vaga {vaga['vaga']}."
                        enviar_notificacao(token_admin, titulo, corpo, {'vaga': str(vaga['vaga']), 'placa': placa, 'tipo': 'reserva'})
                        
                    vaga["notificacao_enviada"] = True


# --- FUNÇÃO DE RELEITURA MANUAL (MANTIDA) ---

def releitura_manual(vaga_id):
    """Simula a releitura forçada da vaga após saída manual do app."""
    print(f"🔄 Releitura manual solicitada para Vaga {vaga_id}")
    
    vaga = next((v for v in vagas_map if v["vaga"] == vaga_id), None)
    if not vaga:
        print(f"Vaga {vaga_id} não encontrada.")
        return
        
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame para releitura.")
        return
        
    placa = ler_placa_da_vaga(vaga, frame)
    
    if placa:
        print(f"⚠️ PLACA AINDA PRESENTE - Vaga {vaga_id}: {placa}")
        
        token_motorista, uid_motorista = buscar_token_motorista(placa)
        token_admin = buscar_token_admin(vaga['id_estacionamento'])
        
        if token_admin:
            titulo = "ALERTA: Saída Manual Incorreta"
            corpo = f"O veículo de placa {placa} ainda está na Vaga {vaga_id} após a saída manual no app."
            enviar_notificacao(token_admin, titulo, corpo, {'vaga': str(vaga_id), 'placa': placa, 'tipo': 'alerta'})
            
        atualizar_vaga_firestore(vaga, placa=placa, ocupada=True, status='OCUPADA')
        
    else:
        print(f"✅ Vaga {vaga_id} confirmada como LIVRE.")
        atualizar_vaga_firestore(vaga, placa=None, ocupada=False, status='LIVRE')


# --- LOOP PRINCIPAL (MANTIDO) ---

print("🚗 Sistema de Estacionamento - Versão Final Corrigida V2")
print(f"🎥 Resolução da Câmera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
print("📍 Aguardando carros estacionarem...")

vagas_ocupadas_simulacao = [False, False, False] 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler frame da câmera.")
        break
    
    frame_original = frame.copy()
    
    carros_detectados = []
    for i, vaga in enumerate(vagas_map):
        if vagas_ocupadas_simulacao[i]:
            vx1, vy1, vx2, vy2 = vaga["coords"]
            margem_w = int((vx2 - vx1) * 0.1)
            margem_h = int((vy2 - vy1) * 0.1)
            carro = (vx1 + margem_w, vy1 + margem_h, vx2 - margem_w, vy2 - margem_h)
            carros_detectados.append(carro)
        else:
            carros_detectados.append(None)
    
    for i, vaga in enumerate(vagas_map):
        processar_vaga(vaga, carros_detectados[i], frame_original)
        
    for vaga in vagas_map:
        vx1, vy1, vx2, vy2 = vaga["coords"]
        ax1, ay1, ax2, ay2 = vaga["area_placa"]
        
        if not vaga["ocupada"]:
            cor_vaga = (0, 255, 0)
            status = "LIVRE"
            cor_texto = (255, 255, 255)
        elif vaga["carro_parado"] and vaga["placa_atual"]:
            cor_vaga = (0, 0, 255)
            status = vaga["placa_atual"]
            cor_texto = (0, 255, 255)
        elif vaga["carro_parado"]:
            cor_vaga = (0, 165, 255)
            status = "LENDO PLACA"
            cor_texto = (255, 255, 255)
        else:
            cor_vaga = (255, 255, 0)
            status = "EM MOVIMENTO"
            cor_texto = (0, 0, 0)
        
        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), cor_vaga, 3)
        
        cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (255, 0, 0), 2)
        cv2.putText(frame, "AREA PLACA", (ax1, ay1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        cv2.rectangle(frame, (vx1, vy1-50), (vx1+220, vy1-10), (0,0,0), -1)
        cv2.putText(frame, f"Vaga {vaga['vaga']}: {status}", 
                   (vx1+5, vy1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_texto, 1)
        
        if vaga["carro_parado"]:
            cv2.putText(frame, "PARADO", (vx1, vy2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        elif vaga["ocupada"]:
            cv2.putText(frame, "MOVIMENTO", (vx1, vy2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
    cv2.putText(frame, "ESTACIONAMENTO - Sistema Automatico", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, "Pressione 1,2,3 para simular entrada/saida", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, "Pressione ESPACO para releitura manual", 
               (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    cv2.imshow("Sistema de Estacionamento", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        for vaga in vagas_map:
            vaga["placa_atual"] = None
        print("🔄 Todas as placas resetadas manualmente")
    elif key == ord(' '):
        releitura_manual(1)
    elif key in [ord('1'), ord('2'), ord('3')]:
        vaga_num = int(chr(key))
        vaga_index = vaga_num - 1
        if 0 <= vaga_index < len(vagas_ocupadas_simulacao):
            vagas_ocupadas_simulacao[vaga_index] = not vagas_ocupadas_simulacao[vaga_index]
            status = "ENTROU" if vagas_ocupadas_simulacao[vaga_index] else "SAIU"
            print(f"🎮 SIMULACAO: Carro {status} na Vaga {vaga_num}")

cap.release()
cv2.destroyAllWindows()