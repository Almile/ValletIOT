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
# Ajuste: cada vaga aumentada em 20 e espaçamento de 50 entre elas
VAGAS_RELATIVAS = [
    # Vaga 1: Aumentada em 20 e com espaçamento
    {
        "vaga": "A1", 
        "coords_rel": (50, 300, 270, 950),  # Aumentada de 200 para 220 (200+20) na largura
        "area_placa_rel": (50, 700, 270, 950),
        "id_estacionamento": "perS63z9MTjKZWRASKt3", 
        "id_vaga_firestore": "zQ5eg0oG72MGDa3BUwT2"
    },
    # Vaga 2: Aumentada em 20 e com espaçamento de 50 da anterior
    {
        "vaga": "B1", 
        "coords_rel": (320, 300, 540, 950),  # 270 + 50 = 320, aumentada para 220 de largura
        "area_placa_rel": (320, 700, 540, 950),
        "id_estacionamento": "perS63z9MTjKZWRASKt3",
        "id_vaga_firestore": "g0rVHsuSaWrpbkQGGwM1"
    },
    # Vaga 3: Aumentada em 20 e com espaçamento de 50 da anterior
    {
        "vaga": "C1", 
        "coords_rel": (590, 300, 810, 950),  # 540 + 50 = 590, aumentada para 220 de largura
        "area_placa_rel": (590, 700, 810, 950),
        "id_estacionamento": "perS63z9MTjKZWRASKt3",
        "id_vaga_firestore": "G63wnmu77VTYm4tJq9i2"
    },
    # Vaga 4: Aumentada em 20 e com espaçamento de 50 da anterior
    {
        "vaga": "D1", 
        "coords_rel": (860, 300, 950, 950),  # 810 + 50 = 860, ajustada para caber na tela
        "area_placa_rel": (860, 700, 950, 950),
        "id_estacionamento": "perS63z9MTjKZWRASKt3",
        "id_vaga_firestore": "QenNjq5bEZKloWEqNIdD"
    }
]


def map_coords(coords_rel, width, height):
    x1_rel, y1_rel, x2_rel, y2_rel = coords_rel
    
    # A lógica original de mapeamento (dividir por 1000) está correta para coordenadas relativas
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
        "confianca_placa_atual": 0.0 # Novo campo para armazenar a confiança da placa atual
    })

cap = cv2.VideoCapture(0)
# Tenta definir a resolução da câmera
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
    """
    Realiza a leitura da placa na área definida da vaga.
    Retorna a placa lida e o nível de confiança (score) do EasyOCR.
    A restrição de tempo foi removida para permitir leitura contínua.
    """
    ax1, ay1, ax2, ay2 = vaga["area_placa"]
    area_placa = frame[ay1:ay2, ax1:ax2]
    
    if area_placa.size == 0:
        return None, 0.0
    
    try:
        # O EasyOCR retorna [bbox, texto, confianca]
        resultados = reader.readtext(
            preprocessar_placa(area_placa),
            detail=1, # Mudar para detail=1 para obter a confiança
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            # Removendo batch_size=1 para evitar overhead em leituras sequenciais
            text_threshold=0.5, # Aumentando um pouco o threshold para focar em leituras mais claras
            low_text=0.4 # Aumentando um pouco o low_text para ser mais tolerante a caracteres individuais
        )
        
        # Lista para armazenar placas válidas e seus scores
        placas_candidatas = []
        
        for (bbox, texto, confianca) in resultados:
            texto_limpo = ''.join(c for c in texto.upper() if c.isalnum())
            
            # A validação de formato estrita pode estar impedindo a leitura.
            # Voltando à validação original do usuário (6 a 8 caracteres, com letras e números)
            if 6 <= len(texto_limpo) <= 8:
                tem_letras = any(c.isalpha() for c in texto_limpo)
                tem_numeros = any(c.isdigit() for c in texto_limpo)
                
                if tem_letras and tem_numeros:
                    # Se a placa for lida, ela é uma candidata.
                    placas_candidatas.append((texto_limpo, confianca))


        
        if placas_candidatas:
            # Seleciona a placa com a maior confiança
            melhor_placa, maior_confianca = max(placas_candidatas, key=lambda item: item[1])
            
            return melhor_placa, maior_confianca
                    
    except Exception as e:
        # print(f"Erro OCR na vaga {vaga['vaga']}: {e}") # Comentado para evitar spam no console
        pass
    
    return None, 0.0


    # Importe a função enviar_notificacao e as dependências necessárias
# from seu_modulo_de_notificacao import enviar_notificacao

def disparar_evento_estacionamento(token_fcm_usuario: str, titulo: str, corpo: str, dados: dict = None):
    """
    Dispara o evento de estacionamento enviando uma mensagem de dados via FCM
    """
    if not token_fcm_usuario:
        print("❌ Token de dispositivo vazio/inválido")
        return False

    print(f"🚀 Disparando evento de estacionamento para token: {token_fcm_usuario[:10]}...")
    
    # 🔥 LOG DOS DADOS QUE ESTÃO SENDO ENVIADOS
    if dados:
        print(f"📦 Dados que serão enviados:")
        for key, value in dados.items():
            print(f"   {key}: {value}")

    # A função enviar_notificacao deve ser definida ou importada
    # Como não está definida no código original, vamos assumir que ela existe
    # e apenas logar a chamada para evitar erro de NameError.
    # Se o usuário precisar da funcionalidade, ele deve garantir que a função esteja disponível.
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

        # Firebase Admin SDK v7+ requires data to be strings
        safe_dados = {k: str(v) for k, v in (dados or {}).items()}

        # DATA ONLY MESSAGE
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
    """Busca o token FCM do motorista pela placa do carro (Coleção 'veiculo')."""
    try:
        print(f"🔍 Buscando token para placa: {placa}")
        
        # Busca o veículo pela placa
        veiculo_doc = db.collection('veiculo').document(placa).get()
        if not veiculo_doc.exists:
            print(f"❌ Veículo com placa {placa} não encontrado")
            return None, None
        
        usuario_email = veiculo_doc.get('usuarioEmail')
        if not usuario_email:
            print(f"❌ Email do usuário não encontrado para placa {placa}")
            return None, None
            
        print(f"📧 Email do usuário: {usuario_email}")

        # Busca o usuário pelo email
        usuarios_ref = db.collection('usuario').where(filter=FieldFilter('email', '==', usuario_email)).limit(1).get()
        
        if not usuarios_ref:
            print(f"❌ Usuário com email {usuario_email} não encontrado")
            return None, None
            
        usuario_doc = usuarios_ref[0]
        usuario_data = usuario_doc.to_dict()
        
        # Verifica se o campo fcm_token existe no documento
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

def gerar_token_para_usuario_existente(usuario_email):
    """
    Gera um token FCM para um usuário existente que não tem token.
    Esta função deve ser chamada manualmente para usuários que já existem no sistema.
    """
    try:
        print(f"🔄 Gerando token FCM para usuário existente: {usuario_email}")
        
        # Busca o usuário pelo email
        usuarios_ref = db.collection('usuario').where(filter=FieldFilter('email', '==', usuario_email)).limit(1).get()
        
        if not usuarios_ref:
            print(f"❌ Usuário com email {usuario_email} não encontrado")
            return False
            
        usuario_doc = usuarios_ref[0]
        usuario_id = usuario_doc.id
        
        # Aqui você precisaria de uma forma de gerar um token FCM
        # Como não podemos gerar tokens do lado do servidor, esta é uma solução temporária
        # O token real deve ser gerado pelo app móvel
        
        print(f"⚠️  Token FCM não pode ser gerado automaticamente.")
        print(f"📱 O usuário {usuario_email} precisa fazer login no app móvel para gerar o token.")
        
        return False
        
    except Exception as e:
        print(f"❌ Erro ao gerar token para usuário existente: {e}")
        return False    

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
            .where(filter=FieldFilter('usuarioId', '==', uid_motorista))\
            .where(filter=FieldFilter('vagaId', '==', id_vaga_firestore))\
            .where(filter=FieldFilter('status', '==', 'ativa'))\
            .limit(1).get()
        # CORREÇÃO: Usa len() para verificar se a lista de resultados não está vazia
        return len(reserva_ref) > 0
    except Exception as e:
        print(f"Erro ao verificar reserva: {e}")
        return False

def verificar_token_admin():
    """Verifica se o admin tem token FCM configurado"""
    try:
        id_estacionamento = "OzBB12ZThIzzsD0JWZmI"  # Seu estacionamento ID
        
        estacionamento_doc = db.collection('estacionamento').document(id_estacionamento).get()
        if estacionamento_doc.exists:
            uid_admin = estacionamento_doc.get('adminUid')
            admin_email = estacionamento_doc.get('adminEmail')
            
            print(f"🔍 Verificando admin:")
            print(f"   UID: {uid_admin}")
            print(f"   Email: {admin_email}")
            
            usuario_doc = db.collection('usuario').document(uid_admin).get()
            if usuario_doc.exists:
                usuario_data = usuario_doc.to_dict()
                token = usuario_data.get('fcm_token')
                
                if token:
                    print(f"✅ Admin TEM token FCM: {token[:20]}...")
                else:
                    print(f"❌ Admin NÃO TEM token FCM")
                    print(f"💡 O admin precisa fazer login no app móvel para gerar o token")
                
                return token
            else:
                print(f"❌ Usuário admin não encontrado no Firestore")
        else:
            print(f"❌ Estacionamento não encontrado")
            
        return None
        
    except Exception as e:
        print(f"❌ Erro ao verificar token admin: {e}")
        return None

# Chame esta função uma vez no início para verificar
print("🔍 Verificando configuração do admin...")
verificar_token_admin()
# --- FUNÇÃO PRINCIPAL DE PROCESSAMENTO (MODIFICADA) ---

def tratar_placa(placa_bruta):
    """
    Trata a placa para garantir que tenha 7 caracteres, 3 letras e 4 caracteres restantes.
    Prioriza a placa bruta se já estiver no formato.
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
            
    # 3. Tenta preencher com 'X' ou '0' se for menor (último recurso)
    # Isso é arriscado, mas pode ser necessário se o OCR falhar.
    # Vamos apenas retornar a placa bruta se não for possível tratar com segurança.
    return placa_bruta

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
        vaga["confianca_placa_atual"] = 0.0 # Reseta a confiança
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
        vaga["confianca_placa_atual"] = 0.0 # Reseta a confiança
        
        if placa_antiga:
            print(f"🚗 CARRO SAIU - Placa {placa_antiga} saiu da Vaga {vaga['vaga']}")
        else:
            print(f"🚗 CARRO SAIU da Vaga {vaga['vaga']}")
            
        # Atualiza Firestore: Livre
        atualizar_vaga_firestore(vaga, ocupada=False, status='LIVRE', placa=None)
        
    # 3. CARRO ESTÁ NA VAGA
    elif vaga_esta_ocupada and vaga["ocupada"]:
        carro_parado = detectar_se_carro_parou(vaga, carro_atual)
        
        if carro_parado and not vaga["carro_parado"]:
            vaga["carro_parado"] = True
            vaga["confianca_placa_atual"] = 0.0 # Reseta a confiança ao parar
            print(f"🅿️  CARRO PAROU na Vaga {vaga['vaga']} - Iniciando leitura contínua...")
        
        elif not carro_parado and vaga["carro_parado"]:
            vaga["carro_parado"] = False
            print(f"🚗 CARRO EM MOVIMENTO na Vaga {vaga['vaga']}")
            
        # A leitura da placa deve ocorrer em todo frame que o carro estiver parado
        if vaga["carro_parado"]:
            placa_lida, confianca = ler_placa_da_vaga(vaga, frame_original)
            
            # Condição para aceitar a nova placa:
            # 1. Uma placa foi lida (placa_lida é diferente de None)
            # 2. A confiança da leitura é maior que a confiança da placa atual registrada
            
            # Se uma placa foi lida e a confiança é maior que a confiança atual
            if placa_lida and confianca > vaga.get("confianca_placa_atual", 0.0):
                
                # A placa lida é a nova melhor placa
                # Trata a placa antes de atualizar o Firestore
                placa_tratada = tratar_placa(placa_lida)
                
                vaga["placa_atual"] = placa_tratada
                vaga["confianca_placa_atual"] = confianca
                vaga["ultima_placa_lida"] = placa_lida # Mantém a placa lida bruta para debug
                
                print(f"✅ PLACA ATUALIZADA - Vaga {vaga['vaga']}: {placa_tratada} (Confiança: {confianca:.2f})")
                
                # Atualiza o Firestore com a melhor placa lida até o momento
                atualizar_vaga_firestore(vaga, placa=placa_tratada)
                
                # Envia notificação apenas na primeira vez que uma placa válida é lida
                if not vaga["notificacao_enviada"]:
                    enviar_notificacoes(vaga, placa_tratada)

    """Função separada para envio de notificações"""
    token_motorista, uid_motorista = buscar_token_motorista(placa_tratada)
    token_admin = buscar_token_admin(vaga['id_estacionamento'])
    
    print(f"🔔 Tentando enviar notificações para placa {placa_tratada}")
    
    # Notificação para o Motorista
    notificacao_enviada = False
    
    if token_motorista and uid_motorista:
        titulo = "Seu veículo foi estacionado!"
        corpo = f"Sua placa {placa_tratada} foi identificada na Vaga {vaga['vaga']}."
        
        # 🔥 CORREÇÃO: Envia o ID da vaga correto (id_vaga_firestore)
        sucesso = disparar_evento_estacionamento(
            token_fcm_usuario=token_motorista,
            titulo=titulo,
            corpo=corpo,
            dados={
                'vagaId': vaga['id_vaga_firestore'],  # 🔥 MUDOU DE 'vaga' PARA 'vagaId'
                'placa': placa_tratada,
                'tipo': 'estacionamento',
                'usuarioId': uid_motorista  # 🔥 ADICIONA O USUARIO ID
            }
        )
        
        if sucesso:
            notificacao_enviada = True
            vaga["notificacao_enviada"] = True
            print(f"✅ Notificação enviada com sucesso para o motorista")
            print(f"📦 Dados enviados: vagaId={vaga['id_vaga_firestore']}, usuarioId={uid_motorista}")
        else:
            print(f"🔄 Token inválido detectado, removendo...")
            remover_token_invalido(placa_tratada)
    else:
        print(f"⚠️  Token do motorista não encontrado para {placa_tratada}")

    # Notificação para o Administrador (se for reserva)
    if token_admin and uid_motorista:
        tem_reserva = verificar_reserva(uid_motorista, vaga['id_vaga_firestore'])
        if tem_reserva:
            titulo = "Reserva Confirmada"
            corpo = f"O veículo reservado entrou na Vaga {vaga['vaga']}."
            
            sucesso_admin = disparar_evento_estacionamento(
                token_fcm_usuario=token_admin,
                titulo=titulo,
                corpo=corpo,
                dados={
                    'vagaId': vaga['id_vaga_firestore'],  # 🔥 MESMA CORREÇÃO
                    'placa': placa_tratada, 
                    'tipo': 'reserva',
                    'usuarioId': uid_motorista
                }
            )
            
            if sucesso_admin:
                print(f"✅ Notificação de reserva enviada para o administrador")
    
    return notificacao_enviada

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
        vaga["confianca_placa_atual"] = 0.0
        print(f"🚗 CARRO ENTROU na Vaga {vaga['vaga']}")
        
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
        vaga["confianca_placa_atual"] = 0.0
        
        if placa_antiga:
            print(f"🚗 CARRO SAIU - Placa {placa_antiga} saiu da Vaga {vaga['vaga']}")
        else:
            print(f"🚗 CARRO SAIU da Vaga {vaga['vaga']}")
            
        atualizar_vaga_firestore(vaga, ocupada=False, status='LIVRE', placa=None)
        
    # 3. CARRO ESTÁ NA VAGA
    elif vaga_esta_ocupada and vaga["ocupada"]:
        carro_parado = detectar_se_carro_parou(vaga, carro_atual)
        
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
                
                # 🔥 **CORREÇÃO: Chamar enviar_notificacoes diretamente**
                if not vaga["notificacao_enviada"]:
                    print(f"🔔 Disparando notificações para placa {placa_tratada}")
                    enviar_notificacoes(vaga, placa_tratada)

def enviar_notificacoes(vaga, placa_tratada):
    """Função separada para envio de notificações"""
    try:
        token_motorista, uid_motorista = buscar_token_motorista(placa_tratada)
        token_admin = buscar_token_admin(vaga['id_estacionamento'])
        
        print(f"🔔 Tentando enviar notificações para placa {placa_tratada}")
        print(f"📞 Token Admin: {token_admin}")
        print(f"📞 Token Motorista: {token_motorista}")
        
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

        # 🔥 **CORREÇÃO CRÍTICA: Notificação para o Administrador**
        # Esta parte estava dentro do if do motorista, mas deve ser separada
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
            
            # 🔥 **ENVIA NOTIFICAÇÃO PARA O ADMIN SEMPRE QUE UM CARRO ESTACIONA**
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
                # 🔥 **SALVA NO FIRESTORE PARA O DJANGO BUSCAR**
                salvar_notificacao_admin_firestore(vaga, placa_tratada, uid_motorista, tipo_notificacao)
            else:
                print(f"❌ Falha ao enviar notificação para admin")
        
        return notificacao_enviada
        
    except Exception as e:
        print(f"❌ Erro em enviar_notificacoes: {e}")
        return False

def salvar_notificacao_admin_firestore(vaga, placa, usuario_id, tipo):
    """Salva notificação no Firestore para o Django sincronizar"""
    try:
        from datetime import datetime
        import firebase_admin
        from firebase_admin import firestore
        
        db = firestore.client()
        
        notificacao_data = {
            'vagaId': vaga['id_vaga_firestore'],
            'placa': placa,
            'usuarioId': usuario_id or 'desconhecido',
            'motivo': f"Veículo {placa} estacionou na Vaga {vaga['vaga']}",
            'timestamp': datetime.now(),
            'tipo': 'sistema_alerta_estacionamento',
            'vagaNumero': vaga['vaga'],
            'estacionamentoId': vaga['id_estacionamento']
        }
        
        # Adiciona à coleção notificacoes_admin
        db.collection('notificacoes_admin').add(notificacao_data)
        print(f"📝 Notificação salva no Firestore para admin: {placa} na Vaga {vaga['vaga']}")
        
    except Exception as e:
        print(f"❌ Erro ao salvar notificação no Firestore: {e}")


def verificar_usuarios_sem_token():
    """Verifica todos os usuários que não possuem token FCM"""
    try:
        print("🔍 Verificando usuários sem token FCM...")
        
        # Busca usuários onde o campo fcm_token não existe ou é nulo
        usuarios_ref = db.collection('usuario').get()
        
        usuarios_sem_token = []
        
        for usuario in usuarios_ref:
            usuario_data = usuario.to_dict()
            if 'fcm_token' not in usuario_data or not usuario_data.get('fcm_token'):
                usuarios_sem_token.append({
                    'id': usuario.id,
                    'email': usuario_data.get('email', 'N/A')
                })
        
        if usuarios_sem_token:
            print(f"⚠️  Encontrados {len(usuarios_sem_token)} usuários sem token FCM:")
            for usuario in usuarios_sem_token:
                print(f"   - {usuario['email']} (ID: {usuario['id']})")
        else:
            print("✅ Todos os usuários possuem token FCM")
            
        return usuarios_sem_token
        
    except Exception as e:
        print(f"❌ Erro ao verificar usuários sem token: {e}")
        return []

def verificar_e_atualizar_token(usuario_email, novo_token):
    """Verifica e atualiza o token FCM do usuário"""
    try:
        usuarios_ref = db.collection('usuario').where(filter=FieldFilter('email', '==', usuario_email)).limit(1).get()
        
        if usuarios_ref:
            usuario_doc = usuarios_ref[0]
            usuario_ref = db.collection('usuario').document(usuario_doc.id)
            usuario_ref.update({'fcm_token': novo_token})
            print(f"✅ Token FCM atualizado para {usuario_email}")
            return True
        return False
    except Exception as e:
        print(f"❌ Erro ao atualizar token: {e}")
        return False

# --- LOOP PRINCIPAL (MODIFICADO) ---
def teste_notificacao_admin():
    """Testa o envio de notificação para o admin"""
    print("🧪 TESTE MANUAL: Enviando notificação para admin...")
    
    # Dados de teste
    vaga_teste = {
        'vaga': 'A1',
        'id_vaga_firestore': 'zQ5eg0oG72MGDa3BUwT2',
        'id_estacionamento': 'perS63z9MTjKZWRASKt3'
    }
    
    enviar_notificacoes(vaga_teste, 'TEST123')
    
    print("✅ Teste concluído")

# Descomente a linha abaixo para testar manualmente
# teste_notificacao_admin()
# Cores em BGR (OpenCV usa BGR)
# Amarelo/Verde Escuro (Dark Yellow/Green)
COR_LIVRE = (0, 100, 0) # Verde Escuro
COR_OCUPADA_PARADA_PLACA = (0, 0, 150) # Vermelho Escuro (para placa lida)
COR_OCUPADA_PARADA_LENDO = (0, 150, 255) # Laranja Escuro (para lendo placa)
COR_OCUPADA_MOVIMENTO = (0, 200, 200) # Amarelo Escuro (para movimento)
COR_TEXTO_CLARO = (255, 255, 255)
COR_TEXTO_ESCURO = (0, 0, 0)
COR_AREA_PLACA = (255, 100, 0) # Azul um pouco mais escuro

print("🚗 Sistema de Estacionamento - Versão Modificada (Leitura Automática e Confiança)")
print(f"🎥 Resolução da Câmera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
print("📍 Aguardando carros estacionarem...")

# Verificar usuários sem token (opcional - pode ser comentado depois)
#verificar_usuarios_sem_token()

# Simulação de detecção de carro (deve ser substituída por um modelo de ML real)
vagas_ocupadas_simulacao = [False, False, False, False] 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler frame da câmera.")
        break
    
    # Centralização da Câmera:
    # O frame lido pode ser maior ou menor que o desejado.
    # Se for maior, cortamos o centro. Se for menor, redimensionamos.
    # Como definimos a resolução, vamos garantir que o frame tenha o tamanho correto.
    # Se a câmera não suportar a resolução, o frame pode ser diferente.
    # Vamos redimensionar para garantir o tamanho.
    frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
    frame_original = frame.copy()
    
    # *** IMPORTANTE: ESTA É UMA SIMULAÇÃO PARA TESTE. SUBSTITUIR POR DETECÇÃO REAL. ***
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
    # FIM DA SIMULAÇÃO
    
    for i, vaga in enumerate(vagas_map):
        processar_vaga(vaga, carros_detectados[i], frame_original)
        
    for vaga in vagas_map:
        vx1, vy1, vx2, vy2 = vaga["coords"]
        ax1, ay1, ax2, ay2 = vaga["area_placa"]
        
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
        
        # Desenha o status da vaga (fundo preto para melhor contraste)
        cv2.rectangle(frame, (vx1, vy1-30), (vx1+220, vy1-5), (0,0,0), -1) # Ajuste de altura para 30px
        cv2.putText(frame, f"Vaga {vaga['vaga']}: {status}", 
                   (vx1+5, vy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_texto, 1) # Ajuste de posição
        
        # Remove os textos "PARADO" e "MOVIMENTO" abaixo da vaga para simplificar a tela
        # if vaga["carro_parado"]:
        #     cv2.putText(frame, "PARADO", (vx1, vy2+20), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        # elif vaga["ocupada"]:
        #     cv2.putText(frame, "MOVIMENTO", (vx1, vy2+20), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
    # Textos de cabeçalho (fundo preto para melhor contraste)
    cv2.rectangle(frame, (0, 0), (CAMERA_WIDTH, 90), (0,0,0), -1)
    
    cv2.putText(frame, "ESTACIONAMENTO - Sistema Automatico", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COR_TEXTO_CLARO, 2)
    cv2.putText(frame, "Pressione 1,2,3,4 para simular entrada/saida", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_TEXTO_CLARO, 1)
    cv2.putText(frame, "Leitura automatica e simultanea de placas ativa", 
               (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_TEXTO_CLARO, 1)
    
    cv2.imshow("Sistema de Estacionamento", frame)
    
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
            atualizar_vaga_firestore(vaga, placa=None, ocupada=False, status='LIVRE')
        print("🔄 Estado das vagas resetado.")
    elif key == ord('1'):
        vagas_ocupadas_simulacao[0] = not vagas_ocupadas_simulacao[0]
        print(f"Simulação Vaga A1: {'OCUPADA' if vagas_ocupadas_simulacao[0] else 'LIVRE'}")
    elif key == ord('2'):
        vagas_ocupadas_simulacao[1] = not vagas_ocupadas_simulacao[1]
        print(f"Simulação Vaga B1: {'OCUPADA' if vagas_ocupadas_simulacao[1] else 'LIVRE'}")
    elif key == ord('3'):
        vagas_ocupadas_simulacao[2] = not vagas_ocupadas_simulacao[2]
        print(f"Simulação Vaga C1: {'OCUPADA' if vagas_ocupadas_simulacao[2] else 'LIVRE'}")
    elif key == ord('4'):
        vagas_ocupadas_simulacao[3] = not vagas_ocupadas_simulacao[3]
        print(f"Simulação Vaga 4: {'OCUPADA' if vagas_ocupadas_simulacao[3] else 'LIVRE'}")

cap.release()
cv2.destroyAllWindows()