import cv2
import easyocr
import numpy as np
from collections import deque
import time

# Configuração do OCR
reader = easyocr.Reader(['en'])

# Mapa das vagas com área específica para leitura da placa
vagas_map = [
    {
        "vaga": 1, 
        "coords": (100, 200, 300, 400),
        "area_placa": (100, 320, 300, 380)  # Área menor para a placa (parte inferior)
    },
    {
        "vaga": 2, 
        "coords": (350, 200, 550, 400),
        "area_placa": (350, 320, 550, 380)
    },
    {
        "vaga": 3, 
        "coords": (600, 200, 800, 400),
        "area_placa": (600, 320, 800, 380)
    }
]

# Inicializa estados - TODAS VAGAS COMEÇAM LIVRES
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
        "ultima_placa_lida": None
    })

cap = cv2.VideoCapture(0)

def carro_na_vaga(carro, vaga):
    x1, y1, x2, y2 = carro
    vx1, vy1, vx2, vy2 = vaga["coords"]
    
    # Calcula interseção
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
    
    # Pelo menos 40% de sobreposição
    return (inter_area / carro_area) > 0.4

def detectar_se_carro_parou(vaga, posicao_atual):
    """Detecta se o carro está parado baseado no histórico"""
    if posicao_atual is None:
        vaga["frames_sem_movimento"] = 0
        return False
    
    vaga["historico_posicoes"].append(posicao_atual)
    
    if len(vaga["historico_posicoes"]) < 5:
        return False
    
    # Calcula variação das últimas posições
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
    
    # Atualiza contador de frames sem movimento
    if deslocamento_medio < 2.0:  # Muito pouco movimento
        vaga["frames_sem_movimento"] += 1
    else:
        vaga["frames_sem_movimento"] = 0
    
    # Considera parado se ficou sem movimento por 10 frames
    return vaga["frames_sem_movimento"] >= 10

def preprocessar_placa(imagem_placa):
    """Pré-processamento para a área da placa"""
    if imagem_placa.size == 0:
        return None
    
    # Converte para cinza
    if len(imagem_placa.shape) == 3:
        cinza = cv2.cvtColor(imagem_placa, cv2.COLOR_BGR2GRAY)
    else:
        cinza = imagem_placa
    
    altura, largura = cinza.shape
    if altura < 20 or largura < 40:
        return None
    
    # Redimensiona
    nova_altura = 60
    fator = nova_altura / altura
    nova_largura = int(largura * fator)
    
    if nova_largura < 40:
        return None
        
    redimensionada = cv2.resize(cinza, (nova_largura, nova_altura))
    
    # Melhora contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrastada = clahe.apply(redimensionada)
    
    return contrastada

def ler_placa_da_vaga(vaga, frame):
    """Tenta ler a placa da área marcada"""
    tempo_atual = time.time()
    
    # Só tenta ler a cada 3 segundos
    if tempo_atual - vaga["ultima_leitura"] < 3.0:
        return None
    
    # Pega a área da placa
    ax1, ay1, ax2, ay2 = vaga["area_placa"]
    area_placa = frame[ay1:ay2, ax1:ax2]
    
    if area_placa.size == 0:
        return None
    
    # MOSTRA a área que está sendo analisada (apenas uma janela por vaga)
    cv2.imshow(f"Placa Vaga {vaga['vaga']}", area_placa)
    
    # Pré-processa
    placa_processada = preprocessar_placa(area_placa)
    
    if placa_processada is None:
        return None
    
    # Tenta OCR
    try:
        resultados = reader.readtext(
            placa_processada,
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
            
            # Filtro para placas
            if 6 <= len(texto_limpo) <= 8:
                tem_letras = any(c.isalpha() for c in texto_limpo)
                tem_numeros = any(c.isdigit() for c in texto_limpo)
                
                if tem_letras and tem_numeros:
                    placas_validas.append(texto_limpo)
        
        # Retorna a placa mais frequente
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

print("🚗 Sistema de Estacionamento - Versão Corrigida")
print("📍 Aguardando carros estacionarem...")

# Variável para controlar quais vagas estão ocupadas na SIMULAÇÃO
vagas_ocupadas_simulacao = [True, True, True]  # Todas ocupadas para teste

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_original = frame.copy()
    
    # SIMULAÇÃO: Lista de carros detectados 
    carros_detectados = []
    for i, vaga in enumerate(vagas_map):
        if vagas_ocupadas_simulacao[i]:  # Só adiciona carro se a vaga está ocupada na simulação
            vx1, vy1, vx2, vy2 = vaga["coords"]
            # Simula um carro em cada vaga (80% do tamanho da vaga)
            margem_w = int((vx2 - vx1) * 0.1)
            margem_h = int((vy2 - vy1) * 0.1)
            carro = (vx1 + margem_w, vy1 + margem_h, vx2 - margem_w, vy2 - margem_h)
            carros_detectados.append((carro, i))  # Guarda também o índice da vaga
        else:
            # Vaga livre na simulação
            carros_detectados.append((None, i))
    
    # PROCESSAMENTO DE CADA VAGA
    for i, vaga in enumerate(vagas_map):
        carro_info, vaga_index = carros_detectados[i]
        carro_atual = carro_info
        vaga_esta_ocupada = carro_atual is not None
        
        if vaga_esta_ocupada:
            # Detecta se o carro está parado
            carro_parado = detectar_se_carro_parou(vaga, carro_atual)
            
            # VERIFICA SE HOUVE MUDANÇA DE ESTADO
            if vaga_esta_ocupada and not vaga["ocupada"]:
                # NOVO CARRO ENTROU NA VAGA
                vaga["ocupada"] = True
                vaga["carro_parado"] = False
                vaga["placa_atual"] = None
                vaga["tempo_estacionado"] = time.time()
                vaga["historico_posicoes"].clear()
                vaga["frames_sem_movimento"] = 0
                print(f"🚗 CARRO ENTROU na Vaga {vaga['vaga']}")
            
            elif vaga_esta_ocupada and vaga["ocupada"]:
                # VAGA CONTINUA OCUPADA - verifica se parou
                if carro_parado and not vaga["carro_parado"]:
                    # CARRO PAROU AGORA
                    vaga["carro_parado"] = True
                    vaga["ultima_leitura"] = 0  # Reseta timer de leitura
                    print(f"🅿️  CARRO PAROU na Vaga {vaga['vaga']} - Lendo placa...")
                
                elif not carro_parado and vaga["carro_parado"]:
                    # CARRO VOLTOU A SE MEXER
                    vaga["carro_parado"] = False
                    print(f"🚗 CARRO EM MOVIMENTO na Vaga {vaga['vaga']}")
            
            # SE CARRO ESTÁ PARADO, TENTA LER PLACA
            if vaga["ocupada"] and vaga["carro_parado"]:
                placa = ler_placa_da_vaga(vaga, frame_original)
                if placa:
                    if vaga["placa_atual"] != placa:
                        vaga["placa_atual"] = placa
                        vaga["ultima_placa_lida"] = placa
                        print(f"✅ PLACA IDENTIFICADA - Vaga {vaga['vaga']}: {placa}")
        
        else:
            # VAGA LIVRE (sem carro detectado)
            if vaga["ocupada"]:
                # CARRO SAIU - LIMPA TUDO
                placa_antiga = vaga["placa_atual"]
                vaga["ocupada"] = False
                vaga["carro_parado"] = False
                vaga["placa_atual"] = None
                vaga["historico_posicoes"].clear()
                vaga["frames_sem_movimento"] = 0
                vaga["ultima_leitura"] = 0
                if placa_antiga:
                    print(f"🚗 CARRO SAIU - Placa {placa_antiga} saiu da Vaga {vaga['vaga']}")
                else:
                    print(f"🚗 CARRO SAIU da Vaga {vaga['vaga']}")
    
    # DESENHA A INTERFACE
    for vaga in vagas_map:
        vx1, vy1, vx2, vy2 = vaga["coords"]
        ax1, ay1, ax2, ay2 = vaga["area_placa"]
        
        # Cor da vaga baseada no status
        if not vaga["ocupada"]:
            cor_vaga = (0, 255, 0)  # Verde - livre
            status = "LIVRE"
            cor_texto = (255, 255, 255)
        elif vaga["carro_parado"] and vaga["placa_atual"]:
            cor_vaga = (0, 0, 255)  # Vermelho - ocupado com placa
            status = vaga["placa_atual"]
            cor_texto = (0, 255, 255)  # Amarelo
        elif vaga["carro_parado"]:
            cor_vaga = (0, 165, 255)  # Laranja - ocupado, lendo placa
            status = "LENDO PLACA"
            cor_texto = (255, 255, 255)
        else:
            cor_vaga = (255, 255, 0)  # Amarelo - carro em movimento
            status = "EM MOVIMENTO"
            cor_texto = (0, 0, 0)
        
        # Desenha vaga
        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), cor_vaga, 3)
        
        # Desenha área da placa (AZUL)
        cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (255, 0, 0), 2)
        cv2.putText(frame, "AREA PLACA", (ax1, ay1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Texto do status com fundo
        cv2.rectangle(frame, (vx1, vy1-50), (vx1+220, vy1-10), (0,0,0), -1)
        cv2.putText(frame, f"Vaga {vaga['vaga']}: {status}", 
                   (vx1+5, vy1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_texto, 1)
        
        # Indicador de movimento
        if vaga["carro_parado"]:
            cv2.putText(frame, "PARADO", (vx1, vy2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        elif vaga["ocupada"]:
            cv2.putText(frame, "MOVIMENTO", (vx1, vy2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
    # Informações gerais
    cv2.putText(frame, "ESTACIONAMENTO - Sistema Automatico", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, "Pressione 1,2,3 para simular entrada/saida", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, "Pressione ESPACO para leitura forçada", 
               (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    cv2.imshow("Sistema de Estacionamento", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset manual
        for vaga in vagas_map:
            vaga["placa_atual"] = None
        print("🔄 Todas as placas resetadas manualmente")
    elif key == ord(' '):
        # Força leitura em todas as vagas ocupadas
        for vaga in vagas_map:
            if vaga["ocupada"] and vaga["carro_parado"]:
                placa = ler_placa_da_vaga(vaga, frame_original)
                if placa:
                    vaga["placa_atual"] = placa
                    print(f"✅ Leitura forçada - Vaga {vaga['vaga']}: {placa}")
    elif key in [ord('1'), ord('2'), ord('3')]:
        # Simula entrada/saída de carros
        vaga_num = int(chr(key)) - 1
        if 0 <= vaga_num < len(vagas_ocupadas_simulacao):
            vagas_ocupadas_simulacao[vaga_num] = not vagas_ocupadas_simulacao[vaga_num]
            status = "ENTROU" if vagas_ocupadas_simulacao[vaga_num] else "SAIU"
            print(f"🎮 SIMULACAO: Carro {status} na Vaga {vaga_num + 1}")

cap.release()
cv2.destroyAllWindows()