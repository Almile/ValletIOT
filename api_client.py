import time
import threading
import requests
from queue import Queue
from config import (API_URL, API_URL_SAIDA, API_URL_ALERTA, API_URL_HEARTBEAT,
                    ID_ESTACIONAMENTO, HEARTBEAT_INTERVALO_S)

api_queue = Queue()

def worker_envio_api():
    """Envia os eventos em segundo plano, com até 3 tentativas cada."""
    while True:
        url, dados, imagem_bytes, descricao = api_queue.get()
        try:
            for tentativa in range(1, 4):
                try:
                    files = {'imagem_corte': ('placa.jpg', imagem_bytes, 'image/jpeg')} if imagem_bytes else None
                    r = requests.post(url, data=dados, files=files, timeout=5)
                    if r.status_code == 200:
                        print(f"🚀 [ENVIADO] {descricao}")
                        break
                    print(f"⚠️ API {r.status_code} ({descricao}): {r.text}")
                    if r.status_code < 500:
                        break
                except Exception as e:
                    print(f"❌ Falha de rede ({descricao}), tentativa {tentativa}/3: {e}")
                time.sleep(2)
        finally:
            api_queue.task_done()

def iniciar_worker_api():
    threading.Thread(target=worker_envio_api, daemon=True).start()


def enviar_placa_para_api(payload, imagem_bytes):
    api_queue.put((API_URL, payload, imagem_bytes, f"placa {payload['placa']}"))


def enviar_saida_para_api(payload):
    api_queue.put((API_URL_SAIDA, payload, None, f"saída {payload['placa']} vaga {payload['numero_vaga']}"))


def enviar_alerta_para_api(tipo, motivo, id_vaga=None, numero_vaga=None):
    dados = {"id_estacionamento": ID_ESTACIONAMENTO, "tipo": tipo, "motivo": motivo}
    if id_vaga:
        dados["id_vaga"] = id_vaga
    if numero_vaga:
        dados["numero_vaga"] = numero_vaga
    api_queue.put((API_URL_ALERTA, dados, None, f"alerta {tipo}"))


def iniciar_heartbeat(ultimo_frame_ts):
    """Função que retorna o instante do último frame processado.
    Se o loop principal travar, o heartbeat para e a API gera o sistema_erro."""
    def loop():
        while True:
            time.sleep(HEARTBEAT_INTERVALO_S)
            if time.time() - ultimo_frame_ts() < 20:
                try:
                    requests.post(API_URL_HEARTBEAT, data={"id_estacionamento": ID_ESTACIONAMENTO}, timeout=3)
                except Exception:
                    pass
    threading.Thread(target=loop, daemon=True).start()


def aguardar_envios(timeout=6):
    """Dá tempo da fila esvaziar antes de encerrar o programa."""
    fim = time.time() + timeout
    while api_queue.unfinished_tasks and time.time() < fim:
        time.sleep(0.2)