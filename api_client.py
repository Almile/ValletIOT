import requests
import threading
from queue import Queue
from config import API_URL

api_queue = Queue()

def worker_envio_api():
    """Thread em segundo plano para envio dos dados HTTP sem travar o feed da câmera."""
    while True:
        dados, imagem_bytes = api_queue.get()
        try:
            files = None
            if imagem_bytes:
                files = {'imagem_corte': ('placa.jpg', imagem_bytes, 'image/jpeg')}
            
            response = requests.post(API_URL, data=dados, files=files, timeout=5)
            if response.status_code == 200:
                print(f"🚀 [ENVIADO COM SUCESSO] Placa {dados['placa']} enviada para a API.")
            else:
                print(f"⚠️ Erro ao enviar para API: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Falha de rede ao conectar na API: {e}")
        finally:
            api_queue.task_done()

def iniciar_worker_api():
    """Inicia a thread consumidora de requisições."""
    thread = threading.Thread(target=worker_envio_api, daemon=True)
    thread.start()

def enviar_placa_para_api(payload, imagem_bytes):
    """Adiciona a leitura na fila de envio."""
    api_queue.put((payload, imagem_bytes))