<div align="center">
   <img height="200" alt="VALLET_logo" src="https://github.com/user-attachments/assets/c016879c-235f-423f-b553-08cb60cf02e3" />
   
   <a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=26&pause=10000&color=3b5042&width=600&height=55&lines=Sistema+de+Gest%C3%A3o+de+Estacionamentos" alt="Typing SVG" /></a>
</div>

<p>
Seja bem-vindo ao projeto <b>Vallet</b>, um sistema multiplataforma que busca otimizar e modernizar o processo de <b>reserva, monitoramento e administração de vagas de estacionamento,</b> beneficiando tanto motoristas quanto os administradores de estacionamentos.
</p>



> <h3> 🚗 Sistema de Estacionamento Inteligente com OCR (EasyOCR + OpenCV) </h3>Este projeto detecta vagas de estacionamento, identifica carros, verifica se estão parados e realiza a **leitura automática da placa** utilizando **visão computacional e OCR**.


<h3>Verifique outras partes do nosso projeto</h3>

 | Módulo / Repositório | Descrição | Link de Acesso |
| :--- | :--- | :--- |
| **ValletWeb** (Gestores) | Dashboards e controle de ocupação para gestores de estacionamento. | [Repositório Web](https://github.com/Pereira-Sah/ValletWeb) |
| **ValletApp** (Android) | Aplicativo móvel voltado para motoristas e usuários finais. | [Repositório Android](https://github.com/lariiscriis/ValletProjeto) |
| **ValletAPI** | API backend para conexão e gerenciamento de todo o ecossistema. | [Repositório API](https://github.com/Almile/ValletAPI) |
| **ValletIOT** | Sistema de leitura de placas (LPR) com uso de webcam. | [Repositório IOT](https://github.com/Almile/ValletIOT) |
| **ValletDesktop** | Sistema desktop voltado para os administradores gerais. | [Repositório Desktop](https://github.com/Almile/valletDesktop) |

<br>
<p align="center">
  <img src="https://img.shields.io/badge/Python-637365?style=for-the-badge&logo=kotlin&logoColor=white" />
  <img src="https://img.shields.io/badge/EasyOCR-3b5042?style=for-the-badge&logo=android&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-cbd9c5?style=for-the-badge&logo=firebase&logoColor=white" />
  <img src="https://img.shields.io/badge/Numpy-637365?style=for-the-badge&logo=google-cloud&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLO-3b5042?style=for-the-badge&logo=architecture&logoColor=white" />
</p>
<br>


---

## 🛠️ Tecnologias Utilizadas

| Biblioteca          | Função                                                      |
| ------------------- | ----------------------------------------------------------- |
| `OpenCV`            | Captura de vídeo, desenho na tela e manipulação de imagens  |
| `EasyOCR`           | Leitura automática das placas dos carros (OCR)              |
| `NumPy`             | Processamento matemático e vetorial                         |

---

## 📦 Instalação

### ✅ 1. Clone o projeto

```bash
git clone https://github.com/almile/ValletIOT.git
cd ValletIOT
```


### 1. Criar o ambiente virtual
```bash
python -m venv venv
```

### 2. Ativar o ambiente
```bash
# Windows: 
venv\Scripts\activate
# Linux/Mac: 
source venv/bin/activate
```

### ✅ 2. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## ▶️ Como Executar

No terminal, execute:

```bash
python main.py
```

O sistema abrirá a webcam e exibirá as vagas desenhadas na tela.

## 🧠 Como o Sistema Funciona

✔️ Cada vaga possui:

* Coordenadas da área da vaga
* Área onde a placa será lida
* Estado de ocupação, placa atual, histórico de movimentos, etc.

✔️ O ciclo do programa:

1. Captura imagem da webcam
2. Simula ou detecta carros nas vagas
3. Verifica se o carro está parado (sem se mover)
4. Quando parado → tenta ler a placa com EasyOCR
5. Exibe status na interface gráfica com OpenCV
6. Envia placa para api usada na comunicação com outros sistemas

---

## 📊 Status das Vagas (cores da interface)

| Cor         | Significado                         |
| ----------- | ----------------------------------- |
| 🟢 Verde    | Vaga Livre                          |
| 🟠 Laranja  | Carro parado, lendo placa           |
| 🔴 Vermelho | Placa detectada com sucesso         |



