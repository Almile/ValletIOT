# ValletWeb - Versão IOT
<img align="right" src="https://github.com/user-attachments/assets/fee4e949-8803-41e0-b21e-f68f17017f75" alt="logo vallet" width="25%" />


<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&pause=1000&color=4CAF50&width=600&height=55&lines=Sistema+de+Gest%C3%A3o+de+Estacionamentos" alt="Typing SVG" /></a>

<p>
Seja bem-vindo ao projeto <b>Vallet</b>, uma aplicação que busca otimizar o processo de <b>reserva, monitoramento e administração de vagas de estacionamento,</b> beneficiando tanto motoristas quanto os administradores de estacionamentos.
</p>


# 🚗 Sistema de Estacionamento Inteligente com OCR (EasyOCR + OpenCV)

Este projeto detecta vagas de estacionamento, identifica carros, verifica se estão parados e realiza a **leitura automática da placa** utilizando **visão computacional e OCR**.

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



