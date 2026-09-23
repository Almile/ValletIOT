# ValletWeb - Versão IOT
<img align="right" src="https://github.com/user-attachments/assets/fee4e949-8803-41e0-b21e-f68f17017f75" alt="logo vallet" width="25%" />


<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&pause=1000&color=4CAF50&width=600&height=55&lines=Sistema+de+Gest%C3%A3o+de+Estacionamentos" alt="Typing SVG" /></a>

<p>
Seja bem-vindo ao projeto <b>Vallet</b>, um aplicativo Android e plataforma web, seguindo a arquitetura MVVM (Model-View-ViewModel) e com integração total ao Firebase.
Ele busca otimizar o processo de <b>reserva, monitoramento e administração de vagas de estacionamento,</b> beneficiando tanto motoristas quanto os administradores de estacionamentos.
</p>

<p>Este repositório inclui as funções para gestor de estacionamento com dashboards e controle de vagas completo.<br> Caso queira acessar a versão para app android, acesse esse repositório: https://github.com/lariiscriis/ValletProjeto</p>

Caso queira acessar a versão para gestor de estacionamento com dashboards e controle de vagas, acesse esse repositório: https://github.com/Pereira-Sah/ValletWeb</p
<br><br>


# 🚗 Sistema de Estacionamento Inteligente com OCR (EasyOCR + OpenCV)

Este projeto detecta vagas de estacionamento, identifica carros, verifica se estão parados e realiza a **leitura automática da placa** utilizando **visão computacional e OCR**.

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca          | Função                                                      |
| ------------------- | ----------------------------------------------------------- |
| `OpenCV`            | Captura de vídeo, desenho na tela e manipulação de imagens  |
| `EasyOCR`           | Leitura automática das placas dos carros (OCR)              |
| `NumPy`             | Processamento matemático e vetorial                         |
| `collections.deque` | Histórico de posições para detectar movimento               |
| `time`              | Controle de tempo de estacionamento e intervalos de leitura |

---

## 📁 Estrutura do Projeto

```
📂 projeto-estacionamento/
├── estacionamento.py               # Código principal do sistema
├── requirements.txt      # Dependências do projeto
└── README.md             # Documentação
```

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
python estacionamento.py
```

O sistema abrirá a webcam e exibirá as vagas desenhadas na tela.

---

## 🎮 Controles da Simulação

| Tecla         | Função                                           |
| ------------- | ------------------------------------------------ |
| `1`, `2`, `3` | Simula entrada/saída de carro nas vagas 1, 2 e 3 |
| `ESPACO`      | Força leitura da placa manualmente               |
| `R`           | Reseta todas as placas identificadas             |
| `Q`           | Encerra o programa                               |

---

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

---

## 📊 Status das Vagas (cores da interface)

| Cor         | Significado                         |
| ----------- | ----------------------------------- |
| 🟢 Verde    | Vaga Livre                          |
| 🟡 Amarelo  | Carro em movimento                  |
| 🟠 Laranja  | Carro parado, lendo placa           |
| 🔴 Vermelho | Placa detectada com sucesso         |
| 🔵 Azul     | Área específica onde a placa é lida |

---

## 📷 Pré-processamento da Placa

Antes de enviar a imagem para OCR:

* Recorte da área inferior da vaga
* Conversão para escala de cinza
* Redimensionamento
* Aumento de contraste com `CLAHE`
* EasyOCR tenta ler somente letras e números


