Aqui vai um **README.md completo, organizado e explicativo** para esse projeto de reconhecimento de placas em vagas de estacionamento.

---

# 🚗 Sistema de Estacionamento Inteligente com OCR (EasyOCR + OpenCV)

Este projeto detecta vagas de estacionamento, identifica carros, verifica se estão parados e realiza a **leitura automática da placa** utilizando **visão computacional e OCR**.

Ideal para protótipos de sistemas automatizados de estacionamento, TCCs e demonstrações em tempo real.

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
├── main.py               # Código principal do sistema
├── requirements.txt      # Dependências do projeto
└── README.md             # Documentação
```

---

## 📦 Instalação

### ✅ 1. Clone o projeto (ou copie os arquivos)

```bash
git clone https://github.com/almile/ValletIOT.git
cd ValletIOT
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

---

## ✅ Resultados Esperados no Console

```
🚗 CARRO ENTROU na Vaga 1
🅿️  CARRO PAROU na Vaga 1 - Lendo placa...
✅ PLACA IDENTIFICADA - Vaga 1: ABC1D23
🚗 CARRO SAIU - Placa ABC1D23 saiu da Vaga 1
```
