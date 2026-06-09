## 1. Visão Geral do Projeto

O Data Burn é uma plataforma inteligente e de alta performance voltada ao monitoramento, previsão e resposta a focos de incêndios florestais usando dados espectrais e espaciais de satélites/drones. 

Implementamos um pipeline completo em **PyTorch** capaz de receber imagens de satélite e classificá-las em três classes para a conservação e gestão de risco florestal:

1. **Fumaça**: Detecção de colunas de fumaça ativa.
2. **Área Queimada**: Mapeamento de solo escuro e devastado pós-incêndio.
3. **Vegetação em Risco / Floresta**: Cobertura florestal saudável sob monitoramento preventivo.

A solução está diretamente alinhada com as Metas de Desenvolvimento Sustentável da ONU: **ODS 13 (Ação Climática)**, **ODS 9 (Indústria, Inovação e Infraestrutura)** e **ODS 2 (Preservação de Biomas e Terras Agrícolas)**.

### Integrantes
* **Bernardo Rocha** - RM99209
* **Eric Carvalho** - RM550249
* **Manoella Waideman** - RM98906
* **Renato Ichikawa** - RM99242
* **Victor Hugo Andrade** - RM550996

> [!IMPORTANT]
> ### Vídeo de Apresentação do Projeto
> Assista ao pitch de apresentação e à demonstração funcional da plataforma **Data Burn**:
> 
> [![Assista ao vídeo](https://img.youtube.com/vi/NpL4_zlQqYU/0.jpg)](https://www.youtube.com/watch?v=NpL4_zlQqYU)
> 
> 🔗 **[Clique aqui para assistir no YouTube](https://www.youtube.com/watch?v=NpL4_zlQqYU)**
> 
> 💡 *Nota: O PowerPoint/apresentação utilizado no vídeo está disponível na pasta [slide/](file:///d:/GS1SEM4ANO/ComputerVision/slide).*

---

## 2. Diferenciais Acadêmicos e Engenharia de IA

### A. Arquiteturas Desenvolvidas 100% Do Zero

No processo desenvolvemos duas arquiteturas autorais em PyTorch sob o arquivo `src/models.py`:

*   **`FireNet_Lite` (Sequencial Leve)**: CNN clássica com blocos sequenciais `Conv2d` -> `BatchNorm` -> `ReLU` -> `MaxPool2d`. Possui alta velocidade de treinamento e é excelente para inferências rápidas em borda ou hardware de baixo custo.

*   **`SpaceFire_DeepCNN` (Profunda com Blocos Residuais do Zero)**: Rede robusta de alta performance inspirada na arquitetura ResNet. Desenvolvemos o módulo

**`ResidualBlock`** do zero, permitindo conexões residuais de atalho (*shortcuts*) para evitar o desaparecimento de gradiente e forçar a convergência estável em tarefas complexas de textura espectral, ultrapassando facilmente a meta mínima de **88% de acurácia**.

### B. Gestão Acadêmica de Overfitting e Underfitting

Identificamos os seguintes desafios clássicos em imagens de satélite de média/alta resolução e os mitigamos ativamente:

1.  **Overfitting (Decorar Fundos Verdes/Uniforme)**: 
    *   *Mitigação*: Aplicação de **Data Augmentation espacial** severo no carregador (`src/dataset.py`). Como imagens aéreas são invariantes à orientação da gravidade, implementamos `RandomHorizontalFlip(p=0.5)`, `RandomVerticalFlip(p=0.5)` e `RandomRotation(degrees=45)`.

    *   *Mitigação*: Inclusão de **Dropout de 40% a 50%** nas camadas totalmente conectadas das redes para forçar o aprendizado distribuído.
2.  **Instabilidade de Gradiente (Treinamento Lento / Divergência)**:
    *   *Mitigação*: Uso de **Batch Normalization (BatchNorm2d)** após cada operação de convolução nas duas redes. O BatchNorm estabiliza a distribuição das ativações internas, permitindo convergir em menos épocas e tolerar taxas de aprendizado estáveis (`learning_rate=0.001`).

### C. Filtro de Confiança e Out-of-Distribution (OOD)
Implementamos um limiar de confiança de **70%** tanto no dashboard Streamlit quanto na API FastAPI. Imagens que não pertencem ao domínio do problema (por exemplo, fotos de animais, pessoas ou objetos não relacionados) ou que possuem baixa assinatura espectral de risco são automaticamente classificadas como **"Desconhecido / Anômalo"**, prevenindo falsos positivos perigosos na detecção de incêndios.

---

## 3. Estrutura de Pastas do Repositório

```text
ComputerVision/
├── checkpoints/                # Pasta gerada pelo treino para salvar pesos (.pth) e histórico (.json)
├── data/                       # Diretório reservado para imagens
│   ├── download_real_data.py   # Script para baixar e limpar o dataset real do Hugging Face
│   └── generate_dummy_data.py  # Script gerador de dataset sintético (para testes rápidos)
├── imagens-teste/              # Pasta contendo imagens e testes de predição
│   ├── imagens usadas/         # Imagens utilizadas nos testes de predição
│   └── testes/                 # Prints de testes de predição dos modelos
├── reports/                    # Relatórios técnicos, imagens de matrizes e gráficos de avaliação
├── requirements.txt            # Dependências oficiais (PyTorch, Streamlit, etc.)
├── slide/                      # Pasta contendo a apresentação (PowerPoint/PDF) do projeto
├── src/                        # Código Fonte do Módulo de Visão Computacional
│   ├── dataset.py              # Pré-processamento, Data Augmentation e DataLoaders
│   ├── models.py               # Definição das duas CNNs customizadas (Lite e Deep)
│   ├── train.py                # Pipeline e loop unificado de treinamento
│   ├── evaluate.py             # Métricas, Curvas de Treino e Matriz de Confusão
│   └── app.py                  # Dashboard Web de Demonstração em Streamlit
└── README.md                   # Este guia oficial de entrega
```

### 📂 Detalhes Adicionais de Pastas
*   **`imagens-teste/`**: Contém subpastas organizadas para validação prática dos modelos:
    *   **`imagens usadas/`**: Pasta com as imagens que foram submetidas aos testes de predição.
    *   **`testes/`**: Pasta com capturas de tela (*prints*) demonstrando os resultados de predição dos modelos.
*   **`slide/`**: Contém a apresentação do projeto utilizada no vídeo de demonstração.

---

## 4. Como Configurar e Executar o Pipeline (Passo a Passo)

### Passo 1: Preparar o Ambiente
Recomendamos criar um ambiente virtual (venv) para isolar as dependências do projeto.
```bash
# 1. Criação do Ambiente Virtual
python -m venv venv

# 2. Ativação do Ambiente Virtual
# No Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# No Linux/macOS:
source venv/bin/activate

# 3. Instalar Dependências do Projeto
pip install -r requirements.txt
```

### Passo 2: Obter o Dataset (Real ou Sintético)

Você tem duas opções para alimentar o pipeline de treino e testes:

#### Opção A: Dataset de Imagens Reais (Recomendado para Banca e Produção)
Para treinar o modelo com imagens aéreas e de satélite reais de alta qualidade do dataset `EdBianchi/SmokeFire` (Hugging Face), execute o script downloader automatizado. Ele baixa os dados em formato Parquet, limpa imagens corrompidas e distribui exatamente 600 imagens para treino, 150 para validação e 150 para teste em cada classe:
```bash
python data/download_real_data.py
```

#### Opção B: Dataset Sintético (Para Testes Rápidos)
Se quiser apenas testar o código rapidamente sem realizar downloads externos:
```bash
python data/generate_dummy_data.py
```
*Isso gerará 480 imagens sintéticas de 128x128 pixels sob a pasta `data/` simulando visualmente assinaturas de fumaça, vegetação e terras queimadas.*

### Passo 3: Treinar os Modelos CNN
Você pode treinar tanto o modelo rápido (`lite`) quanto o profundo residual (`deep`).
```bash
# Treinar a rede profunda customizada (Recomendado para superar 88% de acurácia)
python src/train.py --model deep --epochs 10 --lr 0.001

# Treinar a rede sequencial leve
python src/train.py --model lite --epochs 10 --lr 0.001
```
*Os pesos ótimos serão gravados em `checkpoints/best_model_deep.pth` e o histórico de loss/acurácia em `checkpoints/history_deep.json`.*

### Passo 4: Avaliar e Gerar Relatórios Científicos
Após treinar o modelo, execute o script de validação de teste para gerar a Matriz de Confusão e as curvas de aprendizado:
```bash
# Avaliar a rede profunda
python src/evaluate.py --model deep

# Avaliar a rede leve
python src/evaluate.py --model lite

# Gerar o gráfico comparativo de acurácia se você treinou ambas as redes
python src/evaluate.py --compare
```
*Os resultados, gráficos e matrizes de calor em alta resolução serão gerados sob a pasta `reports/`.*

### Passo 5: Inicializar o Dashboard de Demonstração (Streamlit App)
Para demonstrar o projeto em uma interface gráfica premium, moderna e interativa:
```bash
streamlit run src/app.py
```
*A aplicação web abrirá no seu navegador. Nela, você poderá escolher qual CNN usar, subir qualquer imagem aérea e obter o resultado da classificação probabilística, conectada aos alertas de emergência e aos ODS correspondentes.*

### Passo 6: Inicializar o Microsserviço de API REST para RPA (FastAPI)
Para expor a CNN como um serviço para o robô de RPA da **Data Burn** consumir:
```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```
*   **Documentação Interativa (Swagger UI)**: Acesse **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** no seu navegador para testar requisições diretamente pela interface visual da API.
*   **JSON Contrato**: O robô de RPA receberá um payload estruturado contendo a classe predita (`predicted_class`), o score de confiança (`confidence`), a probabilidade detalhada por classe (`probabilities`) e o nível de alerta operacional de severidade com ação recomendada (`operational_alert`).

---

## 5. Check-list de Atendimento de Requisitos (Para a Banca Examinadora)

- [x] **Divisão Adequada de Dataset:** Separado em Train/Val/Test com proporção 70/15/15 estruturado em pastas separadas.
- [x] **CNNs do Zero:** Duas arquiteturas autorais (`FireNet_Lite` e `SpaceFire_DeepCNN`) desenvolvidas puramente em PyTorch sem importar pacotes pré-treinados.
- [x] **Acurácia > 88%:** A arquitetura residual profunda alcança excelente poder de generalização, atingindo o patamar acadêmico de acurácia de referência no teste.
- [x] **Métricas Completas:** Plota curvas de aprendizado (Loss/Accuracy), exibe relatório de classificação detalhado e gera gráficos premium de Matriz de Confusão.
- [x] **Demonstração Streamlit:** Interface gráfica funcional e elegante para o upload e inferência em tempo real.
- [x] **Integração de RPA via API REST:** API REST robusta em FastAPI documentada no Swagger, retornando payloads JSON detalhados para ações automatizadas do robô Data Burn.

