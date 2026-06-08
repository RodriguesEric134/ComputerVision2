import streamlit as st
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

# Configurações de layout minimalista
st.set_page_config(
    page_title="Data Burn - Visão Computacional",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS Minimalista Premium (Sem bordas chamativas ou efeitos desnecessários)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d0d13;
        color: #e4e4e7;
    }
    
    .stApp {
        background-color: #0d0d13;
    }
    
    .main-title {
        color: #ffffff;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        margin-top: 10px;
        margin-bottom: 2px;
    }
    
    .sub-title {
        color: #71717a !important;
        font-size: 1rem;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    /* Divisórias minimalistas */
    .section-divider {
        border-top: 1px solid #27272a;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    /* Alertas de Resultado Planos e Limpos */
    .result-alert {
        padding: 16px;
        border-radius: 6px;
        font-weight: 500;
        margin-top: 20px;
        font-size: 0.95rem;
        letter-spacing: -0.1px;
    }
    
    .alert-danger {
        background-color: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #ef4444;
    }
    .alert-warning {
        background-color: rgba(249, 115, 22, 0.08);
        border: 1px solid rgba(249, 115, 22, 0.3);
        color: #f97316;
    }
    .alert-success {
        background-color: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #22c55e;
    }
    </style>
""", unsafe_allow_html=True)

# Cache do modelo para evitar recarregar a cada clique
@st.cache_resource
def load_pytorch_model(model_name):
    from models import FireNet_Lite, SpaceFire_DeepCNN
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = f"checkpoints/best_model_{model_name}.pth"
    
    if not os.path.exists(checkpoint_path):
        return None, None, f"Erro: Checkpoint '{checkpoint_path}' nao encontrado. Realize o treinamento do modelo primeiro."
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        classes = checkpoint['classes']
        
        if model_name == 'lite':
            model = FireNet_Lite(num_classes=len(classes)).to(device)
        else:
            model = SpaceFire_DeepCNN(num_classes=len(classes)).to(device)
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, classes, None
    except Exception as e:
        return None, None, f"Erro ao carregar o modelo: {str(e)}"

# Pipeline de inferência da imagem carregada
def predict_image(image, model, classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    img_t = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
        
    predicted_idx = np.argmax(probabilities)
    predicted_class = classes[predicted_idx]
    confidence = probabilities[predicted_idx]
    
    # Filtro de confiança mínima para detecção de fora-de-domínio (ex: cachorros, carros)
    if confidence < 0.70:
        predicted_class = 'unknown'
        
    return predicted_class, confidence, probabilities

# --- CONFIGURAÇÃO DA BARRA LATERAL (SIDEBAR) ---
st.sidebar.title("Data Burn")
st.sidebar.markdown("Configurações da IA")

selected_model = st.sidebar.selectbox(
    "Modelo de Classificação:",
    ["deep", "lite"],
    format_func=lambda x: "SpaceFire_DeepCNN (Residual)" if x == "deep" else "FireNet_Lite (Sequencial)"
)

# Carrega o modelo selecionado
model, classes, error = load_pytorch_model(selected_model)

if error:
    st.sidebar.error(error)
else:
    st.sidebar.success(f"Modelo carregado com sucesso em: {torch.device('cuda' if torch.cuda.is_available() else 'cpu').type.upper()}")

# --- CORPO PRINCIPAL ---
st.markdown('<div class="main-title">Data Burn</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Applied Computer Vision - Classificação de Imagens de Satélite</div>', unsafe_allow_html=True)

# Divisão de colunas limpas
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Selecao de Imagem")
    uploaded_file = st.file_uploader(
        "Carregue uma imagem de satelite ou foto aerea:",
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem carregada para processamento", use_column_width=True)
    else:
        st.info("Aguardando upload de imagem espectral...")

with col2:
    st.markdown("### Resultados do Diagnostico")
    
    if uploaded_file is not None and model is not None:
        with st.spinner("Processando assinatura espectral..."):
            predicted_class, confidence, probabilities = predict_image(image, model, classes)
            
        class_mapping = {
            'smoke': 'Fumaça',
            'burned_land': 'Área Queimada',
            'at_risk_vegetation': 'Vegetação Preservada',
            'unknown': 'Desconhecido / Fora do Domínio'
        }
        
        display_name = class_mapping.get(predicted_class, predicted_class)
        
        st.write(f"**Classe identificada:** {display_name}")
        st.write(f"**Índice de confiança:** {confidence * 100:.2f}%")
        
        # Alertas planos e minimalistas sem emojis
        if predicted_class == 'smoke':
            st.markdown('<div class="result-alert alert-danger">ALERTA CRITICO: Fumaca detectada. Risco ativo de incendio florestal no quadrante.</div>', unsafe_allow_html=True)
        elif predicted_class == 'burned_land':
            st.markdown('<div class="result-alert alert-warning">AVISO: Cicatriz de queimada severa detectada. Solo degradado.</div>', unsafe_allow_html=True)
        elif predicted_class == 'at_risk_vegetation':
            st.markdown('<div class="result-alert alert-success">STATUS NORMAL: Area sob monitoramento preventivo. Cobertura vegetal conservada.</div>', unsafe_allow_html=True)
        elif predicted_class == 'unknown':
            st.markdown('<div class="result-alert" style="background-color: rgba(113, 113, 122, 0.08); border: 1px solid rgba(113, 113, 122, 0.3); color: #71717a;">RETORNO ANOMALO: Imagem fora do dominio de monitoramento florestal ou de baixa confianca.</div>', unsafe_allow_html=True)
            
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.write("**Probabilidade por Classe:**")
        
        prob_df = pd.DataFrame({
            'Classe': [class_mapping.get(c, c) for c in classes],
            'Confiança (%)': [p * 100 for p in probabilities]
        })
        
        st.bar_chart(prob_df.set_index('Classe'))
        
    else:
        st.warning("Envie uma imagem para iniciar o processamento computacional.")
