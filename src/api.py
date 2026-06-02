from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os
import sys
# Adiciona o diretório deste script (src/) ao path do Python para evitar erros de importação com o Uvicorn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import FireNet_Lite, SpaceFire_DeepCNN


app = FastAPI(
    title="Data Burn - REST API de Visão Computacional",
    description="Interface de Integração de IA com o Robô de RPA da Data Burn para Classificação de Imagens de Satélite. Alinhada aos ODS 13, 9 e 2.",
    version="1.0.0"
)


# Caminhos dos Checkpoints
CHECKPOINTS = {
    'lite': 'checkpoints/best_model_lite.pth',
    'deep': 'checkpoints/best_model_deep.pth'
}

# Cache de modelos na memória para inferência instantânea
models_cache = {}

def load_model_instance(model_type: str):
    """
    Carrega o modelo do checkpoint correspondente se não estiver no cache.
    """
    if model_type in models_cache:
        return models_cache[model_type]
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = CHECKPOINTS.get(model_type)
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise HTTPException(
            status_code=503, 
            detail=f"Modelo '{model_type}' indisponível no servidor. Execute o treinamento 'train.py' primeiro."
        )
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        classes = checkpoint['classes']
        
        if model_type == 'lite':
            model = FireNet_Lite(num_classes=len(classes)).to(device)
        else:
            model = SpaceFire_DeepCNN(num_classes=len(classes)).to(device)
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Guarda no cache
        models_cache[model_type] = (model, classes, device)
        print(f" -> Modelo '{model_type.upper()}' carregado no cache com sucesso no dispositivo: {device}")
        return models_cache[model_type]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno ao carregar pesos do modelo '{model_type}': {str(e)}"
        )

# Mapeamento estético para as classes operacionais
ALERT_MAPPING = {
    'smoke': {
        'level': 'VERMELHO',
        'severity': 'CRÍTICO',
        'description': 'Assinatura térmica e visual de fumaça ativa detectada no quadrante. Incêndio em andamento.',
        'rpa_trigger_action': 'ATIVAR_BRIGADA_EMERGENCIA'
    },
    'burned_land': {
        'level': 'LARANJA',
        'severity': 'MÉDIO',
        'description': 'Cicatriz severa de queimada identificada. Solo degradado pós-fogo detectado.',
        'rpa_trigger_action': 'ORDEM_SERVICO_RESTAURACAO'
    },
    'at_risk_vegetation': {
        'level': 'VERDE',
        'severity': 'NENHUMA (ROTINA)',
        'description': 'Vegetação sob monitoramento preventivo. Sem assinaturas de combustão visual.',
        'rpa_trigger_action': 'CONTINUAR_MONITORAMENTO'
    },
    'unknown': {
        'level': 'CINZA',
        'severity': 'DESCONHECIDA',
        'description': 'Retorno espectral anômalo ou imagem fora do domínio de monitoramento (baixa confiança).',
        'rpa_trigger_action': 'REQUISITAR_ANALISE_MANUAL'
    }
}

@app.get("/")
def read_root():
    """
    Retorna o status geral da API e alinhamento com ODS.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return {
        "app": "Data Burn - Applied Computer Vision API",
        "status": "ONLINE",
        "device_used": device.type.upper(),
        "ods_alignment": [
            "ODS 13 - Ação Contra a Mudança Global do Clima",
            "ODS 9 - Indústria, Inovação e Infraestrutura",
            "ODS 2 - Fome Zero e Agricultura Sustentável"
        ],
        "endpoints": {
            "/predict": "POST - Envio de imagem espectral para classificação de risco de queimadas (multipart/form-data)"
        }
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Imagem aérea ou satélite (.png, .jpg, .jpeg)"),
    model_type: str = Query("deep", choices=["deep", "lite"], description="Selecione a CNN: 'deep' (Residual) ou 'lite' (Sequencial)")
):
    """
    Recebe uma imagem aérea ou de satélite, executa inferência através da CNN customizada 
    selecionada e retorna um JSON estruturado para tomada de decisão pelo robô de RPA.
    """
    # 1. Validação básica de arquivo
    content_type = file.content_type
    if content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400, 
            detail="Formato de arquivo inválido. A API aceita apenas imagens no formato PNG, JPG ou JPEG."
        )

    # 2. Carrega a instância do modelo do cache
    model, classes, device = load_model_instance(model_type)
    
    # 3. Leitura e decodificação da imagem usando Pillow
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Conversão de escala de cores (ex: RGBA -> RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Arquivo corrompido ou formato de imagem não decodificável."
        )
        
    # 4. Pré-processamento espectral idêntico ao pipeline de testes
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 5. Execução do Forward Pass na CNN
    try:
        with torch.no_grad():
            outputs = model(img_tensor)
            # Função Softmax para gerar a certeza/confiança probabilística exata
            probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
            
        predicted_idx = int(probabilities.argmax())
        predicted_class = classes[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Filtro de confiança mínima para detecção de fora-de-domínio (ex: cachorros, carros)
        if confidence < 0.70:
            predicted_class = 'unknown'
            
        # Estruturação do dicionário de probabilidades de todas as classes
        class_probabilities = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        
        # Mapeamento do alerta operacional correspondente
        alert_info = ALERT_MAPPING.get(predicted_class, {
            'level': 'DESCONHECIDO',
            'severity': 'NÃO_MAPEADA',
            'description': 'Retorno espectral anômalo. Necessita análise humana.',
            'rpa_trigger_action': 'REQUISITAR_ANALISE_MANUAL'
        })
        
        # 6. Retorno de JSON estruturado de acordo com as necessidades do RPA
        return {
            "success": True,
            "model_used": "SpaceFire_DeepCNN" if model_type == "deep" else "FireNet_Lite",
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in class_probabilities.items()},
            "operational_alert": alert_info
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Falha técnica durante a inferência na rede neural: {str(e)}"
        )
