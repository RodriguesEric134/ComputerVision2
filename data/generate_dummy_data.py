import os
import numpy as np
from PIL import Image

def create_synthetic_image(category, size=(128, 128)):
    """
    Gera uma imagem sintética com padrões de cores representativos de cada classe.
    - at_risk_vegetation: Tons predominantemente verdes e alguns amarelos (floresta/seca).
    - smoke: Tons de cinza claro/branco sobrepostos em fundo florestal (fumaça).
    - burned_land: Tons escuros, carvão, preto e marrom queimado.
    """
    # Inicializa com ruído leve para simular textura natural
    img_data = np.random.randint(0, 15, (size[0], size[1], 3), dtype=np.uint8)
    
    if category == 'at_risk_vegetation':
        # Tons verdes e amarelos
        img_data[:, :, 1] = np.random.randint(120, 220, (size[0], size[1]), dtype=np.uint8) # Verde forte
        img_data[:, :, 0] = np.random.randint(40, 140, (size[0], size[1]), dtype=np.uint8)   # Um pouco de vermelho (amarelado)
        img_data[:, :, 2] = np.random.randint(10, 50, (size[0], size[1]), dtype=np.uint8)    # Baixo azul
        
    elif category == 'smoke':
        # Fundo verde escuro com nuvens/fumaça cinza-branca no centro
        img_data[:, :, 1] = np.random.randint(40, 100, (size[0], size[1]), dtype=np.uint8)
        img_data[:, :, 0] = np.random.randint(10, 50, (size[0], size[1]), dtype=np.uint8)
        
        # Cria uma fumaça circular centralizada de pixels cinza-esbranquiçados
        x, y = np.ogrid[:size[0], :size[1]]
        center_x, center_y = size[0] // 2, size[1] // 2
        radius = size[0] // 3
        mask = (x - center_x)**2 + (y - center_y)**2 < radius**2
        
        # Onde há máscara, substitui por cinza claro (fumaça)
        img_data[mask, 0] = np.random.randint(180, 240, np.sum(mask), dtype=np.uint8)
        img_data[mask, 1] = np.random.randint(180, 240, np.sum(mask), dtype=np.uint8)
        img_data[mask, 2] = np.random.randint(180, 240, np.sum(mask), dtype=np.uint8)
        
    elif category == 'burned_land':
        # Tons escuros de carvão e marrom seco
        img_data[:, :, 0] = np.random.randint(20, 70, (size[0], size[1]), dtype=np.uint8)   # Vermelho escuro
        img_data[:, :, 1] = np.random.randint(15, 50, (size[0], size[1]), dtype=np.uint8)   # Verde baixo (terra seca)
        img_data[:, :, 2] = np.random.randint(10, 35, (size[0], size[1]), dtype=np.uint8)   # Azul baixíssimo
        
    return Image.fromarray(img_data)

def generate_dataset(base_dir='data', splits=None):
    if splits is None:
        splits = {
            'train': 100, # 100 imagens por classe para treino rápido
            'val': 30,    # 30 imagens para validação
            'test': 30    # 30 imagens para teste
        }
        
    categories = ['at_risk_vegetation', 'smoke', 'burned_land']
    
    print(f"Iniciando a criação do dataset sintético em '{base_dir}'...")
    
    for split, count in splits.items():
        for category in categories:
            dir_path = os.path.join(base_dir, split, category)
            os.makedirs(dir_path, exist_ok=True)
            
            print(f"Gerando {count} imagens sintéticas para: {split}/{category}...")
            for i in range(count):
                img = create_synthetic_image(category)
                img.save(os.path.join(dir_path, f"{category}_{i:04d}.png"))
                
    print("Dataset sintético estruturado com absoluto sucesso!")

if __name__ == '__main__':
    # Define o diretório atual do script como base caso seja chamado diretamente
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generate_dataset(current_dir)
