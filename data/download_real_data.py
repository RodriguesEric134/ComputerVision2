import os
import urllib.request
import pandas as pd
import io
import shutil
from PIL import Image

# Configuração de limites para extração rápida
LIMITS = {
    'train': 600,       # Máximo de imagens por classe para treino
    'val': 150,         # Máximo de imagens por classe para validação
    'test': 150         # Máximo de imagens por classe para teste
}

URLS = {
    'train': "https://huggingface.co/datasets/EdBianchi/SmokeFire/resolve/main/data/train-00000-of-00001-d06229d7b437106c.parquet",
    'val': "https://huggingface.co/datasets/EdBianchi/SmokeFire/resolve/main/data/validation-00000-of-00001-d1cff2c5a713ab12.parquet",
    'test': "https://huggingface.co/datasets/EdBianchi/SmokeFire/resolve/main/data/test-00000-of-00001-42d26e110d541931.parquet"
}

# Mapeamento do dataset original para as pastas do projeto
CLASS_MAP = {
    0: 'burned_land',          # Fire -> burned_land
    1: 'at_risk_vegetation',   # Normal -> at_risk_vegetation
    2: 'smoke'                 # Smoke -> smoke
}

import time

def clean_directory(dir_path):
    """Remove todos os arquivos e subpastas de um diretório de forma robusta contra travamento de processos."""
    if os.path.exists(dir_path):
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                for i in range(5):
                    try:
                        os.chmod(file_path, 0o777)
                        os.remove(file_path)
                        break
                    except Exception:
                        time.sleep(0.2)
            for name in dirs:
                dir_to_remove = os.path.join(root, name)
                for i in range(5):
                    try:
                        shutil.rmtree(dir_to_remove)
                        break
                    except Exception:
                        time.sleep(0.2)
        try:
            shutil.rmtree(dir_path)
        except Exception:
            pass
    os.makedirs(dir_path, exist_ok=True)


def download_and_extract():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data') if 'data' in base_dir else base_dir
    data_dir = os.path.abspath(data_dir)
    
    print(f"Diretório de dados: {data_dir}")
    
    # Criar pastas e limpar as antigas
    for split in ['train', 'val', 'test']:
        for class_name in CLASS_MAP.values():
            folder_path = os.path.join(data_dir, split, class_name)
            clean_directory(folder_path)
            
    # Processar cada split
    for split, url in URLS.items():
        temp_parquet = os.path.join(data_dir, f"temp_{split}.parquet")
        print(f"\nBaixando split '{split}' do Hugging Face...")
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req) as response, open(temp_parquet, 'wb') as out_file:
                out_file.write(response.read())
            print("Download concluído! Processando imagens...")
            
            # Carrega o parquet
            df = pd.read_parquet(temp_parquet)
            
            # Contadores por classe
            counts = {c: 0 for c in CLASS_MAP.values()}
            limit = LIMITS[split]
            
            total_saved = 0
            for idx, row in df.iterrows():
                label_idx = int(row['label'])
                if label_idx not in CLASS_MAP:
                    continue
                    
                class_name = CLASS_MAP[label_idx]
                if counts[class_name] >= limit:
                    continue
                
                # Salvar imagem
                img_data = row['image']
                if 'bytes' in img_data:
                    img_bytes = img_data['bytes']
                    try:
                        img = Image.open(io.BytesIO(img_bytes))
                        img.load() # Força o carregamento completo dos pixels para filtrar imagens truncadas do dataset original
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        img_filename = f"{class_name}_{counts[class_name]:04d}.png"
                        img_path = os.path.join(data_dir, split, class_name, img_filename)
                        img.save(img_path)
                        
                        counts[class_name] += 1
                        total_saved += 1
                    except Exception as img_err:
                        print(f"Erro ao salvar imagem {idx} da classe {class_name} (provavelmente corrompida no dataset original): {img_err}")
            
            print(f"Extraído com sucesso para o split '{split}':")
            for c, count in counts.items():
                print(f" - {c}: {count} imagens")
                
        except Exception as e:
            print(f"Erro ao processar split '{split}': {e}")
        finally:
            if os.path.exists(temp_parquet):
                os.remove(temp_parquet)
                
    print("\nDataset real baixado e estruturado com absoluto sucesso!")

if __name__ == '__main__':
    download_and_extract()
