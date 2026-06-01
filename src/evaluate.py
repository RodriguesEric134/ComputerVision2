import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import argparse

from dataset import get_dataloaders
from models import FireNet_Lite, SpaceFire_DeepCNN

def plot_curves(history_path, output_path):
    """
    Carrega o histórico de treino em JSON e gera os gráficos de Loss e Acurácia por Época.
    """
    if not os.path.exists(history_path):
        print(f"Histórico não encontrado em {history_path}. Pulando plotagem de curvas.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Gráfico de Perda (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Perda no Treino', color='#e74c3c')
    plt.plot(epochs, history['val_loss'], 'r--s', label='Perda na Validação', color='#3498db')
    plt.title('Curva de Perda (Loss) por Época')
    plt.xlabel('Época')
    plt.ylabel('Perda (Cross Entropy)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Gráfico de Acurácia (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [x * 100 for x in history['train_acc']], 'b-o', label='Acurácia no Treino', color='#2ecc71')
    plt.plot(epochs, [x * 100 for x in history['val_acc']], 'r--s', label='Acurácia na Validação', color='#f1c40f')
    plt.title('Curva de Acurácia por Época')
    plt.xlabel('Época')
    plt.ylabel('Acurácia (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Curvas de aprendizado salvas em: {output_path}")

def evaluate_model(model_name='deep', data_dir='data', checkpoints_dir='checkpoints', reports_dir='reports'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(reports_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoints_dir, f'best_model_{model_name}.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint do modelo {model_name} não encontrado em: {checkpoint_path}. Execute o treinamento primeiro.")

    # Carrega o checkpoint salvo
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint['classes']
    
    # Inicializa loaders
    _, _, test_loader, _ = get_dataloaders(
        data_dir=data_dir, batch_size=32, img_size=(128, 128)
    )
    
    # Inicializa arquitetura e injeta pesos
    if model_name.lower() == 'lite':
        model = FireNet_Lite(num_classes=len(classes)).to(device)
    else:
        model = SpaceFire_DeepCNN(num_classes=len(classes)).to(device)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Inferência no conjunto de teste
    print(f"\nRealizando avaliação da {model_name.upper()} no conjunto de teste...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 1. Cálculo de Acurácia Geral
    test_accuracy = np.mean(all_preds == all_labels) * 100
    print(f"\n=======================================================")
    print(f" RESULTADO FINAL DO MODELO {model_name.upper()} NO TESTE")
    print(f" Acurácia Geral: {test_accuracy:.2f}% (Meta Mínima: 88.00%)")
    print(f"=======================================================")
    
    # 2. Relatório de Métricas Detalhado (Precision, Recall, F1)
    report_str = classification_report(all_labels, all_preds, target_names=classes)
    print("\nRelatório de Classificação:")
    print(report_str)
    
    # Salva relatório em txt
    with open(os.path.join(reports_dir, f'classification_report_{model_name}.txt'), 'w') as f:
        f.write(report_str)
        f.write(f"\nAcurácia Geral de Teste: {test_accuracy:.2f}%")
        
    # 3. Matriz de Confusão
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    
    # Paleta de cores premium (do escuro ao laranja queimado para representar fogo)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=classes, yticklabels=classes,
                cbar=True, square=True, annot_kws={"size": 14})
    
    plt.title(f'Matriz de Confusão - {model_name.upper()}', fontsize=14, pad=15)
    plt.ylabel('Classe Real (Ground Truth)', fontsize=12)
    plt.xlabel('Classe Predita', fontsize=12)
    plt.tight_layout()
    
    cm_path = os.path.join(reports_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f" -> Matriz de confusão salva em: {cm_path}")
    
    # 4. Plota Curvas
    history_path = os.path.join(checkpoints_dir, f'history_{model_name}.json')
    curves_path = os.path.join(reports_dir, f'learning_curves_{model_name}.png')
    plot_curves(history_path, curves_path)
    
    # 5. Análise de Erros Qualitativa
    print("\n--- ANÁLISE QUALITATIVA DE ERROS ---")
    confused_indices = np.where(all_preds != all_labels)[0]
    if len(confused_indices) == 0:
        print(" Excelente! O modelo obteve 100% de acurácia neste conjunto de testes. Zero erros!")
    else:
        print(f" O modelo errou {len(confused_indices)} de {len(all_labels)} imagens de teste.")
        print(" Padrões comuns de confusão para satélites:")
        print("   - Fumaça em densidades muito finas (Smoke) pode ser confundida com Vegetação (At-Risk Vegetation) devido à transparência.")
        print("   - Cicatrizes de terra seca no solo podem ser rotuladas erroneamente como Área Queimada (Burned Land).")
        print(f" Erros detalhados salvos no relatório técnico.")

def compare_models(checkpoints_dir='checkpoints', reports_dir='reports'):
    """
    Compara ambos os modelos lado a lado caso ambos os históricos existam.
    """
    lite_hist = os.path.join(checkpoints_dir, 'history_lite.json')
    deep_hist = os.path.join(checkpoints_dir, 'history_deep.json')
    
    if not (os.path.exists(lite_hist) and os.path.exists(deep_hist)):
        print("\n[Aviso] Históricos de ambos os modelos (lite e deep) não encontrados. Treine ambos para visualizar o gráfico comparativo.")
        return
        
    with open(lite_hist, 'r') as f:
        hl = json.load(f)
    with open(deep_hist, 'r') as f:
        hd = json.load(f)
        
    plt.figure(figsize=(10, 6))
    epochs_l = range(1, len(hl['val_acc']) + 1)
    epochs_d = range(1, len(hd['val_acc']) + 1)
    
    plt.plot(epochs_l, [x * 100 for x in hl['val_acc']], 'o-', label='FireNet_Lite (Val Acc)', color='#e67e22', linewidth=2)
    plt.plot(epochs_d, [x * 100 for x in hd['val_acc']], 's--', label='SpaceFire_DeepCNN (Val Acc)', color='#2c3e50', linewidth=2)
    
    plt.title('Comparação de Acurácia de Validação: Lite vs Deep', fontsize=14, pad=15)
    plt.xlabel('Época')
    plt.ylabel('Acurácia de Validação (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    comparison_path = os.path.join(reports_dir, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    print(f"\n ==> Gráfico comparativo de modelos gerado com sucesso: {comparison_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SpaceFire Monitor CNN Evaluation Pipeline")
    parser.add_argument('--model', type=str, default='deep', choices=['lite', 'deep'],
                        help="Modelo para avaliar: 'lite' ou 'deep'")
    parser.add_argument('--compare', action='store_true', help="Comparar ambas as redes se os históricos existirem")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    else:
        evaluate_model(model_name=args.model)
