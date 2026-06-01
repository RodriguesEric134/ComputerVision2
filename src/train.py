import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from dataset import get_dataloaders
from models import FireNet_Lite, SpaceFire_DeepCNN

def train_model(model_name='deep', epochs=15, batch_size=32, lr=0.001, data_dir='data', checkpoints_dir='checkpoints'):
    # Auto-detecção de GPU CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executando o treinamento no dispositivo: {device.type.upper()}")
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Inicializa loaders
    train_loader, val_loader, _, classes = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size, img_size=(128, 128)
    )
    
    # Inicializa a arquitetura selecionada
    if model_name.lower() == 'lite':
        model = FireNet_Lite(num_classes=len(classes)).to(device)
        print("-> Carregada Arquitetura: FireNet_Lite (Sequencial Leve)")
    elif model_name.lower() == 'deep':
        model = SpaceFire_DeepCNN(num_classes=len(classes)).to(device)
        print("-> Carregada Arquitetura: SpaceFire_DeepCNN (Residual Customizada)")
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}. Escolha 'lite' ou 'deep'.")
        
    # Função de perda e Otimizador Adam com Weight Decay (Regularização L2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Dicionário para armazenar o histórico de métricas
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    checkpoint_path = os.path.join(checkpoints_dir, f'best_model_{model_name}.pth')
    
    for epoch in range(1, epochs + 1):
        # ----------------------------------------------------
        # FASE DE TREINO
        # ----------------------------------------------------
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train_samples = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zerar os gradientes do otimizador
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass e otimização
            loss.backward()
            optimizer.step()
            
            # Acúmulo de métricas
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_train_samples += inputs.size(0)
            
        epoch_train_loss = running_loss / total_train_samples
        epoch_train_acc = running_corrects / total_train_samples
        
        # ----------------------------------------------------
        # FASE DE VALIDAÇÃO
        # ----------------------------------------------------
        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0
        total_val_samples = 0
        
        with torch.no_grad(): # Desativa cálculo de gradientes
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_val_corrects += torch.sum(preds == labels.data).item()
                total_val_samples += inputs.size(0)
                
        epoch_val_loss = running_val_loss / total_val_samples
        epoch_val_acc = running_val_corrects / total_val_samples
        
        # Armazena no histórico
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc * 100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc * 100:.2f}%")
        
        # Salva o melhor checkpoint com base na acurácia de validação
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'classes': classes
            }, checkpoint_path)
            print(f" ==> Novo melhor modelo salvo com acurácia de validação: {epoch_val_acc * 100:.2f}%")
            
    # Salva o histórico de treinamento em formato JSON para plots posteriores
    history_path = os.path.join(checkpoints_dir, f'history_{model_name}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"\nTreinamento da {model_name.upper()} concluído com sucesso!")
    print(f"Melhor acurácia de validação obtida: {best_val_acc * 100:.2f}%")
    print(f"Métricas salvas em: {history_path}")
    print(f"Checkpoint de pesos salvo em: {checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SpaceFire Monitor CNN Training Pipeline")
    parser.add_argument('--model', type=str, default='deep', choices=['lite', 'deep'],
                        help="Modelo a ser treinado: 'lite' (FireNet_Lite) ou 'deep' (SpaceFire_DeepCNN)")
    parser.add_argument('--epochs', type=int, default=10, help="Quantidade de épocas de treinamento")
    parser.add_argument('--batch_size', type=int, default=32, help="Tamanho do lote (batch size)")
    parser.add_argument('--lr', type=float, default=0.001, help="Taxa de aprendizado (learning rate)")
    
    args = parser.parse_args()
    
    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
