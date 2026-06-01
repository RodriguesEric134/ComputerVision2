import torch
import torch.nn as nn

# =====================================================================
# ARQUITETURA 1: FireNet_Lite (CNN Sequencial Leve)
# =====================================================================
class FireNet_Lite(nn.Module):
    """
    CNN sequencial clássica construída do zero.
    Projetada para ser leve, de rápida convergência, ideal para hardware modesto.
    """
    def __init__(self, num_classes=3):
        super(FireNet_Lite, self).__init__()
        
        # Bloco Convolucional 1: Entrada 3x128x128 -> Saída 32x64x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # Estabiliza treinamento e previne gradientes nulos
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bloco Convolucional 2: Entrada 32x64x64 -> Saída 64x32x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Bloco Convolucional 3: Entrada 64x32x32 -> Saída 128x16x16
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Pooling adaptativo garante compatibilidade com qualquer tamanho de imagem
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4)) # Saída: 128 x 4 x 4 = 2048
        
        # Classificador Densa com Dropout agressivo para combater overfitting
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4), # Zera aleatoriamente 40% das ativações no treino
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits


# =====================================================================
# ARQUITETURA 2: SpaceFire_DeepCNN (CNN Profunda com Blocos Residuais)
# =====================================================================
class ResidualBlock(nn.Module):
    """
    Bloco Residual customizado construído do zero.
    Permite o fluxo de gradiente direto através de conexões de atalho (shortcuts),
    resolvendo o problema de desvanecimento de gradiente em redes profundas.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Primeira convolução do bloco
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        # Segunda convolução do bloco
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Atalho de Identidade (Shortcut)
        self.shortcut = nn.Sequential()
        # Se as dimensões espaciais ou canais mudarem, ajustamos via Conv 1x1
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Somatório do atalho antes da ativação final ReLU (Conexão Residual)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SpaceFire_DeepCNN(nn.Module):
    """
    Arquitetura de alta performance com Blocos Residuais próprios.
    Ideal para padrões complexos de imagens aéreas (ex: fumaça difusa e cicatrizes térmicas).
    """
    def __init__(self, num_classes=3):
        super(SpaceFire_DeepCNN, self).__init__()
        
        # Camada Convolucional Inicial
        self.in_channels = 32
        self.conv_init = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Blocos residuais empilhados (extração profunda de feições texturais)
        self.layer1 = self._make_layer(32, stride=1)   # Saída: 32x128x128
        self.layer2 = self._make_layer(64, stride=2)   # Saída: 64x64x64
        self.layer3 = self._make_layer(128, stride=2)  # Saída: 128x32x32
        self.layer4 = self._make_layer(256, stride=2)  # Saída: 256x16x16
        
        # Global Average Pooling extrai a essência do mapa de ativação espacial
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Saída: 256x1x1
        
        # Classificador robusto com regularização dupla
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5), # Regularização severa para forçar generalização
            nn.Linear(128, num_classes)
        )
        
    def _make_layer(self, out_channels, stride):
        # Cada bloco residual é composto por um ResidualBlock
        block = ResidualBlock(self.in_channels, out_channels, stride)
        self.in_channels = out_channels
        return block

    def forward(self, x):
        out = self.conv_init(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        logits = self.classifier(out)
        return logits


if __name__ == '__main__':
    # Teste rápido de integridade dimensional das redes
    print("Testando integridade das CNNs customizadas...")
    test_tensor = torch.randn(2, 3, 128, 128) # Batch size = 2, Imagem 128x128
    
    # 1. Teste FireNet_Lite
    model_lite = FireNet_Lite(num_classes=3)
    out_lite = model_lite(test_tensor)
    print(f" -> FireNet_Lite OK. Output shape: {out_lite.shape} (Esperado: [2, 3])")
    
    # 2. Teste SpaceFire_DeepCNN
    model_deep = SpaceFire_DeepCNN(num_classes=3)
    out_deep = model_deep(test_tensor)
    print(f" -> SpaceFire_DeepCNN OK. Output shape: {out_deep.shape} (Esperado: [2, 3])")
