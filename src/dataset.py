import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_transforms(img_size=(128, 128)):
    """
    Retorna as transformações do PyTorch para treino, validação e testes.
    Aplica técnicas recomendadas de Data Augmentation para imagens de satélite.
    """
    # Média e desvio padrão padrão para normalização (baseado em ImageNet)
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        # Aumentação espacial para imagens de satélite (invariantes à rotação)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        # Aumentação de cor para simular variação solar e névoa
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    return train_transform, val_test_transform

def get_dataloaders(data_dir, batch_size=32, img_size=(128, 128), num_workers=0):
    """
    Carrega as imagens organizadas em pastas estruturadas de treino/val/teste
    e retorna os respectivos DataLoaders do PyTorch.
    """
    train_transform, val_test_transform = get_data_transforms(img_size)

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')
    test_path = os.path.join(data_dir, 'test')

    # Validação da existência dos diretórios
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Diretório não encontrado: {path}. Certifique-se de gerar o dataset primeiro.")

    # Uso do ImageFolder para inferir as classes a partir do nome das pastas
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=val_test_transform)

    # Criação dos DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    print(f"Dataset carregado com sucesso:")
    print(f" - Treinamento: {len(train_dataset)} imagens (Classes: {train_dataset.classes})")
    print(f" - Validação:   {len(val_dataset)} imagens")
    print(f" - Teste:       {len(test_dataset)} imagens")

    return train_loader, val_loader, test_loader, train_dataset.classes
