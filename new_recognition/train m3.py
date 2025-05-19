import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from thop import profile
from torch.cuda.amp import autocast, GradScaler
from method3 import AttentionFusionFoodNet 

# 数据增强和加载（调整归一化参数）
def prepare_dataloaders(data_dir='E://deep learning//ChineseFoodNet//', batch_size=64):
    # EfficientNetV2的推荐归一化参数
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transforms
    )
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=test_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, train_dataset.classes

# 优化后的训练函数
def train_model(model, train_loader, test_loader, class_names, 
                num_epochs=100, lr=3e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 混合精度训练
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                            total_steps=num_epochs*len(train_loader))
    
    best_acc = 0.0
    print_freq = 66  # 每66个batch打印一次
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item()
            
            if (i+1) % print_freq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, LR: {current_lr:.2e}')
        
        # 每个epoch结束后在测试集上验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(inputs)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Acc: {acc:.2f}%')
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'best_model.pth')
    
    print(f'Best Test Accuracy: {best_acc:.2f}%')
    return model

# 更新后的评估函数
def evaluate_model(model, test_loader, input_size=(1,3,224,224)):
    device = next(model.parameters()).device
    
    # 计算准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with autocast():
                outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    
    # 计算FLOPs和参数量
    example_input = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(example_input,), verbose=False)
    
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'FLOPs: {flops/1e9:.2f}G')
    print(f'Params: {params/1e6:.2f}M')
    
    return {
        'accuracy': accuracy,
        'flops': flops,
        'params': params
    }

if __name__ == '__main__':
    # 参数设置
    num_classes = 100  # 根据实际数据集调整
    batch_size = 64    # 根据GPU显存调整
    num_epochs = 100
    lr = 3e-4
    
    # 准备数据
    train_loader, test_loader, class_names = prepare_dataloaders(
        data_dir='E://deep learning//ChineseFoodNet//',  
        batch_size=batch_size
    )
    
    # 初始化混合模型
    model = AttentionFusionFoodNet(
        num_classes=len(class_names),  # 自动获取类别数
        pretrained=True                # 加载预训练权重
    )
    
    # 训练模型
    trained_model = train_model(
        model, train_loader, test_loader, class_names,
        num_epochs=num_epochs, lr=lr
    )
    
    # 最终评估
    metrics = evaluate_model(trained_model, test_loader)