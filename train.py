# train.py
import torch
from torch import nn
from dataloader import train_loader, valid_loader
from config import config
from model import CassavaResNet50
import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchsummary import summary
import subprocess
import webbrowser
import time

# 启动 TensorBoard
def start_tensorboard(log_dir):
    subprocess.Popen(['tensorboard', '--logdir', log_dir, '--port', '3000'])
    # time.sleep(5)  # 等待 TensorBoard 启动
    # webbrowser.open('http://localhost:3000')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CassavaResNet50().to(device)
optimizer = Adam(model.parameters(), lr=config.LR)
criterion = CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config.PATIENCE, factor=config.DECAY_FACTOR, verbose=True)

# 打印模型摘要
summary(model, (3, 224, 224))

# 定義訓練函數
def train_model():
    log_dir = 'runs/second'
    writer = SummaryWriter(log_dir)
    start_tensorboard(log_dir)  # 启动 TensorBoard
    best_valid_loss = float('inf')
    best_train_acc = 0.0
    best_train_loss = float('inf')
    best_valid_acc = 0.0
    best_epoch = 0  # 用于记录最佳模型保存的 epoch

    # Early stopping 參數
    early_stop_patience = config.early_stop_patience  # Early stopping 設為15
    epochs_no_improve = 0

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for images, labels in tqdm.tqdm(train_loader):
            # print(f'images shape: {images.shape}, labels shape: {labels.shape}')
            images, labels = images.to(device), labels.to(device) 
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).float().mean().item()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        if train_loss < best_train_loss:
            best_train_loss = train_loss

        if train_acc > best_train_acc:
            best_train_acc = train_acc

        model.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in tqdm.tqdm(valid_loader):
                images, labels = images.to(device), labels.to(device)  
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                valid_acc += (outputs.argmax(1) == labels).float().mean().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.argmax(1).cpu().numpy())
            valid_loss /= len(valid_loader)
            valid_acc /= len(valid_loader)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            best_epoch = epoch + 1  # 記錄保存最佳模型的 epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'weights/best_model.pth')  # 保存最佳模型
        else:
            epochs_no_improve += 1

        # 將指標記錄到 TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'valid': valid_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'valid': valid_acc}, epoch)

        # 打印訓練和驗證的損失和準確率
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}, "
              f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Epochs without improvement: {epochs_no_improve}")

        # 調整學習率
        scheduler.step(valid_loss)
        print(f"Learning Rate after adjustment: {optimizer.param_groups[0]['lr']:.6f}")

        # 檢查 early stopping 條件
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    writer.close()

    # 混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('images/confusion_matrix.png')
    # plt.show()
    
    # 最好的結果
    print(f"Best Train Accuracy: {best_train_acc:.4f}")
    print(f"Best Train Loss: {best_train_loss:.4f}")
    print(f"Best Validation Loss: {best_valid_loss:.4f}")
    print(f"Best Validation Accuracy: {best_valid_acc:.4f}")
    print(f"Best model saved at epoch: {best_epoch}")

# 訓練模型
train_model()
