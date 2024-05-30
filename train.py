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
import subprocess  # 添加这一行
import webbrowser  # 添加这一行
import time  # 添加这一行

# 启动 TensorBoard
def start_tensorboard(log_dir):
    subprocess.Popen(['tensorboard', '--logdir', log_dir, '--port', '6006'])
    time.sleep(5)  # 等待 TensorBoard 启动
    webbrowser.open('http://localhost:6006')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CassavaResNet50().to(device)
optimizer = Adam(model.parameters(), lr=config.LR)
criterion = CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)

# 打印模型摘要
summary(model, (3, 512, 512))

# 定義訓練函數
def train_model():
    log_dir = 'runs'
    writer = SummaryWriter(log_dir)
    start_tensorboard(log_dir)  # 启动 TensorBoard
    best_valid_loss = float('inf')
    best_train_acc = 0.0
    best_train_loss = float('inf')
    best_valid_acc = 0.0

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for images, labels in tqdm.tqdm(train_loader):
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

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc

        # 將指標記錄到 TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch)

        # 打印訓練和驗證的損失和準確率
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}")

        # 保存最佳模型
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), 'weights/best_model.pth')

        # 調整學習率
        scheduler.step(valid_loss)

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

# 訓練模型
train_model()
