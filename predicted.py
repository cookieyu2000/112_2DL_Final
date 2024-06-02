import torch
from model import CassavaResNet50
from dataloader import test_loader
from config import config
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tqdm
import matplotlib.pyplot as plt
import numpy as np

weights_path = 'weights/best_model.pth'

# 标签映射，假设你的映射如下：
label_map = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2', 3: 'Class 3', 4: 'Class 4'}

# Load the model
def load_model(weights_path):
    model = CassavaResNet50().to(config.DEVICE)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model

# Predict and calculate accuracy
def predict_and_evaluate(model, test_loader, label_map):
    all_preds = []
    all_labels = []
    all_images = []  # 存储所有图像
    all_image_paths = []  # 存储所有图像路径
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader, desc="Predicting"):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(images.cpu().numpy())  # 存储图像
            all_image_paths.extend(test_loader.dataset.dataframe.iloc[:len(images)]['image_path'])  # 存储图像路径
    accuracy = accuracy_score(all_labels, all_preds)
    # 转换为实际标签名称
    all_preds = [label_map[pred] for pred in all_preds]
    all_labels = [label_map[label] for label in all_labels]
    return accuracy, all_preds, all_labels, all_images, all_image_paths

# 反标准化图像
def denormalize(image, mean, std):
    image = image * std[:, None, None] + mean[:, None, None]
    return image

# 绘制前10张图像的真实标签和预测标签
def plot_images_with_labels(images, labels, preds, output_path, mean, std):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            image = denormalize(images[i], mean, std).transpose(1, 2, 0)  # 从 (C, H, W) 转为 (H, W, C)
            image = np.clip(image, 0, 1)  # 裁剪图像数据到 [0, 1] 范围
            ax.imshow(image)
            ax.set_title(f"True: {labels[i]}\nPred: {preds[i]}")
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()


if __name__ == "__main__":
    model = load_model(weights_path)
    accuracy, all_preds, all_labels, all_images, all_image_paths = predict_and_evaluate(model, test_loader, label_map)
    
    print(f"Test Accuracy: {accuracy:.4f}")

    # 生成并保存混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=list(label_map.values()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.values()))
    disp.plot()
    plt.savefig('images/test_confusion_matrix.png')
    # plt.show()
    
    # 保存预测结果到 CSV 文件
    results = pd.DataFrame({'Image_File': all_image_paths, 'Predicted': all_preds, 'Actual': all_labels})
    results.to_csv('data/test_predictions.csv', index=False)
    
    # 图像标准化参数
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # 绘制前10张图像的真实标签和预测标签
    plot_images_with_labels(all_images[:10], all_labels[:10], all_preds[:10], 'images/predictions.png', mean, std)
