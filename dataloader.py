import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
import torch
import json
import pickle

image_paths ='data/augmented_images'
output_images_folder = 'images'
cache_path = 'data/cache.pkl'  # 定义缓存文件路径


# 定义支持缓存的自定义数据集类
class CachedDataset(Dataset):
    def __init__(self, dataframe, label_map, transform=None, cache_file=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label_map = label_map
        self.cache = {}
        self.cache_file = cache_file

        if cache_file and os.path.exists(cache_file):
            self.load_cache()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if idx in self.cache:  # 检查缓存中是否已有该图像
            image, label = self.cache[idx]
        else:
            img_path = self.dataframe.iloc[idx]['image_path']
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            label = self.dataframe.iloc[idx]['label']
            # 将疾病对应到数字
            label = self.label_map[label]
            self.cache[idx] = (image, label)  # 将图像和标签存储到缓存中

        return image, label

    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
        print(f'Cache saved to {self.cache_file}')

    def load_cache(self):
        with open(self.cache_file, 'rb') as f:
            self.cache = pickle.load(f)
        print(f'Cache loaded from {self.cache_file}')


# 定义函数绘制并保存饼图
def plot_pie_chart(data, title, filename):
    label_counts = data['label'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(title)
    
    # 保存图像
    save_path = os.path.join(output_images_folder, filename)
    plt.savefig(save_path)
    plt.close(fig)
    
    # print(f'{title} detailed counts:\n{label_counts}\n')

train_df = pd.read_csv('data/augmented_labels.csv')
# print(f'total number of images: {len(train_df)}')

standard = pd.read_json('data/label_num_to_disease_map.json', typ='series')
standard_map = dict(standard)
# print(f'Number of classes: {len(standard)}')

# 将疾病名称映射到数值标签
reversed_standard_map = {v: int(k) for k, v in standard_map.items()}
train_df['label'] = train_df['label'].map(standard_map)
train_df['image'] = train_df['image_path'].apply(lambda x: x.split('/')[-1]) # Extract the image name from the path
# print(f'total number of images: {len(train_df["image"])}')

train_data, valid_data = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42, shuffle=True)

valid_data, test_data = train_test_split(valid_data, test_size=0.5, stratify=valid_data['label'], random_state=42, shuffle=True)


# # 绘制并保存饼图
# plot_pie_chart(train_data, 'Training Data Distribution', 'train_data_distribution.png')
# plot_pie_chart(valid_data, 'Validation Data Distribution', 'valid_data_distribution.png')
# plot_pie_chart(test_data, 'Test Data Distribution', 'test_data_distribution.png')

# 計算權重
class_counts = train_data['label'].value_counts()
total_samples = len(train_data)
class_weights = {label: total_samples / count for label, count in class_counts.items()}

# # 逐行打印类别权重
# print("Class Weights:")
# for label, weight in class_weights.items():
#     print(f'Class {label}: {weight}')

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 根据需要调整尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 创建数据集实例
train_dataset = CachedDataset(train_data, reversed_standard_map, transform=transform, cache_file=cache_path)
valid_dataset = CachedDataset(valid_data, reversed_standard_map, transform=transform, cache_file=cache_path)
test_dataset = CachedDataset(test_data, reversed_standard_map, transform=transform, cache_file=cache_path)

# 保存缓存
train_dataset.save_cache()
valid_dataset.save_cache()
test_dataset.save_cache()

# print(f'Training dataset size: {len(train_dataset)}')
# print(f'Validation dataset size: {len(valid_dataset)}')
# print(f'Test dataset size: {len(test_dataset)}')

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

# print(f'Training dataset size: {len(train_loader)}')
# print(f'Train dataset shape: {train_loader.dataset[0][0].shape}') # C, H, W
# print(f'Train dataset label: {train_loader.dataset[0][1]}')
# print(f'Validation dataset size: {len(valid_loader)}')
# print(f'Test dataset size: {len(test_loader)}')
