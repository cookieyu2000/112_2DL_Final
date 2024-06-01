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
import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 定義圖像路徑和輸出資料夾
image_paths = 'data/augmented_images'
output_images_folder = 'images'
cache_path = 'data/train_cache.h5'  # 定義快取文件路徑

# 定義支援快取的自定義數據集類
class CachedDataset(Dataset):
    def __init__(self, dataframe, label_map, transform=None, cache_file=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label_map = label_map
        self.cache = {}
        self.cache_file = cache_file

        # 如果快取文件存在，載入快取
        if cache_file and os.path.exists(cache_file):
            self.load_cache()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 檢查快取中是否已有該圖像
        if idx in self.cache:
            image, label = self.cache[idx]
        else:
            # 載入圖像並應用變換
            img_path = self.dataframe.iloc[idx]['image_path']
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            label = self.dataframe.iloc[idx]['label']
            label = self.label_map[label]  # 將疾病對應到數字
            self.cache[idx] = (image, label)  # 將圖像和標籤存儲到快取中

        return image, label

    # 保存快取
    def save_cache(self):
        with h5py.File(self.cache_file, 'w') as f:
            for idx, (image, label) in self.cache.items():
                grp = f.create_group(str(idx))
                grp.create_dataset('image', data=image.numpy())
                grp.create_dataset('label', data=label)
        print(f'Cache saved to {self.cache_file}')

    # 載入快取
    def load_cache(self):
        with h5py.File(self.cache_file, 'r') as f:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._load_single_item, f, key) for key in f.keys()]
                for future in futures:
                    future.result()
        print(f'Cache loaded from {self.cache_file}')

    def _load_single_item(self, f, key):
        idx = int(key)
        image = torch.tensor(f[key]['image'][:])
        label = int(f[key]['label'][()])
        self.cache[idx] = (image, label)

# 定義函數繪製並保存餅圖
def plot_pie_chart(data, title, filename):
    label_counts = data['label'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(title)
    
    # 保存圖像
    save_path = os.path.join(output_images_folder, filename)
    plt.savefig(save_path)
    plt.close(fig)
    
    # print(f'{title} detailed counts:\n{label_counts}\n')

# 讀取數據
train_df = pd.read_csv('data/augmented_labels.csv')
# print(f'total number of images: {len(train_df)}')

standard = pd.read_json('data/label_num_to_disease_map.json', typ='series')
standard_map = dict(standard) # 將疾病名稱映射到數值標籤
# print(f'Number of classes: {len(standard)}')

# 將疾病名稱映射到數值標籤
reversed_standard_map = {v: int(k) for k, v in standard_map.items()}
train_df['label'] = train_df['label'].map(standard_map)
train_df['image'] = train_df['image_path'].apply(lambda x: x.split('/')[-1]) # 提取圖像名稱
# print(f'total number of images: {len(train_df["image"])}')

# 劃分數據集
train_data, valid_data = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42, shuffle=True)
valid_data, test_data = train_test_split(valid_data, test_size=0.5, stratify=valid_data['label'], random_state=42, shuffle=True)

# # 繪製並保存餅圖
# plot_pie_chart(train_data, 'Training Data Distribution', 'train_data_distribution.png')
# plot_pie_chart(valid_data, 'Validation Data Distribution', 'valid_data_distribution.png')
# plot_pie_chart(test_data, 'Test Data Distribution', 'test_data_distribution.png')

# 計算類別權重
class_counts = train_data['label'].value_counts()
total_samples = len(train_data)
class_weights = {label: total_samples / count for label, count in class_counts.items()}

# # 逐行打印類別權重
# print("Class Weights:")
# for label, weight in class_weights.items():
#     print(f'Class {label}: {weight}')

# 定義圖像變換
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 調整尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 創建數據集實例
train_dataset = CachedDataset(train_data, reversed_standard_map, transform=transform, cache_file=cache_path)
valid_dataset = CachedDataset(valid_data, reversed_standard_map, transform=transform, cache_file=cache_path)
test_dataset = CachedDataset(test_data, reversed_standard_map, transform=transform, cache_file=cache_path)

# 保存快取
train_dataset.save_cache()
valid_dataset.save_cache()
test_dataset.save_cache()

# print(f'Training dataset size: {len(train_dataset)}')
# print(f'Validation dataset size: {len(valid_dataset)}')
# print(f'Test dataset size: {len(test_dataset)}')

# 創建DataLoader
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

# print(f'Training dataset size: {len(train_loader)}')
# print(f'Train dataset shape: {train_loader.dataset[0][0].shape}') # C, H, W
# print(f'Train dataset label: {train_loader.dataset[0][1]}')
# print(f'Validation dataset size: {len(valid_loader)}')
# print(f'Test dataset size: {len(test_loader)}')
