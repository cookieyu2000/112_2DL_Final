import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, io
from PIL import Image
from config import config
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 导入 tqdm
from concurrent.futures import ThreadPoolExecutor  # 导入多线程模块

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['img_path']
        image = io.read_image(img_path)
        image = transforms.ConvertImageDtype(torch.float32)(image)
        
        if self.transform:
            image = self.transform(image)
        
        label = self.dataframe.iloc[idx]['label']
        label = torch.tensor(label, dtype=torch.long)  # 确保标签是 long 类型的 1D 张量
        return image, label

# Define a function to process a single image
def process_image(dataset, idx, output_folder, angle_step, brightness_levels):
    image, label = dataset[idx]
    image = transforms.ToPILImage()(image)
    records = []
    
    for angle in range(0, 360, angle_step):
        for brightness in brightness_levels:
            augmented_image = transforms.functional.adjust_brightness(image, brightness)
            augmented_image = transforms.functional.rotate(augmented_image, angle)
            file_name = f'image_{idx}_angle_{angle}_brightness_{brightness}.jpg'
            save_path = os.path.join(output_folder, file_name)
            augmented_image.save(save_path)
            
            records.append({
                'image_path': save_path,
                'label': label.item()
            })
    
    return records

# Define a function to augment and save images along with labels
def augment_and_save_images(dataset, output_folder, csv_path, angle_step=90, brightness_levels=[0.5, 1.0, 1.2], num_workers=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    records = []
    
    # 使用 ThreadPoolExecutor 进行并行处理
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_image, dataset, idx, output_folder, angle_step, brightness_levels) for idx in range(len(dataset))]
        for future in tqdm(futures, desc="Augmenting Images"):
            records.extend(future.result())
    
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

# Load the dataset
input_path = 'data'
image_folder = 'train_images'
output_folder = 'data/augmented_images'
csv_path = 'data/augmented_labels.csv'

train_df = pd.read_csv(f'{input_path}/train.csv')
standard = pd.read_json(f'{input_path}/label_num_to_disease_map.json', typ='series')
standard_map = dict(standard)
train_df['disease'] = train_df['label'].map(standard_map)
train_df['img_path'] = (f'{input_path}/{image_folder}/' + train_df['image_id']).apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')

# Convert DataFrame to CustomDataset
train_dataset = CustomDataset(train_df)

# Augment and save images along with labels
augment_and_save_images(train_dataset, output_folder, csv_path, num_workers=10)  
