# dataloader.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, io
from sklearn.model_selection import train_test_split
from config import config
import numpy as np

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

# Define the transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(0.2 * 2 * np.pi),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the dataset
input_path = 'data'
image_folder = 'train_images'
train_df = pd.read_csv(f'{input_path}/train.csv')
standard = pd.read_json(f'{input_path}/label_num_to_disease_map.json', typ='series')
standard_map = dict(standard)
train_df['disease'] = train_df['label'].map(standard_map)
train_df['img_path'] = (f'{input_path}/{image_folder}/' + train_df['image_id']).apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')

train_data, valid_data = train_test_split(
    train_df, 
    test_size=config.VAL_SPLIT, 
    random_state=config.SEED,
    stratify=train_df['label']
)

valid_data, test_data = train_test_split(
    valid_data, 
    test_size=config.TEST_SPLIT, 
    random_state=config.SEED,
    stratify=valid_data['label']
)



train_dataset = CustomDataset(train_data, transform=train_transforms)
valid_dataset = CustomDataset(valid_data, transform=valid_transforms)
test_dataset = CustomDataset(test_data, transform=valid_transforms)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# print(f"Train dataset: {len(train_dataset)}")
# print(f"Valid dataset: {len(valid_dataset)}")
# print(f"Test dataset: {len(test_dataset)}")
