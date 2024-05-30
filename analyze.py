import numpy as np 
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import config
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, BatchSampler, DataLoader
from torchvision import transforms, io

input_path = 'data'
images_path = 'images'
image_folder = 'train_images'

train_df = pd.read_csv(f'{input_path}/train.csv')
print(train_df.shape) # (21397, 2)
print(train_df.head())

print(train_df.nunique()) # image_id-21397 label-5
print(train_df['label'].unique()) # [0 3 1 2 4]

# Visualize the distribution of labels in the dataset
df = train_df['label'].value_counts()


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

standard = pd.read_json(f'{input_path}/label_num_to_disease_map.json', typ='series')
print(standard)

standard_map = dict(standard)

train_df['disease'] = train_df['label'].map(standard_map)
train_df['img_path'] = (f'{input_path}/{image_folder}/' + train_df['image_id']).apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')
print(train_df.head())
print(train_df.shape) # (21397, 4)

total_images_path = (f'{input_path}/{image_folder}/')
img = 0
for _, _, files in os.walk(f'{total_images_path}'):
    img += len(files)
print('Total images in the dataset:' + str(img)) # 21397



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

# Split with stratification on labels, i.e., split labels in equal proportions
train_data, valid_data = train_test_split(
    train_df, 
    test_size=config.VAL_SPLIT, 
    random_state=config.SEED,
    stratify=train_df['label']
)

print(f'Training data: {train_data.shape}\nValidation data: {valid_data.shape}')

# Add 'label_counts' for sampling data later, and reset_index after split
counts_df = train_data['label'].value_counts()
counts_map = dict(counts_df)
train_data['label_counts'] = train_data['label'].map(counts_map)
train_data.reset_index(drop=True, inplace=True)
print(train_data.head())

# Add 'label_counts' for sampling data later, and reset_index after split
counts_df = valid_data['label'].value_counts()
counts_map = dict(counts_df)
valid_data['label_counts'] = valid_data['label'].map(counts_map)
valid_data.reset_index(drop=True, inplace=True)
print(valid_data.sample(n=5))

# Define the augmenter
augmenter = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(0.2 * 2 * np.pi)
])

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
        
        original_image = image.clone()  # Clone the original image for later comparison
        
        if self.transform:
            image = self.transform(image)
        
        label = self.dataframe.iloc[idx]['label']
        return original_image, image, label

# Create train and validation datasets
train_dataset = CustomDataset(train_data, transform=augmenter)
valid_dataset = CustomDataset(valid_data, transform=augmenter)

# Function to visualize and save original and augmented images
def visualize_and_save_samples(dataset, num_samples=2, save_path="augmented_images.png"):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    fig.suptitle("Original and Augmented Images", fontsize=16)
    for i in range(num_samples):
        original_image, augmented_image, label = dataset[random.randint(0, len(dataset)-1)]
        axes[i, 0].imshow(transforms.ToPILImage()(original_image))
        axes[i, 0].set_title(f"Label: {label}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(transforms.ToPILImage()(augmented_image))
        axes[i, 1].set_title(f"Label: {label}")
        axes[i, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    # plt.show()

# Visualize and save samples from the train dataset
visualize_and_save_samples(train_dataset, num_samples=2, save_path="images/train_augmented_images.png")

# Visualize and save samples from the validation dataset
visualize_and_save_samples(valid_dataset, num_samples=2, save_path="images/valid_augmented_images.png")


CustomSampler = WeightedRandomSampler(
    torch.FloatTensor(list(1.0 /train_data['label_counts'])), 
    num_samples = len(train_dataset), 
    replacement = True
)

TrainBatchSampler = BatchSampler(
    sampler = CustomSampler,
    batch_size = config.BATCH_SIZE,
    drop_last = False
)

# Check class distribution while using TrainBatchSampler
batches = list(TrainBatchSampler)
dist = {'0':0, '1':0, '2':0, '3':0, '4':0} # distribution
for batch in batches:    
    for id in batch:
        label = train_data['label'][id]
        dist[str(label)]+=1
print(dist)
print("Batch samples: ",sum(dist.values()))


# Train and valid. dataloader
train_loader = DataLoader(train_dataset, batch_sampler = TrainBatchSampler)
valid_loader = DataLoader(valid_dataset, config.BATCH_SIZE, drop_last = False, shuffle=True )