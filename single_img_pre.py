from model import CassavaResNet50
from config import config
import torch
from PIL import Image
from torchvision import transforms

model_path = 'weights/best_model.pth'
image_path = 'data/test_images/2216849948.jpg'

# Load the model
def load_model(model_path):
    model = CassavaResNet50().to(config.DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

if __name__ == "__main__":
    # Load the image
    image = Image.open(image_path).convert('RGB')
    print(image.size)
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(config.DEVICE)
    print(image.size())
    
    # Load the model
    model = load_model(model_path)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(image)
        preds = outputs.argmax(1)
        print(config.CLASSES[preds.item()])
