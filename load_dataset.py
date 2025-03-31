import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import config
import clip
import torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Transparency expressed in bytes.*")
warnings.filterwarnings("ignore", message=".*Clipping input data to the valid range for imshow with RGB data.*")


img_size = config["img_size"]
mean = config["mean"]
std = config["std"]
batch_size = config["batch_size"]
device = config["device"]


class TextureDataset(Dataset):

    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = sorted(os.listdir(image_folder))  # Trier pour un ordre stable
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)

        # Charger l'image
        image = Image.open(img_path).convert("RGB")

        # Appliquer les transformations si nécessaire
        if self.transform:
            image = self.transform(image)

        # Utiliser le nom du fichier sans extension comme label (prompt)
        prompt = os.path.splitext(img_name)[0]  # Enlever ".jpg", ".png", etc.

        return image, prompt


transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

dataset = TextureDataset("dataset", transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)



# pour obtenir des tokens à partir des noms des textures
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def get_embedding(prompt):
    with torch.no_grad():
        text_tokens = clip.tokenize(prompt).to(device)
        text_embedding = clip_model.encode_text(text_tokens)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding.to(torch.float32)
