import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import clip
from matplotlib.gridspec import GridSpec
from DiT import DiT
from config import config



# hyperparameters

device = config["device"]
img_size = config["img_size"]
img_channels = config["img_channels"]
n_classes = config["n_classes"]
batch_size = config["batch_size"]
n_epochs = config["n_epochs"]
learning_rate = config["learning_rate"]
T = config["T"]



# dataset

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Transparency expressed in bytes.*")
warnings.filterwarnings("ignore", message=".*Clipping input data to the valid range for imshow with RGB data.*")

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


mean, std = torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5])

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

dataset = TextureDataset("block", transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


# pour obtenir des tokens à partir des noms des textures
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def get_embedding(prompt):
    with torch.no_grad():
        text_tokens = clip.tokenize(prompt).to(device)
        text_embedding = clip_model.encode_text(text_tokens)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding.to(torch.float32)



def noise_imgs(x1, t):
    t = t.view(-1, 1, 1, 1)
    x0 = torch.randn_like(x1, device=device)
    xt = (1 - t) * x0 + t * x1
    return xt, x0

def sample(n_imgs, model, labels):
    model.eval()
    with torch.no_grad():
        xt = torch.randn((n_imgs, img_channels, img_size, img_size), device=device)
        for t in torch.linspace(0, 1, T):
            t = t.expand(xt.shape[0]).view(-1, 1).to(device)
            xt = xt + 1/T * model(xt, t, labels)
    model.train()
    return xt



model = DiT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()

loss_list = []

print(f"Num params: {(sum(p.numel() for p in model.parameters())) / 1e6} M")

#d_params = {}
#for p in model.named_parameters():
#    d_params[p[0]] = p[1].numel()
#sorted_d_params = dict(sorted(d_params.items(), key=lambda item: item[1], reverse=True))
#print(sorted_d)



for epoch in tqdm(range(n_epochs)):
    for i, (train_imgs, textes) in enumerate(train_loader):

        x1 = train_imgs.to(device)
        labels = get_embedding(textes)
        t = torch.rand((x1.shape[0], 1), device=device)
        xt, x0 = noise_imgs(x1, t)
        targets = x1 - x0

        preds = model(xt, t, labels)

        loss = mse(preds, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


        if i%1 == 0:
            loss_list.append(loss.item())


    if epoch % 1 == 0:
        s = 5
        loss_list_moyennée = torch.tensor(loss_list[:len(loss_list)//s*s]).view(-1, s).mean(1)
        textes = ('stone_bricks4', 'pumpkin_top5', 'comparator_redstone_on', 'potatoes_stage3', 'spruce_planks4', 'respawn_anchor_side_inner4', 'deepslate_gold_ore4')
        labels = get_embedding(textes)
        sampled_imgs = sample(len(textes), model, labels)

        fig = plt.figure(figsize=(20, 8))
        gs = GridSpec(2, len(textes), figure=fig)

        # Graphiques sur la première ligne
        ax1 = fig.add_subplot(gs[0, :len(textes)//2])  # Fusionne les premières colonnes
        ax1.plot(loss_list, c="blue")
        ax1.set_title("Loss List")

        ax2 = fig.add_subplot(gs[0, len(textes)//2:])  # Fusionne les colonnes restantes
        ax2.plot(loss_list_moyennée[100:], c="blue")
        ax2.set_title("Loss List Moyennée")

        # Images sur la deuxième ligne
        for j in range(len(textes)):
            ax = fig.add_subplot(gs[1, j])
            img = torch.clamp(sampled_imgs[j].cpu() * std.view(-1, 1, 1) + mean.view(-1, 1, 1), 0, 1).permute(1, 2, 0)
            ax.imshow(img)
            ax.axis('off')  # Désactive les axes pour une meilleure présentation

        plt.tight_layout()  # Assure que tout s'affiche bien
        plt.savefig("graph.png")  # Sauvegarde le graphe dans un fichier
        plt.close(fig)  # Ferme la figure pour libérer la mémoire
