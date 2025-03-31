import torch
from config import config
import clip
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
from ViT import ViT
from torch.serialization import add_safe_globals

import torch
from ViT import ViT, TransformerBlock  # Importe les deux classes nécessaires
from config import config
from torch.serialization import add_safe_globals

# Autorise les classes personnalisées utilisées dans le modèle
add_safe_globals({"ViT": ViT, "TransformerBlock": TransformerBlock})

# Chargement du modèle complet
model = torch.load("model_full.pth", map_location=config["device"], weights_only=False)
print("✅ Modèle chargé avec succès")
model.eval()
device = next(model.parameters()).device  # auto-détection du bon device

# === Déduction depuis le modèle
img_channels = model.patchify.in_channels
img_size = model.patchify.kernel_size[0] * model.patchify.weight.shape[2]
T = config["T"]
mean, std = torch.tensor([0.5]*img_channels), torch.tensor([0.5]*img_channels)

# === CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def get_embedding(prompt):
    with torch.no_grad():
        tokens = clip.tokenize(prompt).to(device)
        emb = clip_model.encode_text(tokens)
    emb /= emb.norm(dim=-1, keepdim=True)
    return emb.to(torch.float32)

@torch.no_grad()
def sample(n_imgs, model, labels):
    xt = torch.randn((n_imgs, img_channels, img_size, img_size), device=device)
    for t in torch.linspace(0, 1, T):
        t_batch = t.expand(n_imgs).view(-1, 1).to(device)
        xt = xt + 1 / T * model(xt, t_batch, labels)
    return xt

# === Génération
prompt_list = ["stone_bricks4", "grass_block_top", "obsidian"]
labels = get_embedding(prompt_list)
images = sample(len(prompt_list), model, labels)

# === Affichage et sauvegarde
for i, img in enumerate(images):
    img = img.cpu() * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
    img = torch.clamp(img, 0, 1).permute(1, 2, 0)
    plt.imshow(img)
    plt.axis("off")
    plt.title(prompt_list[i])
    plt.show()

    img_path = f"generated_{prompt_list[i]}.png"
    img_to_save = (img * 255).byte().numpy()
    Image.fromarray(img_to_save).save(img_path)
    print(f"💾 Image sauvegardée sous : {img_path}")