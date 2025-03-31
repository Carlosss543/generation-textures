import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from DiT import DiT
from config import config
from load_dataset import train_loader, get_embedding
from fonctions_diffusion import noise_imgs, sample



# hyperparameters
device = config["device"]
img_size = config["img_size"]
img_channels = config["img_channels"]
mean = config["mean"]
std = config["std"]
n_classes = config["n_classes"]
batch_size = config["batch_size"]
n_epochs = config["n_epochs"]
learning_rate = config["learning_rate"]
T = config["T"]



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



# load pretrained model
start_epoch = 10
if start_epoch > 0:
    model.load_state_dict(torch.load(f"./sauvegardes_entrainement/model_epoch_{start_epoch}.pth"))
    optimizer.load_state_dict(torch.load(f"./sauvegardes_entrainement/optimizer_epoch_{start_epoch}.pth"))
    loss_list = torch.load(f"./sauvegardes_entrainement/loss_list_epoch_{start_epoch}.pth")
    print(f"Model loaded from epoch {start_epoch}")



for epoch in tqdm(range(start_epoch+1, n_epochs)):
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


    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f"./sauvegardes_entrainement/model_epoch_{epoch+1}.pth")
        torch.save(optimizer.state_dict(), f"./sauvegardes_entrainement/optimizer_epoch_{epoch+1}.pth")
        torch.save(loss_list, f"./sauvegardes_entrainement/loss_list_epoch_{epoch+1}.pth")


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
