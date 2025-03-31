import torch
import torch.nn as nn
from config import config



device = config["device"]
img_size = config["img_size"]
img_channels = config["img_channels"]
n_classes = config["n_classes"]
patch_size = config["patch_size"]
n_blocks = config["n_blocks"]
n_heads = config["n_heads"]
embd_dim = config["embd_dim"]
dropout = config["dropout"]



class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(embd_dim, elementwise_affine=False)
        self.mha = nn.MultiheadAttention(embd_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embd_dim, elementwise_affine=False)
        self.ffwd = nn.Sequential(
            nn.Linear(embd_dim, 4*embd_dim),
            nn.GELU(),
            nn.Linear(4*embd_dim, embd_dim),
            nn.Dropout(dropout)
        )
        # scale and shift parameter for ln1 (2*embd_dim)
        # scale and shift parameter for ln2 (2*embd_dim)
        # scale for output of attention prior to residual connection (embd_dim)
        # scale for output of mlp prior to residual connection (embd_dim)
        # total : (6, embd_dim)
        self.adaLN_zero = nn.Sequential(
            nn.Linear(embd_dim, embd_dim),
            nn.SiLU(),
            nn.Linear(embd_dim, 6*embd_dim)
        )
        # initialized at 0 to have the identity function at initialization (all residual connections at 0)
        nn.init.constant_(self.adaLN_zero[2].weight, 0)
        nn.init.constant_(self.adaLN_zero[2].bias, 0)

    def forward(self, x, condition):
        scale_shift_params = self.adaLN_zero(condition).unsqueeze(1).chunk(6, dim=-1)    # returns a tuple of the 6 parameters
        pre_attn_shift, pre_attn_scale, post_attn_scale, pre_mlp_scale, pre_mlp_shift, post_mlp_scale = scale_shift_params
    
        x1 = self.ln1(x) * (1 + pre_attn_shift) + pre_attn_scale
        x1, _ = self.mha(x1, x1, x1, need_weights=False)
        x = x + x1 * post_attn_scale

        x2 = self.ln2(x) * (1 + pre_mlp_scale) + pre_mlp_shift
        x2 = self.ffwd(x2)
        x = x + x2 * post_mlp_scale
        
        return x



class ViT(nn.Module):

    def __init__(self):
        super().__init__()
        self.patchify = nn.Conv2d(img_channels, embd_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding_table = nn.Embedding((img_size//patch_size)**2, embd_dim)
        self.register_buffer("pos_indices", torch.arange((img_size // patch_size) ** 2))
        self.t_embedding = nn.Sequential(
            nn.Linear(1, embd_dim),
            nn.SiLU(),
            nn.Linear(embd_dim, embd_dim)
        )
        if n_classes is not None:
            #self.label_embedding_table = nn.Embedding(n_classes, embd_dim)
            self.label_embedding_table = nn.Linear(512, embd_dim)
        self.transformer_blocks = nn.ModuleList([TransformerBlock() for _ in range(n_blocks)])
        self.final_ln = nn.LayerNorm(embd_dim, elementwise_affine=False)
        self.unpatchify = nn.Linear(embd_dim, patch_size**2 * img_channels)
        # scale and shift for the final layernorm
        self.adaLN_zero_unpatchify = nn.Sequential(
            nn.Linear(embd_dim, embd_dim),
            nn.SiLU(),
            nn.Linear(embd_dim, 2*embd_dim)
        )
        nn.init.constant_(self.adaLN_zero_unpatchify[2].weight, 0)
        nn.init.constant_(self.adaLN_zero_unpatchify[2].bias, 0)
    
    def forward(self, imgs, t, labels):
        token_embd = self.patchify(imgs)
        token_embd = token_embd.view(token_embd.shape[0], token_embd.shape[1], -1).permute(0, 2, 1)
        pos_embd = self.pos_embedding_table(torch.arange((img_size//patch_size)**2, device=device))
        x = token_embd + pos_embd

        time_embd = self.t_embedding(t)
        if labels is not None:
            time_embd += self.label_embedding_table(labels)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, time_embd)

        pre_mlp_scale, pre_mlp_shift = self.adaLN_zero_unpatchify(time_embd).unsqueeze(1).chunk(2, dim=-1)
        x = self.final_ln(x) * (1 + pre_mlp_scale) + pre_mlp_shift
        x = self.unpatchify(x)
        x = x.view(-1, img_size//patch_size, img_size//patch_size, img_channels, patch_size, patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(-1, img_channels, img_size, img_size)

        return x
