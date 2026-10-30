"""Our own Vision Transformer (Lecture 1 implementation) with pos_embed "none", a conv stem and timm-compatible hooks."""

import timm
import torch
import torch.nn.functional as F
from torch import nn


class PatchEmbed(nn.Module):
    """Image -> sequence of patch tokens. (B, C, H, W) -> (B, N, D)"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "image size must be divisible by the patch size"
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class ConvStem(nn.Module):
    """Stride-2 3x3 convs down to the patch grid, then a 1x1 projection (Xiao et al. 2021)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        n_down = patch_size.bit_length() - 1
        assert 2 ** n_down == patch_size, "the conv stem needs a power-of-two patch size"
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        layers, c_in = [], in_chans
        for i in range(n_down):
            c_out = embed_dim // 2 ** (n_down - 1 - i)
            # GroupNorm: a batch holds tiles of only a few soils, so BatchNorm statistics would be biased
            layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, c_out), nn.GELU()]
            c_in = c_out
        layers.append(nn.Conv2d(c_in, embed_dim, 1))
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # If True, attention is computed explicitly and kept for attention maps
        self.store_attn = False
        self.attn_map = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, h, N, d)
        if self.store_attn:
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            self.attn_map = attn.detach()
            x = self.attn_drop(attn) @ v
        else:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)  # concat heads
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DropPath(nn.Module):
    """Stochastic depth: drop the whole residual branch per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
        return x * mask / keep_prob


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, temperature: float = 10000.0) -> torch.Tensor:
    """(grid_size**2, embed_dim) fixed positional embeddings in raster order."""
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos embeddings"
    rows, cols = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing="ij")
    omega = 1.0 / temperature ** (torch.arange(embed_dim // 4) / (embed_dim // 4))
    r = rows.flatten()[:, None].float() * omega[None]
    c = cols.flatten()[:, None].float() * omega[None]
    return torch.cat([c.sin(), c.cos(), r.sin(), r.cos()], dim=1)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        global_pool="avg",       # "token" ([CLS]) or "avg"
        pos_embed_type="learn",  # "learn", "sincos" or "none"
        stem="patch",            # "patch" or "conv"
    ):
        super().__init__()
        assert global_pool in ("token", "avg") and pos_embed_type in ("learn", "sincos", "none")
        self.global_pool = global_pool
        self.num_prefix_tokens = 1 if global_pool == "token" else 0
        self.num_features = embed_dim
        self.pretrained_cfg = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}  # as the AugReg checkpoints

        self.patch_embed = (ConvStem if stem == "conv" else PatchEmbed)(img_size, patch_size, in_chans, embed_dim)
        num_tokens = self.num_prefix_tokens + self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if global_pool == "token" else None
        if pos_embed_type == "learn":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_type == "sincos":
            pe = get_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid_size)
            pe = torch.cat([torch.zeros(self.num_prefix_tokens, embed_dim), pe])[None]
            self.register_buffer("pos_embed", pe, persistent=False)
        else:
            self.register_buffer("pos_embed", torch.zeros(1, num_tokens, embed_dim), persistent=False)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dpr[i]) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def no_weight_decay(self) -> set[str]:
        return {"pos_embed", "cls_token"}

    def group_matcher(self, coarse: bool = False) -> dict:
        """Layer ids for layer-wise LR decay (timm's convention)."""
        return {"stem": r"^cls_token|^pos_embed|^patch_embed", "blocks": [(r"^blocks\.(\d+)", None), (r"^norm", (99999,))]}

    def forward_features(self, x):
        x = self.patch_embed(x)  # (B, N, D)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)  # (B, 1 + N, D)
        x = self.pos_drop(x + self.pos_embed)
        return self.norm(self.blocks(x))

    def forward(self, x):
        x = self.forward_features(x)
        return x[:, 0] if self.global_pool == "token" else x[:, self.num_prefix_tokens:].mean(dim=1)

    def set_store_attn(self, flag: bool = True):
        for blk in self.blocks:
            blk.attn.store_attn = flag
            blk.attn.attn_map = None

    def get_attn_maps(self) -> list[torch.Tensor]:
        return [blk.attn.attn_map for blk in self.blocks]


def load_timm_weights(model: VisionTransformer, name: str) -> None:
    """Copy the weights of a timm ViT checkpoint into our model (the head is dropped)."""
    state = timm.create_model(name, pretrained=True).state_dict()
    state = {k: v for k, v in state.items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, f"architecture differs from {name}: {missing} {unexpected}"


@torch.no_grad()
def attention_rollout(model: VisionTransformer, x: torch.Tensor) -> torch.Tensor:
    """Attention rollout (Abnar & Zuidema 2020) as a (B, grid, grid) map."""
    model.set_store_attn(True)
    model(x)
    maps = model.get_attn_maps()
    model.set_store_attn(False)
    n = maps[0].shape[-1]
    eye = torch.eye(n, device=x.device)
    rollout = eye.expand(len(x), n, n)
    for a in maps:
        a = 0.5 * a.float().mean(dim=1) + 0.5 * eye  # heads averaged, residual added
        rollout = (a / a.sum(dim=-1, keepdim=True)) @ rollout
    p = model.num_prefix_tokens
    out = rollout[:, 0, p:] if p else rollout.mean(dim=1)  # [CLS] row, or mean over patches
    g = model.patch_embed.grid_size
    return out.reshape(len(x), g, g)
