"""
Self-Supervised Pre-training Script for Vision Transformer on EEG Spectrograms
=============================================================================

Masked Autoencoder (MAE) pre-training for a ViT-Small encoder on 128-channel EEG
spectrograms. After pre-training, the encoder is saved and can be frozen during
fine-tuning while a regression head is trained on top.

Key properties of this implementation (vs. a common broken pattern):

  * The image is patchified and MASKED **before** the encoder. The encoder only
    ever processes the *visible* subset of patches (this is what makes it a
    "masked" autoencoder — the encoder must reconstruct from partial context).
  * A separate lightweight decoder receives the encoded visible tokens plus
    learnable mask tokens (placed back in original order) and predicts pixels.
  * The reconstruction loss is computed **only on masked patches**, optionally
    with per-patch pixel normalization (as in He et al., 2022).
  * True cosine LR schedule with linear warmup.

Architecture : ViT-Small encoder, 128-channel input, 224x224, patch 16
Optimizer    : AdamW
LR schedule  : cosine annealing with linear warmup
Loss         : MSE on masked patches (optionally pixel-normalized)

ASSUMPTIONS ABOUT `vision_transformer as vits`
----------------------------------------------
This script assumes a DINO-style ViT whose encoder exposes:
    encoder.patch_embed(x) -> (B, N, D)   # conv/linear patch embedding
    encoder.pos_embed        : (1, N+1, D) # index 0 is the CLS position
    encoder.cls_token        : (1, 1, D)
    encoder.blocks           : nn.ModuleList of transformer blocks, each block(x)->x
    encoder.norm             : final LayerNorm
    encoder.num_features     : embedding dim D
If your module names these differently, adjust `MAEPretrainer.forward_encoder`
and the `_encoder_attr` helper below — nothing else needs to change.
"""

import argparse
import datetime
import math
import pickle
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

import vision_transformer as vits


# ============================================================================
# DATA LOADING
# ============================================================================
class EEGSpectrogramDataset(Dataset):
    """
    Lazy dataset for EEG spectrogram files.

    Indexes (file, window) pairs up front and loads each window from disk on
    demand, so RAM stays flat regardless of how many files/releases are used.
    Each item is a FloatTensor of shape (128, 224, 224).

    Supported layouts per file (auto-detected):
      * .npy  with shape (128, W, 224, 224)  -> W windows, sliced along axis 1
      * .npy  with shape (128, 224, 224)     -> a single window
      * .npz  archive containing one array   -> the array is read with the same
                                                 rules as above (via its key)

    NOTE ON .npz: a .npz is a zip archive, so `np.load(path, mmap_mode="r")`
    returns an NpzFile (a lazy dict of arrays), *not* an ndarray, and mmap does
    not apply. We therefore detect the extension and, for .npz, read through the
    stored array key. True memory-mapping (zero-copy windowed reads) only works
    for .npy. If your data is really per-window .npy, prefer that for speed.
    """

    def __init__(
        self,
        data_dir: str,
        npz_pattern: str = "*.np[yz]",
        max_samples: Optional[int] = None,
        augment: bool = False,
        normalize: bool = True,
        per_channel_norm: bool = True,
        npz_key: Optional[str] = None,
        index_cache: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.normalize = normalize
        self.per_channel_norm = per_channel_norm
        self.npz_key = npz_key

        # Try to load from cache first, then fall back to building index
        if index_cache and Path(index_cache).exists():
            print(f"Loading dataset index from cache: {index_cache}")
            start = time.time()
            with open(index_cache, "rb") as f:
                self.index = pickle.load(f)
            elapsed = time.time() - start
            print(f"✓ Loaded {len(self.index)} samples in {elapsed:.2f}s")
        else:
            # Build index from scratch (original slow path)
            if index_cache:
                print(f"Cache not found: {index_cache} - building index from scratch...")
            
            files = sorted(self.data_dir.glob(npz_pattern))
            print(f"Found {len(files)} spectrogram files in {data_dir}")
            if len(files) == 0:
                raise ValueError(f"No .npy/.npz files found in {data_dir}")

            self.index: list[Tuple[str, int]] = []
            start = time.time()
            for idx, f in enumerate(files):
                if idx > 0 and idx % 500 == 0:
                    elapsed = time.time() - start
                    print(f"  Indexed {idx}/{len(files)} files ({elapsed:.1f}s)")

                try:
                    shape = self._array_shape(f)
                except Exception as e:  # noqa: BLE001 - skip unreadable files, keep going
                    print(f"Error indexing {f}: {e}")
                    continue

                if len(shape) == 4 and shape[0] == 128:
                    for w in range(shape[1]):
                        self.index.append((str(f), w))
                elif len(shape) == 3 and shape[0] == 128 and shape[-1] == 224 and shape[-2] == 224:
                    self.index.append((str(f), -1))
                else:
                    print(f"Skipping {f}: unexpected shape {shape}")

            print(f"Indexed {len(self.index)} spectrogram samples (lazy loading)")
            if len(self.index) == 0:
                raise ValueError("No spectrogram samples found")

        if max_samples:
            self.index = self.index[:max_samples]
            print(f"Limited to {len(self.index)} samples (max_samples={max_samples})")

        # Per-worker single-file cache (see note in _load_array).
        self._cache_path: Optional[str] = None
        self._cache_arr = None

    # ---- shape / array access helpers ------------------------------------
    def _array_shape(self, path: Path) -> tuple:
        if path.suffix == ".npz":
            with np.load(path) as z:
                key = self.npz_key or list(z.files)[0]
                return tuple(z[key].shape)
        # .npy: read header only via mmap, no data copy
        return tuple(np.load(path, mmap_mode="r").shape)

    def _load_array(self, path: str):
        """
        Cache the most recently opened file's array handle.

        For .npy this is a memmap (cheap to keep open); for .npz we must
        materialize the stored array (npz cannot be memmapped). The cache only
        helps when consecutive __getitem__ calls hit the same file, so pair this
        dataset with a *window-contiguous* sampler rather than global shuffling
        (see get_train_loader / GroupedShuffleSampler) to avoid I/O thrashing.
        """
        if path != self._cache_path:
            p = Path(path)
            if p.suffix == ".npz":
                with np.load(p) as z:
                    key = self.npz_key or list(z.files)[0]
                    self._cache_arr = np.asarray(z[key])
            else:
                self._cache_arr = np.load(p)  # full contiguous read (fast on Lustre)
            self._cache_path = path
        return self._cache_arr

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path, w = self.index[idx]
        arr = self._load_array(path)

        if w < 0:
            spec = np.asarray(arr, dtype=np.float32)
        else:
            spec = np.asarray(arr[:, w, :, :], dtype=np.float32)

        if spec.ndim != 3 or spec.shape[0] != 128:
            raise ValueError(
                f"Unexpected spectrogram shape: {spec.shape}, expected (128, 224, 224)"
            )

        if self.normalize:
            spec = self._normalize(spec)
        if self.augment:
            spec = self._augment_spectrogram(spec)

        return torch.from_numpy(np.ascontiguousarray(spec)).float()

    def _normalize(self, spec: np.ndarray) -> np.ndarray:
        if self.per_channel_norm:
            # min-max per channel: preserves inter-channel relative scale better
            # than a single global min/max across all 128 EEG channels.
            smin = spec.min(axis=(1, 2), keepdims=True)
            smax = spec.max(axis=(1, 2), keepdims=True)
            denom = np.where(smax > smin, smax - smin, 1.0)
            out = (spec - smin) / denom
            # channels that were flat become 0
            out = np.where(smax > smin, out, 0.0)
            return out.astype(np.float32)
        smin, smax = spec.min(), spec.max()
        return (spec - smin) / (smax - smin) if smax > smin else np.zeros_like(spec)

    @staticmethod
    def _augment_spectrogram(spec: np.ndarray) -> np.ndarray:
        spec = spec.copy()
        spec = np.clip(spec * np.random.uniform(0.9, 1.1), 0, 1)
        shift = np.random.randint(-2, 3)
        if shift != 0:
            spec = np.roll(spec, shift, axis=-1)
        spec = np.clip(spec + np.random.normal(0, 0.01, spec.shape), 0, 1)
        return spec.astype(np.float32)


class GroupedShuffleSampler(torch.utils.data.Sampler):
    """
    Shuffle *files* but keep windows of the same file contiguous.

    This gives training-time shuffling at the file level while letting the
    dataset's single-file cache actually hit, avoiding a re-open/re-read of a
    new file on nearly every sample (the problem with global shuffle=True on a
    lazily-cached dataset).
    """

    def __init__(self, index: list, seed: int = 0):
        # group dataset positions by their source file, preserving window order
        groups: dict = {}
        for pos, (path, _w) in enumerate(index):
            groups.setdefault(path, []).append(pos)
        self.groups = list(groups.values())
        self.seed = seed
        self.epoch = 0
        self._len = sum(len(g) for g in self.groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.groups), generator=g).tolist()
        for gi in order:
            yield from self.groups[gi]


# ============================================================================
# MAE DECODER BLOCK (self-contained; independent of the encoder's block API)
# ============================================================================
class DecoderBlock(nn.Module):
    """Minimal pre-norm transformer block for the MAE decoder."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# MASKED AUTOENCODER
# ============================================================================
class MAEPretrainer(nn.Module):
    """
    Masked Autoencoder wrapping a ViT encoder.

    The encoder processes only visible patches. A small decoder reconstructs the
    full patch grid and loss is taken on masked patches only.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 384,
        patch_size: int = 16,
        in_chans: int = 128,
        img_size: int = 224,
        mask_ratio: float = 0.75,
        decoder_dim: int = 256,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.img_size = img_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        self.grid = img_size // patch_size          # 14
        self.n_patches = self.grid ** 2             # 196
        self.patch_dim = patch_size * patch_size * in_chans

        # ---- decoder ----
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, decoder_dim)  # +1 for CLS slot
        )
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(decoder_dim, decoder_heads) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_dim, bias=True)

        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

    # ---- encoder attribute access (adjust here for a non-DINO ViT) --------
    def _enc(self, name: str):
        obj = getattr(self.encoder, name, None)
        if obj is None:
            raise AttributeError(
                f"Encoder has no attribute '{name}'. Adjust MAEPretrainer._enc / "
                f"forward_encoder to match your vision_transformer API."
            )
        return obj

    # ---- patch <-> image ------------------------------------------------
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, n_patches, patch_dim)."""
        B, C = imgs.shape[0], self.in_chans
        p, g = self.patch_size, self.grid
        x = imgs.reshape(B, C, g, p, g, p)
        x = x.permute(0, 2, 4, 3, 5, 1)          # B, g, g, p, p, C
        return x.reshape(B, g * g, p * p * C)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_patches, patch_dim) -> (B, C, H, W)."""
        B, C = x.shape[0], self.in_chans
        p, g = self.patch_size, self.grid
        x = x.reshape(B, g, g, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)          # B, C, g, p, g, p
        return x.reshape(B, C, g * p, g * p)

    # ---- masking --------------------------------------------------------
    def random_masking(self, x: torch.Tensor):
        """
        Per-sample random masking by shuffling.

        Args:
            x: (B, N, D) patch tokens (no CLS)
        Returns:
            x_visible: (B, len_keep, D)
            mask:      (B, N) with 1 = masked, 0 = kept
            ids_restore: (B, N) to unshuffle decoder tokens back to grid order
        """
        B, N, D = x.shape
        len_keep = max(1, int(round(N * (1 - self.mask_ratio))))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)      # ascending: keep smallest
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(
            x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)      # unshuffle to grid order

        return x_visible, mask, ids_restore

    # ---- encoder / decoder passes --------------------------------------
    def forward_encoder(self, x: torch.Tensor):
        # patch embed -> (B, N, D)
        tokens = self._enc("patch_embed")(x)
        pos_embed = self._enc("pos_embed")           # (1, N+1, D)
        cls_token = self._enc("cls_token")           # (1, 1, D)

        # add positional embedding to patch tokens (skip CLS slot at index 0)
        tokens = tokens + pos_embed[:, 1:, :]

        # mask BEFORE the encoder — encoder sees only visible patches
        tokens, mask, ids_restore = self.random_masking(tokens)

        # prepend CLS (with its positional embedding)
        cls = cls_token + pos_embed[:, :1, :]
        cls = cls.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)     # (B, 1+len_keep, D)

        for blk in self._enc("blocks"):
            tokens = blk(tokens)
        tokens = self._enc("norm")(tokens)
        return tokens, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        x = self.decoder_embed(x)                    # (B, 1+len_keep, dec_dim)
        B, N = x.shape[0], ids_restore.shape[1]

        n_mask = N + 1 - x.shape[1]                  # patches to fill with mask token
        mask_tokens = self.mask_token.expand(B, n_mask, -1)

        # drop CLS, restore original patch order, re-attach CLS
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)          # (B, N, dec_dim)
        x_ = torch.gather(
            x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[-1])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)                    # (B, N+1, dec_dim)

        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)                     # (B, N+1, patch_dim)
        return x[:, 1:, :]                           # drop CLS -> (B, N, patch_dim)

    def forward(self, imgs: torch.Tensor):
        target = self.patchify(imgs)                 # (B, N, patch_dim)
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)       # (B, N)
        loss = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1)

        # reconstructed image (for logging/visualization)
        pred_img = self.unpatchify(pred)
        return loss, pred_img, imgs, mask


# ============================================================================
# TRAINING UTILITIES
# ============================================================================
def fix_random_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def create_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.05) -> AdamW:
    return AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay, eps=1e-8)


def create_scheduler(optimizer, num_epochs: int, num_steps_per_epoch: int, warmup_epochs: int = 5):
    """Linear warmup then true cosine annealing to ~0."""
    total_steps = max(1, num_epochs * num_steps_per_epoch)
    warmup_steps = max(1, warmup_epochs * num_steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir, tag=None) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    name = f"pretrain_ckpt_{tag}.pth" if tag else f"pretrain_ckpt_ep{epoch:04d}.pth"
    path = checkpoint_dir / name
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(checkpoint_path: str, model, optimizer, scheduler) -> int:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Resuming after epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f}")
    # +1: the saved epoch already completed; continue with the next one.
    return ckpt["epoch"] + 1


# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_epoch(model, data_loader, optimizer, scheduler, device, epoch, log_interval=100) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)

    for batch_idx, batch in enumerate(data_loader):
        if batch_idx == 0:
            print(f"Starting first batch for epoch {epoch}")
        spectrograms = batch.to(device, non_blocking=True)  # (B, 128, 224, 224)

        loss, _pred, _target, _mask = model(spectrograms)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if (batch_idx + 1) % log_interval == 0:
            avg = total_loss / (batch_idx + 1)
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch} [{batch_idx + 1}/{num_batches}] "
                f"Loss: {loss.item():.6f} (Avg: {avg:.6f}) LR: {lr:.2e}"
            )

    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate(model, data_loader, device) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = len(data_loader)
    for batch in data_loader:
        spectrograms = batch.to(device, non_blocking=True)
        loss, _pred, _target, _mask = model(spectrograms)
        total_loss += loss.item()
    return total_loss / max(1, num_batches)


def build_encoder(name: str, patch_size: int, in_chans: int) -> nn.Module:
    builders = {"vit_tiny": vits.vit_tiny, "vit_small": vits.vit_small, "vit_base": vits.vit_base}
    if name not in builders:
        raise ValueError(f"Unknown model: {name}")
    return builders[name](patch_size=patch_size, img_size=[224], in_chans=in_chans, num_classes=0)


def main() -> None:
    parser = argparse.ArgumentParser("MAE pre-training for ViT on EEG spectrograms")

    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--val_data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./pretrain_checkpoints")
    parser.add_argument("--npz_key", type=str, default=None,
                        help="Array key to read from .npz files (default: first key)")
    parser.add_argument("--index_cache", type=str, default=None,
                        help="Path to cached dataset index (pkl file) for fast startup")
    parser.add_argument("--per_channel_norm", action="store_true", default=True)
    parser.add_argument("--global_norm", dest="per_channel_norm", action="store_false")

    # Model
    parser.add_argument("--model", type=str, default="vit_small")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--in_chans", type=int, default=128)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--decoder_dim", type=int, default=256)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--no_norm_pix_loss", dest="norm_pix_loss", action="store_false", default=True)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=10)

    args = parser.parse_args()

    fix_random_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print("=" * 80)
    print("EEG ViT MAE Pre-training")
    print("=" * 80)
    print(f"Timestamp     : {datetime.datetime.now()}")
    print(f"Device        : {device}")
    print(f"Data dir      : {args.data_dir}")
    print(f"Model         : {args.model}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Learning rate : {args.lr}")
    print(f"Mask ratio    : {args.mask_ratio}")
    print(f"norm_pix_loss : {args.norm_pix_loss}")
    print("=" * 80)

    # ---- data ----
    print("\nLoading training data...")
    train_dataset = EEGSpectrogramDataset(
        args.data_dir, augment=True, normalize=True,
        per_channel_norm=args.per_channel_norm, npz_key=args.npz_key,
        index_cache=args.index_cache,
    )
    train_sampler = GroupedShuffleSampler(train_dataset.index, seed=args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = None
    if args.val_data_dir:
        print("Loading validation data...")
        val_dataset = EEGSpectrogramDataset(
            args.val_data_dir, augment=False, normalize=True,
            per_channel_norm=args.per_channel_norm, npz_key=args.npz_key,
            index_cache=args.index_cache,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )

    # ---- model ----
    print(f"\nBuilding model: {args.model}")
    encoder = build_encoder(args.model, args.patch_size, args.in_chans)
    embed_dim = encoder.num_features
    model = MAEPretrainer(
        encoder=encoder, embed_dim=embed_dim, patch_size=args.patch_size,
        in_chans=args.in_chans, mask_ratio=args.mask_ratio,
        decoder_dim=args.decoder_dim, decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads, norm_pix_loss=args.norm_pix_loss,
    ).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- optim ----
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args.epochs, len(train_loader), args.warmup_epochs)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)

    # ---- train ----
    print("\nStarting pre-training...")
    print("=" * 80)
    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args.log_interval
        )
        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")

        if val_loader is not None:
            val_loss = validate(model, val_loader, device)
            print(f"Epoch {epoch}: Val Loss = {val_loss:.6f}")
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, val_loss, args.output_dir, tag="best")

        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, args.output_dir)

    print("\n" + "=" * 80)
    print("Pre-training completed!")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 80)

    # Save just the encoder weights for downstream fine-tuning.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    final_path = Path(args.output_dir) / "pretrain_final_encoder.pth"
    torch.save(model.encoder.state_dict(), final_path)
    print(f"Final encoder saved to: {final_path}")


if __name__ == "__main__":
    main()
