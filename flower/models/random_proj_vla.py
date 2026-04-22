"""
RandomProjVLA — FLOWER VLA ablation that replaces Florence-2-large with a
fixed (frozen) random projection.

Architecture
------------
* Image encoder: images are split into 16×16 patches; each patch vector is
  projected to ``proj_dim`` dimensions by a **frozen** random linear layer.
* Text encoder: text is tokenised with a standard BERT tokeniser and embedded
  via a **frozen** random embedding matrix.
* Action model: identical Flow Matching DiT from the original FLOWER VLA,
  trained from scratch on the CALVIN D→D dataset.

This serves as a control experiment: if the VLM features matter, the random
projection model should perform substantially worse than the Florence-2 model.
"""

import logging
import math
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from flower.models.flower import FLOWERVLA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Random-projection image/text encoder
# ---------------------------------------------------------------------------

class RandomPatchEncoder(nn.Module):
    """
    Splits an image into non-overlapping patches and applies a fixed (frozen)
    random linear projection to each patch.

    Parameters
    ----------
    image_size : int
        Expected spatial resolution (height = width).
    patch_size : int
        Size of each square patch (pixels).
    in_channels : int
        Number of image channels (3 for RGB).
    proj_dim : int
        Output dimensionality of each patch token.
    """

    def __init__(
        self,
        image_size: int = 112,
        patch_size: int = 16,
        in_channels: int = 3,
        proj_dim: int = 1024,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.in_channels = in_channels
        self.proj_dim = proj_dim

        n_patches_per_side = image_size // patch_size
        self.num_patches = n_patches_per_side * n_patches_per_side
        self.patch_dim = in_channels * patch_size * patch_size  # e.g. 3*16*16 = 768

        # Fixed random projection — weight is drawn once and frozen
        self.proj = nn.Linear(self.patch_dim, proj_dim, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / math.sqrt(self.patch_dim))
        for p in self.proj.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W) image tensor.

        Returns
        -------
        tokens : (B, num_patches, proj_dim)
        """
        B, C, H, W = x.shape
        ph = pw = self.patch_size

        # Unfold into patches: (B, C, nh, nw, ph, pw)
        patches = x.unfold(2, ph, ph).unfold(3, pw, pw)
        nh, nw = patches.shape[2], patches.shape[3]

        # (B, nh*nw, C*ph*pw)
        patches = patches.contiguous().view(B, C, nh * nw, ph * pw)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B, nh * nw, C * ph * pw)

        # Random projection: (B, num_patches, proj_dim)
        return self.proj(patches)


class RandomTextEncoder(nn.Module):
    """
    Tokenises text with a BERT tokeniser and embeds each token using a fixed
    (frozen) random embedding matrix.

    Parameters
    ----------
    tokenizer_name : str
        HuggingFace tokeniser identifier.
    proj_dim : int
        Embedding dimension.
    max_length : int
        Maximum number of tokens per sentence.
    """

    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        proj_dim: int = 1024,
        max_length: int = 32,
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        vocab_size = self.tokenizer.vocab_size

        # Fixed random embeddings — frozen
        self.embed = nn.Embedding(vocab_size, proj_dim)
        with torch.no_grad():
            nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        for p in self.embed.parameters():
            p.requires_grad = False

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Parameters
        ----------
        texts : list of B strings.
        device : target device.

        Returns
        -------
        embeds : (B, seq_len, proj_dim)
        """
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(device)
        input_ids = enc["input_ids"].clamp(0, self.embed.num_embeddings - 1)
        return self.embed(input_ids)


# ---------------------------------------------------------------------------
# RandomProjVLA
# ---------------------------------------------------------------------------

class RandomProjVLA(FLOWERVLA):
    """
    FLOWER VLA variant where the Florence-2-large VLM is replaced by a fixed
    random projection encoder.  Only the DiT action model is trained.

    Extra constructor parameters
    ----------------------------
    rp_image_size : int
        Input image resolution (height = width in pixels).
    rp_patch_size : int
        Patch size for the image encoder.
    rp_proj_dim : int
        Projection dimension (replaces ``vlm.config.text_config.d_model``).
    rp_text_max_len : int
        Maximum number of text tokens.
    rp_tokenizer : str
        HuggingFace tokeniser name for text encoding.

    All standard FLOWERVLA constructor parameters are accepted but
    ``vlm_path``, ``freeze_florence``, and ``freeze_vision_tower`` are ignored.
    """

    def __init__(
        self,
        # Random-projection specific
        rp_image_size: int = 112,
        rp_patch_size: int = 16,
        rp_proj_dim: int = 1024,
        rp_text_max_len: int = 32,
        rp_tokenizer: str = "bert-base-uncased",
        # Standard FLOWERVLA params (vlm ones are ignored)
        vlm_path: str = "none",
        freeze_florence: bool = False,
        freeze_vision_tower: bool = False,
        vlm_prompt_style: str = "default",
        token_dropout: float = 0.1,
        # Model structure
        multistep: int = 10,
        num_sampling_steps: int = 4,
        lowdim_obs_dim: int = 7,
        action_dim: int = 7,
        act_window_size: int = 10,
        # Model flags
        use_second_view: bool = True,
        second_view_key: str = "image_wrist",
        action_type_adaln: bool = True,
        use_causal_attention: bool = True,
        use_cross_attn: bool = True,
        use_adaln_cond: bool = False,
        use_readout_token: bool = False,
        use_proprio: bool = False,
        return_act_chunk: bool = False,
        # DiT configuration
        sampling_type: str = "uniform",
        dit_dim: int = 1024,
        n_heads: int = 16,
        n_layers: int = 18,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_pdrop: float = 0.1,
        # RoPE configuration
        use_rope: bool = True,
        use_nope: bool = False,
        query_seq_len: int = 100,
        rope_theta: float = 32.0,
        # Optimizer
        optimizer_type: str = "adamw",
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,
        # No pretrained weights for random projection
        load_pretrained: bool = False,
        pretrained_model_path: str = None,
    ):
        # Store random-projection settings BEFORE super().__init__ is called,
        # because super().__init__ calls self._setup_vlm() via polymorphism.
        self._rp_image_size = rp_image_size
        self._rp_patch_size = rp_patch_size
        self._rp_proj_dim = rp_proj_dim
        self._rp_text_max_len = rp_text_max_len
        self._rp_tokenizer = rp_tokenizer

        super().__init__(
            vlm_path=vlm_path,
            freeze_florence=freeze_florence,
            freeze_vision_tower=freeze_vision_tower,
            vlm_prompt_style=vlm_prompt_style,
            token_dropout=token_dropout,
            multistep=multistep,
            num_sampling_steps=num_sampling_steps,
            lowdim_obs_dim=lowdim_obs_dim,
            action_dim=action_dim,
            act_window_size=act_window_size,
            use_second_view=use_second_view,
            second_view_key=second_view_key,
            action_type_adaln=action_type_adaln,
            use_causal_attention=use_causal_attention,
            use_cross_attn=use_cross_attn,
            use_adaln_cond=use_adaln_cond,
            use_readout_token=use_readout_token,
            use_proprio=use_proprio,
            return_act_chunk=return_act_chunk,
            sampling_type=sampling_type,
            dit_dim=dit_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_pdrop=mlp_pdrop,
            use_rope=use_rope,
            use_nope=use_nope,
            query_seq_len=query_seq_len,
            rope_theta=rope_theta,
            optimizer_type=optimizer_type,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            load_pretrained=False,          # always False for random projection
            pretrained_model_path=None,
        )

    # ------------------------------------------------------------------
    # Override: replace Florence-2 with random projection modules
    # ------------------------------------------------------------------

    def _setup_vlm(self, vlm_path: str, freeze_vision_tower: bool, freeze_florence: bool):
        """Install random-projection encoders instead of Florence-2."""

        logger.info(
            "RandomProjVLA: skipping Florence-2 loading; "
            "using frozen random projection encoder "
            f"(proj_dim={self._rp_proj_dim}, patch_size={self._rp_patch_size})."
        )

        # --- Mock VLM object ------------------------------------------------
        # FLOWERVLA.__init__ reads hidden_dim = self.vlm.config.text_config.d_model
        # after _setup_vlm returns.  We create a lightweight namespace that
        # satisfies this attribute access.
        _proj_dim = self._rp_proj_dim

        class _TextCfg:
            d_model = _proj_dim

        class _Cfg:
            text_config = _TextCfg()

        self.vlm = types.SimpleNamespace(config=_Cfg())

        # --- Random patch encoder -------------------------------------------
        self.rp_image_enc = RandomPatchEncoder(
            image_size=self._rp_image_size,
            patch_size=self._rp_patch_size,
            in_channels=3,
            proj_dim=self._rp_proj_dim,
        )

        # --- Random text encoder --------------------------------------------
        self.rp_text_enc = RandomTextEncoder(
            tokenizer_name=self._rp_tokenizer,
            proj_dim=self._rp_proj_dim,
            max_length=self._rp_text_max_len,
        )
        # Expose the tokenizer at the top level (parent code may reference it)
        self.tokenizer = self.rp_text_enc.tokenizer

        # --- Prompt token ---------------------------------------------------
        self.prompt_embeds = nn.Parameter(
            torch.zeros(1, 1, self._rp_proj_dim),
            requires_grad=False,
        )

        # --- Token dropout --------------------------------------------------
        self.vlm_token_dropout = nn.Dropout(self.token_dropout)

    # ------------------------------------------------------------------
    # Override: encode observations with random projection
    # ------------------------------------------------------------------

    def encode_observations(self, batch: Dict) -> Dict:
        device = self.device
        default_type = next(self.parameters()).dtype

        # --- Primary (static) camera ----------------------------------------
        image_tensor = batch["rgb_obs"]["rgb_static"]          # [B, T, C, H, W]
        B, T, C, H, W = image_tensor.shape

        image_features = self.rp_image_enc(
            image_tensor.view(-1, C, H, W).to(device).to(default_type)
        )  # [B*T, num_patches, proj_dim]
        num_patches = image_features.shape[1]
        image_features = image_features.view(B, T * num_patches, -1)

        # --- Second (gripper) camera ----------------------------------------
        if self.use_second_view:
            image2_tensor = batch["rgb_obs"]["rgb_gripper"]    # [B, T, C, H, W]
            image2_features = self.rp_image_enc(
                image2_tensor.view(-1, C, H, W).to(device).to(default_type)
            )
            image2_features = image2_features.view(B, T * num_patches, -1)
            image_features = torch.cat([image_features, image2_features], dim=1)

        # --- Text embeddings ------------------------------------------------
        text_embeds = self.rp_text_enc(
            batch["lang_text"], device
        ).to(default_type)  # [B, text_seq, proj_dim]

        # --- Concatenate visual + prompt + text -----------------------------
        task_prompt = self.prompt_embeds.expand(B, -1, -1).to(device)
        merged = torch.cat(
            [image_features, task_prompt, text_embeds.to(image_features.device)],
            dim=1,
        )  # [B, total_tokens, proj_dim]

        # --- Token dropout --------------------------------------------------
        features = self.vlm_token_dropout(merged)

        # --- Auxiliary conditioning tensors ---------------------------------
        embed_tensor = torch.zeros(B, 1, 1, device=device)
        action_type_tensor = torch.ones(B, self.act_window_size, 7, device=device)

        frequency_embeds = self.frequency_embedder(
            torch.ones_like(embed_tensor) * 3
        )

        return {
            "features": features,
            "frequency_embeds": frequency_embeds,
            "action_space_embeds": None,
            "action_type": action_type_tensor,
            "proprio": None,
            "attention_mask": torch.ones(B, merged.shape[1], device=device),
        }

    # ------------------------------------------------------------------
    # Override: hooks that reference self.vlm as a nn.Module
    # ------------------------------------------------------------------

    def on_train_start(self):
        """Move trainable components to device; skip vlm (it is not a Module)."""
        self.to(self.device)

    def print_model_parameters(self):
        """Print model parameter counts, distinguishing frozen vs. trainable."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"RandomProjVLA — Total: {total:,}  Trainable: {trainable:,}  Frozen: {frozen:,}")
        for name, mod in self.named_modules():
            if "." not in name or name.count(".") <= 1:
                n = sum(p.numel() for p in mod.parameters())
                if n > 0:
                    print(f"  {name}: {n:,}")
