# -*- coding: utf-8 -*-
"""
Re-TFT Model Definitions
========================
Implements three network classes used by the Re-TFT framework:

  GRN                  -- Gated Residual Network (shared building block)
  CNNFrontTFT          -- Base model: multi-scale dilated CNN → LSTM → MHA → GRN
  CNN2TFTLSTMResidual  -- Residual correction head sharing the CNN+LSTM backbone
                          with the base model (frozen during residual training)

Architecture overview (paper Figure 1):

  Input (T × F)
      │
  ┌───┴──────────────────────┐
  │  Multi-Scale Dilated CNN  │  dilation rates: 1, 2, 4
  │  (3 parallel branches)    │
  └───┬──────────────────────┘
      │ concat + enc_proj
  ┌───┴──────────────────────┐
  │  LSTM encoder             │  produces h_T (global trend summary)
  └───┬──────────────────────┘
      │                          ← FROZEN during residual training stage
  ════╪════════════════════════════════════════════════════════════
      │                          ← trainable below in Base Head
  ┌───┴──────────────────────┐
  │  GRN (context fusion)     │  ctx = GRN([h_T, z_fut])
  │  Multi-Head Attention      │  Q=ctx, K/V=LSTM states
  │  GRN (output fusion)       │
  │  Linear → ŷ_base           │
  └───────────────────────────┘

  Residual Head (same frozen backbone, new lightweight decoder):
      sg(h_T)  ← stop-gradient; backbone params frozen
          │
      GRN([sg(h_T), z_fut])  → Linear → ê_res

  Final prediction:
      ŷ_final = ŷ_base + I_guard · (α · ê_res + β)

  Guard (Eq. 9 in paper — bypassed in benchmark by setting thresholds to 0):
      I_guard = 1  if  ρ(ê_res_val, e_val) > τ_corr  AND  ΔRMSE > δ_gain
              = 0  otherwise  →  fall back to ŷ_base
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Gated Residual Network (GRN)
# Paper Section 2.4; also see original TFT paper (Lim et al., 2021)
# ─────────────────────────────────────────────────────────────────────────────
class GRN(nn.Module):
    """
    GRN(a) = LayerNorm(a + GLU(W2 · ELU(W1·a + b1) + b2))

    Acts as a dynamic operational filter: suppresses grid noise while
    amplifying physically relevant state changes.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        # GLU halves the last dimension: feed [h, h] → output dim = hidden_dim
        self.gate = nn.GLU()
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = self.gate(torch.cat([h, h], dim=-1))
        return self.norm(h + self.skip(x))


# ─────────────────────────────────────────────────────────────────────────────
# Base Model: CNNFrontTFT
# Paper Section 2.4 – 2.5
# ─────────────────────────────────────────────────────────────────────────────
class CNNFrontTFT(nn.Module):
    """
    Multi-scale dilated CNN encoder → LSTM → MHA + GRN decoder.

    The CNN bank captures human-driven diurnal profiles and weather-induced
    periodicities at multiple resolutions (dilation rates 1, 2, 4).
    The LSTM encodes macroscopic cross-day/cross-week load trends.
    MHA retrieves relevant historical context conditioned on known future
    calendar features.
    """

    def __init__(
        self,
        d_enc_in: int,
        d_known_in: int,
        conv_out_each: int = 32,
        conv_kernels: tuple = (3, 5, 7),
        conv_dilations: tuple = (1, 2, 4),
        d_model: int = 64,
        n_heads: int = 4,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        act: str = "relu",
    ):
        super().__init__()
        assert len(conv_kernels) == len(conv_dilations)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # ── Multi-scale dilated convolution bank ──────────────────────────
        self.convs = nn.ModuleList()
        for k, d in zip(conv_kernels, conv_dilations):
            assert k % 2 == 1, "Use odd kernel sizes to preserve sequence length"
            pad = (k - 1) * d // 2
            self.convs.append(
                nn.Conv1d(d_enc_in, conv_out_each, kernel_size=k, dilation=d, padding=pad)
            )
        self.act = nn.ReLU() if act.lower() == "relu" else nn.ELU()
        self.cnn_dropout = nn.Dropout(dropout)

        # ── LSTM sequence encoder (long-range temporal context) ────────────
        total_c = d_enc_in + len(self.convs) * conv_out_each
        self.enc_proj = nn.Linear(total_c, d_model)
        self.enc_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── TFT-style decoder head ─────────────────────────────────────────
        self.dec_known_proj = nn.Linear(d_known_in, d_model)
        self.ctx_grn = GRN(2 * d_model, d_model, dropout=dropout)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.fuse_grn = GRN(2 * d_model, d_model, dropout=dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, enc_seq: torch.Tensor, dec_known: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc_seq:   [B, T, F_enc]  historical encoder sequence
            dec_known: [B, F_known]   known future calendar embeddings

        Returns:
            y_hat: [B]  point forecast (scaled domain)
        """
        x0 = enc_seq
        x_conv = x0.transpose(1, 2)                         # [B, F, T]
        conv_feats = [self.act(c(x_conv)) for c in self.convs]
        c = self.cnn_dropout(torch.cat(conv_feats, dim=1)).transpose(1, 2)  # [B, T, C*K]

        x = torch.cat([x0, c], dim=-1)                      # [B, T, F+C*K]
        x = F.elu(self.enc_proj(x))                         # [B, T, d_model]
        enc_out, (h_n, _) = self.enc_lstm(x)
        h_last = h_n[-1]                                     # [B, d_model]

        kf = F.elu(self.dec_known_proj(dec_known))           # [B, d_model]
        ctx = self.ctx_grn(torch.cat([h_last, kf], dim=-1))  # [B, d_model]

        q = ctx.unsqueeze(1)                                 # [B, 1, d_model]
        attn_out, _ = self.mha(q, enc_out, enc_out)          # [B, 1, d_model]
        z = self.fuse_grn(torch.cat([ctx, attn_out.squeeze(1)], dim=-1))
        return self.fc_out(z).squeeze(-1)                    # [B]


# ─────────────────────────────────────────────────────────────────────────────
# Residual Correction Head: CNN2TFTLSTMResidual
# Paper Section 2.6 — Sequential Residual Correction on Frozen Representations
# ─────────────────────────────────────────────────────────────────────────────
class CNN2TFTLSTMResidual(nn.Module):
    """
    Residual correction head that shares the CNN + LSTM backbone with CNNFrontTFT.

    Key design choices (paper Section 2.6.1):
      - The backbone (convs + enc_proj + enc_lstm) is FROZEN during residual
        training, i.e.,  ∇_{Θ_enc} L_res ≡ 0  (stop-gradient on h_T).
      - This prevents gradients from rare extreme events propagating back into
        the shared encoder and distorting the smooth trend representation needed
        by the Base Head.
      - The residual branch learns ONLY the structured peak-sensitive error
        e_t = y_t − ŷ_{base,t}  on a fixed latent space.

    Lightweight decoder: GRN([sg(h_T), z_fut]) → Linear → ê_res
    """

    def __init__(
        self,
        d_enc_in: int,
        d_known_in: int,
        conv_out_each: int = 32,
        conv_kernels: tuple = (3, 5, 7),
        conv_dilations: tuple = (1, 2, 4),
        d_model: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        act: str = "relu",
        init_from_base_state_dict=None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        assert len(conv_kernels) == len(conv_dilations)

        # ── Shared backbone (identical architecture to CNNFrontTFT encoder) ─
        self.convs = nn.ModuleList()
        for k, d in zip(conv_kernels, conv_dilations):
            pad = (k - 1) * d // 2
            self.convs.append(
                nn.Conv1d(d_enc_in, conv_out_each, kernel_size=k, dilation=d, padding=pad)
            )
        self.act = nn.ReLU() if act.lower() == "relu" else nn.ELU()
        self.cnn_dropout = nn.Dropout(dropout)

        total_c = d_enc_in + len(self.convs) * conv_out_each
        self.enc_proj = nn.Linear(total_c, d_model)
        self.enc_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── Lightweight residual decoder (only these are trained) ──────────
        self.dec_known_proj = nn.Linear(d_known_in, d_model)
        self.fuse_grn = GRN(2 * d_model, d_model, dropout=dropout)
        self.fc_out = nn.Linear(d_model, 1)

        # ── Weight initialisation from base checkpoint ─────────────────────
        if init_from_base_state_dict is not None:
            self._load_backbone(init_from_base_state_dict)

        # ── Freeze backbone: ∇_{Θ_enc} L_res ≡ 0 ─────────────────────────
        if freeze_backbone:
            for module in (self.convs, self.enc_proj, self.enc_lstm):
                for p in module.parameters():
                    p.requires_grad = False

    @torch.no_grad()
    def _load_backbone(self, base_sd: dict):
        own_sd = self.state_dict()
        n_copied = 0
        for key in base_sd:
            if key.startswith(("convs.", "enc_proj.", "enc_lstm.")):
                if key in own_sd and own_sd[key].shape == base_sd[key].shape:
                    own_sd[key].copy_(base_sd[key])
                    n_copied += 1
        self.load_state_dict(own_sd)
        print(f"[Residual] Backbone weights loaded: {n_copied} tensors copied from base.")

    def forward(self, enc_seq: torch.Tensor, dec_known: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc_seq:   [B, T, F_enc]
            dec_known: [B, F_known]

        Returns:
            e_hat: [B]  predicted residual correction (original-scale units)
        """
        x0 = enc_seq
        x = x0.transpose(1, 2)
        conv_feats = [self.act(c(x)) for c in self.convs]
        c = torch.cat(conv_feats, dim=1).transpose(1, 2)
        x = torch.cat([x0, self.cnn_dropout(c)], dim=-1)
        x = F.elu(self.enc_proj(x))
        _, (h_n, _) = self.enc_lstm(x)
        h_last = h_n[-1]                                     # stop-gradient applied via frozen params

        kf = F.elu(self.dec_known_proj(dec_known))
        z = self.fuse_grn(torch.cat([h_last, kf], dim=-1))
        return self.fc_out(z).squeeze(-1)                    # [B]
