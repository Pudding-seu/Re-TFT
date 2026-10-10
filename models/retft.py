# -*- coding: utf-8 -*-
"""Core Re-TFT modules for the revised forecasting-centred manuscript.

The Base model maps a historical sequence and target-time *calendar* vector to
one scalar endpoint. The residual module copies the Base CNN/LSTM representation,
freezes those copied parameters during residual training, and predicts one
correction. A separately fitted validation-only affine adjustment combines the
Base and residual outputs. This file contains no fallback/guard policy and makes
no claim that freezing alone causes a performance gain.

This inspectable implementation is not a checkpoint or a full reproduction of
the private training searches reported in the paper.
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
    h = Dropout(W2 · ELU(W1·a + b1) + b2)
    GRN(a) = LayerNorm(W_skip·a + b_skip + GLU([h, h]))
    GLU([h, h]) = h * sigmoid(h).

    Nonlinear transform with a projected skip connection via self.skip(x).
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

    Multi-scale CNN and LSTM encode historical load and weather features.
    Attention queries that history using target-time calendar features.
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
    Residual head with an architecture-compatible CNN/projection/LSTM encoder.

    Pretrained encoder weights are copied from the trained Base and frozen.
    Base and Residual do not share parameter objects; their GRNs are independent.
    The copied encoder is frozen during residual-stage optimization;
    the new decoder estimates error remaining after the Base prediction. The
    benchmark compares independently tuned strategies, not a causal isolation
    of the freeze operation.
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

        # ── Architecture-compatible encoder; pretrained weights copied from Base ─
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
