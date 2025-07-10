import torch
import torch.nn as nn
import math
from transformers import LongformerConfig, LongformerModel

# Discrete classes the model is allowed to predict:
ALLOWED_VALUES = [0, 2, 3, 4, 5, 6, 7, 8, -2, -3, -4, -5, -6, -7, -8,]

# ─────────────────── Longformer (classification version) ───────────────────
class Longformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        dim_feedforward: int,
        dropout: float,
        num_features: int,
        attention_window: int = 2048,
        global_positions=None,
        output_size: int = 1,
        num_labels: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        cfg = LongformerConfig(
            max_position_embeddings=4096,
            hidden_size=d_model,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=dim_feedforward,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            attention_window=[attention_window] * num_hidden_layers,
        )
        self.longformer = LongformerModel(cfg)
        self.fc_out = nn.Linear(d_model, output_size * num_labels)
        self.global_positions = global_positions
        # Buffer to map indices → allowed discrete values
        self.register_buffer("allowed", torch.tensor(ALLOWED_VALUES, dtype=torch.float))

    def forward(self, x, attention_mask=None):
        b, s, _ = x.shape
        x = self.input_proj(x)
        if attention_mask is None:
            attention_mask = torch.ones((b, s), dtype=torch.long, device=x.device)

        hidden = self.longformer(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state
        logits = self.fc_out(hidden)  # shape (B, S, output_size)

        # After reshaping logits to (B, S, L, C)
        probs = torch.softmax(logits, dim=-1)
        pred_indices = probs.argmax(dim=-1)  # shape: (B, S, L)
        allowed = self.allowed.to(pred_indices.device)
        valid_mask = (pred_indices >= 0) & (pred_indices < allowed.shape[0])
        preds = torch.zeros_like(pred_indices, dtype=allowed.dtype, device=allowed.device)
        preds[valid_mask] = allowed[pred_indices[valid_mask]]

        return logits, preds



# ─────────────────────── Transformer (classification version) ───────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0)),


    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        num_hidden_layers,
        num_attention_heads,
        dim_feedforward,
        dropout,
        num_features,
        output_size=1,
        max_seq_len=4096,
        num_labels=1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_seq_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_hidden_layers)
        self.fc_out = nn.Linear(d_model, output_size * num_labels)
        self.register_buffer("allowed", torch.tensor(ALLOWED_VALUES, dtype=torch.float))

    def forward(self, x, attention_mask=None):
        x = self.pos_enc(self.input_proj(x))
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        hidden = self.encoder(x, src_key_padding_mask=pad_mask)
        logits = self.fc_out(hidden)  # shape (B, S, output_size)

        probs = torch.softmax(logits, dim=-1)
        pred_indices = torch.argmax(probs, dim=-1)
        preds = self.allowed[pred_indices]
        return logits, preds


# ─────────────────────────────  TCN (classification version)  ────────────────────────────
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        final = out + res
        return final




class TCN(nn.Module):
    def __init__(
        self,
        d_model,
        num_levels,
        kernel_size,
        dropout,
        num_features,
        output_size,
        tcn_padding="same",
        tcn_channel_growth=False,
        num_labels=1,
        max_dilation=128,  # Add this (or pass it in config later)
        num_values=1,
        **kwargs,
    ):
        super().__init__()
        self.output_size = output_size
        self.num_labels = num_labels
        self.num_values = num_values

        self.input_proj = nn.Linear(num_features, d_model)
        channels = [d_model] * num_levels

        self.tcn = _TCNWrapper(
            num_inputs=d_model,
            num_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            tcn_padding=tcn_padding,
            max_dilation=max_dilation  # ← clamp the max dilation
        )

        self.fc_out = nn.Linear(d_model, output_size * num_labels)
        self.register_buffer("allowed", torch.tensor(ALLOWED_VALUES, dtype=torch.float))


    def forward(self, x, attention_mask=None):
        x = self.input_proj(x)       # (B, T, d_model)
        x = x.permute(0, 2, 1)       # (B, d_model, T) for Conv1d
        hidden = self.tcn(x)    # (B, C, T)
        hidden = hidden.permute(0, 2, 1)  # (B, T, C)
        logits = self.fc_out(hidden)

        B, T, _ = logits.shape
        logits = logits.view(B, T, 2, self.num_values)
        preds = torch.argmax(logits, dim=-1)
        
        return logits, preds


    
class _TCNWrapper(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.1, tcn_padding="same", max_dilation=None):
        super().__init__()
        layers = []
        receptive_field = 1
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            if max_dilation is not None:
                dilation = min(dilation, max_dilation)

            padding = (kernel_size - 1) * dilation // 2  # safer for "same"

            rf_this_layer = (kernel_size - 1) * dilation
            receptive_field += rf_this_layer

            print(f"TCN Layer {i+1}: in={in_ch}, out={out_ch}, dilation={dilation}, padding={padding}, RF+={rf_this_layer}")

            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size, 1, dilation, padding, dropout
                )
            )

        print(f"Total Effective Receptive Field: {receptive_field} time steps")
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
