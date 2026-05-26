import math
import torch
import torch.nn as nn
import torchvision


# ---------------------------------------------------
# MASK
# ---------------------------------------------------

def generate_square_subsequent_mask(sz: int):

    mask = torch.triu(
        torch.ones(sz, sz),
        diagonal=1,
    )

    mask = mask.masked_fill(
        mask == 1,
        float("-inf"),
    )

    return mask


# ---------------------------------------------------
# POSITIONAL ENCODING
# ---------------------------------------------------

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=5000,
    ):

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
            ).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(1)

        self.register_buffer("pe", pe)

    def forward(self, x):

        seq_len = x.size(0)

        x = x + self.pe[:seq_len]

        return self.dropout(x)


class PositionalEncodingImage(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=4096,
        dropout=0.1,
    ):

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
            ).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(1)

        self.register_buffer("pe", pe)

    def forward(self, x):

        seq_len = x.size(0)

        x = x + self.pe[:seq_len]

        return self.dropout(x)


# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

TF_DIM = 512
TF_FC_DIM = 2048
TF_DROPOUT = 0.1
TF_LAYERS = 6
TF_NHEAD = 8

RESNET_DIM = 512


class ResnetTransformer(nn.Module):

    def __init__(
        self,
        data_config,
    ):

        super().__init__()

        self.data_config = data_config

        self.input_dims = data_config["input_dims"]

        self.num_classes = len(
            data_config["mapping"]
        )

        inverse_mapping = {
            val: ind
            for ind, val in enumerate(
                data_config["mapping"]
            )
        }

        self.start_token = inverse_mapping["<S>"]

        self.end_token = inverse_mapping["<E>"]

        self.padding_token = inverse_mapping["<P>"]

        self.max_output_length = data_config[
            "output_dims"
        ][0]

        self.dim = TF_DIM

        # ---------------------------------------------------
        # RESNET
        # ---------------------------------------------------

        backbone = torchvision.models.resnet34(
            weights=None
        )

        self.input_proj = nn.Conv2d(
            1,
            3,
            kernel_size=1,
        )

        self.resnet = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        self.resnet[7][0].conv1.stride = (1, 1)

        self.resnet[7][0].downsample[0].stride = (
            1,
            1,
        )

        # ---------------------------------------------------
        # FEATURES
        # ---------------------------------------------------

        self.feature_pool = nn.AdaptiveAvgPool2d(
            (4, 32)
        )

        self.encoder_projection = nn.Linear(
            RESNET_DIM,
            self.dim,
        )

        self.enc_pos_encoder = PositionalEncodingImage(
            d_model=self.dim,
            dropout=TF_DROPOUT,
        )

        # ---------------------------------------------------
        # DECODER
        # ---------------------------------------------------

        self.embedding = nn.Embedding(
            self.num_classes,
            self.dim,
            padding_idx=self.padding_token,
        )

        self.dec_pos_encoder = PositionalEncoding(
            d_model=self.dim,
            dropout=TF_DROPOUT,
            max_len=self.max_output_length,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.dim,
            nhead=TF_NHEAD,
            dim_feedforward=TF_FC_DIM,
            dropout=TF_DROPOUT,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )

        self.transformer_decoder = (
            nn.TransformerDecoder(
                decoder_layer,
                num_layers=TF_LAYERS,
            )
        )

        self.dropout = nn.Dropout(
            TF_DROPOUT
        )

        self.fc = nn.Linear(
            self.dim,
            self.num_classes,
        )

        self.y_mask = generate_square_subsequent_mask(
            self.max_output_length
        )

    # ---------------------------------------------------
    # ENCODE
    # ---------------------------------------------------

    def encode(self, x):

        if x.shape[1] == 1:

            x = self.input_proj(x)

        x = self.resnet(x)

        x = self.feature_pool(x)

        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)

        x = x.reshape(B, H * W, C)

        x = self.encoder_projection(x)

        x = x.permute(1, 0, 2)

        x = self.enc_pos_encoder(x)

        return x

    # ---------------------------------------------------
    # DECODE
    # ---------------------------------------------------

    def decode(
        self,
        memory,
        y,
    ):

        y_padding_mask = (
            y == self.padding_token
        )

        y = self.embedding(y)

        y = y * math.sqrt(self.dim)

        y = y.permute(1, 0, 2)

        y = self.dec_pos_encoder(y)

        seq_len = y.shape[0]

        y_mask = self.y_mask[
            :seq_len,
            :seq_len
        ].to(memory.device)

        output = self.transformer_decoder(
            tgt=y,
            memory=memory,
            tgt_mask=y_mask,
            tgt_key_padding_mask=y_padding_mask,
        )

        output = self.dropout(output)

        output = self.fc(output)

        return output

    # ---------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------

    def forward(self, x):

        B = x.shape[0]

        memory = self.encode(x)

        output_tokens = torch.full(
            (
                B,
                self.max_output_length,
            ),
            fill_value=self.padding_token,
            device=x.device,
            dtype=torch.long,
        )

        output_tokens[:, 0] = self.start_token

        for step in range(
            1,
            self.max_output_length,
        ):

            y = output_tokens[:, :step]

            logits = self.decode(
                memory,
                y,
            )

            next_token = torch.argmax(
                logits[-1],
                dim=-1,
            )

            output_tokens[:, step] = next_token

            if (
                next_token == self.end_token
            ).all():

                break

        return output_tokens