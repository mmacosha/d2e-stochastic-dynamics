import torch
import torch.nn as nn

from models.module import Module
from models.utils import fourier_proj, ModelOutput


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, use_ln: bool = False, 
                 skip_connection: bool = False):
        super().__init__()
        if in_dim != out_dim:
            assert not skip_connection, "Skip connection requires in_dim == out_dim"
        self.skip_connection = skip_connection
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim), 
            nn.LayerNorm(out_dim) if use_ln else nn.Identity(), 
            nn.ELU()
        )
    
    def forward(self, x):
        if self.skip_connection:
            return x + self.block(x)
        return self.block(x)


class SimpleNet(Module):
    def __init__(
            self, 
            x_emb_size: int, 
            in_dim: int = 2,
            t_emb_size: int | None = None ,
            n_main_body_layers: int = 2,
            predict_log_var: bool = False,
        ):
        super().__init__(
            x_emb_size=x_emb_size,
            in_dim=in_dim,
            t_emb_size=t_emb_size,
            n_main_body_layers=n_main_body_layers,
            predict_log_var=predict_log_var,
        )
        self.x_emb_size = x_emb_size
        self.t_emb_size = t_emb_size
        self.predict_log_var = predict_log_var

        self.x_embed = Block(
            in_dim, x_emb_size, 
            use_ln=True, 
            skip_connection=False
        )
        
        combined_hidden_size = x_emb_size
        
        if self.t_emb_size is not None:
            self.t_embed = Block(
                t_emb_size, x_emb_size,
                use_ln=True,
                skip_connection=False)
            combined_hidden_size += x_emb_size

        layers = []
        for i in range(n_main_body_layers):
            layers.append(
                Block(
                    combined_hidden_size if i == 0 else x_emb_size,
                    x_emb_size,
                    use_ln=True,
                    skip_connection=False
                )
            )
        self.main_body = nn.Sequential(*layers)
        
        self.drift_head = nn.Linear(x_emb_size, 2)

        if self.predict_log_var:
            self.log_var_head = nn.Linear(x_emb_size, 2)

    def forward(self, x, t):
        if self.t_emb_size is None:
            x = torch.cat([x, t.view(-1, 1)], dim=1)

        embeddings = self.x_embed(x)

        if self.t_emb_size is not None:
            t_embed = fourier_proj(t, self.t_emb_size)
            t_embed = self.t_embed(t_embed)
            embeddings = torch.cat([embeddings, t_embed], dim=-1)
        
        
        embeddings = self.main_body(embeddings)
        drift = self.drift_head(embeddings)

        if self.predict_log_var:
            log_var = self.log_var_head(embeddings)
            return ModelOutput(drift=drift, log_var=log_var)

        return ModelOutput(drift=drift)


class Energy(Module):
    def __init__(self, in_dim=2, out_dim=1, 
                 hidden_dim=64, n_blocks=3, 
                 use_ln: bool = False, block_type='simple'):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            use_ln=use_ln,
            block_type=block_type,
        )

        self.proj_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ELU()
        )
        self.proj_out = nn.Linear(hidden_dim, out_dim)
        
        self.body = nn.Sequential(
            *[Block(hidden_dim, use_ln, block_type=="res") 
              for _ in range(n_blocks)]
        )

    def forward(self, x):
        x = self.proj_in(x)
        x = self.body(x)
        out = self.proj_out(x)
        return out.squeeze(1)
