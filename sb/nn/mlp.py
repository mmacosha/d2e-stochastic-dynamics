import torch
import torch.nn as nn

from sb.nn.modules import Block, Module
from sb.nn.utils import ModelOutput, fourier_proj


class SimpleNet(Module):
    def __init__(
            self, 
            x_emb_size: int, 
            in_dim: int = 2,
            out_dim: int = 2,
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
        
        self.drift_head = nn.Linear(x_emb_size, out_dim)

        if self.predict_log_var:
            self.log_var_head = nn.Linear(x_emb_size, out_dim)

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
