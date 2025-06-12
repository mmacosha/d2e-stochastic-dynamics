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
            *[Block(hidden_dim, hidden_dim, use_ln, block_type=="res") 
              for _ in range(n_blocks)]
        )

    def forward(self, x):
        x = self.proj_in(x)
        x = self.body(x)
        out = self.proj_out(x)
        return out.squeeze(1)
