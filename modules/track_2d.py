import torch
import torch.nn as nn
from torch_scatter import scatter

class AxialAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        # x dim : [batch, seq_len, dim]
        qkv = self.to_qkv(x).chunk(3, dim = -1) # qkv dim: [batch, seq_len, heads * dim_head]
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, t.shape[2] // self.heads).permute(0, 2, 1, 3), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], -1)
        return self.to_out(out)

class AxialTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                AxialAttention(dim, heads, dim_head), # row attention
                AxialAttention(dim, heads, dim_head), # column attention
                nn.Sequential(
                nn.Linear(dim, 4 * dim),
                nn.GELU(),
                nn.Linear(4 * dim, dim)
                ) # feedforward
            ]))
            
    def forward(self, x):
        # x dim: [batch, row_length, col_length, dim]
        b,r,c,d = x.shape
        for [norm, row_attn, col_attn, ff] in self.layers:
            x_res = x
            x = x + row_attn(norm(x).permute(0,2,1,3).view(b*c,r,d)).view(b,c,r,d).permute(0,2,1,3)
            x = x + col_attn(norm(x).view(b*r,c,d)).view(b,r,c,d)
            x = x + ff(norm(x))
            x = x_res + x
        return x

class MPNNblock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,edge_dim):
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.message_function = nn.Linear(hidden_channels, hidden_channels)
        self.update_function = nn.Linear(hidden_channels, out_channels)
        self.edge_dim = edge_dim
        
    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        edge_attr = self.edge_encoder(edge_attr)
        x = self.node_encoder(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr), edge_attr
    
    def propagate(self, edge_index, x, edge_attr):
        row, col = edge_index
        out = self.message_function(x[col] + edge_attr)
        out = scatter(out, row, dim=0, dim_size=x.size(0), reduce='mean')
        out = self.update_function(out)
        return out

class MPNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,edge_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(MPNNblock(in_channels, hidden_channels, hidden_channels,edge_dim))
        for _ in range(num_layers - 1):
            self.layers.append(MPNNblock(hidden_channels, hidden_channels, hidden_channels,edge_dim))
        self.out_layer = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr, mask=None):
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool).all(dim=-1)
        out = torch.zeros(x.shape[0], x.shape[1], self.out_layer.out_features, device=x.device)
        out[~mask] = float(-1)
        x = x[mask,:]
        for layer in self.layers:
            x, _ = layer(x, edge_index, edge_attr)
        out[mask] = self.out_layer(x)
        return out

if __name__ == '__main__':
    # model = AxialTransformer(64, 6, 8, 8)
    # x = torch.randn(1, 32, 32, 64)
    # print(model(x).shape)
    model = MPNN(5, 64, 7, 2, 6)
    x1 = torch.randn(5, 5)
    x2 = torch.randn(10, 5)
    edge_index_1 = torch.randint(0, 5, (2, 6))
    edge_index_2 = torch.randint(0, 10, (2, 8))
    edge_attr_1 = torch.randn(6, 2)
    edge_attr_2 = torch.randn(8, 2)
    edge_index = torch.concat([edge_index_1,edge_index_2+len(x1)],dim=1)
    max_length = max([len(x1),len(x2)])
    x1 = torch.nn.functional.pad(x1,(0,0,0,max_length - len(x1)), value=float('nan'))
    x2 = torch.nn.functional.pad(x2,(0,0,0,max_length - len(x2)), value=float('nan'))
    x = torch.stack([x1,x2])
    edge_attr = torch.concat([edge_attr_1,edge_attr_2],dim=0)
    mask = ~torch.isnan(x).all(dim=-1)
    print(model(x, edge_index, edge_attr, mask))
    