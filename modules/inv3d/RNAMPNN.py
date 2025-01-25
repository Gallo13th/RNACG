import torch
import torch.nn as nn
from .module import MPNNBlock
from .feature import RNAFeatures, RNA2DFeatures
'''
This part of the code is referenced from proteinmpnn
repo: https://github.com/dauparas/ProteinMPNN
'''

def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


class RNAMPNN(nn.Module):
    def __init__(self, config):
        super(RNAMPNN, self).__init__()
        self.config = config
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.node_features = self.edge_features =  config.hidden
        self.hidden_dim = config.hidden
        self.vocab = config.vocab_size

        self.features = RNAFeatures(
            config.hidden, config.hidden, 
            top_k=config.k_neighbors, 
            dropout=config.dropout,
            node_feat_types=config.node_feat_types, 
            edge_feat_types=config.edge_feat_types,
        )

        self.W_s = nn.Embedding(config.vocab_size, self.hidden_dim)
        self.encoder = nn.ModuleList([
            MPNNBlock(self.hidden_dim, self.hidden_dim*2, dropout=config.dropout)
            for _ in range(config.num_encoder_layers)])
        self.decoder = nn.ModuleList([
            MPNNBlock(self.hidden_dim, self.hidden_dim*2, dropout=config.dropout)
            for _ in range(config.num_decoder_layers)])

        self.proj= nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        )

        self.out = nn.Linear(self.hidden_dim, config.vocab_size, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask):
        X, S, h_V, h_E, E_idx, batch_id = self.features(X, S, mask) 

        for enc in self.encoder:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = enc(h_V, h_EV, E_idx, batch_id)

        for dec in self.decoder:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = dec(h_V, h_EV, E_idx, batch_id)

        graph_embs = []
        for b_id in range(batch_id[-1].item()+1):
            b_data = h_V[batch_id == b_id].mean(0)
            graph_embs.append(b_data)
        graph_embs = torch.stack(graph_embs, dim=0)
        graph_prjs = self.proj(graph_embs)

        logits = self.out(h_V)
        return logits, S, graph_prjs

    def sample(self, X, S, mask=None):
        X, gt_S, h_V, h_E, E_idx, batch_id = self.features(X, S, mask) 

        for enc in self.encoder:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = enc(h_V, h_EV, E_idx)

        for dec in self.decoder:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = dec(h_V, h_EV, E_idx)

        logits = self.out(h_V)
        return logits, gt_S
    
    def forward_embedding(self, X, S, mask=None):
        X, S, h_V, h_E, E_idx, batch_id = self.features(X, S, mask) 
        for enc in self.encoder:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = enc(h_V, h_EV, E_idx)
        for dec in self.decoder:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = dec(h_V, h_EV, E_idx)

        if mask is not None:
            tmp = torch.zeros((mask.shape[0],mask.shape[1],self.hidden_dim)).to(self.device)
            tmp[mask.bool()] = h_V
            h_V = tmp
        return h_V
   
'''
Add by GLT
'''   
   
    
class RNAMPNN2D(RNAMPNN):
    
    def __init__(self, config):
        super(RNAMPNN2D, self).__init__(config)
        self.features = RNA2DFeatures(
            config.hidden, config.hidden, 
            top_k=config.k_neighbors, 
            dropout=config.dropout,
        )
    
    def forward_embedding(self, X, S, mask, edges):
        X, S, h_V, h_E, E_idx, batch_id = self.features(X, S, mask, edges)
        for enc in self.encoder:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = enc(h_V, h_EV, E_idx)
        for dec in self.decoder:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = dec(h_V, h_EV, E_idx)

        if mask is not None:
            tmp = torch.zeros((mask.shape[0],mask.shape[1],self.hidden_dim)).to(X.device)
            tmp[mask.bool()] = h_V
            h_V = tmp
        return h_V