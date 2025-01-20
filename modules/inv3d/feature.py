
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from collections import defaultdict, deque




'''
This part of the code is referenced from proteinmpnn
repo: https://github.com/dauparas/ProteinMPNN
'''

def gather_edges(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)

def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, idx_flat)

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return torch.cat([h_neighbors, h_nodes], -1)

class MPNNBlock(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super(MPNNBlock, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.ReLU()

        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )

    def forward(self, h_V, h_E, edge_idx):
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_E)))))
        dh = scatter_sum(h_message, src_idx, dim=0) / self.scale
        h_V = self.norm1(h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout(dh))
        return h_V

class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias




feat_dims = {
    'node': {
        'angle': 12,
        'distance': 80,
        'direction': 9,
    },
    'edge': {
        'orientation': 4,
        'distance': 96,
        'direction': 15,
    }
}


def nan_to_num(tensor, nan=0.0):
    idx = torch.isnan(tensor)
    tensor[idx] = nan
    return tensor

def _normalize(tensor, dim=-1):
    return nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


class RNAFeatures(nn.Module):
    def __init__(self, edge_features, node_features, node_feat_types=[], edge_feat_types=[], num_rbf=16, top_k=30, augment_eps=0., dropout=0.1):
        super(RNAFeatures, self).__init__()
        """Extract RNA Features"""
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.dropout = nn.Dropout(dropout)
        self.node_feat_types = node_feat_types
        self.edge_feat_types = edge_feat_types

        node_in = sum([feat_dims['node'][feat] for feat in node_feat_types])
        edge_in = sum([feat_dims['edge'][feat] for feat in edge_feat_types])
        self.node_embedding = nn.Linear(node_in,  node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = (1. - mask_2D)*10000 + mask_2D* torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(self.top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        return torch.exp(-((D_expand - D_mu) / D_sigma)**2)

    def _get_rbf(self, A, B, E_idx=None, num_rbf=16):
        if E_idx is not None:
            D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6)
            D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0]
            RBF_A_B = self._rbf(D_A_B_neighbors)
        else:
            D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,:,None,:])**2,-1) + 1e-6)
            RBF_A_B = self._rbf(D_A_B)
        return RBF_A_B

    def _quaternions(self, R):
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        return Q

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        V = X.clone()
        X = X[:,:,:6,:].reshape(X.shape[0], 6*X.shape[1], 3) 
        dX = X[:,1:,:] - X[:,:-1,:]
        U = _normalize(dX, dim=-1)
        u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
        n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
        b_1 = _normalize(u_0 - u_1, dim=-1)
        
        # select C3'
        n_0 = n_0[:,4::6,:] 
        b_1 = b_1[:,4::6,:]
        X = X[:,4::6,:]

        Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
        Q = Q.view(list(Q.shape[:2]) + [9])
        Q = F.pad(Q, (0,0,0,1), 'constant', 0) # [16, 464, 9]

        Q_neighbors = gather_nodes(Q, E_idx) # [16, 464, 30, 9]
        P_neighbors = gather_nodes(V[:,:,0,:], E_idx) # [16, 464, 30, 3]
        O5_neighbors = gather_nodes(V[:,:,1,:], E_idx)
        C5_neighbors = gather_nodes(V[:,:,2,:], E_idx)
        C4_neighbors = gather_nodes(V[:,:,3,:], E_idx)
        O3_neighbors = gather_nodes(V[:,:,5,:], E_idx)

        Q = Q.view(list(Q.shape[:2]) + [3,3]).unsqueeze(2) # [16, 464, 1, 3, 3]
        Q_neighbors = Q_neighbors.view(list(Q_neighbors.shape[:3]) + [3,3]) # [16, 464, 30, 3, 3]

        dX = torch.stack([P_neighbors,O5_neighbors,C5_neighbors,C4_neighbors,O3_neighbors], dim=3) - X[:,:,None,None,:] # [16, 464, 30, 3]
        dU = torch.matmul(Q[:,:,:,None,:,:], dX[...,None]).squeeze(-1) # [16, 464, 30, 3] 邻居的相对坐标
        B, N, K = dU.shape[:3]
        E_direct = _normalize(dU, dim=-1)
        E_direct = E_direct.reshape(B, N, K,-1)
        R = torch.matmul(Q.transpose(-1,-2), Q_neighbors)
        E_orient = self._quaternions(R)
        
        dX_inner = V[:,:,[0,2,3],:] - X.unsqueeze(-2)
        dU_inner = torch.matmul(Q, dX_inner.unsqueeze(-1)).squeeze(-1)
        dU_inner = _normalize(dU_inner, dim=-1)
        V_direct = dU_inner.reshape(B,N,-1)
        return V_direct, E_direct, E_orient

    def _dihedrals(self, X, eps=1e-7):
        # P, O5', C5', C4', C3', O3'
        X = X[:,:,:6,:].reshape(X.shape[0], 6*X.shape[1], 3)

        # Shifted slices of unit vectors
        # https://iupac.qmul.ac.uk/misc/pnuc2.html#220
        # https://x3dna.org/highlights/torsion-angles-of-nucleic-acid-structures
        # alpha:   O3'_{i-1} P_i O5'_i C5'_i
        # beta:    P_i O5'_i C5'_i C4'_i
        # gamma:   O5'_i C5'_i C4'_i C3'_i
        # delta:   C5'_i C4'_i C3'_i O3'_i
        # epsilon: C4'_i C3'_i O3'_i P_{i+1}
        # zeta:    C3'_i O3'_i P_{i+1} O5'_{i+1} 
        # What's more:
        #   chi: C1' - N9 
        #   chi is different for (C, T, U) and (A, G) https://x3dna.org/highlights/the-chi-x-torsion-angle-characterizes-base-sugar-relative-orientation

        dX = X[:, 5:, :] - X[:, :-5, :] # O3'-P, P-O5', O5'-C5', C5'-C4', ...
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]  # O3'-P, P-O5', ...
        u_1 = U[:,1:-1,:] # P-O5', O5'-C5', ...
        u_0 = U[:,2:,:]   # O5'-C5', C5'-C4', ...
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        D = F.pad(D, (3,4), 'constant', 0)
        D = D.view((D.size(0), D.size(1) //6, 6))
        return torch.cat((torch.cos(D), torch.sin(D)), 2) # return D_features

    def forward(self, X, S, mask):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        # Build k-Nearest Neighbors graph
        B, N, _,_ = X.shape
        # P, O5', C5', C4', C3', O3'
        atom_P = X[:, :, 0, :]
        atom_O5_ = X[:, :, 1, :]
        atom_C5_ = X[:, :, 2, :]
        atom_C4_ = X[:, :, 3, :]
        atom_C3_ = X[:, :, 4, :] 
        atom_O3_ = X[:, :, 5, :]

        X_backbone = atom_P
        D_neighbors, E_idx = self._dist(X_backbone, mask)
      
        mask_bool = (mask==1)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x: torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(-1,x.shape[-1])
        node_mask_select = lambda x: torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])

        # node features
        h_V = []
        # angle
        V_angle = node_mask_select(self._dihedrals(X))
        # distance
        node_list = ['O5_-P', 'C5_-P', 'C4_-P', 'C3_-P', 'O3_-P']
        V_dist = []
        for pair in node_list:
            atom1, atom2 = pair.split('-')
            V_dist.append(node_mask_select(self._get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, self.num_rbf).squeeze()))
        V_dist = torch.cat(tuple(V_dist), dim=-1).squeeze()
        # direction
        V_direct, E_direct, E_orient = self._orientations_coarse(X, E_idx)
        V_direct = node_mask_select(V_direct)
        E_direct, E_orient = list(map(lambda x: edge_mask_select(x), [E_direct, E_orient]))

        # edge features
        h_E = []
        # dist
        edge_list = ['P-P', 'O5_-P', 'C5_-P', 'C4_-P', 'C3_-P', 'O3_-P']
        E_dist = [] 
        for pair in edge_list:
            atom1, atom2 = pair.split('-')
            E_dist.append(edge_mask_select(self._get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], E_idx, self.num_rbf)))
        E_dist = torch.cat(tuple(E_dist), dim=-1)

        if 'angle' in self.node_feat_types:
            h_V.append(V_angle)
        if 'distance' in self.node_feat_types:
            h_V.append(V_dist)
        if 'direction' in self.node_feat_types:
            h_V.append(V_direct)

        if 'orientation' in self.edge_feat_types:
            h_E.append(E_orient)
        if 'distance' in self.edge_feat_types:
            h_E.append(E_dist)
        if 'direction' in self.edge_feat_types:
            h_E.append(E_direct)

        # Embed the nodes
        h_V = self.norm_nodes(self.node_embedding(torch.cat(h_V, dim=-1)))
        h_E = self.norm_edges(self.edge_embedding(torch.cat(h_E, dim=-1)))

        # prepare the variables to return
        S = torch.masked_select(S, mask_bool)
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B,1,1) + E_idx
        src = torch.masked_select(src, mask_attend).view(1,-1)
        dst = shift.view(B,1,1) + torch.arange(0, N, device=src.device).view(1,-1,1).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1,-1)
        E_idx = torch.cat((dst, src), dim=0).long()

        sparse_idx = mask.nonzero()
        X = X[sparse_idx[:,0], sparse_idx[:,1], :, :]
        batch_id = sparse_idx[:,0]
        return X, S, h_V, h_E, E_idx, batch_id
    
    
'''
Add by GLT
'''

class RNA2DFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_rbf=16, top_k=30, augment_eps=0., dropout=0.1):
        super(RNA2DFeatures, self).__init__()
        """Extract RNA Features"""
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.dropout = nn.Dropout(dropout)

        edge_in = 16
        self.nt_embedding = nn.Linear(5, node_features//2)
        self.node_pos_embedding = nn.Embedding(1000,node_features//2) # 1000 is the max length of RNA sequence
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)
    
        
    def _dist(self, X, edges, masks, eps=1E-6):
        batch_size, num_nodes,_ = X.shape
        D_batch = []
        E_idx_batch = []

        for batch_idx in range(batch_size):
            edge = edges[batch_idx]
            mask = masks[batch_idx]
            mask_2D = torch.unsqueeze(mask, 0) * torch.unsqueeze(mask, 1)  # 2D掩码

            # 构建邻接表
            adj = defaultdict(list)
            for i in range(num_nodes - 1):
                if mask[i] and mask[i + 1]:  # 只处理有效节点
                    adj[i].append(i + 1)  # 链式连接
                    adj[i + 1].append(i)  # 双向连接

            # 添加额外边
            for i, j in edge:
                if mask[i] and mask[j]:  # 只处理有效节点
                    adj[i].append(j)
                    adj[j].append(i)

            # 使用BFS计算最短路径
            D = torch.full((num_nodes, num_nodes), float('inf'), device=X.device)
            for i in range(num_nodes):
                if not mask[i]:  # 跳过无效节点
                    continue
                queue = deque()
                queue.append((i, 0))  # (当前节点, 距离)
                visited = set()
                visited.add(i)
                while queue:
                    node, dist = queue.popleft()
                    D[i, node] = dist
                    for neighbor in adj[node]:
                        if neighbor not in visited and mask[neighbor]:  # 只处理有效节点
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

            # 调整掩码，将无效节点的距离设置为最大值
            D_max = D[mask_2D == 1].max()  # 有效节点的最大距离
            D_adjust = D.clone()
            D_adjust[mask_2D == 0] = D_max + 1  # 无效节点的距离设置为最大值
            # 选择每个节点的k近邻
            D_neigh, E_idx = torch.topk(D_adjust, min(self.top_k, num_nodes), dim=-1, largest=False)
            D_batch.append(D_neigh)
            E_idx_batch.append(E_idx)

        # 将结果堆叠为一个批次
        D_batch = torch.stack(D_batch, dim=0)
        E_idx_batch = torch.stack(E_idx_batch, dim=0)

        return D_batch, E_idx_batch

    def _rbf(self, D):
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        return torch.exp(-((D_expand - D_mu) / D_sigma)**2)

    def _get_rbf(self, A, B, E_idx=None, num_rbf=16):
        if E_idx is not None:
            D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6)
            D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0]
            RBF_A_B = self._rbf(D_A_B_neighbors)
        else:
            D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,:,None,:])**2,-1) + 1e-6)
            RBF_A_B = self._rbf(D_A_B)
        return RBF_A_B

    def forward(self, X, S, mask, edges):

        # Build k-Nearest Neighbors graph
        B, N, _ = X.shape # B, N (A, U, G, C, N)
        D_neighbors, E_idx = self._dist(X, edges, mask)      
        # gather D_neighbors and E_idx
        
        mask_bool = (mask==1)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x: torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(-1,x.shape[-1])
        node_mask_select = lambda x: torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])
        E_idx.clamp_(0, self.top_k-1)
        E_idx = E_idx * mask_attend.long()
        # node features
        # position
        h_V_pos = torch.arange(N, device=X.device).view(1,-1).expand(B,-1)
        h_V_pos = self.node_pos_embedding(h_V_pos)
        # node category
        h_V_cat = self.nt_embedding(X)
        h_V = torch.cat((h_V_pos, h_V_cat), dim=-1)
        h_V = node_mask_select(h_V)
        h_V = self.norm_nodes(h_V)
        
        # edge features
        h_E = []
        # dist
        E_dist = []

        E_dist.append(edge_mask_select(self._rbf(gather_edges(D_neighbors[...,None], E_idx)[:,:,:,0])))
        E_dist = torch.cat(tuple(E_dist), dim=-1)
        h_E.append(E_dist)

        h_E = self.norm_edges(self.edge_embedding(torch.cat(h_E, dim=-1)))

        # prepare the variables to return
        S = torch.masked_select(S, mask_bool)
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B,1,1) + E_idx
        src = torch.masked_select(src, mask_attend).view(1,-1)
        dst = shift.view(B,1,1) + torch.arange(0, N, device=src.device).view(1,-1,1).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1,-1)
        E_idx = torch.cat((dst, src), dim=0).long()

        sparse_idx = mask.nonzero()
        X = X[sparse_idx[:,0], sparse_idx[:,1]]
        batch_id = sparse_idx[:,0]
        return X, S, h_V, h_E, E_idx, batch_id



if __name__ == '__main__':
    import sys
    sys.path.append('..')
    rna_featurizer = RNAFeatures(128, 128, node_feat_types=['angle', 'distance', 'direction'], edge_feat_types=['orientation', 'distance', 'direction'])
    X = torch.randn(3,63,6,3)
    S = torch.randn(3,63)
    mask = torch.ones(3,63)
    X, S, h_V, h_E, E_idx, batch_id = rna_featurizer(X, S, mask)
    print(X.shape, S.shape, h_V.shape, h_E.shape, E_idx.shape, batch_id.shape)
    rna_featurizer = RNA2DFeatures(128, 128, num_rbf=16, top_k=30, augment_eps=0., dropout=0.1)
    X = torch.randint(0, 4, (3, 50))
    S = torch.randn(3, 50)
    mask = torch.ones(3, 50)
    edges = torch.randint(0, 50, (3, 10, 2))
    X, S, h_V, h_E, E_idx, batch_id = rna_featurizer(X, S, mask, edges)
    print(X.shape, S.shape, h_V.shape, h_E.shape, E_idx.shape, batch_id.shape)