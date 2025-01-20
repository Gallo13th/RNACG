import e3nn
import torch_cluster
import torch
from torch_cluster import radius_graph
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace,soft_unit_step
from torch_scatter import scatter


class SE3TransformerBlock(torch.nn.Module):
    
    def __init__(self,irreps_input,irreps_query,irreps_key,irreps_sh,irreps_output,number_of_basis,max_radius):
        super(SE3TransformerBlock,self).__init__()
        self.h_q = o3.Linear(irreps_input, irreps_query)
        self.tp_k = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_key, shared_weights=False)
        self.fc_k = nn.FullyConnectedNet([number_of_basis, 16, self.tp_k.weight_numel], act=torch.nn.functional.silu)
        self.tp_v = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_output, shared_weights=False)
        self.fc_v = nn.FullyConnectedNet([number_of_basis, 16, self.tp_v.weight_numel], act=torch.nn.functional.silu)
        self.dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "0e")
        self.number_of_basis = number_of_basis
        self.max_radius = max_radius
        self.irreps_sh = irreps_sh
        self.output_dim = o3.Irreps(irreps_output).dim
        
    def forward(self,f,pos):
        # f dim: [batch, nodes, features]
        # pos dim: [batch, nodes, 3]
        f_dim = f.shape
        batch = torch.arange(pos.shape[0], device=pos.device)[:, None].expand(-1, pos.shape[1]).reshape(-1)
        pos = pos.reshape(-1, 3)
        f = f.reshape(-1, f.shape[2])
        mask = (f == 0).all(dim=1)
        f = f[~mask]
        pos = pos[~mask]
        batch = batch[~mask]
        edge_src, edge_dst = radius_graph(pos, r=self.max_radius, batch=batch, max_num_neighbors=64)
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_length = edge_vec.norm(dim=1)

        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='smooth_finite',
            cutoff=True
        )
        edge_length_embedded = edge_length_embedded.mul(self.number_of_basis**0.5)
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / self.max_radius))
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, True, normalization='component')
        q = self.h_q(f)
        k = self.tp_k(f[edge_src], edge_sh, self.fc_k(edge_length_embedded))
        v = self.tp_v(f[edge_src], edge_sh, self.fc_v(edge_length_embedded))
        log_exp = edge_weight_cutoff[:, None] * self.dot(q[edge_dst], k)
        z = scatter(log_exp, edge_dst, dim=0, dim_size=len(f))
        # z[z == 0] = 1
        alpha = log_exp / z[edge_dst]
        assert torch.isnan(alpha).sum() == 0, (alpha[torch.isnan(alpha)], log_exp[torch.isnan(alpha)], z[edge_dst][torch.isnan(alpha)])
        output = torch.zeros(f_dim[0]*f_dim[1],self.output_dim,device=f.device)
        output[~mask] = scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(f))
        return output.reshape(f_dim[0],f_dim[1],-1)


class SE3Transformer(torch.nn.Module):
    
    def __init__(self,irreps_input,irreps_query,irreps_key,irreps_sh,irreps_hidden,irreps_output,number_of_basis,max_radius,number_of_layers=3):
        super(SE3Transformer,self).__init__()
        self.irreps_input = irreps_input
        self.irreps_query = irreps_query
        self.irreps_key = irreps_key
        self.irreps_sh = irreps_sh
        self.number_of_basis = number_of_basis
        self.max_radius = max_radius
        self.blocks = torch.nn.ModuleList()
        self.input_block = SE3TransformerBlock(irreps_input,irreps_query,irreps_key,irreps_sh,irreps_hidden,number_of_basis,max_radius)
        for _ in range(number_of_layers):
            self.blocks.append(SE3TransformerBlock(irreps_hidden,irreps_query,irreps_key,irreps_sh,irreps_hidden,number_of_basis,max_radius))
        self.output_block = SE3TransformerBlock(irreps_hidden,irreps_query,irreps_key,irreps_sh,irreps_output,number_of_basis,max_radius)
    def forward(self,f,pos):
        f = self.input_block(f,pos)
        for block in self.blocks:
            f_ = block(f,pos)
            f = f + f_
        return self.output_block(f,pos)

if __name__ == '__main__':
    model = SE3Transformer("1x0e+2x1o", "10x0e+3x1o", "10x0e+3x1o", "10x0e+3x1o", "10x0e+3x1o", "10x0e+3x1o", 10, 10)
    f = torch.randn(1,12, 1)
    pos = torch.randn(1,12,3) * 5 + torch.arange(12).unsqueeze(0).unsqueeze(-1).float() * 10
    pos_ = torch.randn(1,12,3) * 5 + torch.arange(12).unsqueeze(0).unsqueeze(-1).float() * 10
    # pos[:,2,:] = float('nan')
    pos_[:,3,:] = float('nan')
    pos[:,4,:] = float('nan')
    zero_padding = torch.zeros(1,1,1)
    f = torch.cat([f, zero_padding], dim=1)
    zero_padding = torch.zeros(1,1,3)
    pos = torch.cat([pos, zero_padding], dim=1)
    pos_ = torch.cat([pos_, zero_padding], dim=1)
    pos[torch.isnan(pos)] = pos_[torch.isnan(pos)]
    pos_[torch.isnan(pos_)] = pos[torch.isnan(pos_)]
    f = torch.concat([f,pos,pos_],dim=-1)
    print(f)
    print(model(f, pos).shape,model(f, pos).sum())