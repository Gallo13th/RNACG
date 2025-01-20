from utils import flow_utils
import torch
import math

def expand_simplex(xt, alphas, prior_pseudocount=2):
    prior_weights = (prior_pseudocount / (alphas + prior_pseudocount - 1))[:,None,None]
    return torch.cat([xt * (1 - prior_weights), xt * prior_weights], -1)

class RMSnorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SequenceFlowBlock(torch.nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        # condition
        self.linear_c = torch.nn.Linear(dim, dim)
        self.laynorm_c = torch.nn.LayerNorm(dim)
        self.t_embed_c = torch.nn.Linear(dim, 6)
        self.adaLN_c = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(dim, 6 * dim, bias=True)
        )
        self.rmsnorm_cq = RMSnorm(dim_head)
        self.rmsnorm_ck = RMSnorm(dim_head)
        # x
        self.linear_x = torch.nn.Linear(dim, dim)
        self.laynorm_x = torch.nn.LayerNorm(dim)
        self.t_embed_x = torch.nn.Linear(dim, 6)
        self.adaLN_x = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(dim, 6 * dim, bias=True)
        )
        self.rmsnorm_xq = RMSnorm(dim_head)
        self.rmsnorm_xk = RMSnorm(dim_head)
        # SiLU
        self.silu = torch.nn.SiLU()
        self.to_qkv_x = torch.nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv_c = torch.nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out_c = torch.nn.ModuleList([
            torch.nn.Linear(inner_dim, dim),
            torch.nn.SiLU(dim),
            torch.nn.Linear(dim, dim)
            ])
        self.to_out_x = torch.nn.ModuleList([
            torch.nn.Linear(inner_dim, dim),
            torch.nn.SiLU(dim),
            torch.nn.Linear(dim, dim)
            ])
        
    def forward(self, x, c, t_embed, mask = None):
        shift_msa_c, scale_msa_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = self.adaLN_c(t_embed).chunk(6, dim = -1)
        c_ = self.laynorm_c(c)
        c_ = c_ * (1 + scale_msa_c.unsqueeze(1)) + shift_msa_c.unsqueeze(1)
        c_ = self.linear_c(c_)
        qkv_c = self.to_qkv_c(c_).chunk(3, dim = -1)
        q_c, k_c, v_c = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, t.shape[2] // self.heads).permute(0, 2, 1, 3), qkv_c)
        
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_c(t_embed).chunk(6, dim = -1)
        x_ = self.laynorm_x(x)
        x_ = x_ * (1 + scale_msa_x.unsqueeze(1)) + shift_msa_x.unsqueeze(1)
        x_ = self.linear_x(x_)
        qkv_x = self.to_qkv_x(x_).chunk(3, dim = -1)
        q_x, k_x, v_x = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, t.shape[2] // self.heads).permute(0, 2, 1, 3), qkv_x)
        
        q = self.rmsnorm_cq(q_c) + self.rmsnorm_xq(q_x)
        k = self.rmsnorm_ck(k_c) + self.rmsnorm_xk(k_x)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            dots.masked_fill_(mask[:,None,:,:], float(-1000000))
        attn = dots.softmax(dim=-1)
        v = torch.einsum('bhij,bhjd->bhid', attn, v_c + v_x)
        v = v.permute(0, 2, 1, 3).reshape(v.shape[0], v.shape[2], -1)
        out_c = c + self.to_out_c[0](v) * gate_msa_c.unsqueeze(1)
        out_c = self.to_out_c[1](out_c)
        out_c = out_c * (1 + scale_mlp_c.unsqueeze(1)) + shift_mlp_c.unsqueeze(1) 
        out_c = self.to_out_c[2](out_c)
        out_c = c + out_c * gate_mlp_c.unsqueeze(1)
        
        out_x = x + self.to_out_x[0](v) * gate_msa_x.unsqueeze(1)
        out_x = self.to_out_x[1](out_x)
        out_x = out_x * (1 + scale_mlp_x.unsqueeze(1)) + shift_mlp_x.unsqueeze(1)
        out_x = self.to_out_x[2](out_x)
        out_x = x + out_x * gate_mlp_x.unsqueeze(1)
        
        return out_x, out_c
    
class TimestepEmbedding(torch.nn.Module):
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    def __init__(self, embedding_dim, max_positions=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions

    def forward(self, signal):
        shape = signal.shape
        signal = signal.reshape(-1)
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=signal.device) * -emb)
        emb = signal.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (signal.shape[0], self.embedding_dim)
        return emb.view(*shape, self.embedding_dim )
       
class SequenceFlow(torch.nn.Module):
    def __init__(self, inp_dim, dim, depth, heads, dim_head, condition = None):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        self.input_embed = torch.nn.Linear(inp_dim, dim//2)
        self.t_embed = TimestepEmbedding(dim)
        self.condition = condition
        self.cond_flow = flow_utils.DirichletConditionalFlow(K=5, alpha_min=1, alpha_max=8, alpha_spacing=0.001)
        for _ in range(depth):
            self.layers.append(SequenceFlowBlock(dim, heads, dim_head),)
        self.if_condition = True if condition is not None else False
        self.out =  torch.nn.Linear(dim, inp_dim)
        self.pe = self.biuld_position_embedding(dim//2)
        self.cls_embed = torch.nn.Linear(1, dim//2)
    
    def biuld_position_embedding(self, dim, max_length = 102400):
        position = torch.arange(max_length).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -math.log(10000.0) / dim)
        emb = torch.zeros(max_length, dim)
        emb[:, 0::2] = torch.sin(position * div)
        emb[:, 1::2] = torch.cos(position * div)
        return emb
    
    def forward(self, x, t, cls=None, mask=None, c=None):
        # x dim: [batch, seq_len, dim]
        b, s, d = x.shape
        x = self.input_embed(x)
        x = x + self.pe[None,:s].to(x.device)
        if cls is None:
            cls = torch.zeros(b, 1, 1).to(x.device)
        elif cls.dim() == 1 and cls.shape[0] == b:
            cls = cls[:,None,None]
        else:
            cls = cls
        cls = self.cls_embed(cls)
        x = torch.cat([cls, x], 1)
        x = expand_simplex(x, t)
        t = self.t_embed(t)
        if self.if_condition and c is not None:
            c = self.condition(**c)
            if c.dim() == 2:
                c = c.unsqueeze(1).expand(-1, s+1, -1)
            elif c.dim() == 3 and c.shape[1] == s:
                c = torch.nn.functional.pad(c, (0, 0, 1, 0, 0, 0))
            c = c.to(x.device)
        else:
            c = torch.zeros_like(x)
        for flow in self.layers:
            x, c = flow(x, c, t, mask)
        x_raw = x
        x = self.out(x)
        return x, x_raw, c
    
    @torch.no_grad()
    def flow_forward(self, x, t_span, mask=None, c=None, inverse=False):
        # x dim: [batch, seq_len, dim]
        b, s, d = x.shape
        xt = x.clone()
        eye = torch.eye(d).to(x)
        if inverse:
            t_span = t_span.flip(0)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            if s < t:
                s, t = t, s
            s = s.unsqueeze(-1).repeat(b)
            x_,_,_ = self.forward(xt, s, None, mask, c)
            x_ = x_[:,1:]
            alpha = s[0]
            c_factor = self.cond_flow.c_factor(xt.cpu().numpy(),alpha.cpu().numpy())
            c_factor = torch.from_numpy(c_factor).to(x)
            flow_probs = torch.softmax(x_/1.0, dim = -1)
            cond_flow = (eye - xt.unsqueeze(-1)) *  c_factor.unsqueeze(-2)
            flow = (flow_probs.unsqueeze(-2) * cond_flow).sum(-1)
            if inverse:
                xt = xt - flow * (t - s[0])
            else:
                xt = xt + flow * (t - s[0])
        return xt

if __name__ == '__main__':
    model = SequenceFlow(64, 64 , 6, 8, 16)
    x = torch.randn(3, 32, 64)
    t = torch.randn(3)
    print(model(x, t).shape)
    class condition_net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Linear(1, 64)
        def forward(self, c):
            return self.embed(c)
    c_net = condition_net()
    model = SequenceFlow(64, 64, 6, 8, 8, condition = c_net)
    x = torch.randn(3, 32, 64)
    t = torch.randn(3)
    c = torch.randn(3, 1)
    print(model(x, t, c = {'c': c}))
    print(model.flow_forward(x, torch.linspace(1.01,8,100), c = {'c': c}))
