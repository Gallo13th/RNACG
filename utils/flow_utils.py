import torch
import scipy
import numpy as np

def simplex_proj(seq):
    """
    Project a sequence onto the simplex space.
    Algorithm from https://arxiv.org/abs/1309.1541 by Weiran Wang and Miguel Á. Carreira-Perpiñán.

    Args:
        seq: Input sequence tensor.

    Returns:
        Projected sequence tensor.
    """
    Y = seq.reshape(-1, seq.shape[-1])
    N, K = Y.shape
    X, _ = torch.sort(Y, dim=-1, descending=True)
    X_cumsum = torch.cumsum(X, dim=-1) - 1
    div_seq = torch.arange(1, K + 1, dtype=Y.dtype, device=Y.device)
    Xtmp = X_cumsum / div_seq.unsqueeze(0)

    greater_than_Xtmp = (X > Xtmp).sum(dim=1, keepdim=True)
    row_indices = torch.arange(N, dtype=torch.long, device=Y.device).unsqueeze(1)
    selected_Xtmp = Xtmp[row_indices, greater_than_Xtmp - 1]

    X = torch.max(Y - selected_Xtmp, torch.zeros_like(Y))
    return X.view(seq.shape)

def k_class_norm_dist(k,batch_size=1):
    dists = [torch.distributions.normal.Normal(k,1).sample((batch_size,)) for i in range(k)]
    x = torch.sum(torch.stack(dists),0)
    return x

def sample_conditional_pt(x0, x1, t, sigma):
    """
    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    xt : Tensor, shape (bs, *dim)

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon

def compute_conditional_vector_field(x0, x1):
    """
    Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch

    Returns
    -------
    ut : conditional vector field ut(x1|x0) = x1 - x0

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    return x1 - x0

def dirichlet_sample_cond_prob_path(x1,alpha_scale=2.0,alpha=None):
    B, L , alphabet_size = x1.shape
    if alpha is None:
        alphas = torch.from_numpy(1 + scipy.stats.expon().rvs(size=B) * alpha_scale).to(x1.device).float()
    else:
        alphas = alpha
    alphas_ = torch.ones(B, L, alphabet_size, device=x1.device)
    alphas_ = alphas_ + x1 * (alphas[:,None,None] - 1)
    xt = torch.distributions.Dirichlet(alphas_).sample()
    return xt, alphas

class DirichletConditionalFlow:
    def __init__(self, K=5, alpha_min=1, alpha_max=8, alpha_spacing=0.01):
        self.alphas = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing)
        self.beta_cdfs = []
        self.bs = np.linspace(0, 1, 1000)
        for alph in self.alphas:
            self.beta_cdfs.append(scipy.special.betainc(alph, K-1, self.bs))
        self.beta_cdfs = np.array(self.beta_cdfs)
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / alpha_spacing
        self.K = K

    def c_factor(self, bs, alpha):
        out1 = scipy.special.beta(alpha, self.K - 1)
        # change out1 dim same to bs
        out2 = np.where(bs < 1, out1 / ((1 - bs) ** (self.K - 1)), 0)
        if np.isinf(out2).any() or np.isnan(out2).any():
            print('inf',alpha)
        out = np.where((bs ** (alpha - 1)) > 0, out2 / (bs ** (alpha - 1)), 0)
        I_func = self.beta_cdfs_derivative[np.argmin(np.abs(alpha - self.alphas))]
        interp = -np.interp(bs, self.bs, I_func)
        final = interp * out
        return final

if __name__ == '__main__':
    x1 = torch.randn(1, 3, 5).argmax(-1)
    x1 = torch.nn.functional.one_hot(x1, num_classes=5).float()
    eye = torch.eye(5)
    for _ in range(1):
        xt,alpha = dirichlet_sample_cond_prob_path(x1)
        D = DirichletConditionalFlow()
        c_factor = D.c_factor(xt.cpu().numpy(),alpha.cpu().numpy())
        c_factor = torch.from_numpy(c_factor)
        c_factor = torch.nan_to_num(c_factor)
        flow_probs = torch.softmax(x1/1.0, dim = -1)
        print((eye - xt.unsqueeze(-1)).sum())
        cond_flow = (eye - xt.unsqueeze(-1)) *  c_factor.unsqueeze(-2)
        flow = (flow_probs.unsqueeze(-2) * cond_flow).sum(-1)
        print(flow.sum())