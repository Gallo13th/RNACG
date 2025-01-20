from typing import Iterator
import torch
from torch.nn.parameter import Parameter

class Description(torch.nn.Module):
    def __init__(self, dim: int, vocab=None):
        super(Description, self).__init__()
        self.dim = dim
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = {}
        self.vocab_idx = {k: i for i, k in enumerate(self.vocab)}
        self.embed = torch.nn.Embedding(len(self.vocab), self.dim)
        self.embed.weight = Parameter(torch.zeros(len(self.vocab), self.dim))
    
    def forward(self, x):
        idx = [self.vocab_idx[w] for w in x]
        return self.embed(torch.tensor(idx,device=self.embed.weight.device))
    
        
if __name__ == '__main__':
    desc = Description(10, vocab={'dog':0, 'cat':1, 'bird':2})
    x = desc(['dog', 'cat', 'dog'])
    print(desc.parameters())
    for p in desc.named_parameters():
        print(p)
    print(desc.vocab)
    y = torch.zeros(3,10) + 0.1
    print(x)
    optim = torch.optim.Adam(desc.parameters(), lr=0.1)
    for _ in range(100):
        x = desc(['dog', 'cat', 'dog'])
        loss = (x-y).abs().sum()
        loss.backward()
        optim.step()
        print(loss)
        optim.zero_grad()
    print(desc.vocab)
    