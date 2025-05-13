import torch
from torch import nn
from torch.nn import functional as F

import math

class MHA(nn.Module):
    def __init__(self, d_q, d_k, d_v, one_head_hiddens, n_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.one_head_hiddens = one_head_hiddens 
        
        # d_q == d_k == d_v == n_hiddens
        self.n_hiddens = n_heads * one_head_hiddens
        self.W_q = nn.Linear(d_q, self.n_hiddens, bias=bias)
        self.W_k = nn.Linear(d_k, self.n_hiddens, bias=bias)
        self.W_v = nn.Linear(d_v, self.n_hiddens, bias=bias)

        self.W_o = nn.Linear(self.n_hiddens, self.n_hiddens, bias=bias)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid_lens: torch.Tensor = None) -> torch.Tensor:
        # shape of q, k, v: (batch_size, seq_len, d_q/d_k/d_v where d_q == d_k == d_v)
        # shape of valid_lens: (batch_size, ) or (batch_size, seq_len)
        seq_len = q.shape[1]

        quires: torch.Tensor = self.W_q(q) # (batch_size, seq_len, n_head * one_head_hiddens)
        keys: torch.Tensor = self.W_k(k) # (batch_size, seq_len, n_head * one_head_hiddens)
        values: torch.Tensor = self.W_v(v) # (batch_size, seq_len, n_head * one_head_hiddens)

        quires = quires.view(quires.shape[0], quires.shape[1], self.n_heads, self.one_head_hiddens) # (batch_size, seq_len, n_heads, one_head_hiddens)
        keys = keys.view(keys.shape[0], keys.shape[1], self.n_heads, self.one_head_hiddens) # (batch_size, seq_len, n_heads, one_head_hiddens)
        values = values.view(values.shape[0], values.shape[1], self.n_heads, self.one_head_hiddens) # (batch_size, seq_len, n_heads, one_head_hiddens)

        # transpose to (batch_size, n_heads, seq_len, one_head_hiddens)
        quires = quires.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # reshape to (batch_size * n_head, seq_len, one_head_hiddens)
        quires = quires.reshape(-1, quires.shape[2], quires.shape[3])
        keys = keys.reshape(-1, keys.shape[2], keys.shape[3])
        values = values.reshape(-1, values.shape[2], values.shape[3])

        # scaled dot product attention
        scores = torch.bmm(quires, keys.transpose(-1, -2)) / math.sqrt(self.one_head_hiddens) # (batch_size * n_heads, seq_len, seq_len)

        if valid_lens is not None:
            # transform trajectory of valid_lens: 
            # (batch_size, )
            # (batch_size, seq_len)
            # (batch_size * n_heads, seq_len)
            # (batch_size * n_heads, seq_len, seq_len)
            if valid_lens.dim() == 1:
                valid_lens = valid_lens.repeat_interleave(seq_len, 0).reshape(-1, seq_len) # (batch_size, seq_len)
            
            valid_lens = valid_lens.repeat(self.n_heads, 1) # (batch_size * n_heads, seq_len)
            valid_lens = valid_lens[:, :, None].repeat(1, 1, seq_len) # (batch_size * n_heads, seq_len, seq_len)

            arange_tensor = torch.arange(seq_len)[None, None, :].repeat(valid_lens.shape[0], seq_len, 1) # (batch_size * n_heads, seq_len, seq_len)
            masks = arange_tensor < valid_lens # (batch_size * n_heads, seq_len, seq_len)

            scores = scores.masked_fill(~masks, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1) # (batch_size * n_heads, seq_len, seq_len)
        
        outputs = torch.bmm(self.dropout(attention_weights), values) # (batch_size * n_heads, seq_len, one_head_hiddens)
        outputs = outputs.reshape(q.shape[0], self.n_heads, q.shape[1], self.one_head_hiddens) # (batch_size, n_heads, seq_len, one_head_hiddens)
        outputs = outputs.permute(0, 2, 1, 3) # (batch_size, seq_len, n_heads, one_head_hiddens)
        outputs = outputs.reshape(q.shape[0], q.shape[1], -1) # (batch_size, seq_len, n_heads * one_head_hiddens)

        return self.W_o(outputs) # (batch_size, seq_len, n_heads * one_head_hiddens)
    
    
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    num_hiddens = 20
    n_heads = 2
    dropout = 0.5

    q = torch.randn((batch_size, seq_len, num_hiddens))
    k = torch.randn((batch_size, seq_len, num_hiddens))
    v = torch.randn((batch_size, seq_len, num_hiddens))

    # valid_lens = torch.randint(1, seq_len, (batch_size, ))
    valid_lens = [4, 3]
    valid_lens = [[i if i <= max_len else max_len for i in range(1, seq_len + 1, 1)] for max_len in valid_lens]
    valid_lens = torch.tensor(valid_lens)

    mha = MHA(num_hiddens, num_hiddens, num_hiddens, one_head_hiddens=int(num_hiddens / n_heads), n_heads=n_heads, dropout=dropout)

    print(mha.eval())

    result: torch.Tensor = mha(q, k, v, valid_lens)
    print(result)
    print(result.shape)

