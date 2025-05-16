# %matplotlib inline
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

import copy

# because I ran this code on a GPU server, so cuda:7 for me is correct.


t = torch.arange(0, 50, 0.01)
sin_t = t.sin()
sin_t_discrete = ((sin_t + 1) / 2 * 99 + 1).long()

class my_MaskedMHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope_on: bool = False, base: int = 10000):
        super().__init__()
        assert(d_model % n_heads == 0)
        self.d_per_head = d_model // n_heads
        self.n_heads = n_heads
        self.rope_on = rope_on
        self.base = base

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)


    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, valid_lens: torch.Tensor = None):
        """
        Q, K, V: shape = (batch_size, seq_len, d_model)
        valid_lens: shape = (batch_size, seq_len, seq_len), 1 means retain, 0 means mask
        """

        # after shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = Q.shape
        d_per_head = self.d_per_head
        n_heads = self.n_heads

        # after shape: (batch_size, seq_len, d_model)
        q: torch.Tensor = self.W_q(Q)
        k: torch.Tensor = self.W_k(K)
        v: torch.Tensor = self.W_v(V)

        # after shape: (batch_size * n_heads, seq_len, d_per_head)
        q = q.reshape(batch_size, seq_len, n_heads, d_per_head).permute(0, 2, 1, 3).reshape(batch_size * n_heads, seq_len, d_per_head)
        k = k.reshape(batch_size, seq_len, n_heads, d_per_head).permute(0, 2, 1, 3).reshape(batch_size * n_heads, seq_len, d_per_head)
        v = v.reshape(batch_size, seq_len, n_heads, d_per_head).permute(0, 2, 1, 3).reshape(batch_size * n_heads, seq_len, d_per_head)

        if self.rope_on == True:
            # theta: shape = (d_per_head / 2, )
            theta = 1. / (self.base ** (torch.arange(0, d_per_head, 2).float() / d_model)).to(q.device)
            # seq_idx: shape = (seq_len, )
            seq_idx = torch.arange(seq_len, device=q.device).float().to(q.device)
            # idx_theta: shape = (seq_len, d_per_head / 2)
            idx_theta = torch.einsum("m,d->md", seq_idx, theta)
            # idx_theta_2: shape = (seq_len, d_per_head)
            idx_theta_2 = torch.cat([idx_theta, idx_theta], dim=1)

            # xxx_cached: shape = (batch_size * n_heads, seq_len, d_per_head)
            cos_cached = idx_theta_2.cos()[None, :, :].repeat(batch_size * n_heads, 1, 1)
            sin_cached = idx_theta_2.sin()[None, :, :].repeat(batch_size * n_heads, 1, 1)

            def rotate_half(x: torch.Tensor):
                x_1 = x[..., :x.shape[-1] // 2]
                x_2 = x[..., x.shape[-1] // 2:]
                return torch.cat([-x_2, x_1], dim=-1)
            
            neg_half_q = rotate_half(q)
            q_rope = q * cos_cached + neg_half_q * sin_cached

            neg_half_k = rotate_half(k)
            k_rope = k * cos_cached + neg_half_k * sin_cached

            # score: shape = (batch_size * n_heads, seq_len, seq_len)
            score = torch.bmm(q_rope, k_rope.permute(0, 2, 1)) / math.sqrt(d_per_head)
        else:
            # score: shape = (batch_size * n_heads, seq_len, seq_len)
            score: torch.Tensor = torch.bmm(q, k.permute(0, 2, 1)) / math.sqrt(d_per_head)

        if valid_lens is not None:
            assert(valid_lens.shape == (batch_size, seq_len, seq_len))
            valid_lens = valid_lens.repeat(n_heads, 1, 1)
            score = score.masked_fill(~valid_lens, float('-inf'))

        # attention_weights: shape = (batch_size * n_heads, seq_len, seq_len)
        attention_weights = torch.softmax(score, dim=-1)
        
        # output: shape = (batch_size * n_heads, seq_len, d_per_head)
        output = torch.bmm(attention_weights, v)

        # after shape: (batch_size, seq_len, d_model)
        output = output.reshape(batch_size, n_heads, seq_len, d_per_head).permute(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        return self.W_o(output), attention_weights


class my_DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dim_feedforward: int = 2048, dropout: float = 0.1, rope_on: bool = False):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.mha = my_MaskedMHA(d_model=d_model, n_heads=n_heads, rope_on=rope_on)
        self.dropout_1 = nn.Dropout(dropout)
        self.n_layers = n_layers

        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )


    def forward(self, x: torch.Tensor, valid_lens: torch.Tensor = None) -> torch.Tensor:
        y, attention_weights = self.mha(self.layer_norm_1(x), x, x, valid_lens)
        y = self.dropout_1(y)
        y = x + (1 / math.sqrt(self.n_layers)) * y

        z = self.ffn(self.layer_norm_2(y))
        z = y + (1 / math.sqrt(self.n_layers)) * z
        return z


class my_TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: my_DecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])


    def forward(self, x: torch.Tensor, valid_lens: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_lens)
        return x


class my_Decoder_only_network(nn.Module):
    def __init__(self, vocab_size: int = 100, d_model: int = 768, max_seq_len: int = 10, n_layers: int = 4, pos_encoding: str = None, **kwargs):
        super().__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=1.0)

        self.pos_encoding = pos_encoding
        rope_on: bool = False
        if self.pos_encoding == "learnable":
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
            nn.init.normal_(self.pos_embedding.weight, mean=0, std=1.0)
        elif self.pos_encoding == "RoPE":
            rope_on = True

        self.dropout = nn.Dropout(0.1)
        self.n_layers = n_layers

        decoder_layer = my_DecoderLayer(
            d_model=d_model, 
            n_heads=4, 
            n_layers=n_layers,
            dim_feedforward=2048, 
            dropout=0.1,
            rope_on=rope_on
        )
        self.decoder = my_TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layers)

        self.out_embedding = nn.Linear(d_model, vocab_size, bias=False)
        self.out_embedding.weight = self.token_embedding.weight

    def forward(self, X: torch.Tensor, valid_lens: torch.Tensor = None):
        """
        X: shape = (batch_size, seq_len), each element is the index of token in vocab
        valid_lens: shape = (batch_size) or (batch_size, seq_len)
        """

        if valid_lens is not None:
            batch_size, seq_len = X.shape
            assert(valid_lens.dim() == 1)
            tmp = torch.tril(torch.ones(seq_len, seq_len, device=X.device)).bool()[None, :].repeat(batch_size, 1, 1)
            for i, valid_len in enumerate(valid_lens):
                tmp[i, :, valid_len:] = False
            valid_lens = tmp

        # shape of X: (batch_size, seq_len)
        batch_size, seq_len = X.shape[0], X.shape[1]

        # shape of token_emb: (batch_size, seq_len, d_model)
        token_emb = self.token_embedding(X)

        # shape of pos_idx: (batch_size, seq_len)
        pos_idx = torch.arange(seq_len, device=X.device)[None, :].repeat(batch_size, 1)

        if self.pos_encoding == "learnable":
            # shape of pos_emb: (batch_size, seq_len, d_model)
            pos_emb = self.pos_embedding(pos_idx)
            in_emb = token_emb + pos_emb
        elif self.pos_encoding is None:
            in_emb = token_emb
        elif self.pos_encoding == "RoPE":
            in_emb = token_emb
        
        in_emb = self.dropout(in_emb)

        out_emb = self.decoder.forward(in_emb, valid_lens)
        
        # shape of return: (batch_size, seq_len, vocab_size)
        return self.out_embedding(out_emb)
    

class TokenDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size
    

    def __len__(self):
        return len(self.tokens) - self.block_size - 1
    

    def __getitem__(self, idx):
        x = self.tokens[idx: idx + self.block_size]
        y = self.tokens[idx + 1: idx + self.block_size + 1]
        return x, y


class MaskedCELoss(nn.CrossEntropyLoss):

    def forward(self, pred: torch.Tensor, label: torch.Tensor, valid_lens: torch.Tensor):
        """
        pred: shape = (batch_size, vocab_size, seq_len)
        label: shape = (batch_size, seq_len)
        valid_lens: shape = (batch_size, )
        """
        assert(valid_lens.dim() == 1)
        if valid_lens.dim() == 1:
            batch_size, seq_len = label.shape

            # weights: shape = (batch_size, seq_len)
            weights = torch.where(torch.arange(seq_len)[None, :] < valid_lens[:, None], 1, 0).to(device)

        self.reduction = 'none'

        # unweighted_loss: shape = (batch_size, seq_len)
        unweighted_loss = super().forward(pred, label)

        # weighted_loss: shape = (batch_size, )
        weighted_loss = (unweighted_loss * weights).mean(dim=-1)
        return weighted_loss
    

def train(dataloader: DataLoader, net: my_Decoder_only_network, num_epochs: int, device: str):
    net = net.to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    criterion = MaskedCELoss()
    pad_value = 0

    loss_list = []
    for epoch in range(num_epochs):
        tot_loss = 0
        tot_num = 0
        max_loss = float('-inf')
        min_loss = float('inf')
        print(f"epoch: {epoch} begin to train")
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)

            tmp_batch_size = x.shape[0] # because x.shape[0] may be smaller than prevous batch_size due to the features of DataLoader
            # x: shape = (tmp_batch_size, block_size)
            # y: shape = (tmp_batch_size, block_size)

            # valid_len: shape = (tmp_batch_size, )
            x_valid_lens: torch.Tensor = torch.tensor([block_size]).repeat(tmp_batch_size)
            # print(f"valid_len: {valid_len.shape}")

            # y_hat: shape = (tmp_batch_size, seq_len, vocab_size)
            y_hat: torch.Tensor = net(x, x_valid_lens)
            # print(f"y_hat: {y_hat.shape}")

            # y_valid_len: shape = (tmp_batch_size, )
            y_valid_lens: torch.Tensor = torch.tensor([block_size]).repeat(tmp_batch_size)
            # print(f"y_valid_len: {y_valid_len.shape}")

            loss: torch.Tensor = criterion(y_hat.permute(0, 2, 1), y, y_valid_lens)
            optimizer.zero_grad()

            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                tot_loss += loss.mean()
                max_loss = max(max_loss, loss.mean())
                min_loss = min(min_loss, loss.mean())
                tot_num += 1

        tot_loss /= tot_num
        loss_list.append(tot_loss)
        print(f"epoch: {epoch}, tot_loss: {tot_loss}, max_loss: {max_loss}, min_loss: {min_loss}")

    torch.save(net.state_dict(), 'model_weights.pth')


def test(test_dataloader: DataLoader, net: my_Decoder_only_network, test_cases: int, device: str, orig_len: int = 32, state_dict_path: str = None, fig_name: str = "plot.png"):
    if state_dict_path is not None:
        net.load_state_dict(torch.load(state_dict_path))
    net.to(device)
    net.eval()

    n_row = int(math.sqrt(test_cases))
    n_col = int(test_cases / n_row)
    fig, axes = plt.subplots(n_row, n_col, figsize=(8, 6))
    for idx, (x, _) in enumerate(test_dataloader):
        print(f"test cases {idx} begin")
        if idx >= test_cases:
            break
        x_original = copy.deepcopy(x)
        x: torch.Tensor = x.to(device)
        
        x[:, orig_len:] = 0
        x_valid_len = torch.tensor([orig_len]).repeat(x.shape[0])

        for i in tqdm(range(orig_len, x.shape[1], 1)):
            # y_hat: shape = (tmp_batch_size, seq_len, vocab_size)
            y_hat: torch.Tensor = net.forward(x, x_valid_len)
            y_hat = y_hat.argmax(dim=-1)
            x[:, i] = y_hat[:, i - 1]
            x_valid_len += 1
            # print(f"x: shape = {x.shape}")

        x = x.cpu()

        i = idx // n_col
        j = idx % n_col
        axes[i, j].plot(x[0], label="predict")
        axes[i, j].plot(x_original[0], label='original')
        axes[i, j].legend()

    plt.tight_layout()
    plt.savefig(fig_name)


def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

if __name__ == "__main__":
    device = 'cuda:7'
    vocab_size = 100 + 1
    block_size = 128
    batch_size = 128

    dataset = TokenDataset(tokens=sin_t_discrete, block_size=block_size)
    train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    net = my_Decoder_only_network(vocab_size=vocab_size, max_seq_len=block_size, pos_encoding="RoPE")
    net.apply(init_xavier)
    train(dataloader=train_dataloader, net=net, num_epochs=20, device=device)

    state_dict_path = "model_weights.pth"
    test(test_dataloader=test_dataloader, net=net, device=device, test_cases=16, orig_len=32, state_dict_path=state_dict_path, fig_name="test_with_RoPE")