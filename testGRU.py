import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataLoader import load_data


# 此文件就对SegRNN的复现，属于是好宝宝的测试
class GRUModel(nn.Module):
    def __init__(self, input_size=1, num_layers=1, output_size=1, seg_len = 1, enc_in = 1):
        super(GRUModel, self).__init__()
        self.seq_len = input_size
        self.pred_len = output_size
        self.enc_in = enc_in
        # enc_in为变量数，是输入x的shape[-1]
        self.d_model = 512


        self.hidden_size = self.d_model
        self.num_layers = num_layers
        self.dropout = 0.35
        self.seg_len = seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len


        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        # 定义GRU层
        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            bidirectional=False
        )
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.hidden_size // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.hidden_size // 2))
        # 定义输出层
        self.predict = nn.Sequential(
            nn.Linear(self.d_model, self.seg_len),
            nn.Dropout(self.dropout),
        )
        # nn.Linear(hidden_size, output_size))

    def encoder(self, x):
        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)
        # print(x.shape)
        # normalization and permute     b,s,c -> b,c,s
        seq_last = x[:, -1:, :].detach()

        x = (x - seq_last).permute(0, 2, 1)  # b,c,s

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))
        # print(x.shape)
        # encoding
        _, hn = self.gru(x)  # bc,n,d  1,bc,d
        # print("here comes hn.shape:")
        # print(hn.shape)
        # m,d//2 -> 1,m,d//2 -> c,m,d//2
        # c,d//2 -> c,1,d//2 -> c,m,d//2
        # c,m,d -> cm,1,d -> bcm, 1, d
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)
        # print("here comes pos_emb.shape:")
        # print(pos_emb.shape)
        _, hy = self.gru(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))  # bcm,1,d  1,bcm,d
        # print("here comes hy.shape:")
        # print(hy.shape)
        # 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy)
        # print("here comes y.shape:")
        # print(y.shape)
        y = y.view(-1, self.enc_in, self.pred_len)
        # permute and denorm
        y = y.permute(0, 2, 1)

        y = y + seq_last
        return y

    def forward(self, x):
        # 初始化隐藏状态
        return self.encoder(x)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #
        # # GRU前向传播
        # out, _ = self.gru(x, h0)  # out: [batch_size, seq_length, hidden_size]
        #
        # # 将GRU的输出传入全连接层
        # out = self.predict(out)  # out: [batch_size, seq_length, output_size]
        #
        # return out

# 设置设备
