import torch
import torch.nn as nn
from utils_HBB.functions_TM import series_decomp
# import torch.optim as optim
# import torch.nn.functional as F
# from dataLoader import load_data



# 此文件就对SegRNN的复现，属于是好宝宝的测试
class GRUModel(nn.Module):
    def __init__(self, CONFIG):
        super(GRUModel, self).__init__()
        self.seq_len = CONFIG.input_length
        self.pred_len = CONFIG.output_length
        self.enc_in = CONFIG.enc_in
        # enc_in为变量数，是输入x的shape[-1]
        self.d_model = CONFIG.dmodel


        self.hidden_size = self.d_model
        self.num_layers = CONFIG.num_layers
        self.dropout = CONFIG.dropout
        # self.dropout_residual = 0.5

        self.seg_len = CONFIG.seg_length
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len
        self.use_residual = CONFIG.use_residual

        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.SiLU()
        )
        # 定义GRU层
        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            bidirectional=False,
            # dorpout=0.1
        )
        self.gru_cell = nn.GRUCell(
            input_size=self.d_model,
            hidden_size=self.d_model,
            # num_layers=self.num_layers,
            bias=True,
            # batch_first=True,
        )
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.hidden_size // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.hidden_size // 2))
        # 定义输出层
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len),
        )

        self.residual_projection = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
        )

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
        hn = torch.zeros(1, x.shape[0], x.shape[2]).to(x.device)

        if self.use_residual:
            h_t = torch.zeros(x.shape[0], x.shape[2]).to(x.device)
            for i in range(self.seg_num_x):
                x_t = x[:, i, :]
                h_t = self.gru_cell(x_t, h_t)
                h_t = x_t + self.residual_projection(h_t)
            hn = h_t.unsqueeze(0)
        #     加入了残差和隐藏状态的dropout，想要测试一下这样会不会正则化强一点
        else :
            _, hn = self.gru(x)  # bc,n,d  1,bc,d




        # print("here comes hn.shape:")
        # print(h_t.shape)
        # print(hn.shape)
        # print(x.shape)
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

# 设置设备
