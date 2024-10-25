import torch
import torch.nn as nn
from utils_HBB.functions_TM import series_decomp
from models_HBB.timeMixer import PastDecomposableMixing
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

        self.use_gruCell = CONFIG.use_gruCell
        self.use_hirarchical = CONFIG.use_hirarchical

        self.d_model = CONFIG.dmodel
        self.preprocess = series_decomp(CONFIG.moving_avg)
        # 增加了趋势分解操作函数。
        if self.use_hirarchical:
            self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(CONFIG)
                                         for _ in range(CONFIG.e_layers)])
        # 增加了历史历史分解模块
        self.hidden_size = self.d_model
        self.num_layers = CONFIG.num_layers
        self.dropout = CONFIG.dropout
        # self.dropout_residual = 0.5
        self.use_decompose = CONFIG.use_decompose
        self.series_decompose = series_decomp(CONFIG.moving_avg)

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
        )
        self.gru_cell = nn.GRUCell(
            input_size=self.d_model,
            hidden_size=self.d_model,
            # num_layers=self.num_layers,
            bias=True,
            # batch_first=True,
        )
        if CONFIG.use_decompose:
            self.gru_cell_second = nn.GRUCell(
                input_size=self.d_model,
                hidden_size=self.d_model,
                # num_layers=self.num_layers,
                bias=True,
            )
        #     当需要趋势分解的时候，就进行两次GRU层

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
        ) if CONFIG.use_residual else nn.Identity()

    def pre_enc(self, x_list):
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def encoder(self, x):
        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)
        # print(x.shape)
        # normalization and permute     b,s,c -> b,c,s
        seq_last = x[:, -1:, :].detach()

        x = (x - seq_last).permute(0, 2, 1)  # b,c,s
        # print("xshape is on the way：")
        # print(x.shape)
        x_s_preEmbed, x_t_preEmbed = self.series_decompose(x)
        # x_s_preEmbed, x_t_preEmbed = series_decomp(x)

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x_s = self.valueEmbedding(x_s_preEmbed.reshape(-1, self.seg_num_x, self.seg_len))
        x_t = self.valueEmbedding(x_t_preEmbed.reshape(-1, self.seg_num_x, self.seg_len))

        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding

        if self.use_decompose:
            # 使用了趋势分解，加入了一层趋势分解。
            # x_s, x_t = series_decomp(x)
            # series_decomp作用在X的最后一维度，应该在valueEmbedding之前使用
            h_t_seasonal = torch.zeros(x_s.shape[0], x.shape[2]).to(x.device)
            h_t_trend = torch.zeros(x_t.shape[0], x.shape[2]).to(x.device)
            for i in range(self.seg_num_x):
                x_t_seasonal = x_s[:, i, :]
                x_t_trend = x_t[:, i, :]
                h_t_seasonal = self.gru_cell(x_t_seasonal, h_t_seasonal)
                h_t_trend = self.gru_cell_second(x_t_trend, h_t_trend)
                # h_t = x_t + self.residual_projection(h_t)
            h_t_final = h_t_seasonal + h_t_trend
            hn = h_t_final.unsqueeze(0)


        elif self.use_gruCell and not self.use_hirarchical:
            h_t = torch.zeros(x.shape[0], x.shape[2]).to(x.device)
            for i in range(self.seg_num_x):
                x_t = x[:, i, :]
                h_t = self.gru_cell(x_t, h_t)
                h_t = x_t + self.residual_projection(h_t)
            hn = h_t.unsqueeze(0)
        #     加入了残差和隐藏状态的dropout，想要测试一下这样会不会正则化强一点
        elif not self.use_hirarchical:
            _, hn = self.gru(x)  # bc,n,d  1,bc,d
        elif self.use_hirarchical:
            h_t = torch.zeros(x.shape[0], x.shape[2]).to(x.device)
            for i in range(self.seg_num_x):
                x_t = x[:, i, :]
                h_t = self.gru_cell(x_t, h_t)
                h_t = x_t + self.residual_projection(h_t)
            hn = h_t.unsqueeze(0)



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
