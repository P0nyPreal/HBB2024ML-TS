import torch
import torch.nn as nn
from utils_HBB.functions_TM import series_decomp
from models_HBB.timeMixer import PastDecomposableMixing
from utils_HBB.Embed import DataEmbedding_wo_pos


# import torch.optim as optim
# import torch.nn.functional as F
# from dataLoader import load_data


# 此文件是对之前多尺度和分解RNN效果不好的另一个复现，准备直接在testGRU上面改
class TestDecompRNN(nn.Module):
    def __init__(self, CONFIG):
        super(TestDecompRNN, self).__init__()
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
        # self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.enc_in)

        self.seg_len = CONFIG.seg_length
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len
        self.use_residual = CONFIG.use_residual

        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
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
            self.gru_second = nn.GRU(
                input_size=self.d_model,
                hidden_size=self.d_model,
                num_layers=self.num_layers,
                bias=True,
                batch_first=True,
                bidirectional=False,
            )
        #     当需要趋势分解的时候，就进行两次GRU层

        self.pos_emb_trend = nn.Parameter(torch.randn(self.seg_num_y, self.hidden_size // 2))
        self.channel_emb_trend = nn.Parameter(torch.randn(self.enc_in, self.hidden_size // 2))

        self.pos_emb_seasonal = nn.Parameter(torch.randn(self.seg_num_y, self.hidden_size // 2))
        self.channel_emb_seasonal = nn.Parameter(torch.randn(self.enc_in, self.hidden_size // 2))

        # 定义输出层

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len),
        )

        self.predict_seasonal = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len),
        )

        self.predict_trend = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len),
        )

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

        # 使用了趋势分解，加入了一层趋势分解。
        # x_s, x_t = series_decomp(x)
        # series_decomp作用在X的最后一维度，应该在valueEmbedding之前使用
        h_t_seasonal = torch.zeros(x_s.shape[0], self.d_model).to(x.device)
        h_t_trend = torch.zeros(x_t.shape[0], self.d_model).to(x.device)
        for i in range(self.seg_num_x):
            x_t_seasonal = x_s[:, i, :]
            x_t_trend = x_t[:, i, :]
            h_t_seasonal = self.gru_cell(x_t_seasonal, h_t_seasonal)
            h_t_trend = self.gru_cell_second(x_t_trend, h_t_trend)
            # h_t = (h_t_seasonal + h_t_trend).unsqueeze(1)
            # h_t_seasonal, h_t_trend = self.series_decompose(h_t)
            # h_t_seasonal = h_t_seasonal[:, 0, :]
            # h_t_trend = h_t_trend[:, 0, :]
        h_t_final = h_t_seasonal + h_t_trend
        hn = h_t_final.unsqueeze(0)

        h_t_seasonal = h_t_seasonal.unsqueeze(0)
        h_t_trend = h_t_trend.unsqueeze(0)


        pos_emb_trend = torch.cat([
            self.pos_emb_trend.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb_trend.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)

        pos_emb_seasonal  = torch.cat([
            self.pos_emb_seasonal.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb_seasonal.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)


        # print("here comes pos_emb.shape:")
        # print(pos_emb.shape)

        hn = hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)

        h_t_seasonal = h_t_seasonal.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)
        h_t_trend = h_t_trend.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)

        # _, hy = self.gru(pos_emb, hn)  # bcm,1,d  1,bcm,d

        output_list = []
        output_trend_list = []
        output_seasonal_list = []
        for i in range(self.seg_num_y):
            hn_step_length = hn.size(1) // self.seg_num_y
            out_put_current_seasonal = self.gru_cell(pos_emb_seasonal[:, i, :],
                                            h_t_seasonal[0, i * hn_step_length: (i + 1) * hn_step_length])
            out_put_current_trend = self.gru_cell_second(pos_emb_trend[:, i, :],
                                            h_t_trend[0, i * hn_step_length: (i + 1) * hn_step_length])
            output_trend_list.append(out_put_current_trend.unsqueeze(0))
            output_seasonal_list.append(out_put_current_seasonal.unsqueeze(0))
            output_list.append((out_put_current_seasonal + out_put_current_trend).unsqueeze(0))


        hy = torch.cat(output_list, dim=1)
        h_trend = torch.cat(output_trend_list, dim=1)
        h_seasonal = torch.cat(output_seasonal_list, dim=1)
        y = self.predict_trend(h_trend) + self.predict_seasonal(h_seasonal)

        # 1,bcm,d -> 1,bcm,w -> b,c,s
        # y = self.predict(y)
        # y = self.predict(hy)

        y = y.view(-1, self.enc_in, self.pred_len)
        # permute and denorm
        y = y.permute(0, 2, 1)

        y = y + seq_last
        return y

    def forward(self, x):
        # 初始化隐藏状态
        return self.encoder(x)

# 设置设备
