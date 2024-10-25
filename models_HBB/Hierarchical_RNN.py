import torch
import torch.nn as nn
# from .configClass import config

# import torch.optim as optim
# import torch.nn.functional as F
# from dataLoader import load_data


# 此文件是好宝宝的多层segRNN的尝试，主要创新点是Hierarchical结构。
class Hierarch_RNN(nn.Module):
    def __init__(self, CONFIG):
        super(Hierarch_RNN, self).__init__()
        self.seq_len = CONFIG.input_length
        self.pred_len = CONFIG.output_length
        self.enc_in = CONFIG.enc_in
        # enc_in为变量数，是输入x的shape[-1]

        self.d_model = CONFIG.dmodel
        self.hidden_size = self.d_model
        self.dropout = CONFIG.dropout

        self.seg_len = CONFIG.seg_length
        self.hierarch_layers = CONFIG.hierarch_layers
        self.hierarch_scale = CONFIG.hierarch_scale

        self.seg_len_list = [self.seg_len // (self.hierarch_scale ** i) for i in range(self.hierarch_layers)]
        # seg_len_list = [96, 48, 24]
        self.d_modelSize_list = [self.d_model // (self.hierarch_scale ** i) for i in range(self.hierarch_layers)]
        #      dmodel = [512,256,128]

        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len
        self.seg_num_y_list = [self.pred_len // self.seg_len_list[i] for i in range(self.hierarch_layers)]

        # 定义embeding层，也具有多尺度，反正就是多尺度我超
        self.valueEmbedding_Hierarchical = nn.ModuleList([
            nn.Sequential(
                # 考虑一下这个要不要加一个dropout，可以以后测试一下
                nn.Linear(self.seg_len_list[i], self.d_modelSize_list[i]),
                nn.ReLU()
            )
            # dmodel = [512,256,128]
            for i in range(self.hierarch_layers)
        ])

        # 定义多尺度的GRU层
        self.gru_cells = nn.ModuleList([
            nn.GRUCell(input_size=self.d_modelSize_list[i],
                       hidden_size=self.d_modelSize_list[i],
                       bias=True)
            # dmodel = [512,256,128]
            for i in range(self.hierarch_layers)
        ])

        # 定义多尺度的输出层，这个感觉还可以呢
        self.predict_Hierarchical = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_modelSize_list[i], self.seg_len_list[i]),
            )
            for i in range(self.hierarch_layers)
        ])


        self.pos_emb_List = nn.ParameterList([
            nn.Parameter(torch.randn(self.seg_num_y_list[i], self.d_modelSize_list[i] // 2))
            for i in range(self.hierarch_layers)
        ])

        self.channel_emb_List = nn.ParameterList([
            nn.Parameter(torch.randn(self.enc_in, self.d_modelSize_list[i] // 2))
            for i in range(self.hierarch_layers)
        ])

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


        if self.use_gruCell and not self.use_hirarchical:
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

        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)

        _, hy = self.gru(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))  # bcm,1,d  1,bcm,d

        y = self.predict(hy)

        y = y.view(-1, self.enc_in, self.pred_len)
        # permute and denorm
        y = y.permute(0, 2, 1)

        y = y + seq_last
        return y

    def forward(self, x):
        # 初始化隐藏状态
        return self.encoder(x)

# 设置设备
if __name__ == '__main__':
    CONIFG = config()
    model = model_dict[CONFIG.model_name](CONFIG).to(device)