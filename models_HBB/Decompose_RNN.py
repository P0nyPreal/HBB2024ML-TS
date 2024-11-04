import torch
import torch.nn as nn
from utils_HBB.functions_TM import series_decomp
from utils_HBB.norm import Normalize
from utils_HBB.Embed import DataEmbedding_wo_pos


# from .configClass import config

# import torch.optim as optim
# import torch.nn.functional as F
# from dataLoader import load_data
class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.input_length
        self.pred_len = configs.output_length
        self.down_sampling_window = configs.down_sampling_window
        # self.batch_size = configs.ba


        self.layer_norm = nn.LayerNorm(configs.dmodel)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        self.decomposition = series_decomp(configs.moving_avg)

        if not configs.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.dmodel, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.dmodel),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mixing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.dmodel, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.dmodel),
        )

    def forward(self, x_list):

        length_list = []
        for x in x_list:

            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


# 此文件是好宝宝的多层segRNN的尝试，主要创新点是Hierarchical结构。
class Decompose_RNN(nn.Module):
    def __init__(self, CONFIG):
        super(Decompose_RNN, self).__init__()
        self.seq_len = CONFIG.input_length
        self.pred_len = CONFIG.output_length
        self.enc_in = CONFIG.enc_in
        self.batch_size = CONFIG.batch_size
        # enc_in为变量数，是输入x的shape[-1]

        self.d_model = CONFIG.dmodel
        self.hidden_size = self.d_model
        self.dropout = CONFIG.dropout
        self.use_mixing = CONFIG.use_mixing
        if self.use_mixing:
            self.mixing_route = CONFIG.mixing_route

        self.seg_len = CONFIG.seg_length
        self.hierarch_layers = CONFIG.hierarch_layers
        self.hierarch_scale = CONFIG.hierarch_scale
        self.down_sampling_method = CONFIG.down_sampling_method
        self.down_sampling_window = CONFIG.hierarch_scale
        self.multi_scale_process_inputs = CONFIG.multi_scale_process_inputs
        self.use_rand_emb = CONFIG.use_rand_emb

        self.use_decompose = CONFIG.use_decompose
        self.series_decompose = series_decomp(CONFIG.moving_avg)

        self.seg_len_list = [self.seg_len // (self.hierarch_scale ** i) for i in range(self.hierarch_layers)]
        # seg_len_list = [96, 48, 24]
        self.d_modelSize_list = [self.d_model // (self.hierarch_scale ** i) for i in range(self.hierarch_layers)]
        # self.d_modelSize_list = [self.d_model for i in range(self.hierarch_layers)]
        #      dmodel = [512,256,128]

        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_x_list = [self.seq_len // self.seg_len_list[0] for _ in range(self.hierarch_layers)]

        self.seg_num_y = self.pred_len // self.seg_len
        self.seg_num_y_list = [self.pred_len // self.seg_len_list[0] for _ in range(self.hierarch_layers)]

        # 定义了pdm模块
        self.pdm_blocks = PastDecomposableMixing(CONFIG)

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.enc_in, affine=True, non_norm=True if not CONFIG.use_norm else False)
                for _ in range(CONFIG.hierarch_layers)
            ]
        )

        # 定义embeding层，也具有多尺度，反正就是多尺度我超
        self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.enc_in,)

        # 定义多尺度的GRU层
        self.gru_cells = nn.ModuleList([
            nn.GRUCell(input_size=self.seg_len_list[i],
                       hidden_size=self.seg_len_list[i],
                       bias=True)
            # dmodel = [512,256,128]
            for i in range(self.hierarch_layers)
        ])

        # 定义多尺度的输出层，这个感觉还可以呢
        self.predict_Hierarchical = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.seg_len_list[i], self.seg_len),
            )
            for i in range(self.hierarch_layers)
        ]) if self.multi_scale_process_inputs else nn.ModuleList([
            nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.seg_len_list[i], self.seg_len_list[i]),
            )
            for i in range(self.hierarch_layers)
        ])

        self.predict_output = nn.Sequential(
            nn.Dropout(0.3),
            # nn.Dropout(self.dropout),
            nn.Linear(self.pred_len * self.hierarch_layers, self.pred_len),
        )

        self.pos_emb_List = nn.ParameterList([
            nn.Parameter(torch.randn(self.seg_num_y_list[i], self.seg_len_list[i] // 2))
            for i in range(self.hierarch_layers)
        ])

        self.channel_emb_List = nn.ParameterList([
            nn.Parameter(torch.randn(self.enc_in, self.seg_len_list[i] // 2))
            for i in range(self.hierarch_layers)
        ])

        self.rand_emb_List = nn.ParameterList([
            nn.Parameter(torch.randn(self.enc_in, 1, self.seg_len_list[i]).repeat(self.batch_size, 1, 1))
            for i in range(self.hierarch_layers)
        ])

        if self.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.down_pool = nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in,
                                       kernel_size=3, padding=padding,
                                       stride=self.down_sampling_window,
                                       padding_mode='circular',
                                       bias=False)


    def __multi_scale_process_inputs(self, x_enc):
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == 'conv':
            down_pool = self.down_pool
        else:
            return x_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc

        x_enc_sampling_list = [x_enc.permute(0, 2, 1)]

        for i in range(self.hierarch_layers - 1):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

        x_enc = x_enc_sampling_list
        return x_enc

    def encoder(self, x):
        batch_size = x.shape[0]
        seq_last = x[:, -1:, :].detach()
        # x_s_preEmbed, x_t_preEmbed = self.series_decompose(x)
        x_seged_list = []
        x_seged_list_preNorm = self.__multi_scale_process_inputs(x)
        # 这里的__multi_scale_process_inputs(x)就是用的Timemixer的多尺度化方式
        x_seged_list_preEmbed = []
        x_seged_list_preprocess = []

        for i, x in zip(range(len(x_seged_list_preNorm)), x_seged_list_preNorm):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            # if self.channel_independence:
            #     x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_seged_list_preEmbed.append(x)

        # 多尺寸输入嵌入===========================================
        for i, x in zip(range(len(x_seged_list_preEmbed)), x_seged_list_preEmbed):
            # enc_out = self.enc_embedding(x)  # [B,T,C]
            # # 通道维度的特征嵌入真的需要吗？？
            # x_seged_list.append(enc_out.reshape(batch_size * self.enc_in, self.seg_num_x, -1))
            x_seged_list.append(x.reshape(batch_size * self.enc_in, self.seg_num_x, -1))


        # x_seged_list = [B,T,C]
        # x_seged_list = self.pdm_blocks(x_seged_list)

        # encoding这里就是多尺度encoding的过程======================

        # hn_list = []
        # for layer in range(self.hierarch_layers):
        #     h_t = torch.zeros(x_seged_list[layer].shape[0], x_seged_list[layer].shape[2]).to(x.device)
        #     for i in range(self.seg_num_x_list[layer]):
        #         x_t = x_seged_list[layer][:, i, :]
        #         h_t = self.gru_cells[layer](x_t, h_t)
        #         # h_t = x_t + h_t
        #     hn_list.append(h_t.unsqueeze(0))

        hn_list_instance = []
        # 逐步循环过程中的初始化
        for i in range(self.hierarch_layers):
            h_t_coarsest = torch.zeros(x_seged_list[i].shape[0], x_seged_list[i].shape[2] ).to(x.device)
            hn_list_instance.append(h_t_coarsest)

        for i in range(self.seg_num_x):
            for layer_now in range(self.hierarch_layers):
                x_t = x_seged_list[layer_now][:, i, :]
                hn_list_instance[layer_now] = self.gru_cells[layer_now](x_t, hn_list_instance[layer_now])
            # hn_list_instance = self.pdm_blocks(hn_list_instance)

            # 这里就是多尺度之间的mixing过程===========================

        hn_list = hn_list_instance

        # 多尺度的可学习通道和位置输出初始化向量生成===================
        pos_emb_list = []
        if not self.use_rand_emb:
            for i in range(self.hierarch_layers):
                pos_emb = torch.cat([
                    self.pos_emb_List[i].unsqueeze(0).repeat(self.enc_in, 1, 1),
                    self.channel_emb_List[i].unsqueeze(1).repeat(1, self.seg_num_y_list[i], 1)
                ], dim=-1).view(-1, 1, self.seg_len_list[i]).repeat(self.batch_size, 1, 1)
                pos_emb_list.append(pos_emb)
        else:
            pos_emb_list = [param[:self.enc_in * batch_size] for param in self.rand_emb_List]

        # 多尺度RNN结构的输出了属于是===============================
        RNN_output_list = []
        for i in range(self.hierarch_layers):
            layer_output_list = []
            hn_now = hn_list[i].repeat(1, 1, self.seg_num_y_list[i]).view(1, -1, self.seg_len_list[i])[0, :, :]
            for step in range(self.seg_num_y_list[i]):
                step_length = pos_emb_list[i].shape[0] // self.seg_num_y_list[i]
                hn_stop_length = hn_now.shape[0] // self.seg_num_y_list[i]
                # print(hn_stop_length)
                pos_emb_input = pos_emb_list[i][step * step_length: (step + 1) * step_length][:, 0, :]
                hn_input = hn_now[step * hn_stop_length: (step + 1) * hn_stop_length, :]
                hy = self.gru_cells[i](pos_emb_input, hn_input)
                layer_output_list.append(hy)
            out_put_this_layer = torch.stack(layer_output_list, dim=0)
            # 在第0个维度上concat输出
            RNN_output_list.append(out_put_this_layer.view(1, -1, self.seg_len_list[i]))

        # 最后的多尺度输出投影了属于是==============================
        output = torch.zeros(seq_last.size(0), self.pred_len, seq_last.size(2)).to(x.device)
        last_layer_list = []
        for i in range(self.hierarch_layers):
            y = self.predict_Hierarchical[i](RNN_output_list[i])
            y = y.view(-1, self.enc_in, self.pred_len)
            y = y.permute(0, 2, 1)
            last_layer_list.append(y)
            output += (y + seq_last)
        # output_last_layer = self.predict_output(torch.cat(last_layer_list, dim=1).permute(0, 2, 1))

        # return output_last_layer.permute(0, 2, 1)
        return output / self.hierarch_layers

    def forward(self, x):
        # 初始化隐藏状态
        return self.encoder(x)

# 设置设备
# if __name__ == '__main__':
#     CONIFG = config()
#     model = model_dict[CONFIG.model_name](CONFIG).to(device)
