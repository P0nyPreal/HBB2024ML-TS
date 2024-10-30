import torch
import torch.nn as nn
from utils_HBB.functions_TM import series_decomp


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

        if not self.multi_scale_process_inputs:
            self.seg_num_x = self.seq_len // self.seg_len
            self.seg_num_x_list = [self.seq_len // self.seg_len_list[i] for i in range(self.hierarch_layers)]

            self.seg_num_y = self.pred_len // self.seg_len
            self.seg_num_y_list = [self.pred_len // self.seg_len_list[i] for i in range(self.hierarch_layers)]
        else:
            self.seg_num_x = self.seq_len // self.seg_len
            self.seg_num_x_list = [self.seq_len // self.seg_len_list[0] for i in range(self.hierarch_layers)]

            self.seg_num_y = self.pred_len // self.seg_len
            self.seg_num_y_list = [self.pred_len // self.seg_len_list[0] for i in range(self.hierarch_layers)]

        # 定义embeding层，也具有多尺度，反正就是多尺度我超
        if not self.multi_scale_process_inputs:
            self.valueEmbedding_Hierarchical = nn.ModuleList([
                nn.Sequential(
                    # 考虑一下这个要不要加一个dropout，可以以后测试一下
                    nn.Linear(self.seg_len_list[i], self.d_modelSize_list[i]),
                    nn.ReLU()
                )
                # dmodel = [512,256,128]
                for i in range(self.hierarch_layers)
            ])
        else:
            self.valueEmbedding_Hierarchical = nn.ModuleList([
                nn.Sequential(
                    # 考虑一下这个要不要加一个dropout，可以以后测试一下
                    nn.Linear(self.seg_len_list[i], self.d_modelSize_list[i]),
                    nn.ReLU()
                )
                # dmodel = [512,256,128], seg_len_list = [96, 48, 24]
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
                nn.Linear(self.d_modelSize_list[i], self.seg_len),
            )
            for i in range(self.hierarch_layers)
        ]) if self.multi_scale_process_inputs else nn.ModuleList([
            nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_modelSize_list[i], self.seg_len_list[i]),
            )
            for i in range(self.hierarch_layers)
        ])

        self.predict_output = nn.Sequential(
            nn.Dropout(0.3),
            # nn.Dropout(self.dropout),
            nn.Linear(self.pred_len * self.hierarch_layers, self.pred_len),
        )

        self.pos_emb_List = nn.ParameterList([
            nn.Parameter(torch.randn(self.seg_num_y_list[i], self.d_modelSize_list[i] // 2))
            for i in range(self.hierarch_layers)
        ])

        self.channel_emb_List = nn.ParameterList([
            nn.Parameter(torch.randn(self.enc_in, self.d_modelSize_list[i] // 2))
            for i in range(self.hierarch_layers)
        ])

        self.rand_emb_List = nn.ParameterList([
            nn.Parameter(torch.randn(self.enc_in, 1, self.d_modelSize_list[i]).repeat(self.batch_size, 1, 1))
            for i in range(self.hierarch_layers)
        ])

        if self.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.down_pool = nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)


        if self.use_mixing:
            # coarse to fine coarse to fine scale mixing多尺度的mixing层——尺度为从大到小了属于是
            self.scale_mixing_coarse2fine = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(0.3),
                    # nn.Dropout(self.dropout),
                    nn.Linear(self.d_modelSize_list[i], self.d_modelSize_list[i + 1]),
                    nn.ReLU(),
                )
                for i in range(self.hierarch_layers - 1)
            ])

            self.scale_mixing_fine2coarse = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(0.3),
                    # nn.Dropout(self.dropout),
                    nn.Linear(self.d_modelSize_list[i + 1], self.d_modelSize_list[i]),
                    nn.ReLU(),
                )
                for i in range(self.hierarch_layers - 1)
            ])

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
        x_seged_list = []


        # 多尺寸输入嵌入===========================================
        if not self.multi_scale_process_inputs:
            x = (x - seq_last).permute(0, 2, 1)  # b,c,s
            for i in range(self.hierarch_layers):
                x_seged_instance = self.valueEmbedding_Hierarchical[i](
                    x.reshape(-1, self.seg_num_x_list[i], self.seg_len_list[i]))
                #   [512,256,128]   [10, 20, 40]  [96, 48, 24] [480. 240. 120]
                x_seged_list.append(x_seged_instance)
        else:
            # print("Now use multi scale processing input==========")
            x_seged_list_preprocess = self.__multi_scale_process_inputs(x)
        # 这里的__multi_scale_process_inputs(x)就是用的Timemixer的多尺度化方式
            for i in range(self.hierarch_layers):
                x_seged_instance = self.valueEmbedding_Hierarchical[i](
                    x_seged_list_preprocess[i].reshape(-1, self.seg_num_x_list[i],
                                                       self.seg_len_list[i]))
                #   [512,256,128]   [10, 20, 40]  [96, 48, 24]
                x_seged_list.append(x_seged_instance)
        # else:



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
        for i in range(self.hierarch_layers):
            h_t_coarsest = torch.zeros(x_seged_list[i].shape[0], x_seged_list[i].shape[2]).to(x.device)
            hn_list_instance.append(h_t_coarsest)

        for i in range(self.seg_num_x_list[0]):
            x_t_coarsest = x_seged_list[0][:, i, :]
            hn_list_instance[0] = self.gru_cells[0](x_t_coarsest, hn_list_instance[0])

            for layer_now in range(self.hierarch_layers - 1):
                layer = layer_now + 1
                if not self.multi_scale_process_inputs:
                    for j in range(self.hierarch_scale ** (layer + 1)):
                        x_t = x_seged_list[layer][:, j, :]
                        hn_list_instance[layer] = self.gru_cells[layer](x_t, hn_list_instance[layer])
                else:
                        x_t = x_seged_list[layer][:, i, :]
                        hn_list_instance[layer] = self.gru_cells[layer](x_t, hn_list_instance[layer])

            # 这里就是多尺度之间的mixing过程===========================
            if self.use_mixing and self.hierarch_layers > 1:
                if self.mixing_route == "coarse2fine":
                    for o_now in range(self.hierarch_layers - 1):
                        o = o_now + 1
                        hn_list_instance[o] += self.scale_mixing_coarse2fine[o_now](hn_list_instance[o_now])
                        # hn_list_instance[o_now] /= 2
                #         思考一下这个地方需不需要整一个norm操作
                elif self.mixing_route == "fine2coarse":
                    for o_now in range(self.hierarch_layers - 2, -1, -1):
                        o = o_now + 1
                        hn_list_instance[o_now] += self.scale_mixing_fine2coarse[o_now](hn_list_instance[o])
                        # hn_list_instance[o_now] /= 2

        hn_list = hn_list_instance

        # 多尺度的可学习通道和位置输出初始化向量生成===================
        pos_emb_list = []
        if not self.use_rand_emb:
            for i in range(self.hierarch_layers):
                pos_emb = torch.cat([
                    self.pos_emb_List[i].unsqueeze(0).repeat(self.enc_in, 1, 1),
                    self.channel_emb_List[i].unsqueeze(1).repeat(1, self.seg_num_y_list[i], 1)
                ], dim=-1).view(-1, 1, self.d_modelSize_list[i]).repeat(self.batch_size, 1, 1)
                pos_emb_list.append(pos_emb)
        else:
            pos_emb_list = [param[:self.enc_in * batch_size] for param in self.rand_emb_List]

        # 多尺度RNN结构的输出了属于是===============================
        RNN_output_list = []
        for i in range(self.hierarch_layers):
            layer_output_list = []
            hn_now = hn_list[i].repeat(1, 1, self.seg_num_y_list[i]).view(1, -1, self.d_modelSize_list[i])[0, :, :]
            for step in range(self.seg_num_y_list[i]):
                step_length = pos_emb_list[i].shape[0] // self.seg_num_y_list[i]
                hn_stop_length = hn_now.shape[0] // self.seg_num_y_list[i]

                pos_emb_input = pos_emb_list[i][step * step_length: (step + 1) * step_length][:, 0, :]
                hn_input = hn_now[step * hn_stop_length: (step + 1) * hn_stop_length, :]
                hy = self.gru_cells[i](pos_emb_input, hn_input)
                layer_output_list.append(hy)
            out_put_this_layer = torch.stack(layer_output_list, dim=0)
            # 在第0个维度上concat输出
            RNN_output_list.append(out_put_this_layer.view(1, -1, self.d_modelSize_list[i]))

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
