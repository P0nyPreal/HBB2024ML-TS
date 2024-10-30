import subprocess
class config:
    def __init__(self):
        self.input_length = 480
        self.output_length = 96
        # 实验设置output_length应该为enumerate(96, 192, 336, 720)
        self.seg_length = 96
        # 分割窗口的大小
        # self.train_ratio = 0.7
        self.dropout = 0.5
        self.dmodel = 256
        self.enc_in = 7
        self.num_layers = 1

        # self.use_residual = True
        self.use_residual = False
        self.use_gruCell = True
        self.use_decompose = True
        # gru_cell就是将GRU模型拆开一层一层运行的结果
        self.use_hirarchical = True
        if self.use_hirarchical:
            self.hierarch_layers = 3
            self.hierarch_scale = 2
            # self.decomp_method = "moving_avg"
            self.e_layers = 1
            self.seq_len = self.input_length
            # e_layer是每次进行PDM的层数
            self.moving_avg = 5
            self.down_sampling_window = 5
            self.channel_independence = True
            self.d_ff = 256
        #     dff是 bottel-neck的大小属于是
            self.down_sampling_layers = 2
            self.use_mixing = True

            self.down_sampling_method = 'avg'
            self.multi_scale_process_inputs = True
            self.use_rand_emb = True

            if self.use_mixing:
                self.mixing_route = "fine2coarse"

        self.batch_size = 512
        self.num_epochs = 50
        self.learning_rate = 0.01

        # self.model_name = "SegRNN"
        self.model_name = "HierarchRNN"
        self.data_set = "ETTh1"
        # 数据集应该为enumerate(ETTh1, ETTh2, ETTm1, ETTm2)
        self.data_mother_dir = "./dataSets/"
        self.filepath = self.data_mother_dir + self.data_set + ".csv"
        self.Global_exp_logger_path = "Global_Logger.txt"

        if self.model_name == "Timemixer":
            self.seq_len = self.input_length
            # self.
            self.decomp_method = "moving_avg"
            self.moving_avg = 5
            # moving_avg是平均池化的窗口大小
            self.top_k = 5
            self.d_ff = 256
            # bottle_neck的尺寸大小
            self.down_sampling_layers = 3
            self.down_sampling_method = "avg"
            self.down_sampling_window = 2
            self.channel_independence = 7
            self.use_norm = 0
            self.embed = 0
            self.c_out = 1
            self.freq = 1


if __name__ == '__main__':
    config = config()
    subprocess.run(['python', 'run.py'])
