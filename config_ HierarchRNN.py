import subprocess
class config:
    def __init__(self):
        self.input_length = 720
        self.output_length = 96
        # 实验设置output_length应该为enumerate(96, 192, 336, 720)
        self.seg_length = 96
        # 分割窗口的大小
        # self.train_ratio = 0.7
        self.dropout = 0.2
        self.dmodel = 512
        self.enc_in = 7
        self.num_layers = 1

        self.hierarch_layers = 3
        # that means seg size form 96-48-24

        # self.use_residual = True
        self.use_residual = False
        self.use_gruCell = True
        self.use_decompose = True
        # gru_cell就是将GRU模型拆开一层一层运行的结果
        self.use_hirarchical = True
        if self.use_hirarchical:
            self.decomp_method = "moving_avg"
            self.e_layers = 1
            self.seq_len = self.input_length
            # e_layer是每次进行PDM的层数
            self.moving_avg = 5
            self.down_sampling_window = 5
            self.channel_independence = True
            self.d_ff = 256
        #     dff是 bottel-neck的大小属于是
            self.down_sampling_layers = 2

        self.batch_size = 256
        self.num_epochs = 50
        self.learning_rate = 0.001

        self.model_name = "HierarchRNN"
        self.data_set = "ETTh1"
        # 数据集应该为enumerate(ETTh1, ETTh2, ETTm1, ETTm2)
        self.data_mother_dir = "./dataSets/"
        self.filepath = self.data_mother_dir + self.data_set + ".csv"
        self.Global_exp_logger_path = "Global_Logger.txt"


if __name__ == '__main__':
    config = config()
    subprocess.run(['python', 'run.py'])
