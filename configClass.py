import subprocess
class config:
    def __init__(self):
        self.input_length = 96
        self.output_length = 96
        # 实验设置output_length应该为enumerate(96, 192, 336, 720)
        self.seg_length = 48
        # 分割窗口的大小
        # self.train_ratio = 0.7
        self.dropout = 0.2
        self.dmodel = 512
        self.enc_in = 7
        self.num_layers = 1
        # self.use_residual = True
        self.use_residual = False

        self.batch_size = 256
        self.num_epochs = 30
        self.learning_rate = 0.0001


        # self.model_name = "SegRNN"
        self.model_name = "Timemixer"
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
            self.e_layers = 2
            self.use_norm = 0
            self.embed = 0
            self.c_out = 1
            self.freq = 1


if __name__ == '__main__':
    config = config()
    subprocess.run(['python', 'run.py'])
