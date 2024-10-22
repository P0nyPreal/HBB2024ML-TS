import subprocess


class config:
    def __init__(self):
        self.input_length = 96
        self.output_length = 720
        # 实验设置output_length应该为enumerate(96, 192, 336, 720)
        self.seg_length = 48
        # 分割窗口的大小
        self.train_ratio = 0.7
        self.dropout = 0.5
        self.dmodel = 512
        self.enc_in = 7
        self.num_layers = 1
        # self.use_residual = True
        self.use_residual = False

        self.batch_size = 256
        self.num_epochs = 50
        self.learning_rate = 0.001

        self.model_name = "SegRNN"
        self.data_set = "ETTh1"
        # 数据集应该为enumerate(ETTh1, ETTh2, ETTm1, ETTm2)
        self.data_mother_dir = "./dataSets/"
        self.filepath = self.data_mother_dir + self.data_set + ".csv"
        self.Global_exp_logger_path = "Global_Logger.txt"



if __name__ == '__main__':
    config = config()
    subprocess.run(['python', 'run.py'])
