import subprocess


class config:
    def __init__(self):
        self.input_length = 720
        self.output_length = 720
        self.seg_length = 48
        # 分割窗口的大小
        self.train_ratio = 0.7
        self.dropout = 0.5
        self.dmodel = 512
        self.enc_in = 7
        self.num_layers = 1

        self.batch_size = 256
        self.num_epochs = 40
        self.learning_rate = 0.001

        self.filepath = "./dataSets/ETTh1.csv"
        self.exp_logger_path = "Experimental_Logger.txt"

if __name__ == '__main__':
    config = config()
    subprocess.run(['python', 'run.py'])
