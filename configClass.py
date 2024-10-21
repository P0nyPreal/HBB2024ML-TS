import subprocess
class config:
    def __init__(self):
        self.input_length = 720
        self.output_length = 192
        self.seg_length = 48
        # 分割窗口的大小
        self.train_ratio = 0.8
        self.dropout = 0.7
        self.dmodel = 512
        self.enc_in = 7
        self.num_layers = 1

        self.batch_size = 128
        self.num_epochs = 30
        self.learning_rate = 0.001
        self.filepath = "ETTh1.csv"

if __name__ == '__main__':
    config = config()
    subprocess.run(['python', 'run.py'])
