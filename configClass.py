class config:
    def __init__(self):
        self.input_length = 720
        self.output_length = 96
        self.seg_length = 48
        # 分割窗口的大小
        self.train_ratio = 0.8
        self.dropout = 0.6
        self.dmodel = 512
        self.enc_in = 7
        self.num_layers = 1

        self.batch_size = 64
        self.num_epochs = 30
        self.learning_rate = 0.001
        self.filepath = "ETTh1.csv"
