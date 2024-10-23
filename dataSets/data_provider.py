from torch.utils.data import DataLoader
from dataSets.data_Loader import Dataset_ETT_hour, Dataset_ETT_minute


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    # 'custom': Dataset_Custom,
    # 'm4': Dataset_M4,
    # 'PSM': PSMSegLoader,
    # 'MSL': MSLSegLoader,
    # 'SMAP': SMAPSegLoader,
    # 'SMD': SMDSegLoader,
    # 'SWAT': SWATSegLoader,
    # 'UEA': UEAloader
}

def data_provider(data_set, embed, batch_size, freq, root_path, data_path, seq_len, label_len, pred_len, features, target, num_workers, flag):
    Data = data_dict[data_set]
    timeenc = 0 if embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False  # fix bug
        batch_size = batch_size
        freq = freq
    elif flag == 'pred':
        # shuffle_flag = False
        # drop_last = False
        # batch_size = 1
        # freq = freq
        # Data = Dataset_Pred
        pass
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = batch_size
        freq = freq

    data_set = Data(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last)
    return data_set, data_loader
