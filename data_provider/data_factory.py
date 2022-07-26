from torch.utils.data import DataLoader
from pathlib import Path

def data_provider(config, flag, turb_id):

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1

    else:
        shuffle_flag = True
        drop_last = True
        batch_size = config['batch_size']


    data_set = DatasetNN(
        data_path=Path(config['data_path'], config['filename']),
        flag=flag,
        target=config['target'],
        day_len=config['day_len'],
        train_days=config['train_days'],
        val_days=config['val_days'],
        test_days=config['test_days'],
        features_to_use=config['features_to_use'],
        turb_id=turb_id,
        input_len=config['input_len'],
        label_len=config['label_len'],
        pred_len=config['output_len'],
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=config['num_workers'],
        drop_last=drop_last)
    return data_set, data_loader
