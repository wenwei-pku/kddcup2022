"""
Description: Prepare the experimental settings
"""
import torch


def prep_env():
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        ## General setting
        "path_to_test_x": "./datasets/test_x/0001in.csv",
        "path_to_test_y": "./datasets/test_y/0001out.csv",
        "data_path": "datasets",
        "filename": "wtbdata_245days.csv",
        "target": "Patv",
        "input_len": 144,
        "output_len": 288,
        "label_len": 0,
        "start_col": 3,  # required
        "day_len": 144,
        "train_days": 200,
        "val_days": 30,
        "test_days": 15,
        # "total_days": 245,
        "capacity": 134,
        "features_to_use": ('Wspd', 'Etmp'),
        "cuda_name": 0,

        ## General model setting
        "checkpoints": "checkpoints",  # required
        "model_type": 'lgb',

        ## lgb specific settings
        "n_group": 1,
        "in_var": 10,
        "out_var": 1,

        ## General Training Parameters
        "num_workers": 5,
        "train_epochs": 10,
        "batch_size": 32,
        "patience": 3,
        "learning_rate": 1e-3,
        "lradj": "type1",
        "gpu": 0,
        "dropout": 0.2,
        "seed": 3407,

        ## Prediction settings
        "pred_file": "predict.py",  # required
        "framework": "base",  # required
        "result_path": "results",

    }
    ###

    # Prepare the GPUs
    if torch.cuda.is_available():
        settings["use_gpu"] = True

    else:
        settings["use_gpu"] = False

    return settings
