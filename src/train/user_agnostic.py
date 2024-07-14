import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from src.train.trainer import Trainer
from src.dataloader import Dataloader
from src.utils import update_dict
from src.data_preprocessor.user_agnostic import UserAgnosticPreprocessor
from torch.optim import Adam
from src.model.custom_typenet import CustomTypeNet 
import argparse

def train(args):
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    seq_len = args.seq_len
    dataset_root = args.dataset_root
    gpu_device = args.gpu_device
    dropout = args.dropout

    files = [
        'fixed_data.txt',
        'free_data.txt',
        'Raw_Temp_Gay_Marriage_Fixed.json',
        'Raw_Temp_Gay_Marriage_Free.json',
        'Raw_Temp_Gun_Control_Fixed.json',
        'Raw_Temp_Gun_Control_Free.json',
        'Raw_Temp_Restaurant_Review_Fixed.json',
        'Raw_Temp_Restaurant_Review_Free.json'
    ]

    dataloader = Dataloader(
      datasets=files,
      ROOT=dataset_root
    )

    fixed_data, free_data, gay_marriage_fixed, gay_marriage_free, gun_control_fixed, gun_control_free, rest_fixed, rest_free = dataloader.load_data()

    # with open(os.path.join(ROOT,'Buffalo_Fixed.json')) as f :
    #   buffalo_fixed = json.load(f)
    
    # with open(os.path.join(ROOT,'Buffalo_Free.json')) as f :
    #   buffalo_free = json.load(f)

    preprocessor = UserAgnosticPreprocessor(seq_len, batch_size)

    gm_fixed = preprocessor.process_data(gay_marriage_fixed)
    gm_free = preprocessor.process_data(gay_marriage_free)
    gnc_fixed = preprocessor.process_data(gun_control_fixed)
    gnc_free = preprocessor.process_data(gun_control_free)
    rs_free = preprocessor.process_data(rest_fixed)
    rs_fixed = preprocessor.process_data(rest_free)
    free_data = preprocessor.process_data(free_data,1)
    fixed_data = preprocessor.process_data(fixed_data,1)

    ## Train : Combined Dataset
    fixed_data_test = {}
    free_data_test = {}

    fixed_data_train = {}
    free_data_train = {}

    # fixed_data_train.update(buffalo_fixed)
    fixed_data_train = update_dict(fixed_data_train,fixed_data)
    fixed_data_train.update(gm_fixed)
    fixed_data_train = update_dict(fixed_data_train,rs_fixed)
    fixed_data_train = update_dict(fixed_data_train,gnc_fixed)

    # free_data_train.update(buffalo_free)
    free_data_train = update_dict(free_data_train,free_data)
    free_data_train.update(gm_free)
    free_data_train = update_dict(free_data_train,rs_free)
    free_data_train = update_dict(free_data_train,gnc_free)

    train_loader, val_loader, test_loader = preprocessor.get_train_test_sets(fixed_data_train, free_data_train, fixed_data_test, free_data_test)


    if gpu_device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('device', device)

    model = CustomTypeNet(seq_len,3,128,128,128,dropout)
    model.to(device)
    loss = nn.BCELoss()

    optimizer = Adam(model.parameters(),lr=learning_rate,weight_decay=0.001)
    ModelTrainer = Trainer(model, optimizer, device, loss, None)
    ModelTrainer.train(train_loader, num_epochs, val_loader, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the model.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length for typenet.')
    parser.add_argument('--gpu_device', type=int, default=1, help='GPU device ID.')
    parser.add_argument('--dropout', type=float, default=0.001, help='Dropout for typenet.')
    parser.add_argument('--dataset_root', type=str, default='', help='Dataset filepath.')

    args = parser.parse_args()
    train(args)
