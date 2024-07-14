import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from src.train.trainer import Trainer
from src.dataloader import Dataloader
from src.data_preprocessor.keyboard_agnostic import KeyboardAgnosticPreprocessor
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
        'Buffalo_Fixed.json',
        'Buffalo_Free.json'
    ]

    dataloader = Dataloader(
      datasets=files,
      ROOT=dataset_root
    )

    buffalo_fixed, buffalo_free = dataloader.load_data()
    
    preprocessor = KeyboardAgnosticPreprocessor(seq_len, batch_size)

    free_data = preprocessor.process_data_buffalo(buffalo_free)
    fixed_data = preprocessor.process_data_buffalo(buffalo_fixed)

    key_0_free = {}
    key_1_free = {}
    key_2_free = {}
    key_3_free = {}

    key_0_fixed = {}
    key_1_fixed = {}
    key_2_fixed = {}
    key_3_fixed = {}

    for key in free_data.keys() :
        new_key = key[:4]
        
        if key[4] == '0' :
            key_0_free[new_key] = free_data[key]
        
        elif key[4] == '1' : 
            key_1_free[new_key] = free_data[key]

        elif key[4] == '2' :
            key_2_free[new_key] = free_data[key]
            
        elif key[4] == '3' :
            key_3_free[new_key] = free_data[key]
            
    for key in fixed_data.keys() :
        new_key = key[:4]
        
        if key[4] == '0' :
            key_0_fixed[new_key] = fixed_data[key]
        
        elif key[4] == '1' : 
            key_1_fixed[new_key] = fixed_data[key]

        elif key[4] == '2' :
            key_2_fixed[new_key] = fixed_data[key]
            
        elif key[4] == '3' :
            key_3_fixed[new_key] = fixed_data[key]
            


    # Train-Test Dataset Formation & Splitting

    '''
        We can create any combination of datasets for training and 
        testing in this pipeline to create the training and testing sets.
        
        key_0_free : Keyboard - 0 Free Data
        key_1_free : Keyboard - 1 Free Data
        key_2_free : Keyboard - 2 Free Data
        key_3_free : Keyboard - 3 Free Data
        
        key_0_fixed : Keyboard - 0 Fixed Data
        key_1_fixed : Keyboard - 1 Fixed Data
        key_2_fixed : Keyboard - 2 Fixed Data
        key_3_fixed : Keyboard - 3 Fixed Data
    '''

    ## Train Data : Keyboard - 1 + Keyboard - 2 + Keyboard - 3, Test Data : Keyboard - 0

    fixed_data_test = {}
    free_data_test = {}

    fixed_data_test.update(key_0_fixed)
    free_data_test.update(key_0_free)

    fixed_data_train = {}
    free_data_train = {}

    fixed_data_train.update(key_3_fixed)
    fixed_data_train.update(key_2_fixed)
    fixed_data_train.update(key_1_fixed)
    free_data_train.update(key_3_free)
    free_data_train.update(key_2_free)
    free_data_train.update(key_1_free)

    train_loader, val_loader, test_loader = preprocessor.get_train_test_sets(fixed_data_train, free_data_train, fixed_data_test, free_data_test)

    if gpu_device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

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
