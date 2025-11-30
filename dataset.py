import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Subset, TensorDataset

from fedlab.utils.dataset.partition import VisionPartitioner, BasicPartitioner



def get_all_labels(dataset):
    all_labels = []
    # Iterate over the dataset to collect labels
    for _, labels in dataset:
        all_labels.append(labels)

    # Convert the list of labels to a single tensor
    all_labels_tensor = torch.stack(all_labels)

    return all_labels_tensor

def get_unsw_nb15(data_path: str = "./data"):
    
    train = pd.read_csv(data_path + '/unsw_train.csv')
    test = pd.read_csv(data_path + '/unsw_test.csv')

    train_target = torch.tensor(train['targets'].values.astype(np.float32))
    train = torch.tensor(train.drop('targets', axis = 1).values.astype(np.float32))
    train_tensor = TensorDataset(train, train_target)

    test_target = torch.tensor(test['targets'].values.astype(np.float32))
    test = torch.tensor(test.drop('targets', axis = 1).values.astype(np.float32))
    test_tensor = TensorDataset(test, test_target)

    return train_tensor , test_tensor

def get_cicids(data_path: str = "./data"):
    
    train = pd.read_csv(data_path + '/cicids_train_binary.csv')
    test = pd.read_csv(data_path + '/cicids_test_binary.csv')

    train_target = torch.tensor(train['Label'].values.astype(np.float32))
    train = torch.tensor(train.drop('Label', axis = 1).values.astype(np.float32))
    train_tensor = TensorDataset(train, train_target)

    test_target = torch.tensor(test['Label'].values.astype(np.float32))
    test = torch.tensor(test.drop('Label', axis = 1).values.astype(np.float32))
    test_tensor = TensorDataset(test, test_target)

    return train_tensor , test_tensor


def get_ciciomt(data_path: str = "./data"):
    
    train = pd.read_csv(data_path + '/ciciomt_train_data_binary.csv', engine="pyarrow")
    test = pd.read_csv(data_path + '/ciciomt_test_data_binary.csv', engine="pyarrow")

    train_target = torch.tensor(train['label'].values.astype(np.float32))
    train = torch.tensor(train.drop('label', axis = 1).values.astype(np.float32))
    train_tensor = TensorDataset(train, train_target)

    test_target = torch.tensor(test['label'].values.astype(np.float32))
    test = torch.tensor(test.drop('label', axis = 1).values.astype(np.float32))
    test_tensor = TensorDataset(test, test_target)

    return train_tensor , test_tensor

def get_ciciomt_multi(data_path: str = "./data"):
    
    train = pd.read_csv(data_path + '/ciciomt_train_data_multi.csv', engine="pyarrow")
    test = pd.read_csv(data_path + '/ciciomt_test_data_multi.csv', engine="pyarrow")

    train_target = torch.tensor(train['label'].values.astype(np.float32))
    train = torch.tensor(train.drop('label', axis = 1).values.astype(np.float32))
    train_tensor = TensorDataset(train, train_target)

    test_target = torch.tensor(test['label'].values.astype(np.float32))
    test = torch.tensor(test.drop('label', axis = 1).values.astype(np.float32))
    test_tensor = TensorDataset(test, test_target)

    return train_tensor , test_tensor

def get_IoTIDS(data_path: str = "./data"):

    train = pd.read_csv(data_path + '/iotids_train_data_binary.csv')
    test = pd.read_csv(data_path + '/iotids_train_data_binary.csv')

    train_target = torch.tensor(train['Label'].values.astype(np.float32))
    train = torch.tensor(train.drop('Label', axis = 1).values.astype(np.float32))
    train_tensor = TensorDataset(train, train_target)

    test_target = torch.tensor(test['Label'].values.astype(np.float32))
    test = torch.tensor(test.drop('Label', axis = 1).values.astype(np.float32))
    test_tensor = TensorDataset(test, test_target)

    return train_tensor , test_tensor

def get_IoTIDS_multi(data_path: str = "./data"):

    train = pd.read_csv(data_path + '/iotids_train_data_multi.csv')
    test = pd.read_csv(data_path + '/iotids_test_data_multi.csv')

    train_target = torch.LongTensor(train['Label_Num'].values)
    train = torch.FloatTensor(train.drop('Label_Num', axis = 1).values)
    train_tensor = TensorDataset(train, train_target)

    test_target = torch.LongTensor(test['Label_Num'].values)
    test = torch.FloatTensor(test.drop('Label_Num', axis = 1).values)
    test_tensor = TensorDataset(test, test_target)

    return train_tensor , test_tensor



def partition_and_prepare_dataset(train_set, test_set, num_clients, 
                                   batch_size, csv, partition='iid', dir_alpha=None,
                                   major_classes_num=0, val_ratio=0.1):
    
    if csv:
        all_labels = get_all_labels(train_set)
        num_classes = len(set(all_labels))
        
        # Use VisionPartitioner for multi-class, BasicPartitioner only for binary
        if num_classes == 2:
            partitioner = BasicPartitioner(all_labels, num_clients, 
                                          partition=partition, 
                                          dir_alpha=dir_alpha,
                                          major_classes_num=major_classes_num, 
                                          seed=2023)
        else:
            partitioner = VisionPartitioner(all_labels, num_clients, 
                                           partition=partition, 
                                           dir_alpha=dir_alpha,
                                           major_classes_num=major_classes_num, 
                                           seed=2023)
    else:
        # Use VisionPartitioner for image datasets (works for any number of classes)
        partitioner = VisionPartitioner(train_set.targets, num_clients, 
                                       partition=partition, 
                                       dir_alpha=dir_alpha,
                                       major_classes_num=major_classes_num, 
                                       seed=2023)
    
    trainloaders = []
    valloaders = []
    
    # Determine number of classes for flip mapping
    if csv:
        all_labels = get_all_labels(train_set)
        num_classes = len(set(all_labels))
    else:
        num_classes = len(set(train_set.targets))
    

    
    for client in range(num_clients):
        idx = partitioner.client_dict[client]
        
        trainset_ = Subset(train_set, idx)
        num_total = len(idx)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        
        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )
        
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False)
        )
    
    testloader = DataLoader(test_set, batch_size=batch_size)
    
    return trainloaders, valloaders, testloader

    