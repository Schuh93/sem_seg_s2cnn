import gzip, pickle, torch
import numpy as np


def load_train_data(path):
    
    with gzip.open(path, 'rb') as f:
        dataset = pickle.load(f)
        
    train_data = torch.from_numpy(dataset["images"][:, None, :, :].astype(np.float32))
    train_labels = torch.from_numpy(dataset["labels"].astype(np.int64))

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    
    return train_dataset
    
def load_test_data(path):
    
    with gzip.open(path, 'rb') as f:
        dataset = pickle.load(f)
        
    test_data = torch.from_numpy(dataset["test"]["images"][:, None, :, :].astype(np.float32))
    test_labels = torch.from_numpy(dataset["test"]["labels"].astype(np.int64))

    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    
    return test_dataset