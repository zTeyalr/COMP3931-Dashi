import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import random
from torch_lr_finder import LRFinder
from model import ECGNet
from data_loader import ECGDataset, FocalLoss
from data_loader import load_ecg_data, load_features_folder
import os

# Not useful after Preprocessing was redone
def convert_features_to_tensor(features):
    features_list = []
    for feature_dict in features:
        feature_dict.pop('qrs')
        array_features = []

        for key, value in feature_dict.items():
            if isinstance(value, np.ndarray):
                tensor_value = torch.tensor(value).float()
            
                if tensor_value.ndim == 0:
                    tensor_value = tensor_value.unsqueeze(0)
                array_features.append(tensor_value)

        array_features_padded = pad_sequence(array_features, batch_first=False, padding_value=0)
        
        features_tensor = torch.cat((array_features_padded,), dim=0)
        features_list.append(features_tensor)

    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0)
    return features_padded

# Not useful anymore as I switched it directly in the training phase
def collate_fn(batch, dataset):
    indices = batch

    data = []
    labels = []
    masks = []
    features_paths = []

    for i in indices:
        d, l, m, features_paths = dataset[i]
        data.append(d)
        labels.append(l)
        masks.append(m)

        features = []
     
        with np.load(features_paths, allow_pickle=True) as npz_data:
            features.append(dict(npz_data))
    features_tensor = convert_features_to_tensor(features)
    data = torch.stack(data)
    labels = torch.stack(labels)
    masks = torch.stack(masks)

    return data, labels, masks, features_tensor

def train_model(num_workers = 0):
    sample_data = 'preprocess_bit'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecg_data = []
    header = []
    counter  = 0
    age_array = []
    num_data = 0
    ecg_data_paths =  []

    amodel = ECGNet()
    DATA_PATH = 'preprocessed_data'
    dataset = ECGDataset(ecg_data_all=ecg_data, ecg_data_paths=ecg_data_paths ,header_all=header, transform=None)

    train_data_dir = os.path.join(DATA_PATH, "train")
    val_data_dir = os.path.join(DATA_PATH, "val")
    test_data_dir = os.path.join(DATA_PATH, "test")
    features_dir = os.path.join(DATA_PATH, "features")

    train_data, _, _, _ = load_ecg_data(train_data_dir)
    val_data, _, _, _ = load_ecg_data(val_data_dir)
    test_data, _, _, _ = load_ecg_data(test_data_dir)

    train_dataset = TensorDataset(torch.tensor(train_data))
    val_dataset = TensorDataset(torch.tensor(train_data))
    test_dataset = TensorDataset(torch.tensor(train_data))
    train_features = load_features_folder(features_dir, subset_name="train")
    val_features = load_features_folder(features_dir, subset_name="val")
    test_features = load_features_folder(features_dir, "test")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: collate_fn(batch, dataset), pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers, collate_fn=lambda batch: collate_fn(batch, dataset), pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers, collate_fn=lambda batch: collate_fn(batch, dataset), pin_memory=True, drop_last=True)
    alpha_values = [0.000676] * 30 + [0.00951] * 103
    alpha_tensor = torch.tensor(alpha_values).to(device)
    criterion = FocalLoss(alpha=alpha_tensor)
    optimizer = optim.Adam(amodel.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=7)

    # TRAINING
    NUM_EPOCHS = 15
    
    amodel.to(device)

    for epoch in range(NUM_EPOCHS):
        amodel.train()
        running_loss = 0.0
        for i, (data, labels, features) in enumerate(train_loader):

            data = data.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            features = features.to(device)

            masked_data = data * masks
            outputs = amodel(data, features.size(1))

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        
        # VALIDATION
        amodel.eval()
        predicted_labels = []
        true_labels = []
        val_loss = 0.0
        with torch.no_grad():
            print("LINE 66")
            for data, labels, masks, features in val_loader:
                data = data.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                features = features.to(device)
                masked_data = data * masks
                outputs = amodel(masked_data, features.size(1))

                _, predicted = torch.max(outputs.data, 1)
                
                predicted_labelszzz = torch.zeros(predicted.size(0), 133).to(device)
                predicted_labelszzz.scatter_(1, predicted.unsqueeze(1), 1)
                predicted_labels.extend(predicted_labelszzz.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                val_loss += criterion(outputs, labels).item()
        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(true_labels[0], predicted_labels[0], average = 'weighted')
        val_f1_macro = f1_score(true_labels[0], predicted_labels[0], average = 'macro')
        val_precision = precision_score(true_labels[0], predicted_labels[0], average='weighted')
        val_recall = recall_score(true_labels[0], predicted_labels[0], average='weighted')

        scheduler.step(val_f1)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}],\n Validation F1-Score: {val_f1:.4f}')
        print(f'Validation F1 MACRO: {val_f1_macro:.4f}')
        print(f'Validation Precision: {val_precision:.4f}')
        print(f'Validation Recall: {val_recall:.4f}')

    # TESTING
    amodel.eval()
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for data, labels, masks, features in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            features = features.to(device)
            masked_data = data * masks
            outputs = amodel(masked_data, features.size(1))

    _, predicted = torch.max(outputs.data, 1)
    predicted_labelszzz = torch.zeros(predicted.size(0), 133).to(device)
    predicted_labelszzz.scatter_(1, predicted.unsqueeze(1), 1)
    predicted_labels.extend(predicted_labelszzz.cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

    test_f1 = f1_score(true_labels[0], predicted_labels[0], average='weighted')
    test_f1_macro = f1_score(true_labels[0], predicted_labels[0], average = 'macro')
    test_precision = precision_score(true_labels[0], predicted_labels[0], average='weighted')
    test_recall = recall_score(true_labels[0], predicted_labels[0], average='weighted')
    print(f'TEST F1-Score: {test_f1:.4f}')
    print(f'TEST F1-Score Macro: {test_f1_macro:.4f}')
    print(f'TEST Precision: {test_precision:.4f}')
    print(f'TEST: {test_recall:.4f}')

    torch.save(amodel.state_dict(), 'ecg_12lead_model.pth')
