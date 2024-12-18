import torch
import torch.nn as nn

class ECGNet(nn.Module):
    def __init__(self, leads=12, classes=133, features_dim=None):
        super(ECGNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=leads, out_channels=32, kernel_size=5, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, padding='same'),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)


        )
        
        with torch.no_grad():
            dummy_input = torch.randn(1, leads, 8192)
            conv_out_size = self.conv_layers(dummy_input).view(1, -1).size(1)
        if features_dim is not None:
            self.feature_layer = nn.Linear(in_features=features_dim, out_features=128)
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=conv_out_size + 128, out_features=128),
                nn.ReLU(),
                nn.Dropout(0.6),
                nn.Linear(in_features=128, out_features=classes)
            )
        else:
            self.feature_layer = None
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=conv_out_size, out_features=128),
                nn.ReLU(),
                nn.Dropout(0.6),
                nn.Linear(in_features=128, out_features=classes)
            )

    
    def forward(self, data, features=None):
        data = self.conv_layers(data)
        if features is not None:
            if self.feature_layer is not None:
                features = self.feature_layer(features)
                data = data.view(data.size(0), -1)
                data = torch.cat((data, features), dim=1)
            
        data = self.fc_layers(data)
        return data

class ECGNet1Lead(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ECGNet1Lead).__init__(*args, **kwargs)

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        