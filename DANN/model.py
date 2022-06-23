import torch
from torch import nn
from functions import ReverseLayerF

import pytorch_lightning as pl


lr = 1e-3


class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super(self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        features = self.base(input_data)
        features = features.view(-1, 50 * 4 * 4)

        return features


class Class_Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100), 
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout2d(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.LogSoftmax()
        )

    def forward(self, features):
        class_output = self.base(features)
        
        return class_output





class Domain_Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, reverse_feature):
        domain_output = self.base(reverse_feature)
        
        return domain_output


class DANN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.class_classifier = Class_Classifier()
        self.domain_classifier = Domain_Classifier()

    def forward(self, input_data, alpha):
        features = self.feature_extractor(input_data)
        reverse_features = ReverseLayerF.apply(features, alpha)
        class_output = self.class_classifier(features)
        domain_output = self.domain_classifier(reverse_features)

        return class_output, domain_output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    def training_step(self, batch , batch_idx):
        