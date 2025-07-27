from torch import nn


class MLP(nn.Module):
    def __init__(self, input_features, dropout=0.2):
        super(MLP, self).__init__()
        self.input_features = input_features
        self.layers = nn.Sequential(
            nn.BatchNorm1d(self.input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(self.input_features, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)
