import torch
import torch.nn as nn

class DetectSpamV0(nn.Module):
    def __init__(self, vocab_size):
        super(DetectSpamV0, self).__init__()
        self.layer1 = nn.Linear(100, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Simple feed-forward for text features
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.layer4(x))
        return x