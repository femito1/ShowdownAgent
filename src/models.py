import torch.nn as nn

# Neural network models for PPO
class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.LeakyReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.policy_head(features)

class ValueNetwork(nn.Module):
    def __init__(self, input_shape):
        super(ValueNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x) 