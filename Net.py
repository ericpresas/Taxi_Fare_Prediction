import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
	    x = F.relu(self.fc1(x))
	    x = F.relu(self.fc2(x))
	    x = self.fc3(x)
	    return F.relu(x)