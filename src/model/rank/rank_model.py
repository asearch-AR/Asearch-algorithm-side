import torch
from torch.functional import F

class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.input = torch.nn.Linear(n_input, n_hidden)
        self.hidden = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x_input):
        x_hidden = F.relu(self.input(x_input))
        x_hidden = F.relu(self.hidden(x_hidden))
        x_hidden = F.relu(self.hidden(x_hidden))
        x_predict = self.predict(x_hidden)
        return F.softmax(x_predict,dim=1)
