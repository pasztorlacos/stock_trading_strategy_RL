import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.input_layer = nn.Linear(8, 128)
        self.hidden_1 = nn.Linear(128, 128)
        self.hidden_2 = nn.Linear(32,31)
        # self.hidden_state = torch.tensor(torch.zeros(2,1,32)).cuda()
        self.hidden_state = torch.tensor(torch.zeros(2,1,32))
        self.rnn = nn.GRU(128, 32, 2)
        self.action_head = nn.Linear(31, 5)
        self.value_head = nn.Linear(31, 1)
        self.saved_actions = []
        self.rewards = []

    def reset_hidden(self):
        # self.hidden_state = torch.tensor(torch.zeros(2,1,32)).cuda()
        self.hidden_state = torch.tensor(torch.zeros(2,1,32))
        
    def forward(self, x):
        # x = torch.tensor(x).cuda()
        x = torch.tensor(x)
        x = torch.sigmoid(self.input_layer(x))
        x = torch.tanh(self.hidden_1(x))
        x, self.hidden_state = self.rnn(x.view(1,-1,128), self.hidden_state.data)
        x = F.relu(self.hidden_2(x.squeeze()))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values
    
    def act(self, state):
        probs, state_value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        # if action == 1 and env.state[0] < 1: action = torch.LongTensor([2]).squeeze().cuda()
        # if action == 4 and env.state[1] < 1: action = torch.LongTensor([2]).squeeze().cuda()
        if action == 1 and env.state[0] < 1: action = torch.LongTensor([2]).squeeze()
        if action == 4 and env.state[1] < 1: action = torch.LongTensor([2]).squeeze()
        self.saved_actions.append((m.log_prob(action), state_value))
        return action.item()