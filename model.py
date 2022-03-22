import torch
from torch import nn
from torch.optim import Adam
import numpy as np


class Net(nn.Module):
    
    def __init__(self, hid_dim, mlp_dim, feature_dim, lr=1e-3, dropout=0.1):
        super().__init__()
        
        # exp by ME
        self.rnn = nn.LSTM(input_size=1+feature_dim+2,
                            hidden_size=hid_dim,
                            batch_first=True,
                            bidirectional=False)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=hid_dim, out_features=mlp_dim),
            nn.Dropout(p=dropout)   
        )
        # self.mark_linear = nn.Linear(in_features=config.mlp_dim, out_features=config.event_class)
        self.time_linear = nn.Linear(in_features=mlp_dim, out_features=1)
        self.optimizer = Adam(self.parameters(), lr=lr)
        
        self.intensity_w_self = nn.Parameter(torch.tensor(0.1, dtype=torch.float))
        self.intensity_b_self = nn.Parameter(torch.tensor(0.1, dtype=torch.float))
        self.intensity_w_nonself = nn.Parameter(torch.tensor(0.1, dtype=torch.float))
        self.intensity_b_nonself = nn.Parameter(torch.tensor(0.1, dtype=torch.float))

        self.w_b_dict = {
            'self': (self.intensity_w_self, self.intensity_b_self),
            'nonself': (self.intensity_w_nonself, self.intensity_b_nonself)
        }
        
        # exp by xh_z
        # n_factor = 20
        # self.time_layer = nn.Sequential(
        #     nn.Linear(1, n_factor),
        #     nn.ReLU(),
        #     nn.Linear(n_factor, 1),
        # )

    def log_likelihood(self, past_inf, target_inter_t, count, event_type='self'):
        '''
        the log likelihood of observing `target_inter_t` 
          (the batch of target interarrival time sequences)
        '''
        # current influence term in Equation (11) of the paper
        w, b = self.w_b_dict[event_type]
        cur_inf =  w * target_inter_t 

        # event-level likelihood, corresponds to term $\log f^{*}(t)$ in paper equation (12)
        # suffix `ll` means log-likelihood
        event_ll = past_inf + cur_inf + b
        non_event_ll = torch.exp(past_inf + b) / w - \
                         torch.exp(past_inf + cur_inf + b) / w
        # sequence-level likelihood
        # account for simultaneous occurrence
        seq_ll = count*event_ll + non_event_ll
        return seq_ll

    def lambda_integral(self, past_inf, inter_t, event_type='self'):
        '''
        past_inf: [batch_size, 1, 1]
        inter_t: [batch_size, seq_len]
        '''
        w, b = self.w_b_dict[event_type]
        # `inter_t`: [batch_size, seq_len] => [batch_size, seq_len, 1]
        inter_t = inter_t.unsqueeze(2)
        # return a tensor of shape [batch_size, seq_len, 1]
        return  torch.exp(past_inf + w * inter_t  + b) / w  - torch.exp(past_inf + b) / w

    def forward(self, b_inter_t, b_marks, b_mask, b_features, *args):
        """
        batch: a tuple, (b_inter_t, b_marks, b_mask, b_features)
          b_inter_t: [batch_size, seq_len], interarrival times
          b_marks: [batch_size, seq_len, 2], event marks
          b_mask: [batch_size, seq_len], whether an event is not padding
          b_features: [batch_size, seq_len, feature_dim], firm features
        """
        # [batch_size, seq_len-1] => [batch_size, seq_len-1, 1]
        x = b_inter_t[:, :-1].unsqueeze(-1)
        x = torch.cat((x, b_features[:, :-1], b_marks[:, :-1].float()), dim=-1)
        # exclude the last event's information, 
        # we do not have true label of interarrival time for the last event
        # hidden_state: [batch_size, seq_len-1, hid_dim]
        hidden_state, _ = self.rnn(x)
        mlp_output = self.mlp(hidden_state)
        past_inf = self.time_linear(mlp_output)
        # x[i, 0] =>  hidden_state[i, 0] pred=> b_inter_t[0, 1]
        # x[i, -1] =>  hidden_state[i, -1] pred=> b_inter_t[0, -1]
        target_inter_t = b_inter_t[:, 1:].unsqueeze(-1)
        
        # event_logits = self.event_linear(mlp_output)
        
        c1 = b_marks[:, 1:, [0]] # [batch_size, seq_len-1, 1]
        c2 = b_marks[:, 1:, [1]] # [batch_size, seq_len-1, 1]
        
        # loss_self, loss_nonself: of shape [batch_size, seq_len-1, 1]
        loss_self = -self.log_likelihood(past_inf, target_inter_t, c1, 'self')
        loss_nonself = -self.log_likelihood(past_inf, target_inter_t, c2, 'nonself')
        
        # [batch_size, seq_len-1] => [batch_size, seq_len-1, 1]
        # we do not have prediction for the first event
        b_mask = b_mask[:, 1:].unsqueeze(-1)
        
        loss_self = loss_self * b_mask
        loss_nonself = loss_nonself * b_mask
        
        # average across batch instances and time steps
        return loss_self.mean(), loss_nonself.mean()

    def train_batch(self, batch):
        """
        batch: a tuple, (b_inter_t, b_marks, b_mask, b_features)
          b_inter_t: [batch_size, seq_len], interarrival times
          b_marks: [batch_size, seq_len, 2], event marks
          b_mask: [batch_size, seq_len], whether an event is not padding
          b_features: [batch_size, feature_dim], firm features
        """
        loss_self, loss_nonself = self.forward(*batch)
        loss = loss_self + loss_nonself
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_self.item(), loss_nonself.item()
    
    def pred(self, batch_hist, batch_pred_inter_t):
        """
        following the way of predicting future citation count as used in paper
            Liu, X. et al. (2017) ‘On Predictive Patent Valuation: Forecasting Patent Citations and Their Types’, 
            Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence. 
            Available at: http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14385.
        
        does not consider the triggering effect of predicted citations
        
        batch_pred_inter_t: of shape [batch_size, seq_len],
        a numpy array of interarrival times relative to the last historical event
        """
        b_inter_t = batch_hist[0]
        b_marks = batch_hist[1]
        b_mask = batch_hist[2]
        b_features = batch_hist[3]
        batch_size = b_inter_t.size(0)
        batch_pred_inter_t = torch.tensor(batch_pred_inter_t, dtype=torch.float32)
        with torch.no_grad():
            x = b_inter_t.unsqueeze(-1)
            x = torch.cat((x, b_features, b_marks.float()), dim=-1)
            # hidden_state: [batch_size, seq_len-1, hid_dim]
            hidden_state, _ = self.rnn(x)
            seq_len = b_mask.long().sum(dim=-1) # [batch_size,]
            rows = torch.arange(batch_size)
            # last_hidden_state: [batch_size, hid_dim] => [batch_size, 1, hid_dim]
            last_hidden_state = hidden_state[rows, seq_len-1].unsqueeze(1)
            
            # mlp_output: [batch_size, 1, 1]
            mlp_output = self.mlp(last_hidden_state)
            # past_inf: [batch_size, 1, 1]
            past_inf = self.time_linear(mlp_output)
            # E_self, E_nonself: [batch_size, seq_len, 1]
            E_self = self.lambda_integral(past_inf, batch_pred_inter_t, event_type='self')
            E_nonself = self.lambda_integral(past_inf, batch_pred_inter_t, event_type='nonself')
            E = torch.cat((E_self, E_nonself), dim=-1)
        return E.cpu().numpy()











