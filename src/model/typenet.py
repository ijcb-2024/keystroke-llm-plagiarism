import torch 
import torch.nn as nn 

class TypeNet(nn.Module):
    def __init__(self, sequence_length, in_dim, hidden_dim_1, hidden_dim_2, output_dim, dropout):
        super(TypeNet, self).__init__()
        self.dropout = dropout
        self.lstm1 = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim_1, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=hidden_dim_1, hidden_size=hidden_dim_2, batch_first=True, dropout=0.2)
        self.bn1 = nn.BatchNorm1d(num_features=in_dim)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim_1)
        self.bn3 = nn.BatchNorm1d(num_features=hidden_dim_2*sequence_length)
        self.bn4 = nn.BatchNorm1d(num_features=output_dim)
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()
        self.act3 = nn.Tanh()
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(sequence_length*hidden_dim_2, output_dim)
        self.softmax = nn.Softmax(dim=1)

        # Weight initialization
        for name, param in self.lstm1.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.kaiming_uniform_(param.data)
        for name, param in self.lstm2.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.kaiming_uniform_(param.data)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x, length):
        x = torch.movedim(x, 2, 1)
        out = self.bn1(x)
        out = torch.movedim(out, 2, 1)

        out = nn.utils.rnn.pack_padded_sequence(out, length, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm1(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = self.act1(out)
        out = self.dropout_1(out)

        out = torch.movedim(out, 2, 1)
        out = self.bn2(out)
        out = torch.movedim(out, 2, 1)

        out_p = nn.utils.rnn.pack_padded_sequence(out, length, batch_first=True, enforce_sorted=False)
        out_p, _ = self.lstm2(out_p)
        out_p, _ = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True)

        out_p = self.act2(out_p)
        out_p = torch.reshape(out_p, (out.shape[0], out.shape[1]*out.shape[2]))
        out = self.bn3(out_p)
        out = self.fc1(out)

        return out