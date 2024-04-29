import torch
from torch import nn

class Muti_slice_fusion_layer(nn.Module):
    def __init__(self, hidden_size:int, num_heads:int=16) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
    
    def forward(self, x:torch.Tensor):
        out_put, _ = self.attention(x, x, x)
        out_put = out_put + x

        out_put = out_put.mean(dim=1)
        return out_put

class Multi_slice_Model(nn.Module):
    def __init__(self, encoder:nn.Module, hidden_size:int, n_classes:int=2, fusion_mode:str='mean') -> None:
        super().__init__()
        self.encoder = encoder
        if fusion_mode == 'mean':
            self.fusion_layer = None
        if fusion_mode == 'transformer':
            self.fusion_layer = Muti_slice_fusion_layer(hidden_size=hidden_size, num_heads=hidden_size//32)
        if fusion_mode == 'rnn':
            self.fusion_layer = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, X:torch.Tensor, device=torch.device('cuda:0')):
        # (batch_size, slices, channel, H, W) -> (slices, batch_size, channel, H, W)
        X = X.permute(1, 0, 2, 3, 4)
        output = torch.Tensor([])
        output = output.to(device)
        for x in X:
            # x(batch_size, channel, H, W)
            # temp (batch_size, hidden_size)
            temp = self.encoder(x)
            # (batch_size, hidden_size) -> (1, batch_size, hidden_size)
            temp = temp.unsqueeze(0)
            output = torch.concat([output, temp])
        # (n_slice, batch_size, hidden_size) -> (batch_size, n_slice, hidden_size)
        output = output.permute(1, 0, 2)
        if self.fusion_layer is None:
            output = output.mean(dim=1)
        elif isinstance(self.fusion_layer, nn.LSTM):
            output, _ = self.fusion_layer(output)
            output = output.mean(dim=1)
        else:
           output = self.fusion_layer(output)
        output = self.fc(output)
        return output