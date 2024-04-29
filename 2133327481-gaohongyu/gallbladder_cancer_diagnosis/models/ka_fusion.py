import torch
from torch import nn
import transformers.models.bert

class Fusion_Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, ct, clinical):
        return torch.concat([ct, clinical], dim=1)


class Scale_Dot_Attention(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(1, num_heads=1, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4), nn.GELU(), nn.Linear(hidden_dim*4, hidden_dim))
        self.norm1 = nn.LayerNorm(hidden_dim, 1e-5)
        self.norm2 = nn.LayerNorm(hidden_dim, 1e-5)
    
    def forward(self, qkv):
        query, key, value = qkv
        q = query.unsqueeze(-1)
        k = key.unsqueeze(-1)
        v = self.norm1(value).unsqueeze(-1)

        attn_v, _ = self.attention(q, k, v)

        output = self.norm2(attn_v.squeeze(-1) + v.squeeze(-1))
        kv = output + self.mlp(output)

        return (query, kv, kv)

class Knowledge_aware_Fusion_Layer(nn.Module):
    def __init__(self, ct_dim, clinical_dim, hidden_dim, n_layers:int=12) -> None:
        super().__init__()
        self.ct_embedding = nn.Linear(ct_dim, hidden_dim)
        self.clinical_embedding = nn.Linear(clinical_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim*2, 1e-5)

        self.att1 = []
        self.att2 = []

        for _ in range(n_layers):
            self.att1.append(Scale_Dot_Attention(hidden_dim))
            self.att2.append(Scale_Dot_Attention(hidden_dim))
        
        self.att1 = nn.Sequential(*self.att1)
        self.att2 = nn.Sequential(*self.att2)

    def forward(self, ct, clinical):
        ct = self.ct_embedding(ct)
        clinical = self.clinical_embedding(clinical)
        _, ct_attn, _ = self.att1((clinical, ct, ct))
        _, clinical_attn, _ = self.att2((ct, clinical, clinical))

        return self.norm(torch.concat([ct_attn, clinical_attn], dim=1))

class KA_Module(nn.Module):
    def __init__(self, ct_encoder, clinical_encoder, ct_dim, clinical_dim, hidden_dim,
                 n_class:int=2, dropout_rate:float=0, fusion_mode:str='knowledge_aware') -> None:
        super().__init__()

        self.ct_encoder = ct_encoder
        self.clinical_encoder = clinical_encoder

        if fusion_mode == 'knowledge_aware':
            self.fusion_layer = Knowledge_aware_Fusion_Layer(ct_dim, clinical_dim, hidden_dim)
            self.cls = nn.Linear(hidden_dim*2, n_class)
        else:
            self.fusion_layer = Fusion_Layer()
            self.cls = nn.Linear(ct_dim+clinical_dim, n_class)

        for m in self.fusion_layer.children():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.trunc_normal_(self.cls.weight, std=0.02)
    
    def forward(self, ct, clinical):
        ct_vector = self.ct_encoder(ct)
        clinical_vector = self.clinical_encoder(clinical)

        fusion_output = self.fusion_layer(ct_vector, clinical_vector)

        return self.cls(fusion_output)