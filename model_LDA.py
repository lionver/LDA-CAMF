import torch.nn as nn
import torch



class CA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        feature_dim = 8
        self.embedding_layer_L = nn.Linear(1, feature_dim)
        self.embedding_layer_D = nn.Linear(1, feature_dim)
        self.mix_attention_layer = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8, batch_first=True)
        self.L_avg_pool = nn.AvgPool1d(feature_dim)
        self.D_avg_pool = nn.AvgPool1d(feature_dim)
    def forward(self, s_x):
        s_x = s_x.float()
        if s_x.shape[1] == 239:
            LFea = s_x[:, :82]
            DFea = s_x[:, 82:]
        else:
            LFea = s_x[:, :89]
            DFea = s_x[:, 89:]
        L_QKV = LFea.unsqueeze(2)
        L_QKV = self.embedding_layer_L(L_QKV)
        D_QKV = DFea.unsqueeze(2)
        D_QKV = self.embedding_layer_D(D_QKV)
        L_att, _ = self.mix_attention_layer(L_QKV, D_QKV, D_QKV)
        D_att, _ = self.mix_attention_layer(D_QKV, L_QKV, L_QKV)
        LConv = L_QKV + L_att
        DConv = D_QKV + D_att
        LConv = self.L_avg_pool(LConv).squeeze(2)
        DConv = self.D_avg_pool(DConv).squeeze(2)
        lnc_dis_matrix = torch.matmul(LConv.T, DConv)
        pair = torch.cat([LConv, DConv], dim=1)
        return lnc_dis_matrix, pair


