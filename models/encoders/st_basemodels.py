import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) :
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            # print("LSTM------",m.named_parameters())
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)  # initializing the lstm bias with zeros
        else:
            print(m,"************")



class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
        
class MLP_gate(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP_gate, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.sigmoid(hidden_states)
        return hidden_states

class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states
        

class Temporal_Encoder(nn.Module):
    """Construct the sequence model"""

    def __init__(self, feature_size, hidden_size, encoder_layers=3, encoder_head=10, skip_conv=False):
        super(Temporal_Encoder, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.skip_conv = skip_conv

        # self.conv1d = nn.Conv1d(self.feature_size, self.hidden_size, kernel_size=3, stride=1, padding=1)
        
        self.mlp1 = MLP(hidden_size=feature_size, out_features=hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=encoder_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        self.conv1d = nn.Conv1d(self.feature_size, self.hidden_size, kernel_size=3, stride=1, padding=1)
        
        # self.lstm = nn.LSTM(input_size=self.hidden_size,
        #                   hidden_size=self.hidden_size,
        #                   num_layers=1,
        #                   bias=True,
        #                   batch_first=True,
        #                   dropout=0,
        #                   bidirectional=False)
        # initialize_weights(self.conv1d.modules())

    def forward(self, x):
        self.x_dense=x
        if not self.skip_conv:
            
            # print("before self.x_dense.shape",self.x_dense.shape)

            self.x_dense=self.mlp1(self.x_dense)


        # print("after self.x_dense.shape",self.x_dense.shape)
        self.x_dense_in = self.transformer_encoder(self.x_dense) + self.x_dense  #[N, H, D]

        # print("encoder self.x_dense_in.shape",self.x_dense_in.shape)


        if self.skip_conv:
            return self.x_dense_in
        else:
            self.x_dense_in=self.conv1d(self.x_dense_in)
            # print("conv last self.x_dense_in.shape",self.x_dense_in.shape)
            return self.x_dense_in
        # output, (hn, cn) = self.lstm(self.x_dense_in)
        # self.x_state, cn = hn.squeeze(0), cn.squeeze(0) #[N, D]
        # self.x_endoced=self.mlp(self.x_state) + self.x_state#[N, D]
        # return self.x_endoced, self.x_state, cn
    

    # def forward_packed(self, x_packed):
    #     # Forward pass through MLP
    #     # x_packed = self.mlp(x_packed.data)
    #     self.x_dense = x_packed.data
    #     print(self.x_dense.shape)
    #     self.x_dense = self.mlp1(self.x_dense) + self.x_dense #[N, H, dim]
    #     print("yessssssssssssssssss")
    #     self.x_dense_in = self.transformer_encoder(self.x_dense) + self.x_dense  #[N, H, D]
    #     output, (hn, cn) = self.lstm(self.x_dense_in)
    #     self.x_state, cn = hn.squeeze(0), cn.squeeze(0) #[N, D]
    #     self.x_endoced=self.mlp(self.x_state) + self.x_state#[N, D]
    #     return self.x_endoced, self.x_state, cn
    



# class GATraj(nn.Module):
#     def __init__(self, args):
#         super(GATraj, self).__init__()
#         self.args = args
#         self.Temperal_Encoder=Temperal_Encoder(self.args)
#         self.Laplacian_Decoder=Laplacian_Decoder(self.args)
#         if self.args.SR:
#             message_passing = []
#             for i in range(self.args.pass_time):
#                 message_passing.append(Global_interaction(args))
#             self.Global_interaction = nn.ModuleList(message_passing)
#         if self.args.ifGaussian:
#             self.reg_loss = GaussianNLLLoss(reduction='mean')
#         else:
#             self.reg_loss = LaplaceNLLLoss(reduction='mean')
#         self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')

#     def forward(self, inputs, epoch, iftest=False):
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         batch_abs_gt, batch_norm_gt, nei_list_batch, nei_num_batch, batch_split = inputs # #[H, N, 2], [H, N, 2], [B, H, N, N], [N, H], [B, 2]
#         self.batch_norm_gt = batch_norm_gt
#         if self.args.input_offset:
#             train_x = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length-1, :, :] #[H, N, 2]
#         elif self.args.input_mix:
#             offset = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length-1, :, :] #[H, N, 2]
#             position = batch_norm_gt[:self.args.obs_length, :, :] #[H, N, 2]
#             pad_offset = torch.zeros_like(position).to(device)
#             pad_offset[1:, :, :] = offset
#             train_x = torch.cat((position, pad_offset), dim=2)
#         elif self.args.input_position:
#             train_x = batch_norm_gt[:self.args.obs_length, :, :] #[H, N, 2]
#         train_x = train_x.permute(1, 2, 0) #[N, 2, H]
#         train_y = batch_norm_gt[self.args.obs_length:, :, :].permute(1, 2, 0) #[N, 2, H]
#         self.pre_obs=batch_norm_gt[1:self.args.obs_length]
#         self.x_encoded_dense, self.hidden_state_unsplited, cn=self.Temperal_Encoder.forward(train_x)  #[N, D], [N, D]
#         self.hidden_state_global = torch.ones_like(self.hidden_state_unsplited, device=device)
#         cn_global = torch.ones_like(cn, device=device)
#         if self.args.SR:
#             for b in range(len(nei_list_batch)):
#                 left, right = batch_split[b][0], batch_split[b][1]
#                 element_states = self.hidden_state_unsplited[left: right] #[N, D]
#                 cn_state = cn[left: right] #[N, D]
#                 if element_states.shape[0] != 1:
#                     corr = batch_abs_gt[self.args.obs_length-1, left: right, :2].repeat(element_states.shape[0], 1, 1) #[N, N, D]
#                     corr_index = corr.transpose(0,1)-corr  #[N, N, D]
#                     nei_num = nei_num_batch[left:right, self.args.obs_length-1] #[N]
#                     nei_index = torch.tensor(nei_list_batch[b][self.args.obs_length-1], device=device) #[N, N]
#                     for i in range(self.args.pass_time):
#                         element_states, cn_state = self.Global_interaction[i](corr_index, nei_index, nei_num, element_states, cn_state)
#                     self.hidden_state_global[left: right] = element_states
#                     cn_global[left: right] = cn_state
#                 else:
#                     self.hidden_state_global[left: right] = element_states
#                     cn_global[left: right] = cn_state
#         else:
#             self.hidden_state_global = self.hidden_state_unsplited
#             cn_global = cn
#         mdn_out = self.Laplacian_Decoder.forward(self.x_encoded_dense, self.hidden_state_global, cn_global, epoch)
#         GATraj_loss, full_pre_tra = self.mdn_loss(train_y.permute(2, 0, 1), mdn_out, 1, iftest)  #[K, H, N, 2]
#         return GATraj_loss, full_pre_tra

#     def mdn_loss(self, y, y_prime, goal_gt, iftest):
#         batch_size=y.shape[1]
#         y = y.permute(1, 0, 2)  #[N, H, 2]
#         # [F, N, H, 2], [F, N, H, 2], [N, F]
#         out_mu, out_sigma, out_pi = y_prime 
#         y_hat = torch.cat((out_mu, out_sigma), dim=-1)
#         reg_loss, cls_loss = 0, 0
#         full_pre_tra = []
#         l2_norm = (torch.norm(out_mu - y, p=2, dim=-1) ).sum(dim=-1)   # [F, N]
#         best_mode = l2_norm.argmin(dim=0)
#         y_hat_best = y_hat[best_mode, torch.arange(batch_size)]
#         reg_loss += self.reg_loss(y_hat_best, y)
#         soft_target = F.softmax(-l2_norm / self.args.pred_length, dim=0).t().detach() # [N, F]
#         cls_loss += self.cls_loss(out_pi, soft_target)
#         loss = reg_loss + cls_loss
#         #best ADE
#         sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
#         full_pre_tra.append(torch.cat((self.pre_obs,sample_k), axis=0))
#         # best FDE
#         l2_norm_FDE = (torch.norm(out_mu[:,:,-1,:] - y[:,-1,:], p=2, dim=-1) )  # [F, N]
#         best_mode = l2_norm_FDE.argmin(dim=0)
#         sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
#         full_pre_tra.append(torch.cat((self.pre_obs,sample_k), axis=0))
#         return loss, full_pre_tra
