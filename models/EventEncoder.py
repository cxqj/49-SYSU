import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from itertools import chain

class Basic_Encoder(nn.Module):
    def __init__(self, opt):
        super(Basic_Encoder, self).__init__()
        self.opt = opt
        self.hidden_dim = self.opt.hidden_dim
        opt.event_context_dim = self.opt.feature_dim
        opt.clip_context_dim = self.opt.feature_dim

    def forward(self, feats, vid_idx, featstamps, event_seq_idx=None, timestamps=None, vid_time_len=None):
        clip_feats, clip_mask =  self.get_clip_level_feats(feats, vid_idx, featstamps)
        event_feats = (clip_feats * clip_mask.unsqueeze(2)).sum(1) / (clip_mask.sum(1, keepdims=True) + 1e-5)
        return event_feats, clip_feats, clip_mask

    def get_clip_level_feats(self, feats, vid_idx, featstamps):
        max_att_len = max([(s[1] - s[0] + 1) for s in featstamps])   # 最长的事件长度
        clip_mask = feats.new(len(featstamps), max_att_len).zero_()  # (Prop_N, max_len)
        clip_feats = feats.new(len(featstamps), max_att_len, feats.shape[-1]).zero_()  # (Prop_N, max_len, 512)
        for i, soi in enumerate(featstamps):
            v_idx = vid_idx[i]
            selected = feats[v_idx][soi[0]:soi[1] + 1].reshape(-1, feats.shape[-1])  # (T,512)
            clip_feats[i, :len(selected), :] = selected
            clip_mask[i, :len(selected)] = 1
        return clip_feats, clip_mask  # (Prop_N, max_event_len, 512), (Prop_N, max_event_len)

# 获得每一对提议的position矩阵后，将其编码为向量矩阵
def extract_position_embedding(position_mat, feat_dim, wave_length=10000):  # position_mat:(Porp_N, Prop_N, 2), feat_dim:512
    # position_mat, [num_rois, nongt_dim, 2]
    num_rois, nongt_dim, _ = position_mat.shape
    feat_range = np.arange(0, feat_dim / 4)  
    # np.full() 第一个参数为要填充的矩阵形状，第二个形状是填充的数值
    # np.power(x,y)  计算x的y次方 
    dim_mat = np.power(np.full((1,), wave_length), (4. / feat_dim) * feat_range)  # (128,)
    dim_mat = np.reshape(dim_mat, newshape=(1, 1, 1, -1))   # (1,1,1,128)
    position_mat = np.expand_dims(100.0 * position_mat, axis=3)  # (Prop_N,Prop_N,2,1)
    div_mat = np.divide(position_mat, dim_mat)   # (Prop_N,Prop_N,2,128)
    sin_mat = np.sin(div_mat)     # (Prop_N,Prop_N,2,128)
    cos_mat = np.cos(div_mat)     # (Prop_N,Prop_N,2,128)
    # embedding, [num_rois, nongt_dim, 2, feat_dim/2]
    embedding = np.concatenate((sin_mat, cos_mat), axis=3)   # (Prop_N,Prop_N,2,256)
    # embedding, [num_rois, nongt_dim, 2, feat_dim/2]
    embedding = np.reshape(embedding, newshape=(num_rois, nongt_dim, feat_dim))    # (Prop_N,Prop_N,512)
    return embedding     # (Prop_N,Prop_N,512)

# 编码了每一对提议的相对长度和相对距离信息
def extract_position_matrix(bbox, nongt_dim):  # bbox:(Prop_N,2)
    start, end = np.split(bbox, 2, axis=1)     # start:(Prop_N,1), end:(Prop_N,1)
    center = 0.5 * (start + end)               # center:(Prop_N,1)
    length = (end - start).astype('float32')   # length: (Prop_N,1)
    length = np.maximum(length, 1e-1)  

    delta_center = np.divide(center - np.transpose(center), length)  # (Prop_N,Prop_N)
    delta_center = delta_center
    delta_length = np.divide(np.transpose(length),length)   # (Prop_N,Prop_N)
    delta_length = np.log(delta_length)
    delta_center = np.expand_dims(delta_center, 2)   # (Prop_N,Prop_N,1)
    delta_length = np.expand_dims(delta_length, 2)   # (Prop_N,Prop_N,1)
    position_matrix = np.concatenate((delta_center, delta_length), axis=2)  # (Prop_N,Prop_N,2)
    return position_matrix


class TSRM_Encoder(Basic_Encoder):
    def __init__(self, opt):
        super(TSRM_Encoder, self).__init__(opt)
        self.opt = opt
        self.hidden_dim = opt.hidden_dim  # 512
        self.group = opt.group_num  # 16
        self.pre_map = nn.Sequential(nn.Linear(opt.event_context_dim, opt.hidden_dim),
                                     nn.ReLU(),
                                     nn.Dropout(0.5))  # event_context_dim=512
        ### 这里也用了Q,K,V
        self.key_map = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.query_map = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.value_map = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.drop = nn.Dropout(0.5)
        self.use_posit_branch = opt.use_posit_branch
        if self.use_posit_branch:
            self.posit_hidden_dim = opt.hidden_dim
            self.pair_pos_fc1 = nn.Linear(opt.hidden_dim, opt.hidden_dim)
            self.pair_pos_fc2 = nn.Linear(opt.hidden_dim, opt.group_num)

        opt.event_context_dim = 2 * self.hidden_dim + 100
        opt.clip_context_dim = self.opt.feature_dim

    """
    feats: (B,T,512)
    vid_idx: [0,0,0,..,Prop_N]  指示当前proposal属于哪一个batch
    featstamps:  (Prop_N,2)
    event_seq_idx: [0,1,2,....,Prop_N]
    timestamps: [[0,7.67],[7.67,31.94],[31.94,50.25],[50.25,61.32],[61.32,73.67],[73.67,81.25]]
    vid_time_len: 85.17
    """
    def forward(self, feats, vid_idx, featstamps, event_seq_idx, timestamps, vid_time_len):
        clip_feats, clip_mask = self.get_clip_level_feats(feats, vid_idx, featstamps)  # (Prop_N, max_event_len, 512), (Prop_N, max_event_len)
        ###### 对应semantic ralation branch的mean pooling操作 ####
        event_feats = (clip_feats * clip_mask.unsqueeze(2)).sum(1) / (clip_mask.sum(1, keepdims=True) + 1e-5)  # (Prop_N, max_event_len, 512)-->(Prop_N, 512)
        event_feats = self.pre_map(event_feats)   # (Prop_N, 512)-->(Prop_N,512)

        batch_size = len(event_seq_idx)  # 1
        event_num = sum([_.shape[0] * _.shape[1] for _ in event_seq_idx])  # event_seq_idx:(1,Prop_N)
        event_feats_expand = feats.new_zeros(event_num, event_feats.shape[-1])  # (Prop_N,512)
        mask = feats.new_zeros(event_num, event_num)  # (Prop_N,Prop_N)  因为要计算当前proposal和其它proposal的关系，因此需要拓展mask
        timestamp_expand = []   
        vid_time_len_expand = []

        total_idx = 0
        for i in range(batch_size):
            vid_start_idx = (vid_idx < i).sum().item()  # 0
            for j in range(event_seq_idx[i].shape[0]):
                event_idx = vid_start_idx + event_seq_idx[i][j]  # [0,1,2,3,...,Prop_N]
                event_feats_expand[total_idx: total_idx + len(event_idx)] = event_feats[event_idx]  # (Prop_N,512) 
                mask[total_idx: total_idx + len(event_idx), total_idx: total_idx + len(event_idx)] = 1  # (Prop_N,Prop_N)
                timestamp_expand.extend([timestamps[ii] for ii in event_idx])  # [[0,7.67],[7.67,31.69],...[73.17,83.67]]
                vid_time_len_expand.extend([vid_time_len[i].item() for jj in event_idx])  # [85.16,85.16,...,85.16]
                total_idx += len(event_idx)

        """
         计算不同提议间的语义相似度
         adopt scaled dot-product attention to calculate the semantic similarity between proposals.
        """
        query_mat = self.query_map(event_feats_expand).reshape(event_num, self.group,
                                                     int(self.hidden_dim / self.group)).transpose(0, 1)  # (Prop_N,512)-->(Prop_N,16,32)-->(16, Prop_N, 32)
        key = self.key_map(event_feats_expand).reshape(event_num, self.group, int(self.hidden_dim / self.group)).transpose(0,
                                                                                                                     1)  # (Prop_N,512)-->(Prop_N,16,32)-->(16, Prop_N, 32)

        cos_sim = torch.bmm(query_mat, key.transpose(1, 2))  # [16, Prop_N, Prop_N]
        cos_sim = cos_sim / math.sqrt(self.hidden_dim / self.group)  
        sim = cos_sim
        mask = mask.unsqueeze(0)  # (1,Prop_N,Prop_N)
    
        """
        计算不同提议间的时序相关性
        对应论文中Temporal Relation 分支  ######
        """
        if self.use_posit_branch:
            pos_matrix = extract_position_matrix(np.array(timestamp_expand), event_num)   # (Prop_N,2) 
            pos_feats = extract_position_embedding(pos_matrix,self.posit_hidden_dim)  # [Prop_N, Prop_N, 512]
            pos_feats = feats.new_tensor(pos_feats).reshape(-1, self.posit_hidden_dim)  # (Prop_N x Prop_N, 512)
            pos_sim = self.pair_pos_fc2(torch.tanh(self.pair_pos_fc1(pos_feats))).reshape(event_num, event_num, self.group)  # 经过两个全连接层编码 (Prop_N, Prop_N, 16)
            pos_sim = pos_sim.permute(2, 0, 1)   # (16,Prop_N,Prop_N)
            sim = cos_sim + pos_sim   # (16,Prop_N,Prop_N)    融合语义和时序相似度信息

        sim = F.softmax(sim, dim=2)
        sim = (sim * mask) / (1e-5 + torch.sum(sim * mask, dim=2, keepdim=True))   # (16,Prop_N,Prop_N)

        value = self.value_map(event_feats_expand).reshape(event_num, self.group, -1).transpose(0, 1)  # (Prop_N,512)-->(Prop_N,16,32)-->(16,Prop_N,32)
        event_ctx = torch.bmm(sim, value)  # (16, Prop_N, Prop_N) x (16, Prop_N, 32)-->(16, Prop_N, 32)
        event_ctx = event_ctx.transpose(0, 1).reshape(event_num, -1)  # (Prop_N,512)
        event_ctx = torch.cat((F.relu(event_ctx), event_feats_expand), 1)  # (Prop_N,1024)    融合mean pooling特征
        event_ctx = self.drop(event_ctx)    # 最终的事件特征

        # positional feature vector for each event
        pos_feats = feats.new(len(timestamp_expand), 100).zero_()  # (Prop_N,100)
        for i in range(len(timestamp_expand)):    
            s, e = timestamp_expand[i]
            duration = vid_time_len_expand[i]
            s, e = min(int(s / duration * 99), 99), min(int(e / duration * 99), 99)
            pos_feats[i, s: e + 1] = 1
        event_ctx = torch.cat([event_ctx, pos_feats], dim=1)  # (Prop_N,1024) + (Prop_N,100)-->(Prop_N,1124)

        ### event_cxt = weighted_event_feature + event_feats_expand + positional_feature_vector
        return event_ctx, clip_feats, clip_mask   # (Prop_N,1124), (Prop_N, max_event_len, 512), (Prop_N, max_event_len)


if __name__ == '__main__':
    pass
