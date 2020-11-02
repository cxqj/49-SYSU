import pdb
import sys
import torch
from torch import nn
import models
import numpy as np
from collections import OrderedDict
from itertools import chain

sys.path.append("densevid_eval3/coco-caption3")

from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

Meteor_scorer = None


# Cider_scorer = None

class EncoderDecoder(nn.Module):
    def __init__(self, opt):
        super(EncoderDecoder, self).__init__()
        self.opt = opt
        if opt.feature_dim > 1024:
            self.frame_reduce_dim_layer = nn.Sequential(nn.Linear(opt.feature_dim, self.opt.hidden_dim),
                                                        nn.ReLU())  # 3072-->512
            self.opt.raw_feature_dim = self.opt.feature_dim  # 3072
            self.opt.feature_dim = self.opt.hidden_dim  # 512

        self.event_encoder_type = opt.event_encoder_type  # tsrm  temporal-semantic relation modul(TSRM) to capture rich relationships between events in terms of both temporal structure and semantic meaning.
        self.event_encoder = models.setup_event_encoder(opt)   # 事件编码器
        self.caption_decoder = models.setup_caption_decoder(opt)   # caption解码器

    def forward(self, dt, mode, loader=None):

        if 'hrnn' in self.opt.caption_decoder_type:
            return self.forward_hrnn(dt, mode, loader)
        else:
            return self.forward_rnn(dt, mode, loader)

    def get_features(self, dt, soi_select_list):
        # assert type(soi_select_list) == list
        soi_select_list = np.array(soi_select_list)  # (proposal_num,2)
        # (Prop_N,1124), (Prop_N, max_event_len, 512), (Prop_N, max_event_len)
        event, clip, clip_mask = self.event_encoder(dt['video_tensor'],
                                                    dt['lnt_gt_idx'][:, 1], soi_select_list, dt['lnt_event_seq_idx'],
                                                    list(chain(*dt['lnt_timestamp'])), dt['video_length'][:, 1])
        return event, clip, clip_mask

    def forward_hrnn(self, dt, mode='train', loader=None):
        '''
        Support caption model with hierarchical RNN, note that batch_size must be 1 (one video)
        '''
        assert self.opt.batch_size == 1
        assert self.opt.train_proposal_type in ['learnt_seq', 'gt']

        FIRST_DIM = 0
        event_seq_idx = dt['lnt_event_seq_idx'][FIRST_DIM]  # [0,1,2,3,4,...N]
        if mode == 'train' or mode == 'train_rl':
            seq_gt_idx = dt['lnt_seq_gt_idx'][FIRST_DIM]    # [0,1,2,3,4,...N]
            cap_raw = dt['cap_raw'][FIRST_DIM]
            cap_big_ids = dt['lnt_gt_idx'][:, 0]   # [0,1,2,3,...N]

        # 特征降维(B,T,3072)通过全连接层降到(B,T,512)
        if hasattr(self, 'frame_reduce_dim_layer'):
            vid_num, vid_len, _ = dt['video_tensor'].shape
            tmp = dt['video_tensor'].reshape(vid_num * vid_len, -1)
            dt['video_tensor'] = self.frame_reduce_dim_layer(tmp).reshape(vid_num, vid_len, -1)

        # Event Encoder, (Prop_N,1124), (Prop_N, max_event_len, 512), (Prop_N, max_event_len)
        event, clip, clip_mask = self.get_features(dt, dt['lnt_featstamps'])   # 对应着论文中的TSRM  

        event_feat_expand_flag = self.event_encoder_type in ['tsrm']  # 是否添加上下文TSRM特征

        if mode == 'train':
            # event(TSRM): (Prop_N,1124)  clip:(Prop_N, max_event_len, 512)  clip_mask: (Prop_N, max_event_len, 512)
            # dt['cap_tensor'][cap_big_ids]: (Prop_N,max_sent_length)  event_seq_idx:[0,1,2,...,Prop_N]  event_feat_expand_flag = 1
            cap_prob = self.caption_decoder(event, clip, clip_mask, dt['cap_tensor'][cap_big_ids],
                                           event_seq_idx, event_feat_expand_flag)  # (1,Prop_N,max_sent_len-1,5478)
            cap_prob = cap_prob.reshape(-1, cap_prob.shape[-2], cap_prob.shape[-1])  # (Prop_N,max_sent_len-1,5478)
            caption_tensor = dt['cap_tensor'][:, 1:][seq_gt_idx.reshape(-1)]  # (Prop_N,max_sent_len-1)
            caption_mask = dt['cap_mask'][:, 1:][seq_gt_idx.reshape(-1)]   # (Prop_N,max_sent_len-1, 1)
            loss = self.caption_decoder.build_loss(cap_prob, caption_tensor, caption_mask)
            return loss, torch.zeros(1), torch.zeros(1)

        ##### 强化学习只在解码阶段使用  #######
        elif mode == 'train_rl':
            # gen_result: (eseq_num, eseq_len, ~cap_len), sample_logprobs:(eseq_num, eseq_len, ~cap_len)
            gen_result, sample_logprobs = self.caption_decoder.sample(event, clip, clip_mask, event_seq_idx,
                                                                      event_feat_expand_flag, opt={'sample_max': 0})
            self.caption_decoder.eval()
            with torch.no_grad():
                # SCST scheme in paper "Self-critical Sequence Training for Image Captioning"
                greedy_res, _ = self.caption_decoder.sample(event, clip, clip_mask, event_seq_idx,
                                                            event_feat_expand_flag)
                # # RL scheme in paper "streamlined dense video captioning"
                # video_bl, event_bl, clip_bl, clip_mask_bl, _ = self.get_features(dt, dt['gt_featstamps'])
                # greedy_res, _ = self.caption_model.sample(video_bl, event_bl, clip_bl, clip_mask_bl, seq_gt_idx,
                #                                               event_feat_expand_flag)
            self.caption_decoder.train()
            gen_result = gen_result.reshape(-1, gen_result.shape[-1])
            greedy_res = greedy_res.reshape(-1, greedy_res.shape[-1])
            gt_caption = [loader.dataset.translate(cap, max_len=50) for cap in cap_raw]
            gt_caption = [gt_caption[i] for i in seq_gt_idx.reshape(-1)]
            reward, sample_meteor, greedy_meteor = get_caption_reward(greedy_res, gt_caption, gen_result, self.opt)
            reward = np.repeat(reward[:, np.newaxis], gen_result.size(1), 1)
            caption_loss = self.caption_decoder.build_rl_loss(sample_logprobs, gen_result.float(),
                                                              sample_logprobs.new_tensor(reward))
            return caption_loss, sample_meteor, greedy_meteor

        elif mode == 'eval':
            with torch.no_grad():  
                # event:(Prop_N,1124) clip:(Prop_N, max_event_length, 512) clip_mask:(Prop_N, max_event_length)
                # event_seq_idx: (0,1,2,...,N)  event_feat_expand_flag:True
                seq, cap_prob = self.caption_decoder.sample(event, clip, clip_mask, event_seq_idx,
                                                           event_feat_expand_flag)  
            return seq, cap_prob  # (batch_size, Prop_N, max_pred_sent_length), (batch_size, Prop_N, max_pred_sent_length)

        else:
            raise AssertionError

    def forward_rnn(self, dt, mode='train', loader=None):
        '''
        Support caption model with single-level RNN, batch_size can be larger than 1
        '''
        assert self.opt.train_proposal_type in ['learnt', 'gt']

        if hasattr(self, 'frame_reduce_dim_layer'):
            vid_num, vid_len, _ = dt['video_tensor'].shape
            dt['video_tensor'] = self.frame_reduce_dim_layer(dt['video_tensor'].reshape(vid_num * vid_len, -1)).reshape(
                vid_num, vid_len, -1)

        cap_bigids, cap_vid_ids, cap_event_ids = dt['lnt_gt_idx'][:, 0], dt['lnt_gt_idx'][:, 1], dt['lnt_gt_idx'][:, 2]
        event, clip, clip_mask = self.get_features(dt, dt['lnt_featstamps'])

        if mode == 'train':
            cap_prob = self.caption_decoder(event, clip, clip_mask, dt['cap_tensor'][cap_bigids])
            cap_prob = cap_prob.reshape(-1, cap_prob.shape[-2], cap_prob.shape[-1])
            caption_tensor = dt['cap_tensor'][:, 1:][cap_bigids]
            caption_mask = dt['cap_mask'][:, 1:][cap_bigids]
            loss = self.caption_decoder.build_loss(cap_prob, caption_tensor, caption_mask)
            return loss, torch.zeros(1), torch.zeros(1)

        elif mode == 'train_rl':
            # gen_result: (eseq_num, eseq_len, ~cap_len), sample_logprobs :(eseq_num, eseq_len, ~cap_len)
            #### 随机采样得到的结果 ####
            gen_result, sample_logprobs = self.caption_decoder.sample(event, clip, clip_mask,
                                                                      opt={'sample_max': 0})  # (batch_size, Prop_N, Caption_Len)  sample_max = 0(随机采样)
            self.caption_decoder.eval()
            with torch.no_grad():
                #### 使用贪心算法得到的结果，也就是测试结果 ####
                greedy_res, _ = self.caption_decoder.sample(event, clip, clip_mask)  # 贪心法采样
                # video_bl, event_bl, clip_bl, clip_mask_bl, _ = self.get_features(dt, dt['gt_featstamps'])
                # greedy_res, _ = self.caption_model.sample(video_bl, event_bl, clip_bl, clip_mask_bl)
            self.caption_decoder.train()  # 训练模式
            gen_result = gen_result.reshape(-1, gen_result.shape[-1])   # (batch_size*Prop_N, Caption_Len)  随机采样生成的句子
            greedy_res = greedy_res.reshape(-1, greedy_res.shape[-1])   # (batch_size*Prop_N, Caption_Len)  贪心采样生成的句子

            if True:
                gt_caption = [[loader.dataset.translate(cap, max_len=50) for cap in caps] for caps in dt['cap_raw']] #将原始语句转为index
                gt_caption = [gt_caption[cap_vid_ids[i]][cap_event_ids[i]] for i in range(len(cap_vid_ids))]  
                # rewards为随机采样与贪心采样meteor得分的差，随机采样方式的meteor得分，贪心采样方式的meteor得分
                reward, sample_meteor, greedy_meteor = get_caption_reward(greedy_res, gt_caption, gen_result, self.opt)    
            reward = np.repeat(reward[:, np.newaxis], gen_result.size(1), 1)  # (1,Caption_Len)
            # new_tensor()可以将源张量中的数据复制到目标张量（数据不共享），同时提供了更细致的属性控制：
            caption_loss = self.caption_decoder.build_rl_loss(sample_logprobs, gen_result.float(),
                                                              sample_logprobs.new_tensor(reward)) # 将reward复制到sample_logprobs中
            return reward, caption_loss, sample_meteor, greedy_meteor  # reward: (1,Caption_Len)  Caption_Loss = Sample_meteor = Greedy_meteor = (1)

        elif mode == 'eval':
            with torch.no_grad():
                seq, cap_prob = self.caption_decoder.sample(event, clip, clip_mask)
            return seq, cap_prob
        else:
            raise AssertionError


def init_scorer():
    global Meteor_scorer
    Meteor_scorer = Meteor()
    # global Cider_scorer
    # Cider_scorer = Cider()


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

"""
SCST的思想就是用当前模型在测试阶段生成的词的reward作为baseline，梯度就变成了：
      (r(ws)-r(w^))(p-1ws)   1ws表示单词的one-hot向量表示，其中r(w^)=argmaxwtp(wt|ht)，就是在测试阶段使用greedy decoding取概率最大的词来生成句子；  
   而r(ws)是通过根据概率来随机sample词，如果当前概率最大的词的概率为60%，那就有60%的概率选到它，而不是像greedy decoding一样100%选概率最大的。
   公式的意思就是：对于如果当前sample到的词比测试阶段生成的词好，那么在这次词的维度上，整个式子的值就是负的（因为后面那一项一定为负），这样梯
   度就会上升，从而提高这个词的分数st；而对于其他词，后面那一项为正，梯度就会下降，从而降低其他词的分数。
   参考：https://blog.csdn.net/sinat_26253653/article/details/78458894
"""
def get_caption_reward(greedy_res, gt_captions, gen_result, opt):
    greedy_res = greedy_res.detach().cpu().numpy()  # (batch_size*Prop_N, Caption_Len)  贪心采样生成的句子
    gen_result = gen_result.detach().cpu().numpy()  # (batch_size*Prop_N, Caption_Len)  随机采样生成的句子
    batch_size = len(gen_result)

    ## 将两种方式生成的句子存入字典 ##
    res = OrderedDict()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(gt_captions)):
        gts[i] = [array_to_str(gt_captions[i][1:])]

    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}

    _, meteor_score = Meteor_scorer.compute_score(gts, res__)  ## 计算meteor得分
    scores = np.array(meteor_score)
    rewards = scores[:batch_size] - scores[batch_size:]  # rewards为随机采样与贪心采样meteor得分的差

    return rewards, scores[:batch_size], scores[batch_size:]  # rewards为随机采样与贪心采样meteor得分的差，随机采样方式的meteor得分，贪心采样方式的meteor得分 


