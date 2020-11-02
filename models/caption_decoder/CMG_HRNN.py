import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

# HRNN为一个sent RNN + 一个word RNN
class CMG_HRNN(nn.Module):
    def __init__(self, opt):
        super(CMG_HRNN, self).__init__()

        self.opt = opt
        assert opt.batch_size == 1

        self.vocab_size = opt.vocab_size  # 5747
        self.input_encoding_size = opt.input_encoding_size  # 512(词嵌入维度)
        self.rnn_size = opt.rnn_size  # 512
        self.num_layers = opt.num_layers  # 1
        self.drop_prob_lm = opt.drop_prob # 0.5
        self.max_caption_len = opt.max_caption_len  #30
        self.ss_prob = 0.0
        self.sent_rnn_size = self.rnn_size  # 512
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)  # (5748,512)词嵌入矩阵

        self.sent_rnn = nn.LSTM(opt.hidden_dim + opt.hidden_dim,
                                self.sent_rnn_size, 1, bias=False,
                                dropout=self.drop_prob_lm)  # 1024-->512

        self.gate_layer = nn.Sequential(nn.Linear(2 * opt.hidden_dim + self.rnn_size, opt.hidden_dim),
                                        nn.Sigmoid())  # 1536-->512
        self.global_proj = nn.Sequential(nn.Linear(self.sent_rnn_size, opt.hidden_dim),
                                         nn.Tanh())    # 512-->512
        self.local_proj = nn.Sequential(nn.Linear(opt.event_context_dim, opt.hidden_dim),
                                        nn.Tanh())     # 1124-->512
        self.para_transfer_layer = nn.Linear(self.sent_rnn_size, self.rnn_size * self.num_layers)  # 512-->512

        self.gate_drop = nn.Dropout(p=opt.drop_prob)  # 0.5

        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)  # 512-->5748 
        self.dropout = nn.Dropout(self.drop_prob_lm)  # 0.5
        self.init_weights()  # 初始化词嵌入和logit全连接层的参数

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):  # batch_size:1
        weight = next(self.parameters()).data  # (5748,512)
        return (weight.new(self.num_layers, batch_size, self.rnn_size).zero_(),
                weight.new(self.num_layers, batch_size, self.rnn_size).zero_())  # (h0, c0) [(1,1,512),(1,1,512)]

    def build_loss(self, input, target, mask):  # input:(Prop_N,max_sent_len-1,5748) target:(Prop_N,max_sent_len-1) mask:(Prop_N,max_sent_len-1,1)
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = input.reshape(-1, input.size(2))   # (Prop_N * max_sent_len-1,5748)
        target = target.reshape(-1, 1)   # (Prop_N * max_sent_len-1, 1)
        mask = mask.reshape(-1, 1)  # (Prop_N * max_sent_len-1, 1) 
        output = - input.gather(1, target) * mask  # (Prop_N * max_sent_len-1, 1)
        output = torch.sum(output) / (torch.sum(mask) + 1e-6)  # 8.76
        return output

    def build_rl_loss(self, input, seq, reward):
        input = (input).reshape(-1)
        reward = (reward).reshape(-1)
        mask = (seq > 0).float()
        mask = (torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / (torch.sum(mask) + 1e-6)
        return output

    ### 解码时先进行CMG_HRNN的前向传播
     # event(TSRM): (Prop_N,1124)  clip:(Prop_N, max_event_len, 512)  clip_mask: (Prop_N, max_event_len, 512)
     # dt['cap_tensor'][cap_big_ids]: (Prop_N,max_sent_length)  event_seq_idx:[0,1,2,...,Prop_N]  event_feat_expand_flag = 1
    def forward(self, event, clip, clip_mask, seq, event_seq_idx, event_feat_expand=False):
        # TODO: annotation
        eseq_num, eseq_len = event_seq_idx.shape  # (1,Prop_N)
        para_state = self.init_hidden(eseq_num)  # return Zero hidden state  (1,1,512)  (1,1,512)   word RNN state
        last_sent_state = clip.new_zeros(eseq_num, self.rnn_size)  # return Zero hidden state  (1,512)   sent RNN state
        para_outputs = []
        seq = seq.long()  # 转为torch.int64类型

        if event is None:
            event = (clip * clip_mask.unsqueeze(2)).sum(1) / (clip_mask.sum(1, keepdims=True) + 1e-5)

        if not event_feat_expand:
            assert len(event.shape) == 2
        else:
            event = event.reshape(eseq_num, eseq_len, event.shape[-1])  # (1,Prop_N,1124)

        ###  遍历每一个提议 ######
        for idx in range(eseq_len):

            event_idx = event[event_seq_idx[:, idx]] if not event_feat_expand else event[:, idx]  # (1,1124)
            clip_idx = clip[event_seq_idx[:, idx]]   # (1,max_event_len,512)
            clip_mask_idx = clip_mask[event_seq_idx[:, idx]]    # (1,max_event_len)
            seq_idx = seq[event_seq_idx[:, idx]]  # (1,max_seq_len)

            ##########################   计算初始state状态信息   ############################
            # cross-modal fusion
            # last_sent_state: linguistic information of previous events
            # event_idx: visual information of previous events  (event idx = weighted_event_feature + event_feats_expand + positional_feature_vector )
            prev_state_proj = self.global_proj(last_sent_state)  # (1,152)-->(1,512)  上一个句子的整体隐状态信息
            event_proj = self.local_proj(event_idx)  # (1,1124)-->(1,512)   事件上下文信息
            """
             1) the position embedding li of proposal pi (event_proj)
             2) the proposal’s feature vector zi,which is the concatenation of the output of TSRM module and the mean pooling of the frame-level features within pi, (event_proj)
             3) the last hidden state sii1 in the word RNN of the previous sentence, and  (para_state[0][-1])
             4) the previous hidden state hii1 of the  sentence RNN.  (prev_state_proj)
            
            """
            gate_input = torch.cat((para_state[0][-1], prev_state_proj, event_proj), 1)  # (1,1536)   para_state为word RNN隐状态，prev_state_proj为上一个提议的整体句子特征，event_proj为事件上下文信息 
            gate = self.gate_layer(self.gate_drop(gate_input))  # (1,1536)-->(1,512)
            gate = torch.cat((gate, 1 - gate), dim=1)  #(1,1024)  gate control visual information, 1-gate control linguistic information
            """
             we propose a cross modal gating (CMG) block to adaptively balance the visual and linguistic information of sent RNN.
            """
            sent_rnn_input = torch.cat((prev_state_proj, event_proj), dim=1)  # (1,1024) prev_state_proj对应linguistic info, event_proj对应visual info
            sent_rnn_input = sent_rnn_input * gate  # (1,1024)
            
            ###### 使用sent RNN ###########
            _, para_state = self.sent_rnn(sent_rnn_input.unsqueeze(0), para_state)  # para_state中保存的是senRNN的状态

            para_c, para_h = para_state  # (1,1,512), (1,1,512)
            num_layers, batch_size, para_dim = para_h.size()  # 1, 1, 512
            init_h = self.para_transfer_layer(para_h[-1]).reshape(self.num_layers, batch_size, para_dim)  # (1,1,512)
            state = (init_h, init_h)  # [(1,1,512),(1,1,512)]

            outputs = []
            seq_len_idx = (seq_idx > 0).sum(1) + 2  # 当前语句的真实长度

            last_sent_state = clip.new_zeros(eseq_num, self.rnn_size)   # (1,512)
            ########################   使用word RNN   ###############################
            for i in range(seq_idx.size(1) - 1):
                if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
                    sample_prob = clip.new(eseq_num).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq_idx[:, i].clone().long() 
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq_idx[:, i].clone().long()
                        prob_prev = torch.exp(outputs[-1].data)  # fetch prev distribution: shape Nx(M+1)
                        it.index_copy_(0, sample_ind,
                                       torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                        it = Variable(it, requires_grad=False)
                else:
                    it = seq_idx[:, i].clone().long()   # word_index
                    # break if all the sequences end
                if i >= 1 and seq_idx[:, i].data.sum() == 0:
                    break
                # event_idx:(1,1124)  clip_idx:(1,max_event_len,512) clip_mask_idx:(1,max_event_len) state:[(1,1,512),(1,1,512)]
                output, state = self.get_logprobs_state(it, event_idx, clip_idx, clip_mask_idx, state)  # output:(1,5748) state:[(1,1,512),(1,1,512)]

                interest = (seq_len_idx == i + 2)

                if interest.sum() > 0:
                    end_id = interest.nonzero().squeeze(1)
                    last_sent_state[end_id] = state[0][
                        -1, end_id]  # state[0].shape = (num_layer, batch_size, rnn_size))
                outputs.append(output)  # output: (batch, vocab_size+1); outputs: (cap_seq_len, batch, vocab_size+1 )
            para_outputs.append(
                torch.stack(outputs, 0))  # para_outputs: (Prop_N, cap_seq_len, 1, 5748)
        ###### 将para_outputs放入固定大小的tensor中 ######
        para_output_tensor = clip.new_zeros(self.opt.batch_size, eseq_len, seq.size(1) - 1, eseq_num,
                                            self.vocab_size + 1)  # (1,Prop_N, max_sent_len, 1, 5748)
        para_output_tensor = para_output_tensor.squeeze(0)  # (Prop_N, max_sent_len, 1, 5748)

        for i in range(para_output_tensor.shape[0]):
            para_output_tensor[i, :len(para_outputs[i])] = para_outputs[i]
        para_output_tensor = para_output_tensor.permute(2, 0, 1,
                                                        3)  # (1,Prop_N,max_sent_len,5748)

        return para_output_tensor  # (1, Prop_N, max_sent_len-1, 5748)

    def get_logprobs_state(self, it, event, clip, clip_mask, state):
        xt = self.embed(it)  # (1,512)
        # 这里调用了ShowAttendTellCore的forward函数
        output, state = self.core(xt, event, clip, clip_mask, state)  # (1,512), [(1,1,512),(1,1,512)]
        logprobs = F.log_softmax(self.logit(self.dropout(output)), dim=1)  # (1,512)-->(1,5748) 
        return logprobs, state  # (1,5748)  [(1,1,512),(1,1,512)]

    ##### sample是评估和强化训练的时候使用的 #####
    def sample(self, event, clip, clip_mask, event_seq_idx, event_feat_expand=False, opt={}):
        # event:(Prop_N,1124) clip:(Prop_N,max_event_len,512) clip_mask:(Prop_N, max_event_len)  event_seq_idx:(1,Prop_N)
        # event_feat_expand = True
        sample_max = opt.get('sample_max', 1)  # 1  贪心法生成句子，不使用集束搜索
        temperature = opt.get('temperature', 1.0)  # 1.0

        eseq_num, eseq_len = event_seq_idx.shape  # 1, Prop_N

        para_state = self.init_hidden(eseq_num)  # [(1,1,512),(1,1,512)]
        last_sent_state = clip.new_zeros(eseq_num, self.rnn_size)  # (1,512)

        para_seqLogprobs = []
        para_seq = []

        if event is None:
            event = (clip * clip_mask.unsqueeze(2)).sum(1) / clip_mask.sum(1, keepdims=True)
        if not event_feat_expand:
            assert len(event.shape) == 2
        else:
            event = event.reshape(eseq_num, eseq_len, event.shape[-1])  # (1,Prop_N,1124)

        for idx in range(eseq_len):
            event_idx = event[event_seq_idx[:, idx]] if not event_feat_expand else event[:, idx]  # (1,1124)
            clip_idx = clip[event_seq_idx[:, idx]]  # (1, max_event_len, 512)
            clip_mask_idx = clip_mask[event_seq_idx[:, idx]]  # (1, max_event_len)

            prev_state_proj = self.global_proj(last_sent_state)  # (1,512)-->(1,512)
            event_proj = self.local_proj(event_idx)  # (1,1536)-->(1,512)
            gate_input = torch.cat((para_state[0][-1], prev_state_proj, event_proj), 1)   # (1,1536)
            gate = self.gate_layer(self.gate_drop(gate_input)) # (1,1536)-->(1,512)  
            gate = torch.cat((gate, 1 - gate), dim=1)  # (1,1024)
            sent_rnn_input = torch.cat((prev_state_proj, event_proj), dim=1)
            sent_rnn_input = sent_rnn_input * gate  
            _, para_state = self.sent_rnn(sent_rnn_input.unsqueeze(0), para_state)  # [(1,1,512),(1,1,512)]

            para_c, para_h = para_state    # (1,1,512),(1,1,512)
            num_layers, batch_size, para_dim = para_h.size()  # 1, 1, 512
            init_h = self.para_transfer_layer(para_h[-1]).reshape(self.num_layers, batch_size, para_dim)  # (1,1,512)
            state = (init_h, init_h)   # [(1,1,512),(1,1,512)]

            seq = []
            seqLogprobs = []
            last_sent_state = clip.new_zeros(eseq_num, self.rnn_size)  # (1,512)

            for t in range(self.max_caption_len + 1):
                if t == 0:  # input <bos>
                    it = clip.new_zeros(eseq_num).long()  # 初始单词
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = logprobs.gather(1, Variable(it,
                                                                 requires_grad=False))  # gather the logprobs at sampled positions
                    it = it.view(-1).long()  # and flatten indices for downstream processing

                logprobs, state = self.get_logprobs_state(it, event_idx, clip_idx, clip_mask_idx, state)   # (1,5748)  [(1,1,512),(1,1,512)] 

                if t >= 1:
                    # stop when all finished
                    if t == 1:
                        unfinished = it > 0     # it>0表示句子还没结束
                        interest = ~unfinished  # interest啥意思？
                    else:
                        new_unfinished = unfinished & (it > 0)
                        interest = new_unfinished ^ unfinished
                        unfinished = new_unfinished
                    it = it * unfinished.type_as(it)
                    seq.append(it)  # seq[t] the input of t+2 time step
                    seqLogprobs.append(sampleLogprobs.view(-1))
                    if unfinished.sum() == 0:
                        break
                    if interest.sum() > 0:
                        end_id = interest.nonzero().squeeze(1)
                        last_sent_state[end_id] = state[0][-1, end_id]
            if len(seq) == 0:
                seq.append(clip.new_zeros(1).long())
                seqLogprobs.append(clip.new_zeros(1, requires_grad=True) - 10)

            para_seqLogprobs.append(torch.stack(seqLogprobs, 0))  # para_seqLogprobs: (eseq_len, seq_len, batch_size)
            para_seq.append(torch.stack(seq, 0))  # para_seq： (eseq_len, seq_len, batch_size)

        max_len = max([p.shape[0] for p in para_seqLogprobs])  # max_pred_sent_len
        para_seqLogprobs_tensor = clip.new_zeros(eseq_len, max_len, eseq_num)   # (Prop_N, max_pred_sent_length, batch_size)
        para_seq_tensor = clip.new_zeros(eseq_len, max_len, eseq_num).int() # (Prop_N, max_pred_sent_length, batch_size)

        for i in range(para_seq_tensor.shape[0]):
            para_seqLogprobs_tensor[i, :len(para_seqLogprobs[i])] = para_seqLogprobs[i]
            para_seq_tensor[i, :len(para_seq[i])] = para_seq[i]
        para_seqLogprobs_tensor = para_seqLogprobs_tensor.permute(2, 0, 1)
        para_seq_tensor = para_seq_tensor.permute(2, 0, 1)
        return para_seq_tensor, para_seqLogprobs_tensor # (batch_size, Prop_N, max_pred_sent_length), (batch_size, Prop_N, max_pred_sent_length)

####  word RNN ########
"""
The word RNN is implemented
as an attention-enhanced RNN, which adaptively select the
salient frames within the proposal pi for word prediction.
相当于选取clip特征中和state关系度高的部分
"""
class ShowAttendTellCore(nn.Module):

    def __init__(self, opt):
        super(ShowAttendTellCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size   #512

        self.rnn_size = opt.rnn_size   # 512
        self.num_layers = opt.num_layers  # 1
        self.drop_prob_lm = opt.drop_prob # 0.5
        # self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.clip_context_dim  # 512
        self.att_hid_size = opt.att_hid_size       # 512

        self.opt = opt
        self.wordRNN_input_feats_type = opt.wordRNN_input_feats_type  # 'C'
        self.input_dim = self.decide_input_feats_dim()  # 512
        # word RNN
        self.rnn = nn.LSTM(self.input_encoding_size + self.input_dim,
                                                      self.rnn_size, self.num_layers, bias=False,
                                                      dropout=self.drop_prob_lm)   # 1024-->512

        if self.att_hid_size > 0:
            self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)  # 512-->512
            self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)   # 512-->512
            self.alpha_net = nn.Linear(self.att_hid_size, 1)  # 512-->1
        else:
            self.ctx2att = nn.Linear(self.att_feat_size, 1)
            self.h2att = nn.Linear(self.rnn_size, 1)

    # 判定单词RNN输入特征的类型，'E'代表event_context_dim=1124  'C'代表clip_context_dim=512
    def decide_input_feats_dim(self):
        dim = 0
        if 'E' in self.wordRNN_input_feats_type:
            dim += self.opt.event_context_dim
        if 'C' in self.wordRNN_input_feats_type:
            dim += self.opt.clip_context_dim
        return dim

    def get_input_feats(self, event, att_clip):  # event:(1,1124)  att_clip:(1,512)
        input_feats = []
        if 'E' in self.wordRNN_input_feats_type:
            input_feats.append(event)
        if 'C' in self.wordRNN_input_feats_type:  # wordRNN_input_feats_type='C'
            input_feats.append(att_clip)

        input_feats = torch.cat(input_feats, 1)
        return input_feats
    # xt:(1,512) event:(1,1124) clip:(1,max_event_len,512) clip_mask:(1,max_event_len) state:[(1,1,512),(1,1,512)]，其中state中第一个为cell state,第二个是hidden state
    ##### 这里主要是计算clip特征和state的关系，相当于关注clip特征中和state关系比较高的部分 ########
    """
    The word RNN is implemented
    as an attention-enhanced RNN, which adaptively select the
    salient frames within the proposal pi for word prediction 
    """
    def forward(self, xt, event, clip, clip_mask, state):  # event的特征在这里没有用到
        att_size = clip.numel() // clip.size(0) // self.opt.clip_context_dim  # max_event_len
        
        #### 通过clip计算得到的attn值 #####
        att = clip.view(-1, self.opt.clip_context_dim)  # (1*max_event_len,512)

        att = self.ctx2att(att)  # (max_event_len,512)
        att = att.view(-1, att_size, self.att_hid_size)  # (1, max_event_len,512)
        
        #### 通过state计算得到的attn值 #####
        att_h = self.h2att(state[0][-1])  # (1,512)
        att_h = att_h.unsqueeze(1).expand_as(att)  # (1, max_event_len,512)
        
        dot = att + att_h  # (1, max_event_len,512)
        dot = torch.tanh(dot)  # (1, max_event_len,512)
        dot = dot.view(-1, self.att_hid_size) # (1*max_event_len,512)
        dot = self.alpha_net(dot)  # (1*max_event_len,1)
        dot = dot.view(-1, att_size)  # (1, max_event_len)

        weight = F.softmax(dot, dim=1)  # (1, max_event_len)
        if clip_mask is not None: 
            weight = weight * clip_mask.view(-1, att_size).float()  # (1, max_event_len)
            weight = weight / (weight.sum(1, keepdim=True) + 1e-6)  # (1, max_event_len)

        att_feats_ = clip.view(-1, att_size, self.att_feat_size)  # (1, max_event_len, 512)
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # (1, 512)

        input_feats = self.get_input_feats(event, att_res)  #(1,512)   event特征的特征并没有用到
        output, state = self.rnn(torch.cat([xt, input_feats], 1).unsqueeze(0), state)  # output:(1,1,512)  state:[(1,1,512),(1,1,512)]
        return output.squeeze(0), state   # (1,512), [(1,1,512),(1,1,512)]


# CMG_HRNN在这里是父类
class ShowAttendTellModel(CMG_HRNN):
    def __init__(self, opt):
        super(ShowAttendTellModel, self).__init__(opt)
        self.core = ShowAttendTellCore(opt)



if __name__ == '__main__':
    import opts

    opt = opts.parse_opts()
    opt.wordRNN_input_feats_type = 'C'
    opt.clip_context_type = 'CC+CH'
    opt.vocab_size = 5767
    opt.max_caption_len = 32
    opt.clip_context_dim = 1012

    model = ShowAttendTellModel(opt)

    video = torch.randn(3, 500)
    event = torch.randn(3, 512)
    clip = torch.randn(3, 20, 1012)
    clip_mask = torch.ones(3, 20)
    seq = torch.randint(0, 27, (3, 32))
    # out = model(video, event, clip, clip_mask, seq)
    out = model.sample(video, event, clip, clip_mask)
    pass

