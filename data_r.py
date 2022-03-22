import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# 辅助类
class Sequence():
    attrs = ['occur_t', 'marks', 'seq_t0', 'seq_features', 'seq_id']

    def __init__(self, occur_t, marks, seq_t0=0, seq_features=None, seq_id=None):
        assert len(occur_t) == len(marks)
        self.occur_t = occur_t
        self.marks = marks
        self.seq_t0 = seq_t0
        self.seq_features = seq_features
        self.seq_id = seq_id

    def __repr__(self):
        info_l = []
        for attr in self.attrs:
            val = getattr(self, attr)
            info_l.append(f"{attr}: {val}")
        return '\n'.join(info_l)

    def __len__(self):
        return len(self.occur_t)

    def trunc(self, T, abs_T=False):
        occur_T = self.occur_t
        if abs_T:
            occur_T = self.occur_t + self.seq_t0
        idx = occur_T <= T
        kwargs = {attr: getattr(self, attr) for attr in self.attrs}
        kwargs['occur_t'] = kwargs['occur_t'][idx]
        kwargs['marks'] = kwargs['marks'][idx]
        return Sequence(**kwargs)
    
    def get_pred_target(self, T_start, n_period=5, abs_T=False):
        occur_T = self.occur_t
        if abs_T:
            occur_T = self.occur_t + self.seq_t0
        idx_l = np.arange(len(self))
        before_start = occur_T <= T_start
        hist_idx = idx_l[before_start]
        T_last = occur_T[hist_idx[-1]]
        
        inter_times = np.arange(n_period) + T_start - T_last + 1
        period_counts = np.zeros((n_period, 2), dtype=np.int64)
        for i in range(n_period):
            T_end = T_start + i + 1
            before_end = occur_T <= T_end
            to_pred_idx = idx_l[(~before_start)&before_end]
            if len(to_pred_idx) > 0:
                period_counts[i] = self.marks[to_pred_idx].sum(axis=0)

        return inter_times, period_counts
        
    def to_pd(self):
        return pd.DataFrame({
            'occur_t_rel': self.occur_t, 
            'occur_t_abs': self.occur_t + self.seq_t0, 
            'n_self_count': self.marks[:, 0], 
            'n_nonself_count': self.marks[:, 1]})
        
class PatentDataset():

    def __init__(self, file_path='./data/patent3.txt', seq_l=None):

        with open(file_path, 'r') as f:
            line_l = f.readlines()
        # ---- generate sequences ----
        '''
        every four lines corresponds to one patent
        consider the following example, 
            4063274,3,4,5,5,5,11
            4063274,3,4,6,6,6,6,7,10,12,12,13,14,15,16,21,22
            4063274,1977
            4063274,0.00778210116732,0.00389105058366,1.0,0.0194552529183, ....
        + the first comma-seperated number in each line is patent id
        + occurrence time is year
        line 1 records the occurrence time of self-citation events 
            (cited by another patent owned by the same firm)
        line 2 records the occurrence time of non self-citation events
            (cited by a patent owned by other firms)
        line 3 records the publish year of the patent
        line 4 records numeric features of the owner firm
        '''
        if seq_l is not None:
            self.seq_l = seq_l
        else:
            seq_l = []
            # for i in range(0, 400, 4):
            for i in tqdm(range(0, len(line_l), 4)):
                one_pat = line_l[i:(i + 4)]
                one_pat = [line.strip('\n').split(',') for line in one_pat]
                # all four lines should have the same patent id
                assert all(line[0] == one_pat[0][0] for line in one_pat)
                patent_id = int(one_pat[0][0])
                '''
                at a particular year, multiple citations could occur, so
                to deal with this kind of simultaneous event,
                we store event marks as an array `mark` which is of shape [seq_len, 2]
                and occurrence times as an array `occur_times` which is of shape [seq_len,]
                + `mark[i, 0]` stores number of self-citations occurred in year `occur_times[i]`
                + `mark[i, 1]` stores number of non self-citations occurred in year `occur_times[i]`
                '''
                seq_a = one_pat[0][1:]
                seq_b = one_pat[1][1:]
                occur_times = np.array(seq_a + seq_b, dtype=np.float32)
                marks = np.repeat(
                    a=np.array([0, 1], dtype=np.int32),
                    repeats=[len(seq_a), len(seq_b)])
                one_pd = pd.DataFrame({
                    'occur_times': occur_times, 'mark': marks})
                count_ps = one_pd.groupby(['mark', 'occur_times'])['mark'].count()
                count_ps.name = 'n'
                one_pd = count_ps.reset_index()
                one_pd = one_pd.pivot_table(index='occur_times', values='n', columns='mark', fill_value=0)
                one_pd = one_pd.rename({0: 'self', 1: 'non-self'}, axis=1).reset_index()
                occur_times = one_pd['occur_times'].values
                marks = one_pd[['self', 'non-self']].values

                t0 = float(one_pat[2][1])
                features = np.array(one_pat[3][1:], dtype=np.float32)
                seq = Sequence(occur_times, marks, t0, features, patent_id)
                seq_l.append(seq)
            '''
            if you print a `seq`, the output should look like
                occur_t: [ 3.  4.  5.  6.  7. 10. 11. 12. 13. 14. 15. 16. 21. 22.]
                marks: [[1 1]
                [1 1]
                [3 0]
                ...
                [0 1]]
                seq_t0: 1977.0
                seq_features: [0.0077821  0.00389105  ...]
                seq_id: 4063274
            '''
            self.seq_l = seq_l
            self.one_pd = one_pd
            
        self.id2idx_pd = pd.Series(index=[seq.seq_id for seq in self.seq_l], data=np.arange(len(self)))

    def __getitem__(self, i):
        return self.seq_l[i]

    def __len__(self):
        return len(self.seq_l)

    def __repr__(self):
        return f'{len(self)} sequences'

    @staticmethod
    def collate_batch(batch):
        # https://androidkt.com/create-dataloader-with-collate_fn-for-variable-length-input-in-pytorch/

        max_len = max([len(seq) for seq in batch])
        batch_size = len(batch)
        # prepare an empty container to hold the batch of interarrival times and marks
        # the 'b' prefix means batch
        # to indicate wether a timestamp is padded or not, also prepare a mask tensor
        _shape = (batch_size, max_len)  # interarrival times
        b_inter_t = np.zeros(shape=_shape, dtype=np.float32)
        b_marks = np.zeros(shape=(batch_size, max_len, 2), dtype=np.int32)
        b_mask = np.ones(shape=_shape, dtype=np.float32)  # 1 means we should keep it
        b_features = np.array([seq.seq_features for seq in batch], dtype=np.float32)
        b_ids = np.array([seq.seq_id for seq in batch], dtype=np.int64)

        for i, seq in enumerate(batch):
            _end = len(seq)
            # first event's interarrival time is its occurrence time
            b_inter_t[i, 0] = seq.occur_t[0]
            b_inter_t[i, 1:_end] = np.diff(seq.occur_t)
            b_marks[i, :_end] = seq.marks
            b_mask[i, _end:] = 0

        b_inter_t = torch.tensor(b_inter_t)
        b_marks = torch.tensor(b_marks)
        b_mask = torch.tensor(b_mask)
        b_features = torch.tensor(b_features)
        b_features = b_features.unsqueeze(1).expand(-1, max_len, -1)
        
        return b_inter_t, b_marks, b_mask, b_features, b_ids






