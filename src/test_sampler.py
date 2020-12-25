import numpy as np
import torch
import os


class Test_Sampler:
    def __init__(self, label, n_way, k_shot, query):

        """  Composing an episode

             *** Never change the sampler init, len, iter part ***

        Returns:
            torch.tensor : (n_way * k_shot, 3, h, w)
        """

        self.txt_file1 = open('sheet1', 'r')

        self.text_list1 = []
        for _ in range(200):
            line = self.txt_file1.readline()
            data = np.array(line.split('_'), dtype=np.int32)
            self.text_list1.append(torch.tensor(data))
            if not line: break

        self.txt_file2 = open('sheet2', 'r')
        self.text_list2 = []
        for _ in range(200):
            line = self.txt_file2.readline()
            data = np.array(line.split('_'), dtype=np.int32)
            self.text_list2.append(torch.tensor(data))
            if not line: break

        self.n_way = n_way
        self.k_shot = k_shot
        self.query = query
        self.idx_ss = []
        label = np.array(label)
        self.idxes = []
        for i in range(min(label), max(label)+1):
            idx = np.argwhere(label == i).reshape(-1)
            idx = torch.from_numpy(idx)
            self.idxes.append(idx)

        self.sample_num = 0

    def __len__(self):
        return self.n_way * self.k_shot + self.query

    def __iter__(self):

        episode = []
        query_set = []

        classes = self.text_list1[self.sample_num][1:self.n_way+1]

        jump = self.k_shot + int(self.query / self.n_way)

        idx = 0
        for c in classes:
            l = self.idxes[c]
            pos = self.text_list1[self.sample_num][self.n_way + 1 + idx*jump: self.n_way + 1 + (idx+1)*jump]
            pos_ss, pos_q = torch.tensor(pos[:self.k_shot], dtype=torch.long), torch.tensor(pos[self.k_shot:], dtype=torch.long)
            episode.append(l[pos_ss])
            self.idx_ss.append([c, pos_ss])
            query_set.append(l[pos_q])

            idx += 1

        episode = torch.stack(episode)
        query_set = torch.stack(query_set).view(-1, self.k_shot)
        query_set = torch.reshape(query_set, [-1])
        query_set = query_set[self.text_list2[self.sample_num][1:].long()]
        query_set = torch.reshape(query_set, [-1, self.k_shot])

        episode = torch.cat((episode, query_set), 0).reshape(-1)

        self.sample_num += 1

        yield episode
