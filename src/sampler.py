import numpy as np
import torch


class Sampler:
    def __init__(self, label, n_way, k_shot, query):

        """  Composing an episode

             *** Never change the sampler init, len, iter part ***

        Returns:
            torch.tensor : (n_way * k_shot, 3, h, w)
        """

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

    def __len__(self):
        return self.n_way * self.k_shot + self.query

    def __iter__(self):

        episode = []
        query_set = []
        classes = torch.randperm(len(self.idxes))[:self.n_way]
        classes = torch.sort(classes).values

        for c in classes:
            l = self.idxes[c]
            pos = torch.randperm(len(l))[:self.k_shot + int(self.query / self.n_way)]
            pos_ss, pos_q = pos[:self.k_shot], pos[self.k_shot:]
            episode.append(l[pos_ss])
            self.idx_ss.append([c, pos_ss])
            query_set.append(l[pos_q])

        episode = torch.stack(episode)
        query_set = torch.stack(query_set).view(-1, self.k_shot)
        episode = torch.cat((episode, query_set), 0).reshape(-1)

        yield episode
