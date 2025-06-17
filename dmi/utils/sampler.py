import torch
import torch.utils.data
import numpy as np

######################################################################################
# This class incorporates code from RobertCsordas/transformer_generalization
# Licenced under the MIT Licence: https://mit-license.org/
######################################################################################
class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, length, train_args, replacement=True, seed=None, bsz=None):
        super().__init__()
        self.length = length
        self.train_args = train_args
        self.replacement = replacement
        self.seed = seed
        self.bsz = bsz

    def __iter__(self):
        n = self.length
        if self.replacement:
            while True:
                yield np.random.randint(0, n, dtype=np.int64)
        else:
            i_list = None
            pos = n
            while True:
                if pos >= n:
                    i_list = np.random.permutation(n).tolist()
                    pos = 0

                sample = i_list[pos]
                pos += 1
                yield sample

    def __len__(self):
        return self.length * self.train_args.epochs
