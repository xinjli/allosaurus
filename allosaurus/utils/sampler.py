import random

class ConcatBatchSampler:

    def __init__(self, datasets, batch_size, shuffle=False):
        self.datasets = datasets

        self.batch_size = batch_size
        self.shuffle = shuffle
        #self.drop_last = drop_last

        self.batches = []

        total_cnt = 0

        for dataset in self.datasets:
            dataset_cnt = len(dataset)

            for i in range(0, dataset_cnt, batch_size):
                self.batches.append(list(range(total_cnt+i, total_cnt+min(i+batch_size, dataset_cnt))))

            total_cnt += dataset_cnt

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        return self.batches[item]

    def __iter__(self):

        batch_lst = list(range(len(self.batches)))

        if self.shuffle:
            random.shuffle(batch_lst)

        for batch_idx in batch_lst:
            yield self.batches[batch_idx]