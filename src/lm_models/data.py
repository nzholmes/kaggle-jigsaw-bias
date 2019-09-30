from torch.utils import data
import torch

class LenMatchBatchSampler(data.BatchSampler):
    def __iter__(self):

        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx][0] == 0)
            count_zeros = int(count_zeros / 64)
            if len(buckets[count_zeros]) == 0:  buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, "produced an inccorect number of batches. expected %i, but yielded %i" \
                                     %(len(self), yielded)


def trim_tensors(tsrs):
    x, y = tsrs
    max_len = torch.max(torch.sum((x != 0), 1))
    if max_len > 2:
        tsrs = [x[:, :max_len], y]
    return tsrs