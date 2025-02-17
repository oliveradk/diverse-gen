import torch 
import random as rnd
import math

class DivisibleBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_size: int, batch_size: int, shuffle: bool = True):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rnd.Random(42)
        
        # Calculate number of complete batches and total samples needed
        self.num_batches = math.ceil(dataset_size / batch_size)
        self.total_size = self.num_batches * batch_size

    def __iter__(self):
        # Generate indices for the entire dataset
        indices = list(range(self.dataset_size))
        
        if self.shuffle:
            # Shuffle all indices
            self.rng.shuffle(indices)
            
        # If we need more indices to make complete batches,
        # randomly sample from existing indices
        if self.total_size > self.dataset_size:
            extra_indices = self.rng.choices(indices, k=self.total_size - self.dataset_size)
            indices.extend(extra_indices)
            
        assert len(indices) == self.total_size
        return iter(indices)

    def __len__(self):
        return self.total_size