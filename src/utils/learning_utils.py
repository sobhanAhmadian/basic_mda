from random import sample


def train_test_sampler(total_samples, train_ratio=0.7):
    train_samples = int(train_ratio * total_samples)
    test_samples = total_samples - train_samples
    train_indices = sample([i for i in range(total_samples)], train_samples)
    test_indices = sample([i for i in range(total_samples)], test_samples)
    return train_indices, test_indices
