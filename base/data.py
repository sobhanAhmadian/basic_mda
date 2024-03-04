import abc

import torch

from .utils import logging as base_logger

logger = base_logger.getLogger(__name__)


class Data(abc.ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def extend(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def subset(self, indices) -> tuple:
        raise NotImplementedError


class TrainTestSplit(abc.ABC):
    @abc.abstractmethod
    def split(self, train_indices, test_indices):  # Return Train and Test Data
        logger.info(f'splitting data')
        logger.info(f'train indices : {train_indices}')
        logger.info(f'test indices : {test_indices}')
        raise NotImplementedError


class SimplePytorchData(Data):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        logger.info(f'Initializing SimplePytorchData with X shape : {X.shape} and y shape : {y.shape}')
        self.X = X
        self.y = y

    def extend(self, X: torch.Tensor, y: torch.Tensor):
        if self.X is None or self.y is None:
            self.X = X
            self.y = y
        else:
            self.X = torch.cat((self.X, X), 0)
            self.y = torch.cat((self.y, y), 0)

    def subset(self, indices) -> tuple:
        return self.y[indices], self.y[indices]


class SimplePytorchDataTrainTestSplit(TrainTestSplit):

    def __init__(self, simple_data):
        logger.info(f'Initializing SimplePytorchDataTrainTestSplit')
        self.X = simple_data.X
        self.y = simple_data.y

    def split(self, train_indices, test_indices):
        train_data = SimplePytorchData(self.X[train_indices], self.y[train_indices])
        test_data = SimplePytorchData(self.X[test_indices], self.y[test_indices])
        return train_data, test_data
