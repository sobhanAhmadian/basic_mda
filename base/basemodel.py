import abc

from .utils import logging as base_logger

logger = base_logger.getLogger(__name__)


class BaseModel(abc.ABC):
    def __init__(self, model_config) -> None:
        self.model_config = model_config
        self.model = None

    @abc.abstractmethod
    def destroy(self):
        logger.info(f'Model {self.model} has been deleted')
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def summary(self):
        raise NotImplementedError


class ModelFactory(abc.ABC):
    @abc.abstractmethod
    def make_model(self) -> BaseModel:
        raise NotImplementedError
