import abc

from .utils import logging as base_logger

logger = base_logger.getLogger(__name__)


class FeatureExtractor(abc.ABC):

    @abc.abstractmethod
    def build(self, **kwargs):
        logger.info(f'Building Feature Extractor')
        raise NotImplementedError
