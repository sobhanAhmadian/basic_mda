import abc

import numpy as np

from base import BaseModel
from base import ModelFactory
from src.methods import MatrixFeatureContainer
from src.utils import prj_logger

logger = prj_logger.getLogger(__name__)


class DiseaseAssessClassifier(BaseModel, abc.ABC):

    def __init__(self, microbe_ids, disease_ids, strategy='constant', keep_empty_features=True, fill_value=0):
        super().__init__(None)
        logger.info(
            f'Initializing AssessClassifier')

        self.fe = MatrixFeatureContainer(microbe_ids, disease_ids, strategy=strategy,
                                         keep_empty_features=keep_empty_features, fill_value=fill_value)

    @abc.abstractmethod
    def build(self, associations):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, disease_id):
        raise NotImplementedError

    def destroy(self):
        pass

    def summary(self):
        pass


class DiseaseAssessClassifierFactory(ModelFactory, abc.ABC):

    def __init__(self, microbe_ids, disease_ids, strategy='constant', keep_empty_features=True,
                 fill_value=0) -> None:
        super().__init__()

        logger.info(f'Initializing AssessClassifierFactory')

        self.microbe_ids = microbe_ids
        self.disease_ids = disease_ids
        self.strategy = strategy
        self.keep_empty_features = keep_empty_features
        self.fill_value = fill_value


class D3Classifier(DiseaseAssessClassifier):

    def __init__(self, microbe_ids, disease_ids, strategy='constant', keep_empty_features=True, fill_value=0):
        super().__init__(microbe_ids, disease_ids, strategy, keep_empty_features, fill_value)
        self.d3 = []

    def build(self, associations):
        logger.info('Building D3Classifier')
        self.fe.build(associations)
        interaction = self.fe.interaction
        temp = interaction.sum(axis=0).copy()
        for _ in range(3):
            self.d3.append(self.fe.disease_ids[np.argmax(temp)])
            temp[np.argmax(temp)] = 0
        logger.info(f'd3 : {self.d3}')

    def predict(self, disease_id):
        if disease_id in self.d3:
            return 1.0
        else:
            return np.random.rand()


class D3ClassifierFactory(DiseaseAssessClassifierFactory):

    def make_model(self) -> D3Classifier:
        return D3Classifier(self.microbe_ids, self.disease_ids, self.strategy, self.keep_empty_features,
                            self.fill_value)


class DPosClassifier(DiseaseAssessClassifier):

    def __init__(self, microbe_ids, disease_ids, strategy='constant', keep_empty_features=True, fill_value=0):
        super().__init__(microbe_ids, disease_ids, strategy, keep_empty_features, fill_value)
        self.pos = []

    def build(self, associations):
        logger.info('Building DPosClassifier')
        self.fe.build(associations)
        positive = self.fe.interaction.sum(axis=0).copy()
        total = self.fe.mask.sum(axis=0).copy()
        for i in range(len(positive)):
            if positive[i] > total[i] - positive[i]:
                self.pos.append(self.fe.disease_ids[i])

        logger.info(f'pos : {self.pos}')

    def predict(self, disease_id):
        if disease_id in self.pos:
            return 1.0
        else:
            return 0.0


class DPosClassifierFactory(DiseaseAssessClassifierFactory):

    def make_model(self) -> DPosClassifier:
        return DPosClassifier(self.microbe_ids, self.disease_ids, self.strategy, self.keep_empty_features,
                              self.fill_value)
