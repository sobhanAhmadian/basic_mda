import abc

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from base import BaseModel, ModelFactory
from src.config import SimpleClassifierConfig, MatrixDecomposerConfig, SklearnClassifierConfig
from src.methods import MDFeatureExtractor, MatrixDummyFeatureExtractor, GaussianSimilarityFeatureExtractor, \
    JaccardSimilarityFeatureExtractor
from src.utils import prj_logger

logger = prj_logger.getLogger(__name__)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1, num_layers=3, dropout=0.1):
        super().__init__()

        logger.info(
            f'''Initial SimpleMLP with {input_dim} input dimension, {hidden_dim} hidden dimension, {output_dim} 
            output dimension, {num_layers} layers and with {dropout} dropout''')

        modules = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for i in range(num_layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        modules.append(nn.Linear(hidden_dim, output_dim))

        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)


class SimpleMDAClassifier(BaseModel):

    def __init__(self, model_config: SimpleClassifierConfig):
        super().__init__(model_config)
        logger.info(f'Initializing SimpleMDAClassifier with model : {model_config.model_name}')
        self.classifier = SimpleMLP(**model_config.get_main_configuration()).to(model_config.device)

    def destroy(self):
        del self.classifier

    def predict(self, X):
        return self.classifier(X).detach()

    def summary(self):
        pass


class SimpleMDAClassifierFactory(ModelFactory):
    def __init__(self, model_config):
        logger.info(f'Initializing SimpleMDAClassifierFactory with model : {model_config.model_name}')
        self.model_config = model_config

    def make_model(self) -> BaseModel:
        return SimpleMDAClassifier(model_config=self.model_config)


class MatrixFeatureBasedClassifier(BaseModel, abc.ABC):

    def __init__(self, model_config, **kwargs) -> None:
        super().__init__(model_config)

        self.classifier = None
        self.fe = self._get_feature_extractor(**kwargs)

    def destroy(self):
        del self.fe
        del self.classifier

    def summary(self):
        pass

    @abc.abstractmethod
    def _get_feature_extractor(self, **kwargs):
        raise NotImplementedError


class MatrixFeatureBasedMDAClassifier(MatrixFeatureBasedClassifier, abc.ABC):

    def __init__(self, model_config: SimpleClassifierConfig, **kwargs) -> None:
        super().__init__(model_config, **kwargs)
        logger.info(f'Initializing MDFeatureBasedMDAClassifier with model : {model_config.model_name}')

        self.device = model_config.device
        self.classifier = SimpleMDAClassifier(model_config)

    def build(self, associations):
        self.fe.build(associations)

    def predict(self, m, d):
        f = torch.tensor(self.fe.pair_feature(m, d), dtype=torch.float32).to(self.device).reshape(1, -1)
        return self.classifier.predict(f)


class MatrixFeatureBasedSklearnClassifier(MatrixFeatureBasedClassifier, abc.ABC):

    def __init__(self, model_config: SklearnClassifierConfig, **kwargs) -> None:
        super().__init__(model_config, **kwargs)
        logger.info(f'Initializing MatrixFeatureBasedSklearnClassifier with model : {model_config.model_name}')

        self.classifier = None
        if model_config.classifier == 'RF':
            self.classifier = RandomForestClassifier(n_estimators=model_config.n_estimators,
                                                     criterion=model_config.criterion,
                                                     max_depth=model_config.max_depth,
                                                     random_state=model_config.random_state)
        elif model_config.classifier == 'AdaBoost':
            self.classifier = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=model_config.max_depth),
                                                 n_estimators=model_config.n_estimators,
                                                 random_state=model_config.random_state)
        elif model_config.classifier == 'Logistic':
            self.classifier = LogisticRegression(random_state=model_config.random_state, penalty=model_config.penalty,
                                                 C=model_config.C, solver=model_config.solver)
        elif model_config.classifier == 'DecisionTree':
            self.classifier = DecisionTreeClassifier(max_depth=model_config.max_depth)

    def build(self, associations):
        self.fe.build(associations)

        logger.info('building data for classifier')
        y = np.array(associations['increased'].tolist())
        X = self.fe.all_pair_features(associations)
        logger.info(f'shape of y : {y.shape}, shape of X : {X.shape}')

        logger.info(f'fitting classifier : {self.model_config.classifier}')
        self.classifier.fit(X, y)

    def predict(self, m, d):
        f = torch.tensor(self.fe.pair_feature(m, d), dtype=torch.float32).numpy().reshape(1, -1)
        return self.classifier.predict_proba(f)[1]


class MDFeatureBasedMDAClassifier(MatrixFeatureBasedMDAClassifier):

    def __init__(self, model_config: SimpleClassifierConfig, md_config: MatrixDecomposerConfig) -> None:
        super().__init__(model_config, md_config=md_config)

    def _get_feature_extractor(self, **kwargs):
        return MDFeatureExtractor(config=kwargs['md_config'])


class DummyFeatureBasedMDAClassifier(MatrixFeatureBasedMDAClassifier):

    def __init__(self, model_config: SimpleClassifierConfig, microbe_ids, disease_ids) -> None:
        super().__init__(model_config, microbe_ids=microbe_ids, disease_ids=disease_ids)

    def _get_feature_extractor(self, **kwargs):
        return MatrixDummyFeatureExtractor(microbe_ids=kwargs['microbe_ids'], disease_ids=kwargs['disease_ids'])


class SimilarityFeatureBasedMDAClassifier(MatrixFeatureBasedMDAClassifier, abc.ABC):

    def __init__(self, model_config: SimpleClassifierConfig, microbe_ids, disease_ids) -> None:
        super().__init__(model_config, microbe_ids=microbe_ids, disease_ids=disease_ids)


class SimilarityFeatureBasedSklearnClassifier(MatrixFeatureBasedSklearnClassifier, abc.ABC):

    def __init__(self, model_config: SklearnClassifierConfig, microbe_ids, disease_ids) -> None:
        super().__init__(model_config, microbe_ids=microbe_ids, disease_ids=disease_ids)


class GaussianSimilarityFeatureBasedMDAClassifier(SimilarityFeatureBasedMDAClassifier):

    def _get_feature_extractor(self, **kwargs):
        return GaussianSimilarityFeatureExtractor(microbe_ids=kwargs['microbe_ids'], disease_ids=kwargs['disease_ids'])


class JaccardSimilarityFeatureBasedMDAClassifier(SimilarityFeatureBasedMDAClassifier):

    def _get_feature_extractor(self, **kwargs):
        return JaccardSimilarityFeatureExtractor(microbe_ids=kwargs['microbe_ids'], disease_ids=kwargs['disease_ids'])


class JaccardSimilarityFeatureBasedSklearnClassifier(SimilarityFeatureBasedSklearnClassifier):

    def _get_feature_extractor(self, **kwargs):
        return JaccardSimilarityFeatureExtractor(microbe_ids=kwargs['microbe_ids'], disease_ids=kwargs['disease_ids'])


class MatrixFeatureBasedMDAClassifierFactory(ModelFactory, abc.ABC):
    def __init__(self, model_config: SimpleClassifierConfig):
        self.model_config = model_config


class MatrixFeatureBasedSklearnClassifierFactory(ModelFactory, abc.ABC):
    def __init__(self, model_config: SklearnClassifierConfig):
        self.model_config = model_config


class MDFeatureBasedMDAClassifierFactory(MatrixFeatureBasedMDAClassifierFactory):
    def __init__(self, model_config: SimpleClassifierConfig, md_config: MatrixDecomposerConfig):
        super().__init__(model_config)
        logger.info(f'Initializing MDFeatureBasedMDAClassifierFactory')
        self.md_config = md_config

    def make_model(self) -> BaseModel:
        return MDFeatureBasedMDAClassifier(model_config=self.model_config, md_config=self.md_config)


class DummyFeatureBasedMDAClassifierFactory(MatrixFeatureBasedMDAClassifierFactory):
    def __init__(self, model_config: SimpleClassifierConfig, microbe_ids, disease_ids):
        super().__init__(model_config)
        logger.info(f'Initializing DummyFeatureBasedMDAClassifierFactory')
        self.microbe_ids = microbe_ids
        self.disease_ids = disease_ids

    def make_model(self) -> BaseModel:
        return DummyFeatureBasedMDAClassifier(model_config=self.model_config, microbe_ids=self.microbe_ids,
                                              disease_ids=self.disease_ids)


class SimilarityFeatureBasedMDAClassifierFactory(MatrixFeatureBasedMDAClassifierFactory, abc.ABC):
    def __init__(self, model_config: SimpleClassifierConfig, microbe_ids, disease_ids):
        super().__init__(model_config)
        logger.info(f'Initializing SimilarityFeatureBasedMDAClassifierFactory')
        self.microbe_ids = microbe_ids
        self.disease_ids = disease_ids


class SimilarityFeatureBasedSklearnClassifierFactory(MatrixFeatureBasedSklearnClassifierFactory, abc.ABC):
    def __init__(self, model_config: SklearnClassifierConfig, microbe_ids, disease_ids):
        super().__init__(model_config)
        logger.info(f'Initializing SimilarityFeatureBasedSklearnClassifierFactory')
        self.microbe_ids = microbe_ids
        self.disease_ids = disease_ids


class GaussianSimilarityFeatureBasedMDAClassifierFactory(SimilarityFeatureBasedMDAClassifierFactory):

    def make_model(self) -> BaseModel:
        return GaussianSimilarityFeatureBasedMDAClassifier(model_config=self.model_config, microbe_ids=self.microbe_ids,
                                                           disease_ids=self.disease_ids)


class JaccardSimilarityFeatureBasedMDAClassifierFactory(SimilarityFeatureBasedMDAClassifierFactory):

    def make_model(self) -> BaseModel:
        return JaccardSimilarityFeatureBasedMDAClassifier(model_config=self.model_config, microbe_ids=self.microbe_ids,
                                                          disease_ids=self.disease_ids)


class JaccardSimilarityFeatureBasedSklearnClassifierFactory(SimilarityFeatureBasedSklearnClassifierFactory):

    def make_model(self) -> BaseModel:
        return JaccardSimilarityFeatureBasedSklearnClassifier(model_config=self.model_config,
                                                              microbe_ids=self.microbe_ids,
                                                              disease_ids=self.disease_ids)
