import abc

import numpy as np
from sklearn.decomposition import NMF, PCA
from sklearn.impute import SimpleImputer

from base import FeatureExtractor
from src.config import MatrixDecomposerConfig
from src.utils import prj_logger

logger = prj_logger.getLogger(__name__)


class MatrixFeatureContainer(FeatureExtractor):

    def __init__(self, microbe_ids, disease_ids, strategy='mean', keep_empty_features=True, fill_value=0):
        super().__init__()
        logger.info(f'Initializing MatrixFeatureExtractor')

        self.imp = SimpleImputer(missing_values=np.nan, strategy=strategy, keep_empty_features=keep_empty_features,
                                 fill_value=fill_value)
        self.microbe_ids = microbe_ids
        self.disease_ids = disease_ids
        self.interaction = None
        self.mask = None

    def build(self, associations):
        logger.info(f'Calling build with associations : {associations}')
        interaction = np.full((len(self.microbe_ids), len(self.disease_ids)), np.nan)
        logger.info(f'interaction matrix with shape {interaction.shape} has built')
        for i in range(associations.shape[0]):
            m = self.microbe_index(associations.iloc[i]['microbe'])
            d = self.disease_index(associations.iloc[i]['disease'])
            a = associations.iloc[i]['increased']
            interaction[m][d] = a
        self.mask = ~np.isnan(interaction)
        logger.info(f'mask matrix with shape {self.mask.shape} has built. This matrix shows not non elements.')
        self.imp.fit(interaction)
        interaction = self.imp.transform(interaction)
        logger.info(f'interaction has been imputed to delete nans')
        self.interaction = interaction

    def microbe_index(self, m):
        return self.microbe_ids.index(m)

    def disease_index(self, d):
        return self.disease_ids.index(d)


class MDFeatureContainer(MatrixFeatureContainer):

    def __init__(self, config: MatrixDecomposerConfig):
        super().__init__(config.microbe_ids, config.disease_ids)
        logger.info(
            f'Initializing MFFeatureExtractor with model : {config.method_name} and decomposer : {config.decomposer}')

        if config.decomposer == 'NMF':
            self.model = NMF(n_components=config.n_components, init='random', random_state=config.random_state)
        elif config.decomposer == 'PCA':
            self.model = PCA(n_components=config.n_components, random_state=config.random_state)
        else:
            logger.error(f'decomposer {config.decomposer} does not exist')


class MatrixFeatureExtractor(MatrixFeatureContainer, abc.ABC):

    @abc.abstractmethod
    def microbe_feature(self, m):
        raise NotImplementedError

    @abc.abstractmethod
    def disease_feature(self, d):
        raise NotImplementedError

    @abc.abstractmethod
    def pair_feature(self, m, d):
        raise NotImplementedError

    def all_pair_features(self, associations):
        X = []
        for i in range(associations.shape[0]):
            X.append(
                self.pair_feature(m=associations['microbe'].iloc[i], d=associations['disease'].iloc[i]).reshape(1, -1))
        return np.concatenate(X, axis=0)


class MDFeatureExtractor(MatrixFeatureExtractor):

    def __init__(self, config: MatrixDecomposerConfig):
        super().__init__(config.microbe_ids, config.disease_ids)
        logger.info(
            f'Initializing MFFeatureExtractor with model : {config.method_name} and decomposer : {config.decomposer}')

        if config.decomposer == 'NMF':
            self.model = NMF(n_components=config.n_components, init='random', random_state=config.random_state)
        elif config.decomposer == 'PCA':
            self.model = PCA(n_components=config.n_components, random_state=config.random_state)
        else:
            logger.error(f'decomposer {config.decomposer} does not exist')

        self.M = None
        self.D = None

    def build(self, associations):
        super().build(associations)
        self.fit(self.interaction)

    def fit(self, interaction):
        self.M = self.model.fit_transform(interaction)
        self.D = self.model.components_

    def microbe_feature(self, m):
        i = self.microbe_ids.index(m)
        return self.M[i]

    def disease_feature(self, d):
        i = self.disease_ids.index(d)
        return self.D[:, i]

    def pair_feature(self, m, d):
        return np.concatenate((self.microbe_feature(m), self.disease_feature(d)), axis=0)


class MatrixDummyFeatureExtractor(MatrixFeatureExtractor):

    def __init__(self, microbe_ids, disease_ids):
        super().__init__(microbe_ids, disease_ids, strategy='constant', fill_value=0)
        logger.info(f'Initializing MatrixDummyFeatureExtractor')

    def microbe_feature(self, m):
        i = self.microbe_ids.index(m)
        return self.interaction[i]

    def disease_feature(self, d):
        i = self.disease_ids.index(d)
        return self.interaction[:, i]

    def pair_feature(self, m, d):
        m_feature = self.microbe_feature(m)
        d_feature = self.disease_feature(d)
        m_feature[self.disease_index(d)] = 0
        d_feature[self.microbe_index(m)] = 0
        return np.concatenate((m_feature, d_feature), axis=0)


class SimilarityFeatureExtractor(MatrixFeatureExtractor, abc.ABC):

    def __init__(self, microbe_ids, disease_ids):
        super().__init__(microbe_ids, disease_ids, strategy='constant', fill_value=0)
        logger.info(f'Initializing SimilarityFeatureExtractor')

        self.disease_similarity_matrix = None
        self.microbe_similarity_matrix = None

    def microbe_feature(self, m):
        i = self.microbe_ids.index(m)
        return self.microbe_similarity_matrix[i]

    def disease_feature(self, d):
        i = self.disease_ids.index(d)
        return self.disease_similarity_matrix[i]

    def pair_feature(self, m, d):
        return np.concatenate((self.microbe_feature(m), self.disease_feature(d)), axis=0)


class GaussianSimilarityFeatureExtractor(SimilarityFeatureExtractor):

    def build(self, associations):
        super().build(associations)
        self.build_gaussian_similarity()

    def build_gaussian_similarity(self):
        logger.info(f'Building Jaccard similarity for diseases')
        self.disease_similarity_matrix = self._get_gaussian_similarity_for_rows(self.interaction.T)
        logger.info(f'Building Jaccard similarity for microbes')
        self.microbe_similarity_matrix = self._get_gaussian_similarity_for_rows(self.interaction)

    @staticmethod
    def _gaussian_kernel(temp, i, j, sigma):
        return np.exp(-(temp[i, i] + temp[j, j] - 2 * temp[i, j]) / (2 * (sigma ** 2)))

    def _get_gaussian_similarity_for_rows(self, interaction):
        n = interaction.shape[0]
        similarity_matrix = np.zeros((n, n))
        sigma = (pow(np.linalg.norm(self.interaction), 2)) / n

        temp = interaction @ interaction.T
        for i in range(n):
            for j in range(i, n):
                similarity = self._gaussian_kernel(temp, i, j, sigma)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
            if i % 500 == 0:
                logger.info(f'similarities calculated for items : {i:4d}')

        return similarity_matrix


class JaccardSimilarityFeatureExtractor(SimilarityFeatureExtractor):

    def build(self, associations):
        super().build(associations)
        self.build_jaccard_similarity()

    def build_jaccard_similarity(self):
        logger.info(f'Building Jaccard similarity for diseases')
        self.disease_similarity_matrix = self._get_jaccard_similarity_for_rows(self.interaction.T)
        logger.info(f'Building Jaccard similarity for microbes')
        self.microbe_similarity_matrix = self._get_jaccard_similarity_for_rows(self.interaction)

    @staticmethod
    def _get_jaccard_similarity_for_rows(interaction):
        dot_product = interaction @ interaction.T
        logger.info(f'calculating dot product :\n{dot_product[:5, :5]}')
        sum_rows = np.sum(interaction, axis=1)
        logger.info(f'sum rows :\n{sum_rows[:5]}')
        union = np.expand_dims(sum_rows, axis=1) + np.expand_dims(sum_rows, axis=0) - dot_product
        logger.info(f'union :\n{union[:5, :5]}')
        similarity_matrix = dot_product / union
        similarity_matrix = np.nan_to_num(similarity_matrix)
        logger.info(f'similarity matrix :\n{similarity_matrix[:5, :5]}')
        return similarity_matrix
