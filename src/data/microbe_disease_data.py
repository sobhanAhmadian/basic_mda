import pandas as pd

from base import Data, TrainTestSplit
from src.utils import prj_logger

logger = prj_logger.getLogger(__name__)


class MicrobeDiseaseAssociationData(Data):

    def __init__(self, associations, **kwargs) -> None:
        super().__init__(**kwargs)
        self.associations = associations

    def extend(self, associations):
        return pd.concat([self.associations, associations])

    def subset(self, indices) -> tuple:
        return self.associations.iloc[indices]


class MicrobeDiseaseAssociationTrainTestSpliter(TrainTestSplit):

    def __init__(self, associations):
        logger.info(f'Initializing MicrobeDiseaseAssociationTrainTestSpliter')
        self.associations = associations

    def split(self, train_indices, test_indices):
        train_data = MicrobeDiseaseAssociationData(self.associations.iloc[train_indices])
        test_data = MicrobeDiseaseAssociationData(self.associations.iloc[test_indices])
        return train_data, test_data
