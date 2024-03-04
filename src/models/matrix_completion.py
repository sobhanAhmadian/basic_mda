from base import BaseModel
from base import ModelFactory
from src.config import MatrixDecomposerConfig
from src.methods import MDFeatureContainer
from src.utils import prj_logger

logger = prj_logger.getLogger(__name__)


class MDMDAClassifier(BaseModel):

    def __init__(self, config: MatrixDecomposerConfig):
        super().__init__(None)
        logger.info(
            f'Initializing MCMDAClassifier')

        self.fe = MDFeatureContainer(config)

    def destroy(self):
        pass

    def predict(self, microbe_id, disease_id):
        m = self.fe.microbe_index(microbe_id)
        d = self.fe.disease_index(disease_id)
        return self.fe.interaction[m][d]

    def summary(self):
        pass


class MDMDAClassifierFactory(ModelFactory):

    def __init__(self, config: MatrixDecomposerConfig) -> None:
        super().__init__()

        logger.info(f'Initializing MCMDAClassifierFactory')

        self.model_config = config

    def make_model(self) -> MDMDAClassifier:
        return MDMDAClassifier(self.model_config)
