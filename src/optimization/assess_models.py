import numpy as np

from base import Trainer, Result, OptimizerConfig, Tester, get_prediction_results
from src.data import MicrobeDiseaseAssociationData
from src.models import DiseaseAssessClassifier
from src.utils import prj_logger

logger = prj_logger.getLogger(__name__)


class DiseaseAssessClassifierTrainer(Trainer):
    def train(self, model: DiseaseAssessClassifier, data: MicrobeDiseaseAssociationData,
              config: OptimizerConfig) -> Result:
        logger.info(f'Call Training with {config.exp_name}')

        model.build(data.associations)

        y = np.array(data.associations['increased'].tolist()).reshape(-1).astype(int)
        y_predict = []
        for d in data.associations['disease']:
            y_predict.append(model.predict(d))

        result = get_prediction_results(y, np.array(y_predict), config.threshold)
        logger.info(f'Result on Train Data : {result.get_result()}')
        return result


class DiseaseAssessClassifierTester(Tester):
    def test(self, model: DiseaseAssessClassifier, data: MicrobeDiseaseAssociationData,
             config: OptimizerConfig) -> Result:
        logger.info(f'Call Testing with {config.exp_name}')

        y = np.array(data.associations['increased'].tolist()).reshape(-1).astype(int)
        y_predict = []
        for d in data.associations['disease']:
            y_predict.append(model.predict(d))

        result = get_prediction_results(y, np.array(y_predict), config.threshold)
        logger.info(f'Result on Test Data : {result.get_result()}')
        return result
