import numpy as np
import torch
from torch_geometric.data import Data as PyGData
from torch_geometric.utils import negative_sampling

from base import SimpleTrainer, SimpleTester, SimplePytorchData
from base import Trainer, BaseModel, Result, backpropagation, OptimizerConfig, Tester, get_prediction_results
from src.config import Node2VecOptimizerConfig, MDMDAClassifierOptimizerConfig
from src.data import MicrobeDiseaseAssociationData
from src.models import MDAKnowledgeGraphEmbedding
from src.models import MDMDAClassifier
from src.models import MatrixFeatureBasedMDAClassifier, MatrixFeatureBasedSklearnClassifier
from src.utils import prj_logger

logger = prj_logger.getLogger(__name__)


def initial_training(model, config, name):
    logger.info(f'Running {name} with {config.exp_name}')
    logger.info(f'Creating {config.optimizer} with lr : {config.lr}')
    optimizer = config.optimizer(model.model.parameters(), lr=config.lr)
    logger.info(f'moving model to {config.device}')
    model.model.to(config.device)
    model.model.train()
    return optimizer


class Node2vecTrainer(Trainer):
    def train(self, model: BaseModel, data: None, config: Node2VecOptimizerConfig) -> Result:
        optimizer = initial_training(model, config, 'Node2vecTrainer')

        loader = model.model.loader(**config.get_main_configuration())

        total_loss = 0
        running_loss = 0
        logger.info('start batch optimizing')
        for j, (pos_rw, neg_rw) in enumerate(loader, 0):
            loss = model.model.loss(pos_rw.to(config.device), neg_rw.to(config.device))
            backpropagation(loss, optimizer)
            running_loss += loss.item()
            total_loss += loss.item()

            if j % config.report_size == config.report_size - 1:
                loss = running_loss / config.report_size
                logger.info(f'loss: {loss:.4f}    [{j + 1:5d}]')
                running_loss = 0

        total_loss = total_loss / len(loader)
        result = Result()
        result.loss = total_loss

        logger.info(f'Result on Train Data : {result.get_result()}')
        return result


class KGETrainer(Trainer):
    def train(self, model: MDAKnowledgeGraphEmbedding, data: PyGData, config: OptimizerConfig) -> Result:
        optimizer = initial_training(model, config, 'KGETrainer')

        loader = model.model.loader(head_index=data.edge_index[0],
                                    rel_type=data.edge_type,
                                    tail_index=data.edge_index[1],
                                    batch_size=config.batch_size,
                                    shuffle=True)

        for epoch in range(1, config.n_epoch + 1):
            total_loss = total_examples = 0
            for head_index, rel_type, tail_index in loader:
                optimizer.zero_grad()
                loss = model.model.loss(head_index, rel_type, tail_index)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * head_index.numel()
                total_examples += head_index.numel()
            total_loss = total_loss / total_examples
            logger.info(f'Epoch: {epoch:03d}, Loss: {total_loss:.4f}')

        result = Result()
        result.loss = total_loss

        logger.info(f'Result on Train Data : {result.get_result()}')
        return result


class GraphAutoEncoderTrainer(Trainer):
    def train(self, model: BaseModel, data: PyGData, config: OptimizerConfig) -> Result:
        optimizer = initial_training(model, config, 'GraphAutoEncoderTrainer')

        logger.info(f'moving data x and edge_index to {config.device}')
        x = data.x.to(config.device)
        pos_edge_index = data.edge_index.to(config.device)

        loss = 0
        for epoch in range(1, config.n_epoch + 1):
            neg_edge_index = negative_sampling(edge_index=data.edge_index, num_nodes=data.num_nodes,
                                               num_neg_samples=pos_edge_index.size(1), ).to(config.device)

            loss = self.predict(model, config, neg_edge_index, pos_edge_index, x)
            backpropagation(loss, optimizer)

            logger.info(f'loss: {loss:.4f}    [epoch: {epoch:5d}]')

        result = Result()
        result.loss = loss
        return result

    @staticmethod
    def predict(model, config, neg_edge_index, pos_edge_index, x):
        pos_pred = model.model(x, pos_edge_index, pos_edge_index)
        neg_pred = model.model(x, pos_edge_index, neg_edge_index)
        loss = config.criterion(torch.cat([pos_pred, neg_pred], dim=0).to(config.device),
                                torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).to(
                                    config.device))
        return loss


class MDMDAClassifierTrainer(Trainer):
    def train(self, model: MDMDAClassifier, data: MicrobeDiseaseAssociationData,
              config: MDMDAClassifierOptimizerConfig) -> Result:
        logger.info(f'Call Training with {config.exp_name}')

        model.fe.build(data.associations)

        inter_bar = model.fe.interaction.copy()

        result = Result()
        new_mse = np.Inf
        i = 0
        while True:
            i += 1
            W = model.fe.model.fit_transform(inter_bar)
            H = model.fe.model.components_
            temp = W @ H

            pre_mse = new_mse
            new_mse = ((temp * model.fe.mask - model.fe.interaction * model.fe.mask) ** 2).sum()

            logger.info(f'interation {i} mse : {new_mse}')

            if pre_mse > new_mse > config.conv_threshold:
                inter_bar = temp * ~model.fe.mask + inter_bar * model.fe.mask
                result.loss = new_mse
            else:
                logger.info('training finished')
                break

        model.fe.interaction = inter_bar

        return result


class MDMDAClassifierTester(Tester):
    def test(self, model: MDMDAClassifier, data: MicrobeDiseaseAssociationData,
             config: MDMDAClassifierOptimizerConfig) -> Result:
        y_test = data.associations['increased'].tolist()
        logger.info(f'y_test has been built : {y_test[:20]} ...')

        y_predict = []
        for i in range(data.associations.shape[0]):
            y_predict.append(model.predict(data.associations.iloc[i]['microbe'], data.associations.iloc[i]['disease']))
        logger.info(f'y_predict has been built : {y_predict[:20]} ...')

        result = get_prediction_results(y_test=y_test, y_predict=y_predict, threshold=config.threshold)
        logger.info(f'Test Result : {result.get_result()}')
        return result


def _simple_data_from_md(model, data, config):
    X = torch.tensor(model.fe.all_pair_features(data.associations), dtype=torch.float32).to(config.device)
    y = torch.tensor(data.associations['increased'].tolist(), dtype=torch.float32).reshape(-1, 1).to(config.device)
    simple_data = SimplePytorchData(X, y)
    return simple_data


class MatrixFeatureBasedMDAClassifierTrainer(Trainer):
    def train(self, model: MatrixFeatureBasedMDAClassifier, data: MicrobeDiseaseAssociationData,
              config: OptimizerConfig) -> Result:
        logger.info(f'Call Training with {config.exp_name}')

        model.build(data.associations)

        simple_data = _simple_data_from_md(model, data, config)

        result = SimpleTrainer().train(model.classifier, data=simple_data, config=config)
        return result


class MatrixFeatureBasedMDAClassifierTester(Tester):
    def test(self, model: MatrixFeatureBasedMDAClassifier, data: MicrobeDiseaseAssociationData,
             config: OptimizerConfig) -> Result:
        logger.info(f'Call Testing with {config.exp_name}')

        simple_data = _simple_data_from_md(model, data, config)

        result = SimpleTester().test(model.classifier, data=simple_data, config=config)
        return result


class MatrixFeatureBasedSklearnClassifierTrainer(Trainer):
    def train(self, model: MatrixFeatureBasedSklearnClassifier, data: MicrobeDiseaseAssociationData,
              config: OptimizerConfig) -> Result:
        logger.info(f'Call Training with {config.exp_name}')

        model.build(data.associations)

        y = np.array(data.associations['increased'].tolist()).reshape(-1).astype(int)
        X = model.fe.all_pair_features(data.associations)
        y_predict = model.classifier.predict_proba(X)[:, 1].reshape(-1)
        result = get_prediction_results(y, y_predict, config.threshold)
        logger.info(f'Result on Train Data : {result.get_result()}')
        return result


class MatrixFeatureBasedSklearnClassifierTester(Tester):
    def test(self, model: MatrixFeatureBasedSklearnClassifier, data: MicrobeDiseaseAssociationData,
             config: OptimizerConfig) -> Result:
        logger.info(f'Call Testing with {config.exp_name}')

        y = np.array(data.associations['increased'].tolist()).reshape(-1).astype(int)
        X = model.fe.all_pair_features(data.associations)
        y_predict = model.classifier.predict_proba(X)[:, 1].reshape(-1)
        result = get_prediction_results(y, y_predict, config.threshold)
        logger.info(f'Result on Test Data : {result.get_result()}')
        return result
