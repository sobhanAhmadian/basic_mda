import abc
import os.path
import random

import torch
from torch.utils.data import TensorDataset, DataLoader

from .basemodel import BaseModel, ModelFactory
from .data import Data, TrainTestSplit, SimplePytorchData
from .config import OptimizerConfig
from .evaluation import Result, get_prediction_results
from .utils import logging as base_logger

logger = base_logger.getLogger(__name__)


class Trainer(abc.ABC):
    @abc.abstractmethod
    def train(self, model: BaseModel, data: Data, config: OptimizerConfig) -> Result:
        raise NotImplementedError


class Tester(abc.ABC):
    @abc.abstractmethod
    def test(self, model: BaseModel, data: Data, config: OptimizerConfig) -> Result:
        raise NotImplementedError


def cross_validation(k: int, data_size: int, train_test_spliter: TrainTestSplit, model_factory: ModelFactory,
                     trainer: Trainer, tester: Tester, config: OptimizerConfig) -> Result:
    logger.info(f'Start {k}-fold Cross Validation with config : {config.exp_name}')

    subsets = dict()
    subset_size = int(data_size / k)
    remain = set(range(0, data_size))
    for i in range(k - 1):
        subsets[i] = random.sample(remain, subset_size)
        remain = remain.difference(subsets[i])
    subsets[k - 1] = remain

    result = Result()
    for i in range(k):
        logger.info(f"---- Fold {i + 1} ----")

        indices = set(range(0, data_size))
        test_indices = list(subsets[i])
        train_indices = list(indices.difference(subsets[i]))

        train_data, test_data = train_test_spliter.split(train_indices, test_indices)

        model = model_factory.make_model()
        trainer.train(model=model, data=train_data, config=config)
        test_result = tester.test(model=model, data=test_data, config=config)
        logger.info(f'Result of fold {i + 1} : {test_result.get_result()}')
        model.destroy()

        result.add(test_result)

    result.divide(k=k)

    logger.info(
        f'{k}-fold result: avg_auc: {result.auc}, avg_acc: {result.acc}, avg_f1: {result.f1}, avg_aupr: {result.aupr}')

    return result


def _batch_optimize(loader, model, config: OptimizerConfig, ):
    model.classifier.train()
    optimizer = config.optimizer(model.classifier.parameters(), lr=config.lr)

    for epoch in range(config.n_epoch):
        running_loss = 0.0
        for j, data in enumerate(loader, 0):
            X, y = data

            loss, running_loss = predict_error(X, config.criterion, model, running_loss, y)
            backpropagation(loss, optimizer)

            if j % config.report_size == config.report_size - 1:
                loss = running_loss / (config.report_size * config.batch_size)
                logger.info(f'loss: {loss:.4f}    [{epoch + 1}, {j + 1:5d}]')
                running_loss = 0

        if config.save:
            if epoch % 5 == 0 or epoch == config.n_epoch - 1:
                m = os.path.join(config.save_path, model.model_config.model_name + ".pth")
                torch.save(model.classifier.state_dict(), m)


def predict_error(X, loss_function, model, running_loss, y):
    pred = model.classifier(X)
    loss = loss_function(pred, y)
    running_loss += loss.item()
    return loss, running_loss


def backpropagation(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def _evaluate(model: BaseModel, loader, config: OptimizerConfig):
    model.classifier.eval()
    total_labels = []
    total_predictions = []
    total_loss = 0.0
    for data in loader:
        inputs, labels = data
        outputs = model.predict(inputs)
        total_loss += config.criterion(outputs, labels).item()
        total_labels.extend(labels.cpu().numpy())
        total_predictions.extend(torch.sigmoid(outputs).cpu().numpy())
    result = get_prediction_results(total_labels, total_predictions, config.threshold)
    result.loss = total_loss / len(loader)
    return result


class SimpleTrainer(Trainer):
    def train(self, model: BaseModel, data: SimplePytorchData, config: OptimizerConfig) -> Result:
        logger.info(f'Running Simple Trainer with config : {config.exp_name}')

        logger.info(f'moving data and model to {config.device}')
        model.classifier = model.classifier.to(config.device)
        dataset = TensorDataset(data.X.to(config.device), data.y.to(config.device))
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        _batch_optimize(loader, model, config)
        result = _evaluate(model, loader, config)

        logger.info(f'Result on Train Data : {result.get_result()}')
        return result


class SimpleTester(Tester):
    def test(self, model: BaseModel, data: SimplePytorchData, config: OptimizerConfig = None) -> Result:
        logger.info(f'Running Simple Tester with config : {config.exp_name}')

        logger.info(f'moving data and model to {config.device}')
        model.classifier = model.classifier.to(config.device)
        dataset = TensorDataset(data.X.to(config.device), data.y.to(config.device))
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        result = _evaluate(model, loader, config)

        logger.info(f'Result on Test Data : {result.get_result()}')
        return result
