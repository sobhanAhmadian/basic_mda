import os

from torch_geometric.nn import GCNConv

from base import ModelConfig, OptimizerConfig, MethodConfig

RAW_DATA_DIR = os.getcwd() + "/data_repository/raw"
PROCESSED_DATA_DIR = os.getcwd() + "/data_repository/processed"
LOG_DIR = os.getcwd() + "/data_repository/log"
MODEL_SAVED_DIR = os.getcwd() + "/data_repository/ckpt"

RAW_RELATION_FILE = os.path.join(RAW_DATA_DIR, "relations.txt")
RAW_ENTITY_FILE = os.path.join(RAW_DATA_DIR, "entities.txt")
RAW_RELATION_TYPE_FILE = os.path.join(RAW_DATA_DIR, "relation_types.txt")
RAW_ASSOCIATION_FILE = os.path.join(RAW_DATA_DIR, "associations.txt")

ENTITY_FILE = os.path.join(PROCESSED_DATA_DIR, "entity.csv")
RELATION_TYPE_FILE = os.path.join(PROCESSED_DATA_DIR, "relation_type.csv")
RELATION_FILE = os.path.join(PROCESSED_DATA_DIR, "relation.csv")
ASSOCIATION_FILE = os.path.join(PROCESSED_DATA_DIR, "association.csv")


class SklearnClassifierConfig(ModelConfig):
    CLASSIFIERS = ['RF', 'AdaBoost', 'Logistic', 'DecisionTree']
    PENALTY = ['l1', 'l2', 'elasticnet']
    CRITERION = ['gini', 'entropy', 'log_loss']

    def __init__(self):
        super().__init__()
        self.classifier = None
        self.random_state = 0
        self.penalty = 'l2'
        self.C = 1.0
        self.n_estimators = 100
        self.criterion = 'gini'
        self.max_depth = None
        self.solver = 'lbfgs'

    def get_configuration(self):
        return {
            "model_name": self.model_name,
            "classifier": self.classifier,
            "random_state": self.random_state,
            "penalty": self.penalty,
            "n_estimators": self.n_estimators
        }

    def get_summary(self):
        return {
        }


class SimpleClassifierConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.input_dim = None
        self.hidden_dim = 32
        self.output_dim = 1
        self.num_layers = 3
        self.dropout = 0.1

    def get_configuration(self):
        return {
            "model_name": self.model_name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout
        }

    def get_summary(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        }

    def get_main_configuration(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout
        }


class Node2VecConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 32
        self.walk_length = 50
        self.context_size = 10
        self.walks_per_node = 10
        self.num_negative_samples = 1
        self.p = 1.0
        self.q = 1.0
        self.num_nodes = None
        self.sparse = True

    def get_configuration(self):
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "walk_length": self.walk_length,
            "context_size": self.context_size,
            "walks_per_node": self.walks_per_node,
            "num_negative_samples": self.num_negative_samples,
            "p": self.p,
            "q": self.q,
            "num_nodes": self.num_nodes,
            "sparse": self.sparse,
        }

    def get_summary(self):
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "walk_length": self.walk_length,
        }

    def get_main_configuration(self):
        return {
            "embedding_dim": self.embedding_dim,
            "walk_length": self.walk_length,
            "context_size": self.context_size,
            "walks_per_node": self.walks_per_node,
            "num_negative_samples": self.num_negative_samples,
            "p": self.p,
            "q": self.q,
            "num_nodes": self.num_nodes,
            "sparse": self.sparse,
        }


class Node2VecOptimizerConfig(OptimizerConfig):
    def __init__(self):
        super().__init__()
        self.shuffle = True
        self.num_workers = 2

    def get_main_configuration(self):
        return {
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
        }


class GraphAutoEncoderConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.input_dim = None
        self.hidden_dim = 32
        self.output_dim = 1
        self.num_layers = 3
        self.dropout = 0.1
        self.with_embedd = False
        self.GCN = GCNConv

    def get_configuration(self):
        return {
            "model_name": self.model_name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "with_embedd": self.with_embedd,
            "GCN": self.GCN
        }

    def get_summary(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        }

    def get_encoder_configuration(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "with_embedd": self.with_embedd,
            "GCN": self.GCN
        }


class MatrixDecomposerConfig(MethodConfig):
    def __init__(self):
        super().__init__()
        self.microbe_ids = None
        self.disease_ids = None
        self.n_components = 10
        self.random_state = 1
        self.decomposer = 'NMF'
        self.dummy = False

    def get_configuration(self):
        return {
            "method_name": self.method_name,
            "microbe_ids": self.microbe_ids,
            "disease_ids": self.disease_ids,
            "n_components": self.n_components,
            "random_state": self.random_state,
            "decomposer": self.decomposer,
        }

    def get_summary(self):
        return {
            "microbe_ids": self.microbe_ids,
            "disease_ids": self.disease_ids,
            "n_components": self.n_components,
        }


class MDMDAClassifierOptimizerConfig(OptimizerConfig):
    def __init__(self):
        super().__init__()
        self.conv_threshold = 0.5


class KGEConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.kge = None
        self.num_nodes = None
        self.num_relations = None
        self.hidden_channels = None

    def get_configuration(self):
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "kge": self.kge,
            "num_nodes": self.num_nodes,
            "num_relations": self.num_relations,
            "hidden_channels": self.hidden_channels,
        }

    def get_summary(self):
        return {
            "model_name": self.model_name,
        }

    def get_kge_configuration(self):
        return {
            "num_nodes": self.num_nodes,
            "num_relations": self.num_relations,
            "hidden_channels": self.hidden_channels,
        }
