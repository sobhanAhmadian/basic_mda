from src.utils import prj_logger

logger = prj_logger.getLogger(__name__)

from torch import nn
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.nn.norm import BatchNorm
from base import BaseModel, ModelFactory
from src.config import GraphAutoEncoderConfig

import torch


class DeepGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=32, num_layers=3, dropout=0.1, with_embedd=False,
                 GCN=GCNConv):
        super().__init__()

        logger.info(f'''Initial GCNEncoder with {input_dim} input_dimension,
            {hidden_dim} hidden dimension, {output_dim} output dimension,
            {num_layers} layers and with {dropout} dropout''')

        self.GCN = GCN

        modules = []
        self.add_first_block(modules, input_dim, hidden_dim, dropout, with_embedd)
        for i in range(num_layers - 2):
            self.add_hidden_block(modules, hidden_dim, dropout)
        self.add_last_block(modules, hidden_dim, output_dim)

        self.seq = Sequential('x, edge_index', modules)

    def add_last_block(self, modules, hidden_dim, output_dim):
        modules.append((self.GCN(hidden_dim, output_dim), 'x, edge_index -> x'))
        modules.append(nn.ReLU(inplace=True))

    def add_hidden_block(self, modules, hidden_dim, dropout):
        modules.append((self.GCN(hidden_dim, hidden_dim), 'x, edge_index -> x'))
        modules.append(nn.ReLU(inplace=True))
        modules.append((BatchNorm(hidden_dim, hidden_dim), 'x -> x'))
        modules.append((nn.Dropout(dropout), 'x -> x'))

    def add_first_block(self, modules, input_dim, hidden_dim, dropout, with_embedd):
        if with_embedd:
            modules.append((nn.Embedding(input_dim, hidden_dim), 'x -> x'))
            modules.append((nn.Dropout(dropout), 'x -> x'))
            modules.append((self.GCN(hidden_dim, hidden_dim), 'x, edge_index -> x'))
        else:
            modules.append((nn.Dropout(dropout), 'x -> x'))
            modules.append((self.GCN(input_dim, hidden_dim), 'x, edge_index -> x'))
        modules.append(nn.ReLU(inplace=True))
        modules.append((BatchNorm(hidden_dim, hidden_dim), 'x -> x'))
        modules.append((nn.Dropout(dropout), 'x -> x'))

    def forward(self, x, edge_index):
        return self.seq(x, edge_index)


class LinkDecoder(nn.Module):
    def __init__(self):
        super(LinkDecoder, self).__init__()
        logger.info(f'''Initial LinkDecoder''')

    def forward(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)


class GraphAutoEncoder(nn.Module):
    def __init__(self, config: GraphAutoEncoderConfig):
        super(GraphAutoEncoder, self).__init__()
        logger.info(f'Initializing GCNAutoEncoder ...')

        self.encoder = DeepGraphEncoder(**config.get_encoder_configuration())
        self.decoder = LinkDecoder()

    def forward(self, x, edge_index, asked_edge_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z, asked_edge_index)

    def get_node_embeddings(self, x, edge_index):
        return self.encoder(x, edge_index)


class GraphAutoEncoderModel(BaseModel):

    def __init__(self, config: GraphAutoEncoderConfig):
        super().__init__(config)
        logger.info(f'Initializing GraphAutoEncoderModel with config : {config.get_configuration()}')

        self.model = GraphAutoEncoder(config)

    def destroy(self):
        logger.info(f'deleting model : {self.model}')
        del self.model

    def predict(self, **kwargs):
        return self.model.get_node_embeddings(**kwargs)

    def summary(self):
        pass


class GraphAutoEncoderModelFactory(ModelFactory):

    def __init__(self, model_config: GraphAutoEncoderConfig):
        logger.info(f'Initializing GraphAutoEncoderModelFactory with model : {model_config.model_name}')
        self.model_config = model_config

    def make_model(self) -> BaseModel:
        return GraphAutoEncoderModel(config=self.model_config)
