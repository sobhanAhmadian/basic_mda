import torch
from torch_geometric.data import Data

from base import OptimizerConfig
from src.config import Node2VecConfig, Node2VecOptimizerConfig, GraphAutoEncoderConfig, KGEConfig
from src.models import MDNode2Vec, GraphAutoEncoderModel, MDAKnowledgeGraphEmbedding
from src.optimization import Node2vecTrainer, GraphAutoEncoderTrainer, KGETrainer
from src.utils import prj_logger
from .raw import get_entities, get_relations, get_associations

logger = prj_logger.getLogger(__name__)


def get_homogeneous_graph_with_gae_embedd(model_config: GraphAutoEncoderConfig, optimizer_config: OptimizerConfig):
    homo = get_homogeneous_graph(optimizer_config.device)
    homo.x = get_gae_embedd_for_all_nodes(model_config, optimizer_config)
    return homo


def get_homogeneous_graph_with_node2vec_embedd(model_config: Node2VecConfig, optimizer_config: Node2VecOptimizerConfig):
    homo = get_homogeneous_graph(optimizer_config.device)
    homo.x = get_node2vec_embedd_for_all_nodes(model_config, optimizer_config)
    return homo


def get_kge_pair_embedd_for_training_data(model_config: KGEConfig, optimizer_config: OptimizerConfig):
    disease_list = get_associations()['disease'].tolist()
    microbe_list = get_associations()['microbe'].tolist()
    return get_kge_pair_embedd(disease_list, microbe_list, model_config, optimizer_config)


def get_node2vec_gae_pair_embedd_for_training_data(node2vec_config, node2vec_optimizer_config: Node2VecOptimizerConfig,
                                                   gae_config: GraphAutoEncoderConfig,
                                                   gae_optimizer_config: OptimizerConfig):
    disease_list = get_associations()['disease'].tolist()
    microbe_list = get_associations()['microbe'].tolist()
    return get_node2vec_gae_pair_embedd(disease_list, microbe_list, node2vec_config, node2vec_optimizer_config,
                                        gae_config, gae_optimizer_config)


def get_gae_pair_embedd_for_training_data(model_config: GraphAutoEncoderConfig, optimizer_config: OptimizerConfig):
    disease_list = get_associations()['disease'].tolist()
    microbe_list = get_associations()['microbe'].tolist()
    return get_gae_pair_embedd(disease_list, microbe_list, model_config, optimizer_config)


def get_node2vec_pair_embedd_for_training_data(model_config: Node2VecConfig, optimizer_config: Node2VecOptimizerConfig):
    disease_list = get_associations()['disease'].tolist()
    microbe_list = get_associations()['microbe'].tolist()
    return get_node2vec_pair_embedd(disease_list, microbe_list, model_config, optimizer_config)


def get_kge_pair_embedd(disease_list, microbe_list, model_config: KGEConfig, optimizer_config: OptimizerConfig):
    logger.info(f'Calling get_node2vec_pair_embedd on {optimizer_config.device} device ...')

    assert len(disease_list) == len(microbe_list), (f'length of microbe list: {len(microbe_list)}'
                                                    f'length of disease list: {len(disease_list)}')

    all_embedd = get_kge_embedd_for_all_nodes(model_config, optimizer_config)
    md_embedd = _get_pair_embedd(all_embedd, disease_list, microbe_list, optimizer_config)
    return md_embedd


def get_node2vec_gae_pair_embedd(disease_list, microbe_list, node2vec_config,
                                 node2vec_optimizer_config: Node2VecOptimizerConfig, gae_config: GraphAutoEncoderConfig,
                                 gae_optimizer_config: OptimizerConfig):
    logger.info(f'Calling get_node2vec_gae_pair_embedd on {gae_optimizer_config.device} device ...')
    assert len(disease_list) == len(microbe_list), (f'length of microbe list: {len(microbe_list)}'
                                                    f'length of disease list: {len(disease_list)}')

    all_embedd = get_node2vec_gae_embedd_for_all_nodes(node2vec_config, node2vec_optimizer_config, gae_config,
                                                       gae_optimizer_config)
    md_embedd = _get_pair_embedd(all_embedd, disease_list, microbe_list, gae_optimizer_config)
    return md_embedd


def get_gae_pair_embedd(disease_list, microbe_list, model_config: GraphAutoEncoderConfig,
                        optimizer_config: OptimizerConfig):
    logger.info(f'Calling get_gae_pair_embedd on {optimizer_config.device} device ...')

    assert len(disease_list) == len(microbe_list), (f'length of microbe list: {len(microbe_list)}'
                                                    f'length of disease list: {len(disease_list)}')

    all_embedd = get_gae_embedd_for_all_nodes(model_config, optimizer_config)
    md_embedd = _get_pair_embedd(all_embedd, disease_list, microbe_list, optimizer_config)
    return md_embedd


def get_node2vec_pair_embedd(disease_list, microbe_list, model_config: Node2VecConfig,
                             optimizer_config: Node2VecOptimizerConfig):
    logger.info(f'Calling get_node2vec_pair_embedd on {optimizer_config.device} device ...')

    assert len(disease_list) == len(microbe_list), (f'length of microbe list: {len(microbe_list)}'
                                                    f'length of disease list: {len(disease_list)}')

    all_embedd = get_node2vec_embedd_for_all_nodes(model_config, optimizer_config)
    md_embedd = _get_pair_embedd(all_embedd, disease_list, microbe_list, optimizer_config)
    return md_embedd


def _get_pair_embedd(all_embedd, disease_list, microbe_list, optimizer_config):
    d_embedd = all_embedd[disease_list]
    m_embedd = all_embedd[microbe_list]
    logger.info(f'disease embedding shape : {d_embedd.shape}')
    logger.info(f'microbe embedding shape : {m_embedd.shape}')
    md_embedd = torch.cat((d_embedd, m_embedd), 1).detach().to(optimizer_config.device)
    logger.info(f'microbe disease combination embedding shape : {md_embedd.shape}')
    return md_embedd


def get_kge_embedd_for_all_nodes(model_config: KGEConfig, optimizer_config: OptimizerConfig):
    node_list = get_homogeneous_graph(optimizer_config.device).x.squeeze().detach().tolist()
    return get_kge_embedd(node_list, model_config, optimizer_config)


def get_node2vec_gae_embedd_for_all_nodes(node2vec_config, node2vec_optimizer_config: Node2VecOptimizerConfig,
                                          gae_config: GraphAutoEncoderConfig, gae_optimizer_config: OptimizerConfig):
    homo = get_homogeneous_graph_with_node2vec_embedd(node2vec_config, node2vec_optimizer_config)

    logger.info(f'set graph auto encoder input_dim to homo.x.shape[1] : {homo.x.shape[1]}')
    gae_config.input_dim = homo.x.shape[1]

    logger.info(f'set graph auto encoder with_embedd to False')
    gae_config.with_embedd = False
    return get_node2vec_gae_embedd(g=homo, homo=homo, gae_config=gae_config, gae_optimizer_config=gae_optimizer_config)


def get_gae_embedd_for_all_nodes(model_config: GraphAutoEncoderConfig, optimizer_config: OptimizerConfig):
    homo = get_homogeneous_graph(optimizer_config.device)
    return get_gae_embedd(homo, model_config, optimizer_config)


def get_node2vec_embedd_for_all_nodes(model_config: Node2VecConfig, optimizer_config: Node2VecOptimizerConfig):
    node_list = get_homogeneous_graph(optimizer_config.device).x.squeeze().detach().tolist()
    return get_node2vec_embedd(node_list, model_config, optimizer_config)


def get_kge_embedd(node_list, model_config: KGEConfig, optimizer_config: OptimizerConfig):
    logger.info(f'Calling get_kge_embedd on {optimizer_config.device} device ...')

    model = _create_and_train_kge_model(model_config, optimizer_config)

    node_embed = model.predict(node_list=node_list)
    logger.info(f'node embedding shape : {node_embed.shape}')
    return node_embed


def get_node2vec_gae_embedd(g, homo, gae_config: GraphAutoEncoderConfig, gae_optimizer_config: OptimizerConfig):
    logger.info(f'Calling get_node2vec_gae_embedd on {gae_optimizer_config.device} device ...')

    model = _create_and_train_node2vec_gae_model(homo, gae_config, gae_optimizer_config)

    node_embed = model.predict(x=g.x, edge_index=g.edge_index)
    logger.info(f'node embedding shape : {node_embed.shape}')
    return node_embed


def get_gae_embedd(g, model_config: GraphAutoEncoderConfig, optimizer_config: OptimizerConfig):
    logger.info(f'Calling get_node2vec_embedd on {optimizer_config.device} device ...')

    model = _create_and_train_gae_model(model_config, optimizer_config)

    node_embed = model.predict(x=g.x.detach().reshape(-1), edge_index=g.edge_index)
    logger.info(f'node embedding shape : {node_embed.shape}')
    return node_embed


def get_node2vec_embedd(node_list, model_config: Node2VecConfig, optimizer_config: Node2VecOptimizerConfig):
    logger.info(f'Calling get_node2vec_embedd on {optimizer_config.device} device ...')

    node2vec_model = _create_and_train_node2vec_model(model_config, optimizer_config)

    node_embed = node2vec_model.predict(node_list=node_list)
    logger.info(f'node embedding shape : {node_embed.shape}')
    return node_embed


def _create_and_train_kge_model(model_config, optimizer_config):

    data = get_knowledge_graph(optimizer_config.device)

    logger.info(f'Setting num relations and num nodes for kge config to {data.num_edge_types} and {data.num_nodes}')
    model_config.num_relations = data.num_edge_types
    model_config.num_nodes = data.num_nodes

    logger.info('Creating KGE model ...')
    kge_model = MDAKnowledgeGraphEmbedding(model_config)

    logger.info('Training KGE ...')
    result = KGETrainer().train(model=kge_model, data=data,
                                config=optimizer_config)
    logger.info(f'loss of KGE model : {result.loss}')
    return kge_model


def _create_and_train_node2vec_model(model_config, optimizer_config):
    logger.info('Creating Node2Vec model ...')
    node2vec_model = MDNode2Vec(model_config, get_homogeneous_graph(optimizer_config.device))
    logger.info('Training Node2Vec ...')
    result = Node2vecTrainer().train(model=node2vec_model, data=None, config=optimizer_config)
    logger.info(f'loss of Node2Vec model : {result.loss}')
    return node2vec_model


def _create_and_train_gae_model(model_config: GraphAutoEncoderConfig, optimizer_config):
    logger.info('Creating GraphAutoEncoderModel ...')
    model = GraphAutoEncoderModel(model_config)

    homo = get_homogeneous_graph(optimizer_config.device)
    homo.x = homo.x.reshape(-1)
    logger.info(f'Reshape Homogeneous graph : {homo.x.shape}')

    logger.info('Training GraphAutoEncoderModel ...')
    result = GraphAutoEncoderTrainer().train(model=model, data=homo, config=optimizer_config)
    logger.info(f'loss of GraphAutoEncoderModel model : {result.loss}')
    return model


def _create_and_train_node2vec_gae_model(homo, gae_config: GraphAutoEncoderConfig,
                                         gae_optimizer_config: OptimizerConfig):
    logger.info('Creating GraphAutoEncoderModel ...')
    model = GraphAutoEncoderModel(gae_config)

    logger.info('Training GraphAutoEncoderModel ...')
    result = GraphAutoEncoderTrainer().train(model=model, data=homo, config=gae_optimizer_config)
    logger.info(f'loss of GraphAutoEncoderModel model : {result.loss}')
    return model


def get_homogeneous_graph(device):
    logger.info(f'Calling get_homogeneous_graph')
    homo = Data()
    homo.x = torch.tensor(get_entities()['id'].tolist()).reshape(-1, 1).to(device)
    homo.edge_index = torch.tensor([get_relations()['head'].tolist(), get_relations()['tail'].tolist()]).to(device)
    logger.info(f'homogeneous data : {homo}')
    return homo


def get_knowledge_graph(device):
    logger.info(f'Calling get_knowledge_graph')
    data = Data()
    data.num_nodes = torch.tensor(get_entities()['id'].shape[0]).to(device)
    data.edge_index = torch.tensor([get_relations()['head'].tolist(), get_relations()['tail'].tolist()]).to(device)
    data.edge_type = torch.tensor(get_relations()['relation'].tolist()).to(device)
    logger.info(f'knowledge graph data : {data}')
    return data


def get_heterogeneous_data():
    pass  # TODO
