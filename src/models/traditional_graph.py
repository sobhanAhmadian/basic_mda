from torch_geometric.nn import Node2Vec

from base import BaseModel
from src.utils import prj_logger

logger = prj_logger.getLogger(__name__)


class MDNode2Vec(BaseModel):

    def __init__(self, model_config, gdata):
        super().__init__(model_config)
        logger.info(f'Initialing MDNode2Vec with model_config {model_config.get_summary()}')

        self.model = Node2Vec(
            gdata.edge_index,
            **model_config.get_main_configuration()
        )

    def destroy(self):
        del self.model

    def predict(self, **kwargs):
        # X should be list
        return self.model()[kwargs['node_list']].detach()

    def summary(self):
        pass
