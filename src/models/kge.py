from base import BaseModel
from src.config import KGEConfig
from src.utils import prj_logger

logger = prj_logger.getLogger(__name__)


class MDAKnowledgeGraphEmbedding(BaseModel):

    def __init__(self, model_config: KGEConfig):
        super().__init__(model_config)
        logger.info(f'Initialing MDATransE with model_config {model_config.get_summary()}')

        self.model = model_config.kge(
            **model_config.get_kge_configuration()
        )

    def destroy(self):
        del self.model

    def predict(self, **kwargs):
        return self.model.node_emb.weight[kwargs['node_list']].detach()

    def summary(self):
        pass
