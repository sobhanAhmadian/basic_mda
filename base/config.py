import abc


class Config(abc.ABC):
    @abc.abstractmethod
    def get_configuration(self):
        pass

    @abc.abstractmethod
    def get_summary(self):
        pass


class ModelConfig(Config):
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.embedding_dim = None  # dimension of embedding
        self.device = None

    def get_configuration(self):
        return {
            "model_name": self.model_name,
            "embed_dim": self.embedding_dim,
        }

    def get_summary(self):
        return {
            "model_name": self.model_name,
            "embed_dim": self.embedding_dim,
        }


class MethodConfig(Config):
    def __init__(self):
        super().__init__()
        self.method_name = None

    def get_configuration(self):
        return {
            "method_name": self.method_name,
        }

    def get_summary(self):
        return {
            "method_name": self.method_name,
        }


class OptimizerConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.optimizer = None
        self.criterion = None
        self.lr = None  # learning rate
        self.batch_size = None
        self.n_epoch = None
        self.exp_name = None
        self.save = False
        self.save_path = None
        self.device = 'cpu'
        self.report_size = 100  # batch to report ratio
        self.threshold = 0.5

    def get_configuration(self):
        return {
            "optimizer": self.optimizer,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "n_epoch": self.n_epoch,
        }

    def get_summary(self):
        return {
            "optimizer": self.optimizer,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "n_epoch": self.n_epoch,
        }
