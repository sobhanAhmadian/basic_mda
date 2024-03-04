import pandas as pd

from src import config
from src.utils import prj_logger

logger = prj_logger.getLogger(__name__)


def get_associations():
    return pd.read_csv(config.ASSOCIATION_FILE, index_col=0)


def get_relations():
    return pd.read_csv(config.RELATION_FILE, index_col=0)


def get_relation_types():
    return pd.read_csv(config.RELATION_TYPE_FILE, index_col=0)


def get_entities():
    return pd.read_csv(config.ENTITY_FILE, index_col=0)


def process_data():
    entity_vocab = _load_entities()['id'].tolist()
    relation_vocab = _load_relation_types()['id'].tolist()
    _load_relations(entity_vocab, relation_vocab)
    _load_associations(entity_vocab)


def _read_id_file(file_path: str):
    logger.info(f"reading id file: {file_path}")
    vocab = []
    with open(file_path, encoding="utf8") as reader:
        count = 0
        for line in reader:
            if count == 0:
                count += 1
                continue
            name, i = line.strip().split("\t")
            vocab.append({'type': name.split('::')[0],
                          'name': name.split('::')[1],
                          'id': int(i)})
    return pd.DataFrame(vocab)


def _read_raw_entities():
    return _read_id_file(config.RAW_ENTITY_FILE)


def _load_entities():
    entities = _read_raw_entities()
    entities.to_csv(config.ENTITY_FILE)
    logger.info(f"entities saved at {config.ENTITY_FILE}")
    return entities


def _read_raw_relation_types():
    return _read_id_file(config.RAW_RELATION_TYPE_FILE)


def _load_relation_types():
    relation_types = _read_raw_relation_types()
    relation_types.to_csv(config.RELATION_TYPE_FILE)
    logger.info(f"relation types saved at {config.RELATION_TYPE_FILE}")
    return relation_types


def _read_raw_associations(entity_vocab: list):
    logger.info(f"reading association file: {config.RAW_ASSOCIATION_FILE}")
    assert len(entity_vocab) > 0
    associations = []
    unknowns = 0
    with open(config.RAW_ASSOCIATION_FILE, encoding="utf8") as reader:
        for idx, line in enumerate(reader):
            d1, d2, flag = line.strip().split('\t')[:3]
            d1, d2, flag = int(d1), int(d2), int(flag)
            if d1 not in entity_vocab or d2 not in entity_vocab:
                unknowns += 1
                continue
            if d1 in entity_vocab and d2 in entity_vocab:
                associations.append({'disease': d1, 'microbe': d2, 'increased': flag})

    associations = pd.DataFrame(associations)
    logger.info(f"size of associations: {associations.shape}")
    logger.info(f"number of unknowns: {unknowns}")
    return associations


def _load_associations(entity_vocab: list):
    _read_raw_associations(entity_vocab).to_csv(config.ASSOCIATION_FILE)
    logger.info(f"associations saved at {config.ASSOCIATION_FILE}")


def _read_raw_relations(entity_vocab: list, relation_vocab: list):
    logger.info(f"reading relations file: {config.RAW_RELATION_FILE}")

    relations = []
    unknown_entities = 0
    unknown_relation_types = 0
    with open(config.RAW_RELATION_FILE, encoding="utf8") as reader:
        count = 0
        for line in reader:
            if count == 0:
                count += 1
                continue

            head, relation, tail = line.strip().split('\t')
            head, relation, tail = int(head), int(relation), int(tail)
            if head not in entity_vocab:
                unknown_entities += 1
                continue
            if tail not in entity_vocab:
                unknown_entities += 1
                continue
            if relation not in relation_vocab:
                unknown_relation_types += 1
                continue

            # undirected graph
            relations.append({'head': head, 'relation': relation, 'tail': tail})
            relations.append({'head': tail, 'relation': relation, 'tail': head})

    logger.info(f"number of unknown entities : {unknown_entities}")
    logger.info(f"number of unknown relation types : {unknown_relation_types}")
    return pd.DataFrame(relations)


def _load_relations(entity_vocab: list, relation_vocab: list):
    _read_raw_relations(entity_vocab, relation_vocab).to_csv(config.RELATION_FILE)
    logger.info(f"relations saved at {config.RELATION_FILE}")
