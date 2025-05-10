from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer

import config


def main(filename: str):
    training_config = config.load(filename)
    print(training_config)
