import requests
import numpy as np

def reranker_similarity(host,sentences1, sentences2):
    """
    Calculate the scores for a given query using an encoder.
    """
    #host = for pass the sentence to reranker
    predict = requests.post(
                url=host,
                json={
                    "inputs": {
                        "input1": list(sentences1),
                        "input2": list(sentences2)
                        }
                    }
                )
    predictions = np.array(predict.json()['outputs']).reshape(-1)
    return predictions
    