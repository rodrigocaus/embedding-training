from datasets import Dataset


def format_retrieval_evaluator(dataset: Dataset):
    filtered_examples = [
        (i, example) for i, example in enumerate(dataset)
        if len(example['similar']) > 0
    ]

    # queries
    queries = {
        f'q_{idx}': str(example['sentence'])
        for idx, example in filtered_examples
    }

    # corpus
    corpus1 = {
        f'doc_{idx}_{j}': str(similar)
        for idx, example in filtered_examples
        for j, similar in enumerate(example['similar'])
    }
    all_texts= sum([x['almost_similar'] + x['dissimilar'] for x in dataset ],[])
    corpus2 = {}
    for i,x in enumerate(all_texts):
        corpus2[f"doc_distractor_{i}"] = x
    # relevant
    corpus = corpus1 | corpus2
    # print(corpus)
    relevant = {
        f'q_{idx}': set(f"doc_{idx}_{j}" for j in range(len(example['similar'])))
        for idx, example in filtered_examples
    }
    return dict(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant
    )

def format_retrieval_gosim3(dataset: Dataset):
    queries = {}
    corpus = {}
    relevant = {}
    for i,item in enumerate(dataset):
        if item['label'] == 'similar':
            queries[f"q_{i}"] = str(item['sentence1'])
            corpus[f"Q_{i}"] = str(item['sentence2'])
            relevant[f"q_{i}"] = set([f"Q_{i}"])
        else:
            corpus[f"c_{i}"] = str(item['sentence1'])
    queries = {
        f'q_{i}': str(sentence)
        for i, sentence in enumerate(dataset['sentence1'])
    }
    return dict(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant
    )
