from datasets import Dataset

DEFAULT_NAME_TO_LABEL = {
    "similar": 1,
    "almost_similar": 0,
    "dissimilar": -1,
}


class EFAQCosineSimilarityGenerator:
    """
        Process a e-faq data in format:
        ```
            sentence: str
            similar: List[str]
            almost_similar: List[str]
            dissimilar: List[str]
        ```
        to a tuple of strings in format:
        ```
            (anchor, candidate, label)
        ```
        Parameters
        ----------
        dataset: Dataset
            The input dataset.
        name_to_label: Dict
            Mapping from column name to label value.
    """

    def __init__(
        self,
        dataset: Dataset,
        anchor_column: str = "sentence",
        name_to_label: dict = DEFAULT_NAME_TO_LABEL
    ) -> None:
        self.dataset = dataset
        self.anchor_column = anchor_column
        self.name_to_label = name_to_label
        self.output_columns = ("anchor", "candidate", "label")

    def as_columns_dict(self, *args):
        return dict(zip(self.output_columns, args))

    def __iter__(self):
        for entry in self.dataset:
            anchor: str = entry[self.anchor_column]
            for name, label in self.name_to_label.items():
                for candidate in entry[name]:
                    yield self.as_columns_dict(anchor, candidate, label)

    def __call__(self):
        return iter(self)
