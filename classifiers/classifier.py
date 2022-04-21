from abc import ABC, abstractmethod
from pandas import DataFrame


class Classifier(ABC):
    @abstractmethod
    def train(self, features_df: DataFrame, label_column: str):
        ...

    @abstractmethod
    def predict(self, features_df: DataFrame) -> DataFrame:
        ...
