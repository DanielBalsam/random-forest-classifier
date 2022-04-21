from random import shuffle
from typing import Any, cast
from pandas import DataFrame, concat
from .classifier import Classifier


class Node:
    def __init__(
        self, dataframe: DataFrame, label_column: str, min_num_to_split=1, max_error=5
    ):
        self.input_data = dataframe
        self.label_column = label_column

        self.features_df = self.input_data.drop(label_column, axis=1)
        self.features = list(self.features_df.columns)

        self.min_num_to_split = min_num_to_split
        self.max_error = max_error

        if len(self.input_data) > self.min_num_to_split:
            best_feature, best_value = self.compute_optimal_split()
            if best_feature:
                self.split_feature = best_feature
                self.split_value = best_value

                self.sorted_data = DataFrame(
                    self.input_data[
                        self.input_data[self.split_feature] == self.split_value
                    ]
                )
                self.unsorted_data = DataFrame(
                    self.input_data[
                        self.input_data[self.split_feature] != self.split_value
                    ]
                )

                self.child_node = Node(self.unsorted_data, label_column)

                return

        self.sorted_data = DataFrame()
        self.unsorted_data = self.input_data

        self.split_feature = None
        self.split_value = None
        self.child_node = None

    def compute_optimal_split(self):
        shuffle(self.features)

        best_num_to_split = self.min_num_to_split
        best_feature: str = ""
        best_value: Any = None
        max_error = self.max_error

        for feature in self.features:
            unique_values = list(self.input_data[feature].unique())
            shuffle(unique_values)

            for value in unique_values:
                after_split_data = self.input_data[self.input_data[feature] == value]
                correctly_sorted = after_split_data[
                    after_split_data[self.label_column] == True
                ]
                incorrectly_sorted = after_split_data[
                    after_split_data[self.label_column] == False
                ]

                if len(correctly_sorted) > best_num_to_split and max_error >= len(
                    incorrectly_sorted
                ):
                    best_feature = feature
                    best_value = value
                    best_num_to_split = len(correctly_sorted)
                    max_error = len(incorrectly_sorted)

        return best_feature, best_value


class DecisionTreeBinaryClassifier(Classifier):
    def __init__(self, min_num_to_split=1, max_error=5):
        self.features: list[str] = []

        self.min_num_to_split = min_num_to_split
        self.max_error = max_error

    def train(self, dataframe: DataFrame, label_column: str):
        features_df = dataframe.drop(label_column, axis=1)
        self.features = list(features_df.columns)
        self.label_column = label_column

        self.root_node = Node(
            dataframe,
            label_column,
            min_num_to_split=self.min_num_to_split,
            max_error=self.max_error,
        )

    def predict(self, dataframe: DataFrame) -> DataFrame:
        if not self.root_node:
            raise BaseException(
                "Must train DecisionTreeBinaryClassifier instance before attempting to predict"
            )

        for feature in self.features:
            if feature not in dataframe.columns:
                raise KeyError(f"Feature with name {feature} not in DataFrame")

        output_dfs = []
        current_node = self.root_node
        negatively_classified_df = dataframe.copy()
        while current_node and current_node.split_feature:
            should_classify_as_postive_mask = (
                negatively_classified_df[current_node.split_feature]
                == current_node.split_value
            )
            positively_classified_df = negatively_classified_df[
                should_classify_as_postive_mask
            ].copy()

            negatively_classified_df = negatively_classified_df.drop(
                negatively_classified_df[should_classify_as_postive_mask].index
            )

            positively_classified_df[self.label_column] = True
            output_dfs.append(positively_classified_df)

            current_node = current_node.child_node

        negatively_classified_df[self.label_column] = False
        output_dfs.append(negatively_classified_df)

        results_df = cast(DataFrame, concat(output_dfs).sort_index())
        results_df[self.label_column] = results_df[self.label_column].astype(bool)

        return results_df
