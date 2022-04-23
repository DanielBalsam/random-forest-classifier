from random import seed, shuffle
from typing import Any
from pandas import DataFrame, Series
from .classifier import Classifier


class Node:
    def __init__(
        self,
        dataframe: DataFrame,
        label_column: str,
        min_samples_per_leaf=1,
        max_impurity=1,
    ):
        self.input_data = dataframe
        self.label_column = label_column

        self.features_df = self.input_data.drop(label_column, axis=1)
        self.features = list(self.features_df.columns)

        self.min_samples_per_leaf = min_samples_per_leaf
        self.max_impurity = max_impurity

        self.majority_class = bool(dataframe[label_column].astype(int).median())
        self.impurity = len(dataframe[dataframe[label_column] != self.majority_class])

        if self.impurity > self.max_impurity:
            best_feature, best_value = self.compute_optimal_split()

            if best_feature:
                self.split_feature = best_feature
                self.split_value = best_value

                left_side_data = DataFrame(
                    self.input_data[
                        self.input_data[self.split_feature] == self.split_value
                    ]
                )
                right_side_data = DataFrame(
                    self.input_data[
                        self.input_data[self.split_feature] != self.split_value
                    ]
                )

                if len(left_side_data) != len(self.input_data):
                    self.left_child = Node(
                        left_side_data,
                        label_column,
                        min_samples_per_leaf=self.min_samples_per_leaf,
                        max_impurity=self.max_impurity,
                    )
                    self.right_child = Node(
                        right_side_data,
                        label_column,
                        min_samples_per_leaf=self.min_samples_per_leaf,
                        max_impurity=self.max_impurity,
                    )
                    self.is_leaf = False

                    return

        self.is_leaf = True

        self.split_feature = None
        self.split_value = None
        self.left_child = None
        self.right_child = None

    def compute_optimal_split(self):
        shuffle(self.features)

        min_samples_per_leaf = self.min_samples_per_leaf
        best_feature: str = ""
        best_value: Any = None
        max_impurity = 100

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

                if len(correctly_sorted) > min_samples_per_leaf and max_impurity >= len(
                    incorrectly_sorted
                ):
                    best_feature = feature
                    best_value = value

                    min_samples_per_leaf = len(correctly_sorted)
                    max_impurity = len(incorrectly_sorted)

        return best_feature, best_value


class DecisionTreeBinaryClassifier(Classifier):
    def __init__(
        self, min_samples_per_leaf=1, max_impurity=1, verbose=False, random_seed=42
    ):
        self.features: list[str] = []

        self.min_samples_per_leaf = min_samples_per_leaf
        self.max_impurity = max_impurity
        self.verbose = verbose

        seed(42)

    def train(self, dataframe: DataFrame, label_column: str):
        features_df = dataframe.drop(label_column, axis=1)
        self.features = list(features_df.columns)
        self.label_column = label_column

        self.root_node = Node(
            dataframe,
            label_column,
            min_samples_per_leaf=self.min_samples_per_leaf,
            max_impurity=self.max_impurity,
        )

    def predict(self, dataframe: DataFrame) -> DataFrame:
        if not self.root_node:
            raise BaseException(
                "Must train DecisionTreeBinaryClassifier instance before attempting to predict"
            )

        for feature in self.features:
            if feature not in dataframe.columns:
                raise KeyError(f"Feature with name {feature} not in DataFrame")

        results_df = dataframe.copy()

        def compute_prediction(row: Series):
            current_node = self.root_node

            while not current_node.is_leaf:
                should_go_left = (
                    row[current_node.split_feature] == current_node.split_value
                )

                if self.verbose:
                    print(
                        f"Assessing if {current_node.split_feature} is {current_node.split_value}"
                    )

                if should_go_left and current_node.left_child:
                    current_node = current_node.left_child
                elif current_node.right_child:
                    current_node = current_node.right_child
                else:
                    # NOTE: in theory we should never get here
                    raise Exception("You goofed up Dan.")

            return current_node.majority_class

        results_df[self.label_column] = results_df.apply(compute_prediction, axis=1)

        return results_df
