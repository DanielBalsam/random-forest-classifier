from dataclasses import dataclass
from random import random, seed
from typing import cast

from pandas import DataFrame

from .classifier import Classifier
from .decision_tree import DecisionTreeBinaryClassifier


@dataclass
class BootstrappedTree:
    classifier: DecisionTreeBinaryClassifier
    features: list[str]


class RandomForestBinaryClassifier(Classifier):
    def __init__(
        self,
        num_trees=100,
        threshold=0,
        min_samples_pre_tree=10,
        min_features_per_tree=2,
        min_samples_per_leaf=1,
        max_impurity=1,
        random_seed=42,
        verbose=False,
    ):
        self.features: list[str] = []
        self.trees: list[BootstrappedTree] = []

        self.num_trees = num_trees
        self.threshold = threshold
        self.min_samples_pre_tree = min_samples_pre_tree
        self.min_features_per_tree = min_features_per_tree
        self.random_seed = random_seed
        seed(self.random_seed)

        self.min_samples_per_leaf = min_samples_per_leaf
        self.max_impurity = max_impurity
        self.verbose = verbose

    def train(self, dataframe: DataFrame, label_column: str):
        features_df = dataframe.drop(label_column, axis=1)
        self.features = list(features_df.columns)
        self.label_column = label_column

        self.trees = []
        for _ in range(0, self.num_trees):
            random_subsample = dataframe.sample(
                n=max(
                    min(
                        round(len(dataframe) / (random() * self.num_trees)),
                        len(dataframe),
                    ),
                    self.min_samples_pre_tree,
                ),
                random_state=self.random_seed,
            )

            random_samples_and_features = random_subsample.drop(
                label_column, axis=1
            ).sample(
                n=round(
                    min(
                        max(random() * len(self.features), len(self.features)),
                        self.min_features_per_tree,
                    )
                ),
                random_state=self.random_seed,
                axis="columns",
            )

            tree = BootstrappedTree(
                classifier=DecisionTreeBinaryClassifier(
                    min_samples_per_leaf=self.min_samples_per_leaf,
                    max_impurity=self.max_impurity,
                    verbose=self.verbose,
                ),
                features=list(random_samples_and_features.columns),
            )

            random_samples_and_features[label_column] = random_subsample[label_column]

            tree.classifier.train(
                cast(DataFrame, random_samples_and_features), label_column
            )

            self.trees.append(tree)

    def predict(self, dataframe: DataFrame):
        results_dataframe = dataframe.copy()
        results_dataframe["times_classified_as_true"] = 0

        if not self.trees:
            raise Exception("Must train before using")

        for tree in self.trees:
            features_df = results_dataframe[tree.features]
            tree_results = tree.classifier.predict(features_df)

            tree_results[self.label_column] = tree_results[self.label_column].astype(
                int
            )
            results_dataframe.times_classified_as_true += tree_results[
                self.label_column
            ]

        mean_times_classified_as_true = results_dataframe[
            "times_classified_as_true"
        ].mean()
        stdev_times_classified_as_true = results_dataframe[
            "times_classified_as_true"
        ].std()

        results_dataframe["standard_devs_from_median"] = (
            results_dataframe["times_classified_as_true"]
            - mean_times_classified_as_true
        ) / stdev_times_classified_as_true

        results_dataframe[self.label_column] = False
        results_dataframe.loc[
            results_dataframe.standard_devs_from_median > self.threshold,
            self.label_column,
        ] = True

        results_dataframe = results_dataframe.drop("standard_devs_from_median", axis=1)
        results_dataframe = results_dataframe.drop("times_classified_as_true", axis=1)

        return results_dataframe
