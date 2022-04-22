from pandas import read_csv, DataFrame
from classifiers.random_forest import RandomForestBinaryClassifier

# Instantiate the classifier
classifier = RandomForestBinaryClassifier(threshold=0.5)

# Read in the data
all_data = read_csv("./data/titanic.csv")

# Drop unusable columns
all_data = all_data.drop("Name", axis=1)
all_data = all_data.drop("Fare", axis=1)

# Train/Test split
test_data = DataFrame(all_data.iloc[0 : round(len(all_data) / 4)])
training_data = DataFrame(all_data.iloc[round(len(all_data) / 4) : len(all_data)])

# Train the model with training data
print("Training...")
classifier.train(training_data, "Survived")

# Predict with test data
print("Analyzing...")
results = classifier.predict(test_data)
test_data["predicted"] = results.Survived

# Compute accuracy

accuracy = len(test_data[(test_data["predicted"] == test_data["Survived"])]) / len(
    test_data
)
print(f"Overall accuracy: {accuracy}")

accuracy = len(
    (
        test_data[
            (test_data["Survived"] == True)
            & (test_data["predicted"] == test_data["Survived"])
        ]
    )
) / len(test_data)
print(f"Accuracy survived: {accuracy}")

accuracy = len(
    (
        test_data[
            (test_data["Survived"] == False)
            & (test_data["predicted"] == test_data["Survived"])
        ]
    )
) / len(test_data)
print(f"Accuracy did not survived: {accuracy}")
