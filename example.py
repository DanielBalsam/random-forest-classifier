from pandas import read_csv, DataFrame
from classifiers.random_forest import RandomForestBinaryClassifier

classifier = RandomForestBinaryClassifier()
all_data = read_csv("./data/titanic.csv")

all_data = all_data.drop("Name", axis=1)
all_data = all_data.drop("Fare", axis=1)

all_data = all_data.sample(frac=1, axis=1).reset_index(drop=True)

test_data = DataFrame(all_data.iloc[0 : round(len(all_data) / 4)])
training_data = DataFrame(all_data.iloc[round(len(all_data) / 4) : len(all_data)])

print("Training...")
classifier.train(training_data, "Survived")

print("Analyzing...")
results = classifier.predict(test_data)
training_data["predicted"] = results.Survived

accuracy = len(
    training_data[(training_data["predicted"] == training_data["Survived"])]
) / len(training_data)
print(f"Overall accuracy: {accuracy}")

accuracy = len(
    (
        training_data[
            (training_data["Survived"] == True)
            & (training_data["predicted"] == training_data["Survived"])
        ]
    )
) / len(training_data)
print(f"Accuracy survived: {accuracy}")

accuracy = len(
    (
        training_data[
            (training_data["Survived"] == False)
            & (training_data["predicted"] == training_data["Survived"])
        ]
    )
) / len(training_data)
print(f"Accuracy did not survived: {accuracy}")
