from pandas import read_csv
from classifiers.random_forest import RandomForestBinaryClassifier

classifier = RandomForestBinaryClassifier()
training_data = read_csv("./data/titanic.csv")
print("Training...")
classifier.train(training_data, "Survived")

print("Analyzing...")
results = classifier.predict(training_data)
training_data["predicted"] = results.Survived

accuracy = len(
    training_data[(training_data["predicted"] == training_data["Survived"])]
) / len(training_data)
print(f"Overall accuracy: {accuracy}")
