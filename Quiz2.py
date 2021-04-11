import pandas as pd
import numpy as np

# Read CSV Files
classes = pd.read_csv("animal_classes.csv")
test = pd.read_csv("animals_test.csv")
train = pd.read_csv("animals_train.csv")

# Filter the animal classes file
classes_filtered = pd.DataFrame(classes["Class_Number"])
class_type = classes["Class_Type"]
classes_filtered["Class_Type"] = class_type

# import train_test_split
from sklearn.model_selection import train_test_split

# Create dataframes without the class number and animal names for equal comparison
train_no_numbers = train.loc[:, train.columns != "class_number"]
dataframe2 = test.loc[:, test.columns != "animal_name"]

# test/train split with the training data without class numbers
x_train, x_test, y_train, y_test = train_test_split(
    train_no_numbers.values, train["class_number"], random_state=11
)

# import knn
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

# fit data
knn.fit(X=x_train, y=y_train)

# Fit the data using the dataframe without the animal names
Animal_Predictions = knn.predict(X=dataframe2)

# create list to hold classifications conversion
animal_classifications = []

# long if statement to convert the numbers into animal classifications
for number in Animal_Predictions:
    if number == 1:
        animal_classifications.append("Mammal")
    elif number == 2:
        animal_classifications.append("Bird")
    elif number == 3:
        animal_classifications.append("Reptile")
    elif number == 4:
        animal_classifications.append("Fish")
    elif number == 5:
        animal_classifications.append("Amphibian")
    elif number == 6:
        animal_classifications.append("Bug")
    elif number == 7:
        animal_classifications.append("Invertebrate")

# create a final dataframe to hold the animal names and their classifications
Final_Output_DF = pd.DataFrame(
    {
        "Animal_Name": test["animal_name"],
        "Class_Type": animal_classifications,
    }
)

# print to csv without the indices
Final_Output_DF.to_csv("Animals_Classified", index=False)
