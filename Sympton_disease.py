# %%
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
from sklearn.model_selection import KFold, cross_val_score
from statistics import mode




# %%

train_data_path="/content/Testing.csv"
test_data_path="/content/Training.csv"
train_data=pd.read_csv(train_data_path).dropna(axis=1)
test_data=pd.read_csv(test_data_path).dropna(axis=1)

test_data

# %%
disase_counts=train_data['prognosis'].value_counts()
temp_dataframe=pd.DataFrame({
    'disease': disase_counts.index,
    'counts':disase_counts.values
})

temp_dataframe
plt.figure(figsize=(15,10))
sns.barplot(x='disease',y='counts',data=temp_dataframe)
plt.xticks(rotation=90)
plt.show


# %%
# we need to convert the data numbers into a value the data in the program because the data is of string type we need to convert them to numeric data type
# We can use labelencoder to convert string data type to numeric data


encoder=LabelEncoder()
train_data['prognosis']=encoder.fit_transform(train_data['prognosis'])
test_data['prognosis']=encoder.fit_transform(test_data['prognosis'])
train_data
test_data

# %%
# here, the data is x as the other desired data, and the y value represents the numerical data under prognasis.
x=train_data.iloc[:,:-1]
y=train_data.iloc[:,-1]

# of data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=24)


# %% [markdown]
# - Using K-Fold Cross-Validation Model

# %%
# first we need to calculate the scoring metric to use in k-fold

def cv_scoring(estimator,x,y):
    return accuracy_score(y, estimator.predict(x))

models={
    "SVC":SVC(),
    "Gaussian NB":GaussianNB(),
    "Random Forest":RandomForestClassifier(random_state=18),
}

# points are generated at this stage

for model_name in models:
    model = models[model_name]
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)  # Regular use of KFold
    scores = cross_val_score(model, x, y, cv=kfold, scoring=cv_scoring, n_jobs=-1)
    print("==" * 30)
    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")

# %%
# let's do the svm model training first
svmModel=SVC()
svmModel.fit(x_train,y_train)
preds=svmModel.predict(x_test)

print(f"Accuracy on test data by SVM Classifier\
: {accuracy_score(y_test, preds)*100}")


cf_matrix=confusion_matrix(y_test,preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM Classifier on Test Data")
plt.show()

# %%
# Naive Bayes Classifier with training data

nb_model=GaussianNB()
nb_model.fit(x_train,y_train)
nbModelPredict=nb_model.predict(x_test)

print(f"Accuracy on train data by Naive Bayes Classifier\
: {accuracy_score(y_train, nb_model.predict(x_train))*100}")

cf_matrix=confusion_matrix(y_test, nbModelPredict)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Navie Bayes Classifier on Test Data")
plt.show()

# %%
# Finally, data training with Random Forest Calssifier

randomForestModel=RandomForestClassifier(random_state=18)
randomForestModel.fit(x_train,y_train)
randomForestModelPredict=randomForestModel.predict(x_test)

print(f"Accuracy on test data by Random Forest Classifier\
: {accuracy_score(y_test, preds)*100}")
cf_matrix=confusion_matrix(y_test, randomForestModelPredict)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix,annot=True)
plt.title("Confusion Matrix for Random Forest Classifier on Test Data")
plt.show()

# %% [markdown]
# - Now we will combine all our data, train on the training data and test on the test data.

# %%

svmModelFit = SVC()
nbModelFit = GaussianNB()
rfModelFit = RandomForestClassifier(random_state=18)

svmModelFit.fit(x, y)
nbModelFit.fit(x, y)
rfModelFit.fit(x, y)

test_x = test_data.iloc[:, :-1]
test_y = test_data.iloc[:, -1]

svmPredicts = svmModelFit.predict(test_x)
nbModelPredicts = nbModelFit.predict(test_x)
rfModelPredicts = rfModelFit.predict(test_x)

# Mode function with handling for no clear mode
def safe_mode(lst):
    try:
        return mode(lst)
    except:
        return max(set(lst), key=lst.count)

final_preds = [safe_mode([i, j, k]) for i, j, k in zip(svmPredicts, nbModelPredicts, rfModelPredicts)]

print(f"Accuracy on Test dataset by the combined model: {accuracy_score(test_y, final_preds) * 100}")

cf_matrix = confusion_matrix(test_y, final_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Combined Model on Test Dataset")
plt.show()


# %%
symptoms = x.columns.values

# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
	symptom = " ".join([i.capitalize() for i in value.split("_")])
	symptom_index[symptom] = index

data_dict = {
	"symptom_index":symptom_index,
	"predictions_classes":encoder.classes_
}


def predictDisease(symptoms):
	symptoms = symptoms.split(",")

	input_data = [0] * len(data_dict["symptom_index"])
	for symptom in symptoms:
		index = data_dict["symptom_index"][symptom]
		input_data[index] = 1


	input_data = np.array(input_data).reshape(1,-1)

	# generating individual outputs
	rf_prediction = data_dict["predictions_classes"][rfModelFit.predict(input_data)[0]]
	nb_prediction = data_dict["predictions_classes"][nbModelFit.predict(input_data)[0]]
	svm_prediction = data_dict["predictions_classes"][svmModelFit.predict(input_data)[0]]

	# making final prediction by taking mode of all predictions
	final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])
	predictions = {
		"rf_model_prediction": rf_prediction,
		"naive_bayes_prediction": nb_prediction,
		"svm_model_prediction": svm_prediction,
		"final_prediction":final_prediction
	}
	return predictions

# Testing the function
test_symptoms = "Palpitations,Irritability,Slurred Speech,Excessive Hunger,Vomiting,Fatigue"
test_predictions = predictDisease(test_symptoms)
print(test_predictions)
warnings.filterwarnings("ignore", category=UserWarning)



