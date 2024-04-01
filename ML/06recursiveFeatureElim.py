
#pip install openpyxl
#pip install imbalanced-learn
#pip install lime
#pip install shap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import imblearn
import lime
import shap
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from lime import lime_tabular, submodular_pick


#Function that does the EDA for categorical data using Seaborn, Outputs a bar graph
def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', verbose=True):
    if x == None:
        column_interested = y
    else:
        column_interested = x
    series = dataframe[column_interested]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        #print(series.value_counts())

    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette)
    #plt.show()



#Function that does the EDA for quantitative data using Seaborn, outputs a boxplot
def quantitative_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', ax=None, verbose=True, swarm=False):
    series = dataframe[y]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        #print(series.value_counts())

    sns.boxplot(x=x, y=y, hue=hue, data=dataframe, palette=palette, ax=ax)

    if swarm:
        sns.swarmplot(x=x, y=y, hue=hue, data=dataframe,
                      palette=palette, ax=ax)

    #plt.show()




'''PREPROCESSING'''
#Read csv file and put it in a DataFrame object
df = pd.DataFrame(pd.read_csv("Test.csv"))

#remove row 0 and 1
df = df.drop(0)
df = df.drop(1)

#Rename column names in the df
df = df.rename(columns={'Dataset for People for their Blood Glucose Level with their Superficial Body Feature readings.': 'Age',
                        'Unnamed: 1': 'Blood Glucose',
                        'Unnamed: 2': 'Diastolic Blood Pressure',
                        'Unnamed: 3': 'Systolic Blood Pressure',
                        'Unnamed: 4': 'Heart Rate',
                        'Unnamed: 5': 'Body Temperature',
                        'Unnamed: 6': 'SPO2',
                        'Unnamed: 7': 'Sweating',
                        'Unnamed: 8': 'Shivering',
                        'Unnamed: 9': 'Diabetic'})

#Change the row indices
df = df.rename(index = lambda x: x-2)

#Change the datatypes of the columns
convert_dict = {'Age': int,
                'Blood Glucose': int,
                'Diastolic Blood Pressure': int,
                'Systolic Blood Pressure': int,
                'Heart Rate': int,
                'Body Temperature': float,
                'SPO2': int,
                'Sweating': int,
                'Shivering': int,
                'Diabetic': str
                }

df = df.astype(convert_dict)

#Check datatypes to see if they were converted correctly
#print(df.dtypes)

#Check for the number of missing values per column
#print(df.isna().sum())




'''EDA'''
#Visualize the data
#Create a histogram using Plotly Express
fig = px.histogram(df, x='Age')
fig = px.histogram(df, x='Blood Glucose')
fig = px.histogram(df, x='Diastolic Blood Pressure')
fig = px.histogram(df, x='Systolic Blood Pressure')
fig = px.histogram(df, x='Heart Rate')
fig = px.histogram(df, x='Body Temperature')
fig = px.histogram(df, x='SPO2')
fig = px.histogram(df, x='Sweating')
fig = px.histogram(df, x='Shivering')
fig = px.histogram(df, x='Diabetic')
#fig.show()

#Create a boxplot using Plotly Express
fig = px.box(df, x='Age')
fig = px.box(df, x='Blood Glucose') #Has
fig = px.box(df, x='Diastolic Blood Pressure')
fig = px.box(df, x='Systolic Blood Pressure') #Has a bit
fig = px.box(df, x='Heart Rate') #Has numerous
fig = px.box(df, x='Body Temperature')
fig = px.box(df, x='SPO2') #Has
fig = px.box(df, x='Sweating')
fig = px.box(df, x='Shivering')
fig = px.box(df, x='Diabetic')
#fig.show()

#EDA using Seaborn and functions
categorical_summarized(df, y='Age')
quantitative_summarized(df, y='Blood Glucose')
quantitative_summarized(df, y='Diastolic Blood Pressure')
quantitative_summarized(df, y='Systolic Blood Pressure')
quantitative_summarized(df, y='Heart Rate')
quantitative_summarized(df, y='Body Temperature')
quantitative_summarized(df, y='SPO2')
categorical_summarized(df, y='Sweating')
categorical_summarized(df, y='Shivering')
categorical_summarized(df, y='Diabetic')

#Make the target Diabetic column have values of 0 or 1
#0=Diabetic, 1=Non-diabetic
lecols = ['Diabetic']
df[lecols] = df[lecols].apply(LabelEncoder().fit_transform)

#Correlation using Seaborn
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', size = 15)
colormap = sns.diverging_palette(10, 220, as_cmap = True)
sns.heatmap(df.corr(),
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linewidths=0.1,vmax=1.0, linecolor='white',
            annot_kws={'fontsize':12 })
#plt.show()




'''MACHINE LEARNING PROPER'''
#80-20 split
target = df['Diabetic']
data = df.drop(['Diabetic'], axis = 1)
#data = data.iloc[:, :-1].values

#Split into training and test data
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

#Count the number of instances the target variable makes
#print(df['Diabetic'].value_counts())

#Count the number of instances the target variable makes before and after SMOTE
#print(Counter(y_train))
#print(Counter(y_train_smote))


#We will now implement Recursive Feature Elimination with Cross Validation to get the optimal features
#We will use a Random Forest model to evaluate the features during the recursive search
rfecv_model = RandomForestClassifier()
rfecv = RFECV(estimator = rfecv_model,step=1, cv=5, scoring="roc_auc")
#Optimal features = 9

#Fit the data to our RFECV model
rfecv.fit(x_train, y_train)

#Print the output of the RFECV model
print("Optimal number of features:", rfecv.n_features_)
print("Best features:", x_train.columns[rfecv.support_])
#Optimal number of features = 8

#Now that we have the features that we need, we can reduce the dataset to these reduced features
x_train_rfe = rfecv.transform(x_train)
x_test_rfe = rfecv.transform(x_test)




'''NAIVE BAYES'''
#Generate the model
# Training the Naive Bayes model on the Training set
classifier = GaussianNB()
classifier.fit(x_train_rfe, y_train) #x_train_smote here is a dataFrame

# Predicting the Test set results
y_pred = classifier.predict(x_test_rfe)

#Metrics
print("NAIVE BAYES")
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("AUROC score: ", roc_auc_score(y_test, y_pred))
print("===================================")



'''LOGISTIC REGRESSION'''
# instantiate the model (using the default parameters)
logreg = LogisticRegression(max_iter=7600)

# fit the model with data
logreg.fit(x_train_rfe, y_train)

#Predict the response for the dataset
y_pred=logreg.predict(x_test_rfe)

#Metrics
print("LOGISTIC REGRESSION")
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("AUROC score: ", roc_auc_score(y_test, y_pred))
print("===================================")



'''KNN'''
# Scale the features using StandardScaler
scaler = StandardScaler()
x_train_2 = scaler.fit_transform(x_train_rfe)
x_test_2 = scaler.transform(x_test_rfe)

'''
#What value of k should we use?
k_values = [i for i in range (1,31)]
k_scores = []

scaler = StandardScaler()
X = data
X = scaler.fit_transform(X)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, target, cv=5)
    k_scores.append(np.mean(score))

#Print the graph to determine the k with the highest accuracy score/s
plt.figure(figsize=(14,12))
plt.title('K Values VS Accuracy', size = 15)
sns.lineplot(x = k_values, y = k_scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
#plt.show()
#k=3 gives us the highest accuracy
'''

#Generate the model
knn = KNeighborsClassifier(n_neighbors=3)

#Train the model using the training sets
knn.fit(x_train_2, y_train) #x_train_smote here is a dataFrame

#Predict response for dataset
y_pred = knn.predict(x_test_2)

#Metrics
print("K NEAREST NEIGHBORS")
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("AUROC score: ", roc_auc_score(y_test, y_pred))
print("===================================")



'''RANDOM FOREST'''
#Generate the model
rf = RandomForestClassifier()

#Convert the x_train_smote dataFrame into an array
#x_train_rf = x_train_rfe.values

#Train the model using the training sets
rf.fit(x_train_rfe, y_train)
model_shap = rf.fit(x_train_rfe, y_train)

#Predict response for dataset
y_pred = rf.predict(x_test_rfe)

# Metrics
print("RANDOM FOREST")
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("AUROC score: ", roc_auc_score(y_test, y_pred))
print("===================================")



'''SVM'''
#Generate the model
#Create a SVM Classifier
clf = svm.SVC(kernel='linear', cache_size=7600) # Linear Kernel

#Try to scale data to [-1,1] in order to increase SVM speed:
scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train_rfe)
x_train_svm = scaling.transform(x_train_rfe)
x_test_svm = scaling.transform(x_test_rfe)

#Train the model using the training sets
clf.fit(x_train_svm, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test_svm)


# Model Accuracy: how often is the classifier correct?
print("SVM: LINEAR KERNEL")
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))
print("AUROC score: ", roc_auc_score(y_test, y_pred))
print("===================================")

#The SVM is taking a really long time to train due to the amount of training data
#It would be best for the data the SVM handles to be scaled
#So far, the best performing model is the Random Forest



