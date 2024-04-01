
#pip install openpyxl
#pip install imbalanced-learn
#pip install lime
#pip install shap
#pip install kaleido
#pip install joblib
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import imblearn
import lime
import shap
import pickle
import joblib
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
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
fig = px.histogram(df, x='Sweating') #0 = yes, 1 = no
fig = px.histogram(df, x='Shivering') #0 = yes, 1 = no
fig = px.histogram(df, x='Diabetic')
#fig.show()

#Create a boxplot using Plotly Express
fig = px.box(df, x='Age')
fig = px.box(df, x='Blood Glucose') 
fig = px.box(df, x='Diastolic Blood Pressure')
fig = px.box(df, x='Systolic Blood Pressure') 
fig = px.box(df, x='Heart Rate') 
fig = px.box(df, x='Body Temperature')
fig = px.box(df, x='SPO2') 
fig = px.box(df, x='Sweating') #0 = yes, 1 = no
fig = px.box(df, x='Shivering') #0 = yes, 1 = no
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
categorical_summarized(df, y='Sweating') #0 = yes, 1 = no
categorical_summarized(df, y='Shivering') #0 = yes, 1 = no
categorical_summarized(df, y='Diabetic')
#plt.show()

#Make the target Diabetic column have values of 0 or 1
#0=Diabetic, 1=Non-diabetic
lecols = ['Diabetic']
df[lecols] = df[lecols].apply(LabelEncoder().fit_transform)

#Change the temp unit from F to C
for i in range(0, len(df['Body Temperature'])):
    df['Body Temperature'][i] = (df['Body Temperature'][i] - 32) / 1.8
#print(df['Body Temperature'])

'''
print(Counter(df['Sweating']))
print(Counter(df['Shivering']))
print(Counter(df['Diabetic']))
'''

#Correlation using Seaborn
plt.figure(figsize=(14,24))
plt.title('Pearson Correlation of Features', size = 14)
colormap = sns.diverging_palette(10, 220, as_cmap = True)
sns.heatmap(df.corr(),
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linewidths=0.1,vmax=1.0, linecolor='white',
            annot_kws={'fontsize': 10})
#sns.set(font_scale=0.5)
plt.show()




'''MACHINE LEARNING PROPER'''
#80-20 split
target = df['Diabetic']
data = df.drop(['Diabetic'], axis = 1)
#data = data.iloc[:, :-1].values

#Split into training and test data
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

#Count the number of instances the target variable makes
#print(df['Diabetic'].value_counts())

#Test the shape
print("x train shape: ", x_train.shape)
print("y train shape: ", y_train.shape)
print("x test shape: ", x_test.shape)
print("y test shape: ", y_test.shape)

#It can be observed we have more Diabetic(0) than Non-diabetic(1) patients in the dataset
#To combat this imbalance, we will attempt to use SMOTE to oversample the minority data class
#which is the Non-diabetic (1) class

#Oversampling the test dataset using SMOTE, do not do anything to the test dataset
sm = SMOTE(random_state = 2)
x_train_smote, y_train_smote = sm.fit_resample(x_train, y_train.ravel())

#Count the number of instances the target variable makes before and after SMOTE
print(Counter(y_train))
print(Counter(y_train_smote))





'''RANDOM FOREST'''
#Generate the model
rf = RandomForestClassifier()

#Convert the x_train_smote dataFrame into an array
x_train_rf = x_train_smote.values

#Train the model using the training sets
rf.fit(x_train_rf, y_train_smote.ravel())
model_shap = rf.fit(x_train_rf, y_train_smote.ravel())

#Predict response for dataset
y_pred = rf.predict(x_test)

# Metrics
print("RANDOM FOREST")
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("AUROC score: ", roc_auc_score(y_test, y_pred))
print("===================================")

#Plot the confusion matrix as a heatmap using seaborn
fig, ax = plt.subplots(figsize=(12, 8))
sns.set(font_scale=1)
sns.heatmap(np.eye(2), 
            annot=confusion_matrix(y_test, y_pred),
            annot_kws={'size': 18},
            xticklabels=['0 - Diabetic', '1 - Non-diabetic'], 
            yticklabels=['0 - Diabetic', '1 - Non-diabetic'], 
            cmap=sns.color_palette(['#3A0CA3','#F72585'], as_cmap=True), 
            fmt='g')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.tick_params(labelsize=14, length=0)

ax.set_title('Seaborn Confusion Matrix', size=22, pad=20)
ax.set_xlabel('Predicted Values', size=14)
ax.set_ylabel('Actual Values', size=14)

plt.show()

#Let us try to predict a new datapoint
#0=Diabetic, 1=Non-diabetic
new_entry = [[9, 100, 96, 110, 87, 38, 94, 0, 0]]
prediction = rf.predict(new_entry)
print("Prediction: ", prediction)
#predicion = 0 means diabetic

'''
#Saving the model using joblib to our local working directory
joblib.dump(rf, 'bestModel')
'''

