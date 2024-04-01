'''TEST'''
'''SUPPORT VECTOR MACHINES'''

#pip install openpyxl
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''Reading the Excel file and converting it to a CSV file'''
#Read the Excel file using pandas
read_file = pd.read_excel(r'C:\Users\Windows\Desktop\test\testdata.xlsx', engine = 'openpyxl')

#Convert to CSV
read_file.to_csv("Test.csv", index = None, header=True)

#Read csv file and put it in a DataFrame object
df = pd.DataFrame(pd.read_csv("Test.csv"))

'''Data Cleaning'''
#print(df)

#remove row 0 and 1
df = df.drop(0)
df = df.drop(1)

#drop columns
#print(df.keys())
#df.pop('Unnamed: 9')

#Check if there are any columns with null values; 1 age value was missing
df.isnull().sum()

# Class/ASD is the dependent variable
dv = df.iloc[:,-1].values

# The rest are independent variables
iv = df.iloc[:,0:-1].values


#print(dv)
#print(iv)
##print(df)

#rename
df = df.rename({'Dataset for People for their Blood Glucose Level with their Superficial Body Feature readings.':'Age'}, axis = 1)
df = df.rename({'Unnamed: 1': 'Blood Glucose Level', 'Unnamed: 2': 'Diastolic BP', 'Unnamed: 3': 'Systolic BP'}, axis = 1)
df = df.rename({'Unnamed: 4': 'Heart Rate', 'Unnamed: 5': 'Body Temp', 'Unnamed: 6': 'SPO2'}, axis = 1)
df = df.rename({'Unnamed: 7': 'Sweating', 'Unnamed: 8': 'Shivering', 'Unnamed: 9': 'Diabetic'}, axis = 1)

#datatypes
#print(df.dtypes)
df['Age'] = df['Age'].astype(int)
df['Blood Glucose Level'] = df['Blood Glucose Level'].astype(int)
df['Diastolic BP'] = df['Diastolic BP'].astype(int)
df['Systolic BP'] = df['Systolic BP'].astype(int)
df['Heart Rate'] = df['Heart Rate'].astype(int)
df['Body Temp'] = df['Body Temp'].astype(float)
df['SPO2'] = df['SPO2'].astype(int)
df['Sweating'] = df['Sweating'].astype(str)
df['Shivering'] = df['Shivering'].astype(str)
df['Diabetic'] = df['Diabetic'].astype(str)

print(df)

'''SVM ALgorithm (LINEAR)'''
#80% training and 20% test
target = df['Diabetic']
data = df.drop('Diabetic', axis = 1)

#Split into training and test data
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

#Generate the model
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

# Model Accuracy: how often is the classifier correct?
print("LINEAR KERNEL")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))