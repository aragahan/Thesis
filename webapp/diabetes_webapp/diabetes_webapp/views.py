from django.shortcuts import render
from django.http import JsonResponse
import os
#import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
#import plotly.express as px
#import imblearn
import lime
#import shap
import pickle
import joblib
import dill
import base64
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
#from sklearn.model_selection import train_test_split
#from sklearn import metrics
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
#from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
#from imblearn.over_sampling import SMOTE
#from imblearn.pipeline import Pipeline as imbpipeline
from lime import lime_tabular, submodular_pick
from io import BytesIO
from PIL import Image

#Import the ML model
rf_saved = joblib.load('diabetes_webapp/bestModel')

#import the LIME Explainer
explainer = ""
with open('diabetes_webapp/lime_explainer', 'rb') as f:
    explainer = dill.load(f)

#Function to convert an image to base64 format so we can put it in an <img> tag in our template
def img_to_base64(filename):
    buff = BytesIO()
    filename.save(buff, format='png')
    filename_str = base64.b64encode(buff.getvalue())
    filename_str = filename_str.decode("utf-8")
    buff.close()
    return filename_str

def index(request):
    context = {}
    return render(request, 'index.html', context)

def predict(request):
    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file', False)
        datapoint = ''
        
        if csv_file == False:
            #Get datapoint thru form input
            datapoint = np.array([request.POST.get('age'),  
                                  request.POST.get('glucose_level'), 
                                  request.POST.get('diastolic_bp'),
                                  request.POST.get('systolic_bp'), 
                                  request.POST.get('heart_rate'), 
                                  request.POST.get('body_temp'), 
                                  request.POST.get('SPO2'), 
                                  request.POST.get('sweating'), 
                                  request.POST.get('shivering')])
        
        else:
            #Convert CSV input to a numpy array datapoint
            datapoint = np.loadtxt(csv_file, delimiter=',')
        

        #Predict the inputs from the form
        model_predicion = rf_saved.predict([datapoint])

        #LIME using the saved Random Forest model
        #Import the saved "explainer" object

        #Change the datapoint into a float
        datapoint_float = datapoint.astype(float)

        #Generate a LIME explainer for the datapoint
        exp = explainer.explain_instance(
            data_row=datapoint_float, 
            predict_fn=rf_saved.predict_proba
        )

        #Show the resulting explainer graph as HTML and as a plotly graph
        result = exp.as_html(labels=None, predict_proba=True, show_predicted_value=False)
        #print(result)
        model = exp.as_pyplot_figure()

        #Save the graphed explainer figure as an image
        plt.figure(model)
        plt.xlabel('', fontsize=12)
        plt.ylabel('', fontsize=12)
        plt.title('Local explanation')
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img2 = Image.open(img)

        #Convert the image to base64
        img3 = img_to_base64(img2)

        context = {'result': result, 'final_image': img3, 'model_prediction': model_predicion}

    return render(request, 'predict.html', context)
