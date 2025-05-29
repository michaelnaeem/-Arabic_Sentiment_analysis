#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle
import PreProcessing
import matplotlib.pyplot as plt


# In[ ]:


with open('models/SVC_Classifier.sav', 'rb') as f:
    model = pickle.load(f)


# In[ ]:


with open('models/vectorizer.sav', 'rb') as f:
    victorizer = pickle.load(f)


# In[ ]:


def predict(sentence):
    sentence = PreProcessing.SentencePreProcess(sentence)
    pred = model.predict(victorizer.transform([sentence]))[0]
    percent = round(model.predict_proba(victorizer.transform([sentence]))[0][0]*100, 1)
    if pred == 'POS':
        return ("هذه الجملة ايجابية", percent)
    else:
        return ("هذه الجملة سلبية", percent)


# In[ ]:


def predict_file(location):
    df = PreProcessing.PreProcess(location)
    predictions = []
    percentages = []
    for sentence in df['preprocessed_text']:
        predictions.append(model.predict(victorizer.transform([sentence]).toarray())[0])
        per = round(model.predict_proba(victorizer.transform([sentence]))[0][0]*100, 1)
        if per < 50 :
            percentages.append(100 - per)
        else: percentages.append(per)
    df['class'] = predictions
    df['class'] = df['class'].replace("POS","Positive").replace("NEG","Negative")
    df['confidence'] = percentages
    return df


# In[ ]:


def piechart(df):
    labels = df['class'].value_counts().index
    sizes = df['class'].value_counts().values
    colors = ['lightcoral', 'yellowgreen']
    explode = (0.08, 0)
    plt.figure(figsize=(2.2, 2.2))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title('Classification Percentages')
    plt.axis('equal')
    
    plt.savefig('assets/frame1/pie_chart.png', transparent = True)

