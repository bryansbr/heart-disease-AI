#!/usr/bin/env python
# coding: utf-8

# <div align="center">
#     <h1>Heart Disease</h1>
# </div>
# 
# <div>
#     <h1>Introduction to Artificial Intelligence | Project 2 | Universidad del Valle</h1>
# </div>
# 
# ![Imagen](https://img.webmd.com/dtmcms/live/webmd/consumer_assets/site_images/articles/health_tools/did_you_know_this_could_lead_to_heart_disease_slideshow/493ss_thinkstock_rf_heart_illustration.jpg)
# 
# ## Authors
# - Bryan Steven Biojó     - 1629366
# - Julián Andrés Castaño  - 1625743
# - Juan Sebastián Saldaña - 1623447
# - Juan Pablo Rendón      - 1623049
# 
# ## Objective
# - Apply the concept of Machine Learning (ML) to solve a **classification problem** using the methods seen in the course.
# 
# ## 1. Importing libraries
# As a first step, the libraries used during the development of the problem will be imported:

# In[29]:


# Common libraries
import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import re
import os
import sys
import warnings
warnings.filterwarnings('ignore')
from pandas.api.types import CategoricalDtype
from matplotlib.legend_handler import HandlerLine2D
from IPython.display import SVG, display
from graphviz import Source

# Sklearn libraries
import sklearn
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
#from sklearn.externals.six import StringIO

# Keras libraries
from keras.models import Sequential
from keras.layers.core import Dense

# Disabling TensorFlow warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Another function that captures warnings
def warn(*args, **kwargs):
    pass


# ## 2. Loading the dataset
# Next, we will load the dataset with the heart diseases which are in a CSV file. Said data was extracted directly from the **Kaggle** website (https://www.kaggle.com/ronitf/heart-disease-uci) and uploaded again to the following **GitHub** repository (https://github.com/bryansbr/heart-disease-AI:

# In[47]:


# Heard disease dataset
url = "https://raw.githubusercontent.com/bryansbr/heart-disease-AI/main/heart.csv"
data = pd.read_csv(url)
print(data.columns)
print(data.shape)
data.head()
#data.describe()


# ## 3. Data description
# In total we have 14 columns with the following information:
# 
# - **age:** Age in years.
# - **sex:** Where (1 = male; 0 = female).
# - **cp:** Chest pain type. Where (1 = angina; 2 = pain without angina; 3 = asymptomatic).
# - **trestbps:** Resting blood pressure (in mm/Hg on admission to the hospital).
# - **chol:** Serum cholesterol of the person in mg/dl.
# - **fbs:** Fasting blood sugar > 120 mg/dl. Where (1 = true; 0 = false).
# - **restecg:** Resting electrocardiographic results. Where (0 = normal; 1 = with ST-T wave abnormality (T wave inversions and/or ST elevation or depression > 0.05 mV); 2 = showing probable or definitive left ventricular hypertrophy according to Romhilt criteria-Estes).
# - **thalach:** Maximum heart rate achieved.
# - **exang:** Exercise induced angina (1 = yes; 0 = no). 
# - **oldpeak:** ST depression induced by exercise relative to rest.
# - **slope:** The slope of the peak exercise ST segment. Where (0 = ascending slope; 1 = flat; 2 = descending slope).
# - **ca:** Number of major vessels (0 - 3) colored by flourosopy.
# - **thal:** Blood disorder known as 'Thalassemia'. Where (3 = normal; 6 = fixed defect; 7 = reversable defect).
# - **target:** Indicates the probability of suffering from heart disease, according to the information in the preceding columns (1 = yes; 0 = no). This is the column that we want to **predict** with our ML models.
# 
# ## 4. Types of variables
# Now, we will group the variable types into numeric or categorical as appropriate. The **numerical variables** are those statistical variables that give, as a result, a numerical value and these can be discrete or continuous, while the **categorical variables** can take one of a limited number, and usually fixed, of possible values that are base of some qualitative characteristic.
# 
# According to the above, the grouping of the variables would be as follows:
# 
# |   **Variable**  |   **Type**  |
# |-----------------|-------------|
# |    **age**      |  numerical  |
# |     **sex**     | categorical |
# |    **cp**       | categorical |
# |   **trestbps**  |  numerical  |
# |     **chol**    |  numerical  |
# |     **fbs**     | categorical |
# |   **restecg**   | categorical |
# |   **thalach**   |  numerical  |
# |    **exang**    | categorical |
# |   **oldpeak**   |  numerical  |
# |    **slope**    | categorical |
# |     **ca**      | categorical |
# |    **thal**     | categorical |
# |   **target**    | categorical |
# 
# 
# ## 5. Grouping of variables according to their type
# Before graphing, we must verify that our variables correspond to the type in which they have been classified (numerical or categorical). There must be no missing or null data, as well as strings of characters where numbers must go and vice versa.
# 
# ### 5.1. Checking for missing or null data
# We check for missing or null data in our dataset. If they exist, we must complete or delete them as appropriate.

# In[48]:


# We check if there is missing or null data
print("NaN data exists in the dataset?: ")
print(data.isna().any().any())
print("---------------------------------")
print("null data exists in the dataset?:")
print(data.isnull().any().any())


# In this case, we see that there is no null or missing data, so we can proceed to make the respective conversions.
# 
# ### 5.2. Grouping and converting variables
# We group the numerical and categorical variables and the target variable.

# In[62]:


# Array for numerical variables
numerical_vars = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Array for categorical variables
categorical_vars = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Target variable
target = "target"

# We convert the corresponding attributes to numerical. In Python, numerical is the default datatype
for columns in numerical_vars:
    data[columns] = pd.to_numeric(data[columns], errors='coerce')
    
# Now, we convert the corresponding attributes to categorical.
for columns in categorical_vars:
    cat_type = CategoricalDtype(categories = None, ordered = False)
    data[columns] = data[columns].astype(dtype = cat_type)


# ## 6. Graphing the variables
# According to the previous information, the graphs of the variables are made. The numerical variables will be represented as **histograms**, while the categorical variables as **pie diagrams**.
# 
# ### 6.1. Checking for missing or null data
# Before graphing, let's check for missing or null data in our dataset. If they exist, we must complete or remove them as appropriate.

# In this case, we see that there are no null or missing data, so we can proceed to graph the variables according to their grouping.
# 
# ### 6.2. Numerical variables
# - age.
# - trestbps.
# - chol.
# - thalach.
# - oldpeak.

# In[50]:


# Figure type object to develop the subplots
fig = plt.figure(figsize = (20, 10))
i = 1

# Graphs of type histogram numerical variables
for num_attrs in numerical_vars:
    ax = fig.add_subplot(3, 5, i)
    data[[num_attrs]].plot(kind = 'hist', ax = ax, rwidth = 1)
    i = i + 1


#  ### 6.3. Categorical variables
# - sex
# - cp
# - fbs
# - restecg
# - exang
# - slope
# - ca
# - thal
# - target

# In[61]:


# Figure type object to develop the subplots
fig = plt.figure(figsize = (15, 45))
i = 1

# Graphs of type histogram categorical variables
for cat_attrs in categorical_vars:
    ax = fig.add_subplot(6, 3, i)
    data[cat_attrs].value_counts().plot(kind = 'pie', ax = ax, startangle = 115, fontsize = 12)
    i = i + 1

# Graphing the target variable individually 'target'
fig = plt.figure(figsize = (15, 45))
ax = fig.add_subplot(1, 2, 1)
data["target"].value_counts().plot(kind = 'pie',
                                   figsize = (12, 10),
                                   autopct = '%1.1f%%', # Add in percentages
                                   startangle = 90, # Start angle 90° (Africa)
                                   shadow = True, # Add shadow
                                  )
plt.title("target")

# Show the pie charts
plt.show()


# ## 7. Preparing the data for classification
# There are some methods that do not accept categorical variables as inputs to algorithms. For this reason, we must convert categorical variables to numeric and for this, we will use dummies variables.

# In[63]:


# We make a copy of the original dataset
data_copy = data.copy()

# We convert categorical variables to numerical
for cat_attrs in categorical_vars:
    dummies = pd.get_dummies(data_copy[cat_attrs], prefix = cat_attrs)
    data_copy = pd.concat([data_copy.drop(cat_attrs, axis = 1), dummies], axis = 1)

# We display the dataset again with the categorical variables converted to numerical
print(data_copy.columns)
print(data_copy.shape)
data_copy.head()
#data_copy.describe()


# ## 8. Generating the training and test datasets
# Next, we will divide the data set into two parts: 80% for training and 20% for testing.

# In[67]:


# We divide the dataset into two parts
features = data_copy.target.values
tg = data_copy.drop("target", axis = 1)
trainX, testX, trainY, testY = train_test_split(tg, features, test_size = 0.2, stratify = features)

# We print the training and test dataset
print("Number of training tuples:")
print(trainX.shape, trainY.shape)
print("--------------------------")
print("Number of test tuples:")
print(testX.shape, testY.shape)


# In[ ]:




