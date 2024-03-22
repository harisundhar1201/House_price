#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction

# In[48]:


#Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import folium
from folium.plugins import FastMarkerCluster

from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

import warnings
warnings.filterwarnings("ignore")


# In[49]:


# Importing the dataset

data = pd.read_csv('home_data.csv')
data


# In[50]:


#droping the unnecessary columns such as id, date

data.drop(['id','date','sqft_above','sqft_basement','sqft_living15','sqft_lot15','zipcode'],axis=1,inplace=True)
data.head()


# In[51]:


data.info()


# In[52]:


data.describe()


# In[53]:


# checking for null values/missing values

data.isnull().sum()


# In[54]:


data.nunique()


# # Data Preprocessing

# In[55]:


# changing float to integer

data['bathrooms'] = data['bathrooms'].astype(int)
data['floors'] = data['floors'].astype(int)

# renaming the column yr_built to age and changing the values to age

data.rename(columns={'yr_built':'age'},inplace=True)
data['age'] = 2023 - data['age']
# print(data["age"])

# changing the column yr_renovated to renovated and changing the values to 0 and 1

data.rename(columns={'yr_renovated':'renovated'},inplace=True)
data['renovated'] = data['renovated'].apply(lambda x: 0 if x == 0 else 1)
# print(data["renovated"])
data.head()


# # Exploratory Data Analysis

# ### Correlation Matrix to find the relationship between the variables

# In[56]:


# using correlation statistical method to find the relation between the price and other features

data.corr()['price'].sort_values(ascending=False)


# In[57]:


plt.figure(figsize=(23,23))
sns.heatmap(data.corr(),annot=True)
plt.show()


# ### Visualizing the coorelation with price

# In[58]:


data.corr()['price'].sort_values().plot(kind='bar')


# ### Visualizing the relation between price and sqft_living, sqft_lot, sqft_above, sqft_basement, sqft_living15, sqft_lot15, age, renovated, bedrooms, bathrooms, floors, waterfront, view, condition, grade

# In[59]:


sns.scatterplot( x = data['bedrooms'], y = data['price'])
plt.show()


# In[60]:


sns.lineplot( x = data['bathrooms'], y = data['price'])
plt.show()


# In[61]:


sns.boxplot( x = data['renovated'], y = data['price'])
plt.show()


# In[62]:


sns.lineplot( x = data['age'], y = data['price'])
plt.show()


# In[63]:


sns.scatterplot( x = data['sqft_living'], y = data['price'])
plt.show()


# In[64]:


sns.scatterplot( x = data['sqft_lot'], y = data['price'])
plt.show()


# In[65]:


sns.barplot( x = data['floors'], y = data['price'])
plt.show()


# In[66]:


sns.boxplot( x = data['waterfront'], y = data['price'])
plt.show()


# In[67]:


sns.barplot( x = data['view'], y = data['price'])
plt.show()


# In[68]:


sns.lineplot( x = data['grade'], y = data['price'])
plt.show()


# In[69]:


sns.scatterplot( x = data['long'], y = data['price'])
plt.show()


# In[70]:


sns.scatterplot( x = data['lat'], y = data['price'])
plt.show()


# In[71]:


sns.barplot( x = data['condition'], y = data['price'])
plt.show()


# In[72]:


sns.lineplot( x = data['age'], y = data['renovated'])
plt.show()


# ### Plotting the location of the houses based on longitude and latitude on the map

# In[73]:


# adding a new column price_range and categorizing the price into 4 categories
price_range = pd.cut(data['price'],bins=[0,321950,450000,645000,1295648],labels=['Low','Medium','High','Very High'])
map = folium.Map(location=[47.560053,-122.213896],zoom_start=5)
marker_cluster = FastMarkerCluster(data[['lat', 'long']].values.tolist()).add_to(map)
map


# # Train/Test Split

# In[74]:


x = data.drop(['price'],axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(x , y,test_size=0.3,random_state=10)


# ## Model Training

# ### Using pipeline to combine the transformers and LinearRegression

# In[75]:


input = [('scale',StandardScaler()),('polynomial', PolynomialFeatures(degree=2)),('model',LinearRegression())]
pipe = Pipeline(input)

pipe.fit(X_train,y_train)


# In[76]:


#testing the model

pipe_pred = pipe.predict(X_test)
r2_score(y_test,pipe_pred)


# ## Ridge Regression

# In[77]:


Ridgemodel = Ridge(alpha = 0.001)
Ridgemodel.fit(X_train,y_train)


# In[78]:


#testing the model

r_pred = Ridgemodel.predict(X_test)
r2_score(y_test,r_pred)


# ## Random Forest Regression

# In[79]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)

regressor.fit(X_train,y_train)


# In[80]:


#testing the model

yhat = regressor.predict(X_test)
r2_score(y_test,yhat)


# ## Model Evalution

# ### Distribution plot from the models predictions and the actual values

# In[81]:


fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# Plot for Linear Regression
sns.distplot(y_test, ax=ax[0])
sns.distplot(pipe_pred, ax=ax[0])
ax[0].legend(['Actual Price', 'Predicted Price'])
ax[0].set_title('Linear Regression')

# Plot for Ridge Regression
sns.distplot(y_test, ax=ax[1])
sns.distplot(r_pred, ax=ax[1])
ax[1].legend(['Actual Price', 'Predicted Price'])
ax[1].set_title('Ridge Regression')

# Plot for Random Forest Regression
sns.distplot(y_test, ax=ax[2])
sns.distplot(yhat, ax=ax[2])
ax[2].legend(['Actual Price', 'Predicted Price'])
ax[2].set_title('Random Forest Regression')

plt.tight_layout()
plt.show()


# ### Error Evaluation

# In[82]:


# Create figure and axes
fig, ax = plt.subplots(3, 1, figsize=(5, 10))

# Plot Mean Absolute Error
sns.barplot(x=['Linear Regression', 'Ridge Regression', 'Random Forest'],
            y=[mean_absolute_error(y_test, pipe_pred),
               mean_absolute_error(y_test, r_pred),
               mean_absolute_error(y_test, yhat)],
            ax=ax[0])
ax[0].set_ylabel('Mean Absolute Error')
ax[0].set_title('Comparison of Mean Absolute Error')

# Plot Mean Squared Error
sns.barplot(x=['Linear Regression', 'Ridge Regression', 'Random Forest'],
            y=[mean_squared_error(y_test, pipe_pred),
               mean_squared_error(y_test, r_pred),
               mean_squared_error(y_test, yhat)],
            ax=ax[1])
ax[1].set_ylabel('Mean Squared Error')
ax[1].set_title('Comparison of Mean Squared Error')

# Plot Root Mean Squared Error
sns.barplot(x=['Linear Regression', 'Ridge Regression', 'Random Forest'],
            y=[np.sqrt(mean_squared_error(y_test, pipe_pred)),
               np.sqrt(mean_squared_error(y_test, r_pred)),
               np.sqrt(mean_squared_error(y_test, yhat))],
            ax=ax[2])
ax[2].set_ylabel('Root Mean Squared Error')
ax[2].set_title('Comparison of Root Mean Squared Error')

plt.tight_layout()
plt.show()


# ### Accuracy Evaluation

# In[83]:


# plot accuracy of all models in the same graph
fig, ax = plt.subplots(figsize=(7,5))
sns.barplot(x=['Linear Regression','Ridge Regression','Random Forest Regression'],y=[metrics.r2_score(y_test,pipe_pred),metrics.r2_score(y_test,r_pred),metrics.r2_score(y_test,yhat)])
ax.set_title('Accuracy of all models')
plt.show()


# ## Predicting the price of a new house

# In[84]:


#input the values

bedrooms = 3
bathrooms = 2
sqft_living = 2000
sqft_lot = 10000
floors = 2
waterfront = 0
view = 0
condition = 3
grade = 8
yr_built = 1990
yr_renovated = 0
lat = 47.5480
long = -121.9836


# In[85]:


#predicting the price using random forest regression
price = regressor.predict([[bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,grade,yr_built,yr_renovated,lat,long]])
print('The price of the house is $',price[0])


# In[86]:


import pandas as pd
from dash import html, dcc, Input, Output, State, Dash

# Initialize the Dash app
app = Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H1("House Price Prediction", style={'text-align': 'center'}),
        
        html.Div([
            dcc.Input(id='bedrooms', type='number', placeholder='Bedrooms',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='bathrooms', type='number', placeholder='Bathrooms',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='sqft_living', type='number', placeholder='Sqft_living',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='sqft_lot', type='number', placeholder='Sqft_lot',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='floors', type='number', placeholder='Floors',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='waterfront', type='number', placeholder='Waterfront',
                      style={'margin': '10px', 'padding': '10px'}),            
            dcc.Input(id='view', type='number', placeholder='View',
                      style={'margin': '10px', 'padding': '10px'}),            
            dcc.Input(id='condition', type='number', placeholder='Condition',
                      style={'margin': '10px', 'padding': '10px'}),            
            dcc.Input(id='grade', type='number', placeholder='Grade',
                      style={'margin': '10px', 'padding': '10px'}),            
            dcc.Input(id='age', type='number', placeholder='Age',
                      style={'margin': '10px', 'padding': '10px'}),            
            dcc.Input(id='renovated', type='number', placeholder='Renovated',
                      style={'margin': '10px', 'padding': '10px'}),            
            dcc.Input(id='lat', type='number', placeholder='Lat',
                      style={'margin': '10px', 'padding': '10px'}),            
            dcc.Input(id='long', type='number', placeholder='Long',
                      style={'margin': '10px', 'padding': '10px'}),
            html.Button('Predict Price', id='predict_button',
                        style={'margin': '10px', 'padding': '10px', 'background-color': '#007BFF', 'color': 'white'}),
        ], style={'text-align': 'center'}),
        
        html.Div(id='prediction_output', style={'text-align': 'center', 'font-size': '20px', 'margin-top': '20px'})
    ], style={'width': '50%', 'margin': '0 auto', 'border': '2px solid #007BFF', 'padding': '20px', 'border-radius': '10px'})
])

# Define callback to update output
@app.callback(
    Output('prediction_output', 'children'),
    [Input('predict_button', 'n_clicks')],
    [State('bedrooms', 'value'), 
     State('bathrooms', 'value'),
     State('sqft_living', 'value'), 
     State('sqft_lot', 'value'),
     State('floors', 'value'), 
     State('waterfront', 'value'),
     State('view', 'value'),
     State('condition', 'value'),
     State('grade', 'value'),
     State('age', 'value'),
     State('renovated', 'value'),
     State('lat', 'value'),
     State('long', 'value')]
)
def update_output(n_clicks, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view,
                  condition, grade, age, renovated, lat, long):
    if n_clicks is not None and n_clicks > 0 and all(v is not None for v in [bedrooms, bathrooms, sqft_living, sqft_lot,
                                                     floors, waterfront, view, condition,
                                                     grade, age, renovated, lat, long]):
        # Prepare the feature vector
        features = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, 
                                  view, condition, grade, age, renovated, lat, long]], 
                                columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                                         'floors', 'waterfront', 'view', 'condition', 'grade',
                                         'age', 'renovated', 'lat', 'long'])
        # Predict
        prediction = regressor.predict(features)[0]
        return f'Predicted House Price: ${prediction:.2f}'
    elif n_clicks is not None and n_clicks > 0:
        return 'Please enter all values to get a prediction'
    return ''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


# In[87]:


data

