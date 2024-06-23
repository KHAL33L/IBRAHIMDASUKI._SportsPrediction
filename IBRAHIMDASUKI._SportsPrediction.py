#!/usr/bin/env python
# coding: utf-8

# ### Importation

# In[1]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import joblib
import pickle as pkl


# ### Data prep and extraction

# In[2]:


#loading the datasets
stats = pd.read_csv("male_players (legacy).csv")


# In[3]:


stats.head()


# In[4]:


# dropping columns with more than 30% na values
threshold = 0.3 * stats.shape[0]
stats = stats.dropna(thresh=threshold, axis=1)
stats.drop(columns=['value_eur', 'wage_eur', 'release_clause_eur', 'international_reputation'], inplace=True)
# dropping the wage columns because they are rather dependant on the overall rating


# ### Subsetting data into the most important feautures

# In[5]:


# picking numeric values for the independent variables only because they are generally more important in football rating
nums = stats.select_dtypes(include=np.number)


# In[6]:


# imputing missing values in the numeric data
imputer = SimpleImputer(strategy='mean')
imputed_numeric_data = pd.DataFrame(imputer.fit_transform(nums), columns=nums.columns, index=nums.index)


# In[7]:


# Separate the data into dependent and independent sets
y = imputed_numeric_data['overall']
X = imputed_numeric_data.drop(columns=['overall'])


# In[8]:


# picking the top 13 most important features
important = SelectKBest(f_regression, k=13)
X_selected = important.fit_transform(X, y)

selected_features = X.columns[important.get_support()]
stats[selected_features]


# In[9]:


# selected features
X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
column_names = X.columns.tolist()


# In[10]:


# Print shape and column names
print("Shape of X:", X.shape)
print("Selected features:", column_names)


# ### Training the models

# In[11]:


# loading the test data set
players_22 = pd.read_csv("players_22.csv")


# In[12]:


# splitting the data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


# models
models = {
    'GradientBoosting': GradientBoostingRegressor(),
    'DecisionTree': DecisionTreeRegressor(),
    'LinearRegression': LinearRegression()
}


# In[14]:


# setting parameters
params = {
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7]
    },
    'DecisionTree': {
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    },
    'LinearRegression': {}
}


# ### Measuring performance and fine-tuning

# In[15]:


# performing GridSearch
best_models = {}
for model_name in models:
    grid_search = GridSearchCV(models[model_name], params[model_name], cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_


# In[16]:


# evaluating models
for model_name in best_models:
    model = best_models[model_name]
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - RMSE: {rmse}, MAE: {mae}, R2: {r2}")


# In[17]:


# evaluating models
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name}:\n RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


# ### Testing with FIFA22 data

# In[18]:


players_22_processed = players_22.select_dtypes(include=[np.number]).dropna(axis=1)
common_features = list(set(X.columns) & set(players_22_processed.columns))   # checking common features

print("Common features:", common_features)


# In[19]:


# subseting
X_common = X[common_features]
X_test_22 = players_22_processed[common_features]

# making sure the "overall" variable is present
if 'overall' in players_22_processed.columns:
    y_test_22 = players_22_processed['overall']
else:
    raise KeyError("The 'overall' column is not found in the new season data.")


# In[20]:


# retraining the models
models = {
    'GradientBoosting': GradientBoostingRegressor(),
    'DecisionTree': DecisionTreeRegressor(),
    'LinearRegression': LinearRegression()}
best_models = {}

for model_name, model in models.items():
    X_train, X_val, y_train, y_val = train_test_split(X_common, y, test_size=0.2, random_state=42)   # splitting the data
    
    model.fit(X_train, y_train)   # training the model
    val_score = model.score(X_val, y_val)   # evaluate on validation set
    print(f"{model_name} - Validation R2 Score: {val_score:.4f}")
    
    best_models[model_name] = model


# In[21]:


# evaluating each model on the new dataset
for model_name, model in best_models.items():
    y_pred_22 = model.predict(X_test_22)
    
    rmse_22 = np.sqrt(mean_squared_error(y_test_22, y_pred_22))
    mae_22 = mean_absolute_error(y_test_22, y_pred_22)
    r2_22 = r2_score(y_test_22, y_pred_22)
    
    print(f"{model_name} for player22 subset:")
    print(f"RMSE: {rmse_22:.4f}, MAE: {mae_22:.4f}, R2: {r2_22:.4f}")
    print()


# In[22]:


# ensemble model
ensemble_model = VotingRegressor([
    ('gb', best_models['GradientBoosting']),
    ('dt', best_models['DecisionTree']),
    ('lr', best_models['LinearRegression'])])

# fitting the ensemble model
ensemble_model.fit(X_train, y_train)

# predicting based on the test set
y_pred_ensemble = ensemble_model.predict(X_val)


# In[23]:


# evaluating the ensemble model
rmse_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))
mae_ensemble = mean_absolute_error(y_val, y_pred_ensemble)
r2_ensemble = r2_score(y_val, y_pred_ensemble)

print("Ensemble Model Performance on Validation Set:")
print(f"RMSE: {rmse_ensemble:.4f}, MAE: {mae_ensemble:.4f}, R2: {r2_ensemble:.4f}")


# In[24]:


# testing on the FIFA22 data
y_pred_22_ensemble = ensemble_model.predict(X_test_22)

rmse_22_ensemble = np.sqrt(mean_squared_error(y_test_22, y_pred_22_ensemble))
mae_22_ensemble = mean_absolute_error(y_test_22, y_pred_22_ensemble)
r2_22_ensemble = r2_score(y_test_22, y_pred_22_ensemble)

print("Ensemble Model Performance on FIFA 22 data:")
print(f"RMSE: {rmse_22_ensemble:.4f}, MAE: {mae_22_ensemble:.4f}, R2: {r2_22_ensemble:.4f}")


# ### Saving the best model

# In[25]:


joblib.dump(ensemble_model, 'fifa_rating_predictor.pkl')


# ### Creating preprocessing and prediction functions

# In[26]:


# preprocessing function
def preprocess(data):
    input_df = pd.DataFrame([data])
    for feature in common_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    input_df = input_df[common_features]
    return input_df


# In[27]:


# prediction function
def predict(data):
    processed_input = preprocess(data)
    prediction = model.predict(processed_input)
    return prediction[0]


# ### Deploying the model to Streamlit
