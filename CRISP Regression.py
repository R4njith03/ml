#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split

import pickle as pk

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold



from sklearn.feature_selection import RFE

# show up charts when export notebooks
get_ipython().run_line_magic('matplotlib', 'inline')


# In[41]:


df_train=pd.read_csv('mpg.csv')
df_train.head()


# In[42]:


df_train.shape


# In[43]:


df_train.info()


# In[44]:


df_train.isna().sum()


# In[45]:


df_train.isna().any()


# In[46]:


df_train.columns


# ## MPG is the target

# In[47]:


plt.figure(figsize=(10,6))
plt.hist(df_train['mpg'],bins=50,ec='black',color='#3F51B5')
plt.xlabel("Miles Per Gallon")
plt.ylabel("Number of vehicles")
plt.show()


# In[48]:


# With outliers 
frequency=df_train['mpg'].value_counts()
plt.figure(figsize=(10,6))
plt.bar(frequency.index,height=frequency)
plt.xlabel("Miles Per Gallon")
plt.ylabel("Number of vehicles")
plt.show()


# In[49]:


frequency


# ## Removing Outliers

# In[50]:


# Outlier function with threshold 2. Function to get list of outliers
def outliers_z_score(df):
    threshold = 2

    mean = np.mean(df)
    std = np.std(df)
    z_scores = [(y - mean) / std for y in df] #Used a Z-Score to remove the outliers
    return np.where(np.abs(z_scores) > threshold)


# In[51]:


# Selecting only the numerical columns in data set
my_list = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_columns = list(df_train.select_dtypes(include=my_list).columns)
numerical_columns = df_train[num_columns]
numerical_columns.head(3)


# In[52]:


# Calling the outlier function and Calculating the outlier of dataset
outlier_list = numerical_columns.apply(lambda x: outliers_z_score(x))
outlier_list


# In[53]:


# Making outlier list ot dataframe
df_of_outlier = outlier_list.iloc[0]
df_of_outlier = pd.DataFrame(df_of_outlier)
df_of_outlier.columns = ['Rows_to_exclude']
df_of_outlier


# In[54]:


# Convert all values from column Rows_to_exclude to a numpy array
outlier_list_final = df_of_outlier['Rows_to_exclude'].to_numpy()

# Concatenate a whole sequence of arrays
outlier_list_final = np.concatenate( outlier_list_final, axis=0 )

# Drop duplicate values
outlier_list_final_unique = set(outlier_list_final)
outlier_list_final_unique


# In[55]:


# Removing outliers from the dataset
filter_rows_to_exclude = df_train.index.isin(outlier_list_final_unique)
clean_data = df_train[~filter_rows_to_exclude]
clean_data


# In[56]:


plt.figure(figsize=(20,15))

# With outliers
plt.subplot(1,2,1)
plt.title("With outliers")
frequency=df_train['mpg'].value_counts()
plt.bar(frequency.index,height=frequency)
plt.xlabel("Miles Per Gallon")
plt.ylabel("Number of vehicles")

# Without outliers
plt.subplot(1,2,2)
plt.title("Without outliers")
frequency=clean_data['mpg'].value_counts()
plt.bar(frequency.index,height=frequency)
plt.xlabel("Miles Per Gallon")
plt.ylabel("Number of vehicles")
plt.show()


# In[57]:


plt.figure(figsize=(20,15))

# With outliers
plt.subplot(1,2,1)
plt.title("With outliers")
sns.histplot(df_train['mpg'],bins=50,kde=True,stat="density",kde_kws=dict(cut=3),alpha=0.4)
plt.xlabel("Miles Per Gallon")
plt.ylabel("Number of vehicles")

# Without outliers
plt.subplot(1,2,2)
plt.title("Without outliers")
sns.histplot(clean_data['mpg'],bins=50,kde=True,stat="density",kde_kws=dict(cut=3),alpha=0.4)
plt.xlabel("Miles Per Gallon")
plt.ylabel("Number of vehicles")
plt.show()


# In[58]:


# Distribution before outlier removal
sns.boxplot(x='mpg', data=df_train)


# In[59]:


# Distribution after outlier removal
sns.boxplot(x='mpg', data=clean_data)


# ## Cleaning Data

# In[60]:


clean_data.head(50)


# In[61]:


#Replacing ? with NaN
clean_data.loc[:,'horsepower'] = clean_data['horsepower'].replace('?', np.nan)


# In[62]:


clean_data.head(50)


# In[63]:


clean_data.info()


# In[64]:


clean_data.isna().any()


# In[65]:


clean_data.isna().sum()


# In[66]:


# Missing Values
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[67]:


missing_values_table(clean_data)


# In[68]:


# Using KNN removing null values
from sklearn.impute import KNNImputer

# Assuming you want to impute missing values in the 'horsepower' column
column_to_impute = 'horsepower'

# Create a KNNImputer object
knn_imputer = KNNImputer(n_neighbors=5)

# Fit and transform only the selected column using KNN imputation
clean_data.loc[:,column_to_impute] = knn_imputer.fit_transform(clean_data[[column_to_impute]])


# In[69]:


clean_data.isna().any()


# In[70]:


clean_data.info()


# In[71]:


clean_data = clean_data.copy()
clean_data['horsepower']=clean_data['horsepower'].astype(float)
clean_data['weight']=clean_data['weight'].astype(float)
clean_data.info()


# ## Dimensionality Reduction

# In[72]:


# Dimensionality reduction PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

# Assuming df contains your dataset with discrete and continuous variables
# Separate discrete and continuous variables
discrete_cols = clean_data.select_dtypes(include=['object']).columns
continuous_cols = clean_data.select_dtypes(include=['int64', 'float64']).columns

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_cols),  # Standardize continuous variables
        ('cat', OneHotEncoder(), discrete_cols)     # One-hot encode discrete variables
    ])

# Define TruncatedSVD models
svd_original = TruncatedSVD(n_components=2)  # TruncatedSVD for original dataset
svd_encoded = TruncatedSVD(n_components=2)    # TruncatedSVD for encoded dataset

# Define pipelines
pipeline_original = Pipeline(steps=[('preprocessor', preprocessor), ('svd', svd_original)])
pipeline_encoded = Pipeline(steps=[('preprocessor', preprocessor), ('svd', svd_encoded)])

# Fit and transform data
X_original_svd = pipeline_original.fit_transform(clean_data)
X_encoded_svd = pipeline_encoded.fit_transform(clean_data)

# Analyze explained variance ratio
explained_variance_ratio_original = svd_original.explained_variance_ratio_
explained_variance_ratio_encoded = svd_encoded.explained_variance_ratio_

# Print explained variance ratio
print("Explained Variance Ratio - Original:", explained_variance_ratio_original)
print("Explained Variance Ratio - Encoded:", explained_variance_ratio_encoded)


# In[73]:


# Visualize principal components (if reduced to 2 components)
if X_original_svd.shape[1] == 2:
    plt.scatter(X_original_svd[:, 0], X_original_svd[:, 1], label='Original Data')
    plt.scatter(X_encoded_svd[:, 0], X_encoded_svd[:, 1], label='Encoded Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('TruncatedSVD Visualization')
    plt.legend()
    plt.show()


# In[74]:


# Feature Selection using Varience Threshold
# Compute the variance of each feature before and after applying the threshold
variances_before = X_encoded_svd.var()
selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X_encoded_svd)
variances_after = pd.Series(selector.variances_)

# Print the variances before and after feature selection
print("Variances Before Threshold:")
print(variances_before)
print("\nVariances After Threshold:")
print(variances_after)


# In[75]:


from sklearn.model_selection import train_test_split

# Assuming y represents the target variable in your dataset
# If you have already separated the target variable from the DataFrame, you can use it directly
# If not, replace 'target_variable_name' with the name of your target variable column
y = clean_data['mpg']
#X = df_processed.drop(columns=['mpg','name'])  # Drop the target variable from the features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded_svd, y, test_size=0.2, random_state=42)


# ## Regression

# In[76]:


get_ipython().system('pip3 install catboost')
get_ipython().system('pip3 install lightgbm')
get_ipython().system('pip3 install xgboost')
# Modelling
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge, HuberRegressor, ElasticNetCV
from sklearn.gaussian_process import GaussianProcessRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "Neural Network": MLPRegressor(),
    "Elastic Net": ElasticNet(),
    "Bayesian Ridge": BayesianRidge(),
    "Huber": HuberRegressor(),
    "Gaussian Process": GaussianProcessRegressor(),
    "CatBoost": CatBoostRegressor(),
    "LightGBM": LGBMRegressor(),
    "Elastic NetCV": ElasticNetCV(),
    "AdaBoost": AdaBoostRegressor()
}

# Evaluate each model
evaluation_results = {}
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Calculate RMSE from MSE
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evaluation_results[model_name] = {"Mean Squared Error": mse, "Root Mean Squared Error": rmse, "Mean Absolute Error": mae, "R2 Score": r2}

# Print evaluation results
print("\nEvaluation Metrics:")
for model_name, metrics in evaluation_results.items():
    print(f"{model_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")


# In[77]:


# Perform Cross Validation
from sklearn.model_selection import cross_val_score

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "Neural Network": MLPRegressor(),
    "Elastic Net": ElasticNet(),
    "Bayesian Ridge": BayesianRidge(),
    "Huber": HuberRegressor(),
    "Gaussian Process": GaussianProcessRegressor(),
    "CatBoost": CatBoostRegressor(),
    "LightGBM": LGBMRegressor(),
    "Elastic NetCV": ElasticNetCV(),
    "AdaBoost": AdaBoostRegressor()
}

# Evaluate each model using cross-validation
cv_results = {}
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    # Perform cross-validation with 5 folds
    scores = cross_val_score(model, X_encoded_svd, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)  # Convert negative MSE to RMSE
    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    cv_results[model_name] = {"Mean RMSE": mean_rmse, "Std RMSE": std_rmse}

# Print cross-validation results
print("\nCross-Validation Results:")
for model_name, results in cv_results.items():
    print(f"{model_name}:")
    print(f"Mean RMSE: {results['Mean RMSE']}, Std RMSE: {results['Std RMSE']}")


# In[79]:


# Performing Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Define hyperparameters grid for each model
param_grid = {
    "Linear Regression": {},
    "Ridge Regression": {"alpha": [0.1, 1.0, 10.0]},
    "Lasso Regression": {"alpha": [0.1, 1.0, 10.0]},
    "Decision Tree Regressor": {"max_depth": [None, 10, 20]},
    "Random Forest Regressor": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]},
    "Gradient Boosting Regressor": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7]},
    "XGBoost": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7]},
    "SVR": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
    "KNN": {"n_neighbors": [3, 5, 7]},
    "Neural Network": {"hidden_layer_sizes": [(100,), (200,), (300,)], "alpha": [0.0001, 0.001, 0.01]},
    "Elastic Net": {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.25, 0.5, 0.75]},
    "Bayesian Ridge": {"alpha_1": [1e-06, 1e-05, 0.0001], "alpha_2": [1e-06, 1e-05, 0.0001], "lambda_1": [1e-06, 1e-05, 0.0001], "lambda_2": [1e-06, 1e-05, 0.0001]},
    "Huber": {"alpha": [0.0001, 0.001, 0.01]},
    "Gaussian Process": {},
    "CatBoost": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7]},
    "LightGBM": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7]},
    "Elastic NetCV": {},
    "AdaBoost": {"n_estimators": [50, 100, 150]}
}

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "Neural Network": MLPRegressor(),
    "Elastic Net": ElasticNet(),
    "Bayesian Ridge": BayesianRidge(),
    "Huber": HuberRegressor(),
    "Gaussian Process": GaussianProcessRegressor(),
    "CatBoost": CatBoostRegressor(),
    "LightGBM": LGBMRegressor(),
    "Elastic NetCV": ElasticNetCV(),
    "AdaBoost": AdaBoostRegressor()
}

# Initialize dictionary to store best models
best_models = {}

# Perform hyperparameter tuning using GridSearchCV
for model_name, model in models.items():
    print(f"Tuning hyperparameters for {model_name}...")
    # Initialize GridSearchCV with the model, hyperparameter grid, and 5-fold cross-validation
    grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    # Fit GridSearchCV to the data
    grid_search.fit(X_train, y_train)
    # Get the best model from GridSearchCV
    best_models[model_name] = grid_search.best_estimator_

# Print best hyperparameters for each model
print("\nBest Hyperparameters:")
for model_name, best_model in best_models.items():
    print(f"{model_name}: {best_model.get_params()}")


# In[80]:


# Evaluation methods
best_model_name = min(evaluation_results, key=lambda k: evaluation_results[k]["Mean Squared Error"])
best_model = best_models[best_model_name]

# Print evaluation results
print("\nEvaluation Metrics:")
for model_name, metrics in evaluation_results.items():
    print(f"{model_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

# Print the best model
print(f"\nBest Model: {best_model_name}")
print(f"Best Model Hyperparameters: {best_model.get_params()}")

