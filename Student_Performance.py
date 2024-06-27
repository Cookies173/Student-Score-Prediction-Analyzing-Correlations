#!/usr/bin/env python
# coding: utf-8

# # Working on data

# ### Loading required library to work upon data in Python

# In[1]:


# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Working with data
import pandas as pd

# Plotting library
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

# tells matplotlib to embed plots within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

#Sk-Learn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


# ### Importing data from the students of Math course & Portuguese course dataset

# In[2]:


Math=pd.read_csv('Data/student-mat.csv', sep=';')
Port=pd.read_csv('Data/student-por.csv', sep=';')


# ### Review the dataset

# In[3]:


Math.head()


# In[4]:


Port.head()


# In[5]:


Math.shape


# In[6]:


Port.shape


# In[7]:


Math.info()


# In[8]:


Port.info()


# In[9]:


Math.describe()


# In[10]:


Math.describe(include="object")


# In[11]:


Port.describe()


# In[12]:


Port.describe(include="object")


# In[13]:


Math.isnull().sum(axis=0)


# In[14]:


Port.isnull().sum()


# In[15]:


Math.hist(bins=10, figsize=(20, 15))
plt.show()


# In[16]:


Port.hist(bins=10, figsize=(20, 15))
plt.show()


# In[17]:


Math.duplicated().sum()


# In[18]:


Port.duplicated().sum()


# ### Adding a feature "subject" to both datasets

# In[19]:


Math["subject"]='M'
Port["subject"]='P'


# In[20]:


temp1=Math.subject
Math.drop('subject', axis=1, inplace=True)
temp2=Port.subject
Port.drop('subject', axis=1, inplace=True)


# In[21]:


Math=pd.concat([temp1, Math], axis=1)
Port=pd.concat([temp2, Port], axis=1)


# In[22]:


# subject=pd.Series([], name='subject', dtype='object')
# Math=pd.concat([subject, Math], axis=1)
# Port=pd.concat([subject, Port], axis=1)
# Math=Math.fillna('M')
# Port=Port.fillna('P')


# In[23]:


Math.head()


# In[24]:


Port.head()


# ### Merging Both the datasets of Math and Portuguese Subject

# In[25]:


df=pd.concat([Math, Port])


# In[26]:


df.head()


# In[27]:


df.info()


# In[28]:


df.describe()


# In[29]:


df.subject.value_counts()


# ### Declare feature vector and target variable

# In[30]:


X=df.drop(['G3'], axis=1)
y=df['G3']


# ### Splitting data into Train and Test datasets

# In[31]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=2022)


# In[32]:


y_test.shape


# # Discover & Visualize the data to gain insights

# ### Create a copy of training dataset

# In[33]:


data=X_train.copy()
data.head()


# ### Looking for correlations

# In[34]:


# data=data.drop(['subject', 'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'], axis=1)
data=data.select_dtypes(exclude='object')
corr_matrix=data.corr()


# In[35]:


corr_matrix['age'].sort_values(ascending=False)


# We observe that famrel(quality of family relationships), health, absences and age are not much correlated with other features

# In[36]:


attributes=['age', 'absences', 'G1', 'G2']
scatter_matrix(data[attributes], figsize=(10, 6))
plt.show()


# G1, G2 are positively correlated

# # Prepare Data for Machine Learning Algorithm

# ### Data Cleaning 
# This part is majorly done in first section, as there are no NA values in the data, we can continue to next part.

# ### Handling text & Categorical Atrributes

# In[37]:


df_train=X_train.copy()


# In[38]:


df_train=df_train.select_dtypes(include='object')


# In[39]:


df_train.describe()


# As the number of unique values of all the features are less than 25, we can Label Encode(OrdinalEncoder) the features and as it is also less than 10, we can make it's sparse matrix(OneHotEncoder) too

# In[40]:


ordinal_encoder = OrdinalEncoder()
ordinal_train = ordinal_encoder.fit_transform(df_train)
ordinal_train


# In[41]:


hot_encoder=OneHotEncoder()
hot_train = hot_encoder.fit_transform(df_train)
hot_train.toarray()


# ### Custom Transformers to apply Encoding efficiently

# In[42]:


df_train=X_train.copy()


# In[43]:


categorical_cols=df_train.select_dtypes(include='object').columns
categorical_cols


# In[44]:


column_transform_ordinal = make_column_transformer((OrdinalEncoder(), categorical_cols), remainder='passthrough')


# In[45]:


column_transform_hot = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), categorical_cols), remainder='passthrough')


# ### Feature Scaling

# In[46]:


df_train=X_train.copy()


# In[47]:


continuous_cols=df_train.select_dtypes(include='int').columns
continuous_cols


# In[48]:


df_train=df_train.select_dtypes(exclude='object')


# In[49]:


stdScaler=StandardScaler()
std_scaler = stdScaler.fit_transform(df_train)
std_scaler


# In[50]:


column_transform_stdScaler = make_column_transformer((StandardScaler(), continuous_cols), remainder='passthrough')


# ### Transformation Pipelines

# In[51]:


continuous_pipeline = make_pipeline(column_transform_stdScaler)
categorical_pipeline = make_pipeline(column_transform_hot)


# In[52]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_pipeline, continuous_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])


# In[53]:


X_train_scaled = preprocessor.fit_transform(X_train)
X_train_scaled


# In[54]:


df=pd.DataFrame(X_train_scaled)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[55]:


X_test_scaled = preprocessor.transform(X_test)
X_test_scaled


# # Select & Train a Model

# In[56]:


# For Classification Models - Creating a Predefined function to assess the accuracy of a model. This will be the scoring function
# def score(model, title = "Default"):
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
# #     print(confusion_matrix(y_test, preds))
#     accuracy = round(accuracy_score(y_test, preds), 5)
#     print('Accuracy for', title, ':', accuracy, '\n')


# ## Linear Models

# #### Linear Regression using Normal Equation

# In[57]:


lin_reg=LinearRegression()


# In[58]:


lin_reg_model = make_pipeline(preprocessor, lin_reg)


# In[59]:


lin_reg_model.fit(X_train, y_train)


# In[60]:


y_pred = lin_reg_model.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[61]:


lin_mae = mean_absolute_error(y_test, y_pred)
lin_mae


# In[62]:


lin_reg_model.score(X_train, y_train)


# In[63]:


lin_reg_model.score(X_test, y_test)


# #### Stochastic Gradient Descent

# In[64]:


sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.01, random_state=42)
sgd_reg_model = make_pipeline(preprocessor, sgd_reg)
sgd_reg_model.fit(X_train, y_train)


# In[65]:


y_pred = sgd_reg_model.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[66]:


lin_mae = mean_absolute_error(y_test, y_pred)
lin_mae


# In[67]:


sgd_reg_model.score(X_train, y_train)


# In[68]:


sgd_reg_model.score(X_test, y_test)


# #### Regularized models

# Ridge(l2) Penalty

# In[69]:


# ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg = SGDRegressor(max_iter=50, penalty="l2", random_state=42)
# ridge_reg = Ridge(alpha=1, solver="sag", random_state=42)
ridge_reg_model = make_pipeline(preprocessor, ridge_reg)
ridge_reg_model.fit(X_train, y_train)


# In[70]:


y_pred = ridge_reg_model.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[71]:


lin_mae = mean_absolute_error(y_test, y_pred)
lin_mae


# In[72]:


ridge_reg_model.score(X_train, y_train)


# In[73]:


ridge_reg_model.score(X_test, y_test)


# Lasso(l1) Penalty

# In[74]:


# lasso_reg = Lasso(alpha=0.01)
lasso_reg = SGDRegressor(max_iter=50, penalty="l1", random_state=42)
lasso_reg_model=make_pipeline(preprocessor, lasso_reg)
lasso_reg_model.fit(X_train, y_train)


# In[75]:


y_pred = lasso_reg_model.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[76]:


lin_mae = mean_absolute_error(y_test, y_pred)
lin_mae


# In[77]:


lasso_reg_model.score(X_train, y_train)


# In[78]:


lasso_reg_model.score(X_test, y_test)


# Elastic Net - Middle Ground between Ridge and Lasso

# In[79]:


elsNet_reg = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elsNet_reg_model = make_pipeline(preprocessor, elsNet_reg)
elsNet_reg_model.fit(X_train, y_train)


# In[80]:


y_pred = elsNet_reg_model.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[81]:


lin_mae = mean_absolute_error(y_test, y_pred)
lin_mae


# In[82]:


elsNet_reg_model.score(X_train, y_train)


# In[83]:


elsNet_reg_model.score(X_test, y_test)


# #### Support Vector Machine

# Linear SVM

# In[84]:


svm_reg = LinearSVR(epsilon=1.5, random_state=42)
svm_reg_model = make_pipeline(preprocessor, svm_reg)
svm_reg_model.fit(X_train, y_train)


# In[85]:


y_pred = svm_reg_model.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[86]:


lin_mae = mean_absolute_error(y_test, y_pred)
lin_mae


# In[87]:


svm_reg_model.score(X_train, y_train)


# In[88]:


svm_reg_model.score(X_test, y_test)


# Non-Linear

# In[89]:


svmPoly_reg = SVR(kernel="poly", degree=2, C=5, epsilon=0.1, gamma="auto")
svmPoly_reg_model = make_pipeline(preprocessor, svmPoly_reg)
svmPoly_reg_model.fit(X_train, y_train)


# In[90]:


y_pred = svmPoly_reg_model.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[91]:


lin_mae = mean_absolute_error(y_test, y_pred)
lin_mae


# In[92]:


svmPoly_reg_model.score(X_train, y_train)


# In[93]:


svmPoly_reg_model.score(X_test, y_test)


# #### Decision Tree

# In[94]:


tree_reg = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_reg_model=make_pipeline(preprocessor, tree_reg)
tree_reg_model.fit(X_train, y_train)


# In[95]:


y_pred = tree_reg_model.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[96]:


lin_mae = mean_absolute_error(y_test, y_pred)
lin_mae


# In[97]:


tree_reg_model.score(X_train, y_train)


# In[98]:


tree_reg_model.score(X_test, y_test)


# #### Ensemble Learning

# Gradient Boosting

# In[99]:


gb_reg = GradientBoostingRegressor(max_depth=2, n_estimators=10, learning_rate=0.3, random_state=42)
gb_reg_model=make_pipeline(preprocessor, gb_reg)
gb_reg_model.fit(X_train, y_train)


# In[100]:


y_pred = gb_reg_model.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[101]:


lin_mae = mean_absolute_error(y_test, y_pred)
lin_mae


# In[102]:


gb_reg_model.score(X_train, y_train)


# In[103]:


gb_reg_model.score(X_test, y_test)


# XG Boost

# In[104]:


xg_reg = XGBRegressor(n_estimators=50, learning_rate=0.5, n_jobs=10)
xg_reg_model=make_pipeline(preprocessor, xg_reg)
xg_reg_model.fit(X_train, y_train)


# In[105]:


y_pred = xg_reg_model.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[106]:


lin_mae = mean_absolute_error(y_test, y_pred)
lin_mae


# In[107]:


xg_reg_model.score(X_train, y_train)


# In[108]:


xg_reg_model.score(X_test, y_test)


# Random Forest

# In[109]:


rf_reg = RandomForestRegressor()
rf_reg_model=make_pipeline(preprocessor, rf_reg)
rf_reg_model.fit(X_train, y_train)


# In[110]:


y_pred = rf_reg_model.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[111]:


lin_mae = mean_absolute_error(y_test, y_pred)
lin_mae


# In[112]:


rf_reg_model.score(X_train, y_train)


# In[113]:


rf_reg_model.score(X_test, y_test)


# # Fine Tuning a Model

# Lets fix a model to fine tune the hyperparameters to increase the score to improve predictions of the model

# In[114]:


scores = cross_val_score(rf_reg_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
display = np.sqrt(-scores)
display


# ### GridSearchCV

# In[115]:


parameters = {
    'regressor__n_estimators': [130, 132, 134, 136, 138]
}
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, y_train)


# In[116]:


grid_search.best_params_


# In[117]:


pd.DataFrame(grid_search.cv_results_)


# ### RandomizedSearchCV

# In[118]:


parameters = {
        'regressor__n_estimators': randint(low=100, high=150)
    }
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
rnd_search = RandomizedSearchCV(pipeline, param_distributions=parameters,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(X_train, y_train)


# In[119]:


rnd_search.best_params_


# In[120]:


pd.DataFrame(rnd_search.cv_results_)


# ### Tune the final model with best parameters

# In[137]:


final_reg = RandomForestRegressor(n_estimators=136)
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', final_reg)
])
final_model.fit(X_train, y_train)


# In[138]:


final_model.score(X_train, y_train)


# In[139]:


final_model.score(X_test, y_test)


# The score of the final model for predicting the scores of the students are 0.9812 for training and 0.7567 for testing dataset.
# The model looks little overfit as the score of training data is greater than testing data.

# ### Feature Importance

# In[140]:


importances=final_model.named_steps['regressor'].feature_importances_


# In[141]:


feat_importances = {}

for i,features in zip(importances,X_train.columns):
    print("{}: {}".format(features,i))
    feat_importances[features] = i


# In[142]:


feat_importances = dict(sorted(feat_importances.items(), key=lambda item: item[1]))


feat_importances


# In[143]:


plt.figure(figsize=(20,20))

plt.bar(range(len(feat_importances)), list(feat_importances.values()), align='center')
plt.xticks(range(len(feat_importances)), list(feat_importances.keys()),  rotation=60, fontsize = 12)

plt.title("Feature Importance")
plt.show()


# Model is giving the "studytime" to be the most weighted feature in the model.
