Regression Project: Boston House Price Prediction

Marks: 60
Welcome to the project on regression. We will use the Boston house price dataset for this project.

Objective

The problem at hand is to predict the housing prices of a town or a suburb based on the features of the locality provided to us. In the process, we need to identify the most important features affecting the price of the house. We need to employ techniques of data preprocessing and build a linear regression model that predicts the prices for the unseen data.

Dataset

Each record in the database describes a house in Boston suburb or town. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. Detailed attribute information can be found below:

Attribute Information:

CRIM: Per capita crime rate by town
ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS: Proportion of non-retail business acres per town
CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX: Nitric Oxide concentration (parts per 10 million)
RM: The average number of rooms per dwelling
AGE: Proportion of owner-occupied units built before 1940
DIS: Weighted distances to five Boston employment centers
RAD: Index of accessibility to radial highways
TAX: Full-value property-tax rate per 10,000 dollars
PTRATIO: Pupil-teacher ratio by town
LSTAT: % lower status of the population
MEDV: Median value of owner-occupied homes in 1000 dollars
Importing the necessary libraries
# Importing libraries for data manipulation
import numpy as np

import pandas as pd

# Importing libraries for data visualization
import seaborn as sns

import matplotlib.pyplot as plt

# Importing libraries for building linear regression model
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

# To scale the data using z-score 
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split


# To ignore warnings
import warnings
warnings.filterwarnings("ignore")
Loading the dataset
boston_house = pd.read_csv("Boston.csv")
# Copying data to another variable to avoid any changes to original data
data = boston_house.copy()
# Checking the first 5 rows of the dataset
data.head()
CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	LSTAT	MEDV
0	0.00632	18.0	2.31	0	0.538	6.575	65.2	4.0900	1	296	15.3	4.98	24.0
1	0.02731	0.0	7.07	0	0.469	6.421	78.9	4.9671	2	242	17.8	9.14	21.6
2	0.02729	0.0	7.07	0	0.469	7.185	61.1	4.9671	2	242	17.8	4.03	34.7
3	0.03237	0.0	2.18	0	0.458	6.998	45.8	6.0622	3	222	18.7	2.94	33.4
4	0.06905	0.0	2.18	0	0.458	7.147	54.2	6.0622	3	222	18.7	5.33	36.2
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 506 entries, 0 to 505
Data columns (total 13 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   CRIM     506 non-null    float64
 1   ZN       506 non-null    float64
 2   INDUS    506 non-null    float64
 3   CHAS     506 non-null    int64  
 4   NOX      506 non-null    float64
 5   RM       506 non-null    float64
 6   AGE      506 non-null    float64
 7   DIS      506 non-null    float64
 8   RAD      506 non-null    int64  
 9   TAX      506 non-null    int64  
 10  PTRATIO  506 non-null    float64
 11  LSTAT    506 non-null    float64
 12  MEDV     506 non-null    float64
dtypes: float64(10), int64(3)
memory usage: 51.5 KB
# Check the shape of the data
data.shape
(506, 13)
# Check for missing values in the data
data.isnull().sum()
CRIM       0
ZN         0
INDUS      0
CHAS       0
NOX        0
RM         0
AGE        0
DIS        0
RAD        0
TAX        0
PTRATIO    0
LSTAT      0
MEDV       0
dtype: int64
data.nunique()
CRIM       504
ZN          26
INDUS       76
CHAS         2
NOX         81
RM         446
AGE        356
DIS        412
RAD          9
TAX         66
PTRATIO     46
LSTAT      455
MEDV       229
dtype: int64
# Check for duplicate values
data.duplicated().sum()
0
Data Overview : Observations

The dataset contains 13 columns and 506 rows.
There are no missing values.
All the variables are numerical type.
#Checking the statistical summary of the data¶
data.describe().T
count	mean	std	min	25%	50%	75%	max
CRIM	506.0	3.613524	8.601545	0.00632	0.082045	0.25651	3.677083	88.9762
ZN	506.0	11.363636	23.322453	0.00000	0.000000	0.00000	12.500000	100.0000
INDUS	506.0	11.136779	6.860353	0.46000	5.190000	9.69000	18.100000	27.7400
CHAS	506.0	0.069170	0.253994	0.00000	0.000000	0.00000	0.000000	1.0000
NOX	506.0	0.554695	0.115878	0.38500	0.449000	0.53800	0.624000	0.8710
RM	506.0	6.284634	0.702617	3.56100	5.885500	6.20850	6.623500	8.7800
AGE	506.0	68.574901	28.148861	2.90000	45.025000	77.50000	94.075000	100.0000
DIS	506.0	3.795043	2.105710	1.12960	2.100175	3.20745	5.188425	12.1265
RAD	506.0	9.549407	8.707259	1.00000	4.000000	5.00000	24.000000	24.0000
TAX	506.0	408.237154	168.537116	187.00000	279.000000	330.00000	666.000000	711.0000
PTRATIO	506.0	18.455534	2.164946	12.60000	17.400000	19.05000	20.200000	22.0000
LSTAT	506.0	12.653063	7.141062	1.73000	6.950000	11.36000	16.955000	37.9700
MEDV	506.0	22.532806	9.197104	5.00000	17.025000	21.20000	25.000000	50.0000
 
 
 
 
fig, axes = plt.subplots(1, 3, figsize = (20, 6))
  
fig.suptitle('Histogram for all numerical variables in the dataset')
  
sns.histplot(x = 'CRIM', data = data , kde = True, ax = axes[0]);

sns.histplot(x = 'ZN', data = data , kde = True, ax = axes[1]);

sns.histplot(x = 'INDUS', data = data , kde = True, ax = axes[2]);

fig, axes = plt.subplots(1, 3, figsize = (20, 6))

sns.histplot(x = 'NOX', data = data, kde = True, ax = axes[0]);

sns.histplot(x = 'CHAS', data = data, kde = True, ax = axes[1]);

sns.histplot(x = 'RM', data = data, kde = True, ax = axes[2]);

fig, axes = plt.subplots(1, 3, figsize = (20, 6))

sns.histplot(x = 'AGE', data = data, kde = True, ax = axes[0]);

sns.histplot(x = 'DIS', data = data, kde = True, ax = axes[1]);

sns.histplot(x = 'RAD', data = data, kde = True, ax = axes[2]);

fig, axes = plt.subplots(1, 2, figsize = (20, 6))

sns.histplot(x = 'TAX', data = data, kde = True, ax = axes[0]);

sns.histplot(x = 'PTRATIO', data = data, kde = True, ax = axes[1]);

fig, axes = plt.subplots(1, 2, figsize = (20, 6))

sns.histplot(x = 'LSTAT', data = data, kde = True, ax = axes[0]);

sns.histplot(x = 'MEDV', data = data, kde = True, ax = axes[1]);

fig = plt.figure(figsize = (18, 6))

sns.scatterplot(x = 'CRIM', y = 'MEDV', data = data);

sns.scatterplot(x = 'CRIM', y = 'MEDV', data = data);

#, ci = None, estimator = 'mean'); ci = None, estimator = 'mean')

fig, axes = plt.subplots(1, 3, figsize = (20, 6))
  

sns.scatterplot(x = 'RM', y = 'MEDV', data = data, ax = axes[0]);

sns.scatterplot(x = 'RAD', y = 'TAX', data = data, ax = axes[1]);

sns.scatterplot(x = 'RAD', y = 'MEDV', data = train_df, ax = axes[2]);

fig, axes = plt.subplots(1, 3, figsize = (20, 6))
  

sns.scatterplot(x = 'NOX', y = 'MEDV', data = data, ax = axes[0]);

sns.scatterplot(x = 'INDUS', y = 'NOX', data = data, ax = axes[1]);

sns.scatterplot(x = 'DIS', y = 'MEDV', data = train_df, ax = axes[2]);

fig, axes = plt.subplots(1, 3, figsize = (20, 6))
  

sns.scatterplot(x = 'PTRATIO', y = 'MEDV', data = data, ax = axes[0]);

sns.scatterplot(x = 'LSTAT', y = 'MEDV', data = data, ax = axes[1]);

sns.scatterplot(x = 'TAX', y = 'MEDV', data = train_df, ax = axes[2]);

fig = plt.figure(figsize = (18, 6))

sns.heatmap(train_data.corr(), annot = True);

plt.xticks(rotation = 45);

Exploratory Data Analysis (EDA)
Univariate Analysis
The variable 'CHAS' has only two unique values and around 500 houses are not on the banks of Charles river. This variable is not adding much value to the analysis and hence we can drop it.
The 'MEDV' follows a normal distribution with a slight right skewness.
Multivariate Analysis
From the heat map, it can be seen that
the dependent variable 'RM' is positively correlated to the dependent variable 'MEDV'
the dependent variables 'LSTAT' and 'PTRATIO' are negatively correlated to the dependent variable 'MEDV'
the independent variable 'CHAS' has no significant correlation with any other variables.
the independent variables 'TAX' and 'RAD' are positively correlated
The variables 'RM' and 'AGE' have a few outliers.
Preparing Data
boston_data = data.drop(columns = 'CHAS')
boston_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 506 entries, 0 to 505
Data columns (total 12 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   CRIM     506 non-null    float64
 1   ZN       506 non-null    float64
 2   INDUS    506 non-null    float64
 3   NOX      506 non-null    float64
 4   RM       506 non-null    float64
 5   AGE      506 non-null    float64
 6   DIS      506 non-null    float64
 7   RAD      506 non-null    int64  
 8   TAX      506 non-null    int64  
 9   PTRATIO  506 non-null    float64
 10  LSTAT    506 non-null    float64
 11  MEDV     506 non-null    float64
dtypes: float64(10), int64(2)
memory usage: 47.6 KB
 
Data Preprocessing
Missing value treatment
Log transformation of dependent variable if skewed
Feature engineering (if needed)
Outlier detection and treatment (if needed)
Preparing data for modeling
Any other preprocessing steps (if needed)
 
Model Building - Linear Regression
# Separating the target variable and other variables
Y = boston_data.MEDV
X = boston_data.drop(columns = ['MEDV'])

# Add the intercept term
X = sm.add_constant(X)
# splitting the data in 70:30 ratio of train to test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 1)
Removing Multicolinearity
#Checking VIF score
vif_series = pd.Series(
    [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
    index = X_train.columns,
    dtype = float)

print("VIF Scores: \n\n{}\n".format(vif_series))
VIF Scores: 

const      535.260829
CRIM         1.914578
ZN           2.738223
INDUS        3.985174
NOX          4.383210
RM           1.854570
AGE          3.146633
DIS          4.341588
RAD          8.195861
TAX          9.947156
PTRATIO      1.940250
LSTAT        2.854597
dtype: float64

Observations

There are two variables RAD and TAX with IVF greater than 5
Hence we drop one of these columns, we drop TAX with the highest IVF.
# Create the model after dropping TAX
X_train_new = X_train.drop('TAX', axis = 1) 
#Checking VIF score
vif_series = pd.Series(
    [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])],
    index = X_train_new.columns,
    dtype = float)

print("VIF Scores: \n\n{}\n".format(vif_series))
VIF Scores: 

const      532.022887
CRIM         1.914323
ZN           2.483364
INDUS        3.270825
NOX          4.354430
RM           1.849910
AGE          3.145997
DIS          4.324493
RAD          2.942153
PTRATIO      1.902409
LSTAT        2.853842
dtype: float64

The VIF of all variables is now less than 5. Hence we assume there is no more multicolinearity.
# Calling the OLS algorithm on the train features and the target variable
ols_model_0 = sm.OLS(y_train, X_train_new)

# Fitting the Model
ols_res_0 = ols_model_0.fit()
print(ols_res_0.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   MEDV   R-squared:                       0.695
Model:                            OLS   Adj. R-squared:                  0.686
Method:                 Least Squares   F-statistic:                     78.29
Date:                Thu, 25 May 2023   Prob (F-statistic):           2.68e-82
Time:                        22:30:05   Log-Likelihood:                -1070.2
No. Observations:                 354   AIC:                             2162.
Df Residuals:                     343   BIC:                             2205.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         48.8136      6.194      7.881      0.000      36.631      60.997
CRIM          -0.1202      0.044     -2.755      0.006      -0.206      -0.034
ZN             0.0478      0.018      2.653      0.008       0.012       0.083
INDUS         -0.0230      0.071     -0.325      0.745      -0.163       0.116
NOX          -22.6341      4.755     -4.760      0.000     -31.986     -13.282
RM             2.8476      0.528      5.393      0.000       1.809       3.886
AGE            0.0056      0.017      0.333      0.739      -0.028       0.039
DIS           -1.5274      0.262     -5.838      0.000      -2.042      -1.013
RAD            0.1405      0.052      2.681      0.008       0.037       0.244
PTRATIO       -1.0721      0.173     -6.206      0.000      -1.412      -0.732
LSTAT         -0.5913      0.062     -9.474      0.000      -0.714      -0.469
==============================================================================
Omnibus:                      135.124   Durbin-Watson:                   1.832
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              554.289
Skew:                           1.629   Prob(JB):                    4.34e-121
Kurtosis:                       8.193   Cond. No.                     2.09e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.09e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
The value of R squared is 0.695 and adjusted R squared is 0.686, which shows the models is a better fit.
The alpha values of the variables 'INDUS' and 'AGE' are greater than 0.05, which shows that these are statistically insignificant. Hence we can drop them.
#drop the most insignificant one and check the multicolinearity.
X_train_new1 = X_train_new.drop(['INDUS','AGE'], axis =1)
#Checking VIF score
vif_series = pd.Series(
    [variance_inflation_factor(X_train_new1.values, i) for i in range(X_train_new1.shape[1])],
    index = X_train_new1.columns,
    dtype = float)

print("VIF Scores: \n\n{}\n".format(vif_series))
VIF Scores: 

const      528.833955
CRIM         1.909008
ZN           2.456118
NOX          3.522378
RM           1.769027
DIS          3.709370
RAD          2.884244
PTRATIO      1.822980
LSTAT        2.435551
dtype: float64

# Calling the OLS algorithm on the train features and the target variable
ols_model_1 = sm.OLS(y_train, X_train_new1)

# Fitting the Model
ols_res_1 = ols_model_1.fit()
print(ols_res_1.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   MEDV   R-squared:                       0.695
Model:                            OLS   Adj. R-squared:                  0.688
Method:                 Least Squares   F-statistic:                     98.34
Date:                Thu, 25 May 2023   Prob (F-statistic):           3.03e-84
Time:                        22:35:23   Log-Likelihood:                -1070.3
No. Observations:                 354   AIC:                             2159.
Df Residuals:                     345   BIC:                             2193.
Df Model:                           8                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         48.8415      6.159      7.930      0.000      36.727      60.956
CRIM          -0.1196      0.043     -2.751      0.006      -0.205      -0.034
ZN             0.0472      0.018      2.641      0.009       0.012       0.082
NOX          -22.8074      4.265     -5.347      0.000     -31.197     -14.418
RM             2.8955      0.515      5.622      0.000       1.883       3.909
DIS           -1.5341      0.242     -6.347      0.000      -2.009      -1.059
RAD            0.1371      0.052      2.649      0.008       0.035       0.239
PTRATIO       -1.0784      0.169     -6.394      0.000      -1.410      -0.747
LSTAT         -0.5855      0.058    -10.181      0.000      -0.699      -0.472
==============================================================================
Omnibus:                      136.454   Durbin-Watson:                   1.827
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              567.287
Skew:                           1.641   Prob(JB):                    6.53e-124
Kurtosis:                       8.262   Cond. No.                         776.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Model Performance Check

The R-Squared value is not affected by the removal of the variables 'AGE' and 'INDUS'.
Since the alpha values of the remaining variables are less than 0.05, all of them are significant.
All the VIF scores are now less than 5 and hence no multicollinearity.
Evaluation Metrics
print(ols_res_1.mse_resid)
25.39598912122027
print(np.sqrt(ols_res_1.mse_resid))
5.039443334458705
# Fitting linear model

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

linearregression = LinearRegression()                                    

cv_Score11 = cross_val_score(linearregression, X_train_new1, y_train, cv = 10)

cv_Score12 = cross_val_score(linearregression, X_train_new1, y_train, cv = 10, 
                             scoring = 'neg_mean_squared_error')                                  


print("RSquared: %0.3f (+/- %0.3f)" % (cv_Score11.mean(), cv_Score11.std()*2))

print("Mean Squared Error: %0.3f (+/- %0.3f)" % (-1*cv_Score12.mean(), cv_Score12.std()*2))
RSquared: 0.665 (+/- 0.241)
Mean Squared Error: 26.344 (+/- 20.195)
Checking Linear Regression Assumptions
In order to make statistical inferences from a linear regression model, it is important to ensure that the assumptions of linear regression are satisfied.

Mean of residuals should be 0
Normality of error terms
Linearity of variables
No heteroscedasticity
# Residuals
residual = ols_res_1.resid 
residual.mean()
-4.935662675802504e-14
This is a very small number and very close to zero. Hence, the corresponding assumption is satisfied.
# Plot histogram of residuals
sns.histplot(residual, kde = True)
<AxesSubplot:ylabel='Count'>

The residuals follow a normal distribution and hence the second assumption is satisfied.
# Predicted values
fitted = ols_res_1.fittedvalues

sns.residplot(x = fitted, y = residual, color = "lightblue")

plt.xlabel("Fitted Values")

plt.ylabel("Residual")

plt.title("Residual PLOT")

plt.show()

The residuals and the fitted values do not form a strong pattern. They are randomly distributed on the x-axis.
Test for heteroscadacity

Goldfeld–Quandt test to check homoscedasticity.

Null hypothesis : Residuals are homoscedastic
Alternate hypothesis : Residuals are hetroscedastic
from statsmodels.stats.diagnostic import het_white

from statsmodels.compat import lzip

import statsmodels.stats.api as sms
import statsmodels.stats.api as sms

from statsmodels.compat import lzip

name = ["F statistic", "p-value"]

test = sms.het_goldfeldquandt(y_train, X_train_new1)

lzip(name, test)
[('F statistic', 1.1726847038950667), ('p-value', 0.15144024614611934)]
As we observe from the above test, the p-value is greater than 0.05, so we fail to reject the null-hypothesis. That means the residuals are homoscedastic.
Thus all the assumptions of the linear regression are satisfied.
Final Model

MEDV = 48.8415 + (-0.1196)CRIM + (0.0472)ZN +(-22.8074)NOX +(2.8955)RM + (-1.5341)DIS + (0.1371)RAD + (-1.0784)PTRATIO +(-0.5855)LSTAT
Actionable Insights and Recommendations
House prices are very low in the places where there is high crime rate, high levels of air pollution , high pupil-teacher ratio and a large population of low income families.

As the number of room increases, the house price also increases.

# Boston-House-Price-Prediction
