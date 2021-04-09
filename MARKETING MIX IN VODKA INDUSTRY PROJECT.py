#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[3]:


data = pd.read_csv(r'C:\Users\kkcha\Downloads\Dataset-Svedka.csv')


# In[4]:


data[:5]


# In[5]:


data.columns


# #### As we can see here there is a column called 'BrandName'. Let's see the different brands present.

# In[6]:


data['BrandName'].unique()


# In[7]:


len(data['BrandName'].unique())


# #### There are 27 distinct types of vodka brands in it.

# In[8]:


data.groupby(['BrandName']).size().reset_index(name='counts')


# #### Let's work one just one brand for now to get some statistical inference.
# 
# 
# 
# ### statsmodels:
# 
# #### The Python module used is 'statsmodels', which provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests, and statistical data exploration.

# In[9]:


# Selected a single brand to work on
Absolut_Vod = data[data['BrandName'] == 'Absolut']
Absolut_Vod.head()


# #### We can say that price is an important indicator (for consumer goods) to understand sales. 

# In[10]:


Price_Absolut = Absolut_Vod[['LnSales','LnPrice']]


# In[11]:


plt.scatter(Price_Absolut['LnPrice'],Price_Absolut['LnSales'])
plt.title('Normalized price vs sales')
plt.xlabel('Price')
plt.ylabel('Sales')
plt.show()


# In[12]:


# Regression Plot using seaborn
import seaborn as sns; sns.set(color_codes=True)
plot = sns.regplot(x = Price_Absolut['LnPrice'],y = Price_Absolut['LnSales'], data=Price_Absolut)


# In[13]:


import statsmodels.formula.api as sm
from statsmodels.compat import lzip
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.regressionplots


# #### Regression model for LnSales (Y - dependent variable) vs LnPrice (X - independent variable)

# In[14]:


reg_result = sm.ols(formula = 'LnSales ~ LnPrice',data = Price_Absolut).fit()
reg_result.summary()


# ### Warnings:
# #### [1] Standard Errors assume that the covariance matrix of the errors is correctly specified. p-value:
# 
# #### The p-value of the price is zero (less than the test statistic), indicates that the price is significant indicator of sales.
# 
# ### R-squared:
# 
# #### An R-squared of 1 or 100% means that all movements of a dependent variable is completely explained by movements of the independent variable we are interested in
# 
# #### In this case, the value of R-squared is 0.688 i.e the price variable indicates nearly 69% of the sales data points.
# 
# #### co-efficient of price (LnPrice - coef):
# 
# #### The co-efficient of price means that, every unit increase in price, there is 1.13 times increase in sales.

# In[15]:


name = sm.ols(formula = 'LnSales ~ LnPrice', data = Price_Absolut)
name.endog_names #dependent variable


# In[16]:


name.exog_names #intercept and predictor


# In[17]:


r = name.fit()
r.params


# In[18]:


name.loglike(r.params)

# We can see the log likelihood ratio from the above summary stats and here


# In[19]:


name.predict(r.params, [[1, 4.7]])

# In terms of linear regression, y = mx + c, c = 2.836674 and m = 1.130972, 
# we are passing two values x = 4.7 and 
# the the value 1 is passed as multiplier for c (so, c remains at 2.836674 as per our model)


# In[20]:


# Statsmodels Regression Plots
fig = plt.figure(figsize=(12,6))
fig = statsmodels.graphics.regressionplots.plot_regress_exog(reg_result, "LnPrice", fig=fig)


# In[21]:


#Let's add more indicators to the regression and to monitor the R-squared value, 
# our aim is to increase R-squared (or to determine the optimum level)
Additional_Absolut = Absolut_Vod[['LnSales','LnMag','LnNews','LnOut','LnBroad','LnPrint','LnPrice']]


# In[22]:


result_2 = sm.ols('LnSales ~ LnMag + LnNews + LnOut + LnBroad + LnPrint + LnPrice',data=Additional_Absolut).fit()


# In[23]:


result_2.summary()


# In[24]:


# Statsmodels Multivariate Regression Plots
fig = plt.figure(figsize=(15,8))
fig = statsmodels.graphics.regressionplots.plot_partregress_grid(result_2, fig=fig)


# #### Since the number of indeicators we have used are more, the Adj. R-squared value is at 0.86+. It is able to explain 87% of the data points.
# 
# #### But here the p-values of some variables are high which can be accounted due to interaction effect and some other factors.
# 
# #### Let's try out the interation effect method between variables LnBroad and LnPrint

# In[25]:


interaction = sm.ols('LnSales ~ LnMag + LnNews + LnOut + LnBroad * LnPrint + LnPrice',data=Additional_Absolut).fit()


# In[26]:


interaction.summary()


# #### The presence of the interaction indicates that the effect of one predictor variable on the response variable is different at different values of the other predictor variable.
# 
# #### The R-squared value has increased.

# In[27]:


# Plots
fig = plt.figure(figsize=(12,6))
fig = statsmodels.graphics.regressionplots.plot_partregress_grid(interaction, fig=fig)


# #### We have noticed in all the summary stats, there is a problem of strong multicollinearity, I will update the notebook soon, addrressing this issue.

# ### Suggested Questions:
# 
# #### 1. Evaluate the independence of the independent variables using appropriate method
# 

# In[175]:


#df_cor = df_l.iloc[:,5:]
df_cor = data
corrmat = df_cor.corr()

plt.figure(figsize=(15, 10))
p =sns.heatmap(corrmat, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))


# #### 2. Compute the descriptive statistics of the variables (individual brands)

# In[165]:


df.groupby(["BrandName"]).sum().sort_values("TotalSales", ascending=False)


# In[164]:


data.groupby(['BrandName']).size().reset_index(name='counts')


# ### 3. Perform outlier analysis (if required)
# 

# In[30]:


def remove_outlier_Hampel(data):
    med=data.median()
    List=abs(data-med)
    cond=List.median()*4.5
    good_list=List[~(List>cond)]
    return good_list

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[31]:


def remove_sign(x,sign):
    if type(x) is str:
        x = float(x.replace(sign,'').replace(',',''))
    return x


# In[32]:


df=data[['Ln2Lsales','LnDiff']]
df=pd.DataFrame(data)


sns.boxplot(y='Ln2Lsales', x='LnDiff',data=data)
plt.xticks(rotation=90)
plt.ylabel('Price ($)')


# ### 4. Is there evidence of linear relationship between each independent variable and the dependent variable. Comment. If not, what is the appropriate method of handling such situations.
# 

# In[33]:


from sklearn.linear_model import LinearRegression


# In[34]:


data.corr()


# In[35]:


X= data.iloc[:,:-1].values
print(X)


# In[36]:


y= data.iloc[:,3].values
print(y)


# In[ ]:


df = data[['LnLSales','Ln2Lsales','LnDiff','LnNews','LnBroad']]

sns.pairplot(data, kind="scatter")
plt.show()


# ### Correlation Test
# 
# #### A correlation test is another method to determine the presence and extent of a linear relationship between two quantitative variables. In our case, we would like to statistically test whether there is a correlation between the applicant’s investment and their work experience. The first step is to visualize the relationship with a scatter plot, which is done using the line of code below.

# In[37]:


plt.scatter(data['LnLSales'], data['Ln2Lsales'])
plt.show()


# In[38]:


np.corrcoef(data['LnLSales'], data['Ln2Lsales'])


# #### The value of 0.98 shows a positive but weak linear relationship between the two variables. Let’s confirm this with the linear regression correlation test, which is done in Python with the linregress() function in the scipy.stats module.

# In[39]:


from scipy.stats import linregress
linregress(data['LnLSales'], data['Ln2Lsales'])


# ## Linear regression

# In[89]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

print(f'Coefficients: {lin_reg.coef_}')
print(f'Intercept: {lin_reg.intercept_}')
print(f'R^2 score: {lin_reg.score(X, y)}')


# In[91]:


import statsmodels.api as sm

X_constant = sm.add_constant(X)
lin_reg = sm.OLS(y,X_constant).fit()
lin_reg.summary()


# In[92]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format ='retina'")
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
sns.set_style('darkgrid')
sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)

def linearity_test(model, y):
    '''
    Function for visually inspecting the assumption of linearity in a linear regression model.
    It plots observed vs. predicted values and residuals vs. predicted values.
    
    Args:
    * model - fitted OLS model from statsmodels
    * y - observed values
    '''
    fitted_vals = model.predict()
    resids = model.resid

    fig, ax = plt.subplots(1,2)
    
    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
    ax[0].set(xlabel='Predicted', ylabel='Observed')

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
    ax[1].set(xlabel='Predicted', ylabel='Residuals')
    
linearity_test(lin_reg, y)   


# In[96]:



get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format ='retina'")
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
sns.set_style('darkgrid')
sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)

def homoscedasticity_test(model):
    '''
    Function for testing the homoscedasticity of residuals in a linear regression model.
    It plots residuals and standardized residuals vs. fitted values and runs Breusch-Pagan and Goldfeld-Quandt tests.
    
    Args:
    * model - fitted OLS model from statsmodels
    '''
    fitted_vals = model.predict()
    resids = model.resid
    resids_standardized = model.get_influence().resid_studentized_internal

    fig, ax = plt.subplots(1,2)

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Residuals vs Fitted', fontsize=16)
    ax[0].set(xlabel='Fitted Values', ylabel='Residuals')

    sns.regplot(x=fitted_vals, y=np.sqrt(np.abs(resids_standardized)), lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Scale-Location', fontsize=16)
    ax[1].set(xlabel='Fitted Values', ylabel='sqrt(abs(Residuals))')

    bp_test = pd.DataFrame(sms.het_breuschpagan(resids, model.model.exog), 
                           columns=['value'],
                           index=['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'])

    gq_test = pd.DataFrame(sms.het_goldfeldquandt(resids, model.model.exog)[:-1],
                           columns=['value'],
                           index=['F statistic', 'p-value'])

    print('\n Breusch-Pagan test ----')
    print(bp_test)
    print('\n Goldfeld-Quandt test ----')
    print(gq_test)
    print('\n Residuals plots ----')

homoscedasticity_test(lin_reg)


# In[97]:


import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(lin_reg.resid, lags=40 , alpha=0.05)
acf.show()


# ### Mandatory Questions

# #### 1. Run a regression of the natural logarithm of sales on all the following price: price, printmarketing expenditure, outdoor marketing expenditure, broadcast marketing expenditure, and previous years’s sale. Evaluate the results. Perform residual analysis to satisfy the assumptions of regression.
# 

# In[40]:


col =data.columns


# In[41]:


col


# In[42]:


Lrmodel = data[col]
Lrmodel.head()


# In[176]:


plt.figure(figsize = (16,16))
sns.pairplot(Lrmodel)
plt.show()


# In[43]:


Lrmodel = data[['PriceRerUnit','News', 'Outdoor',  'Broad','LagTotalSales']]
Lrmodel.head()


# In[45]:


p = sns.pairplot(Lrmodel)


# ### Assumptions for Linear Regression

# In[48]:


# visualize the relationship between the features and the response using scatterplots
p = sns.pairplot(Lrmodel, x_vars=['PriceRerUnit','News','Outdoor','Broad'], y_vars='LagTotalSales', size=7, aspect=0.7)


# #### A tip is to remember to always see the plots from where the dependent variable is on the y axis. Though it wouldn't vary the shape much but that's how linear regression's intuition is, to put the dependent variable as y and independents as x(s).
# 
# #### Now rest of the assumptions require us to perform the regression before we can even check for them. So let's perform regression on it

# ### Fitting the linear model

# In[50]:


x = Lrmodel.drop(["LagTotalSales"],axis=1)
y = Lrmodel.LagTotalSales


# In[51]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)


# In[52]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0,test_size=0.25)


# In[53]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred = regr.predict(X_train)


# In[54]:


print("R squared: {}".format(r2_score(y_true=y_train,y_pred=y_pred)))


# ### Mean of Residuals

# #### Residuals as we know are the differences between the true value and the predicted value. One of the assumptions of linear regression is that the mean of the residuals should be zero. So let's find out

# In[55]:


residuals = y_train.values-y_pred
mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))


# In[72]:


p = sns.scatterplot(y_pred,residuals)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
plt.ylim(-10,10)
plt.xlim(0,26)
p = sns.lineplot([0,26],[0,0],color='blue')
p = plt.title('Residuals vs fitted values plot for homoscedasticity check')


# In[68]:


import statsmodels.stats.api as sms
from statsmodels.compat import lzip
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(residuals, X_train)
lzip(name, test)
from scipy.stats import bartlett
p = sns.distplot(residuals,kde=True)
p = plt.title('Normality of error terms/residuals')


# In[70]:


plt.figure(figsize=(8,4))
p = sns.lineplot(y_pred,residuals,marker='o',color='blue')
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
plt.ylim(-10,10)
plt.xlim(0,26)
p = sns.lineplot([0,26],[0,0],color='red')
p = plt.title('Residuals vs fitted values plot for autocorrelation check')


# In[69]:


import statsmodels.api as sm
sm.graphics.tsa.plot_acf(residuals, lags=40)
plt.show()
# partial autocorrelation
sm.graphics.tsa.plot_pacf(residuals, lags=40)
plt.show()


# #### The results show signs of autocorelation since there are spikes outside the red confidence interval region. This could be a factor of seasonality in the data

# ### 2. Run a regression of the natural logarithm of change in sales on the natural logarithm of previous period’s prices and the natural log of marketing expenditures on print, outdoor and broadcasting. Evaluate the results. Perform residual analysis to satisfy the assumptions of regression.

# In[74]:


Lrmodel = data[['Mag', 'Outdoor',  'Broad']]
Lrmodel.head()
p = sns.pairplot(Lrmodel)


# In[75]:


# visualize the relationship between the features and the response using scatterplots
p = sns.pairplot(Lrmodel, x_vars=['Outdoor','Broad'], y_vars='Mag', size=7, aspect=0.7)


# ### 3. To understand the influence of Vodka quality, run a regression by adding the tier 1 and tier 2 dummy variables (that indicate whether a vodka brand belongs to first or second quality tiers) to the set of independent variables in Q2. Evaluate the results. Perform residual analysis to satisfy the assumptions of regression.
# 

# In[80]:


Lrmodel = data[['Tier1', 'Tier2']]
Lrmodel.head()
p = sns.pairplot(Lrmodel)


# In[84]:


# visualize the relationship between the features and the response using scatterplots
p = sns.pairplot(Lrmodel, x_vars=['Tier1'], y_vars='Tier2', size=7, aspect=0.7)


# ### 4. To understand the influence of competition and brand power, run a regression by adding the sum of sales of all the competing brands in the previous year (“lagtotalminussales”) to the independent variables in Q3. Perform residual analysis to satisfy the assumptions of regression.

# In[143]:


df.groupby(["BrandName"]).sum().sort_values("Brand ID", ascending=False)


# In[148]:


df["TotalSales"] = df["Brand ID"] * df["PriceRerUnit"]
df.head()


# In[151]:


df.groupby("BrandName").sum().sort_values("TotalSales", ascending=False).head()


# In[152]:


group = df.groupby(["Year","BrandName"]).sum()
total_price = group["TotalSales"].groupby(level=0, group_keys=False)
total_price.nlargest(5)


# In[142]:


Lrmodel = data[[ 'Outdoor',  'Broad']]
Lrmodel.head()
p = sns.pairplot(Lrmodel)


# In[110]:


m = data['Outdoor']
n = data['Broad']


# In[119]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(m, n, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[120]:


import statsmodels.api as sm
X_train.head()


# In[121]:


y_train.head()


# In[122]:


# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()


# In[123]:


# Print the parameters, i.e. the intercept and the slope of the regression line fitted
lr.params


# In[124]:


print(lr.summary())


# ### Looking at some key statistics from the summary
# #### The values we are concerned with are -
# 
# ### The coefficients and significance (p-values)
# #### R-squared
# ### F statistic and its significance
# #### 1. The coefficient for outdoor is 0.488, with a very low p value
# #### The coefficient is statistically significant. So the association is not purely by chance.
# 
# #### 2. R - squared is 0.057
# #### Meaning that 5.7% of the variance in Sales is explained by TV
# 
# #### This is a decent R-squared value.

# ## Model Evaluation
# ### Residual analysis 
# 
# 
# #### To validate assumptions of the model, and hence the reliability for inference
# #### Distribution of the error terms
# #### We need to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.

# In[126]:


y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)


# In[133]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


# In[134]:


#Looking for patterns in the residuals
plt.scatter(X_train,res)
plt.show()


# ### 5.To measure the sales growth of new brands compared to the existent ones, include the variable “firstintro” to the independent variable set in Q4. Firstintro is equal to one in the first three years after a brand is introduced and is zero elsewhere

# In[162]:


data['Marketshare'] = df.TotalSales.pct_change()


# In[ ]:




