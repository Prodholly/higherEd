#!/usr/bin/env python
# coding: utf-8

# # Higher Education Students Performance Evaluation
# The data was collected from the Faculty of Engineering and Faculty of Educational Sciences students in 2019. The purpose is to predict students' end-of-term performances using ML techniques.
# 
# Goals:
# 1. Which feature most impact a student's end-of-term performance
# 2. Which feature least impact a student's end-of-term performance
# 3. What impact does having a spouse have on one's end-of-term performance
# 4. Does artistic or sports activity impact end-of-term performance
# 
# Grade (0: Fail, 1: DD, 2: DC, 3: CC, 4: CB, 5: BB, 6: BA, 7: AA)

# Source: https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation

# In[85]:


# Import dependencies
import re
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split


# In[86]:


df = pd.read_csv('stdperf.csv')
df


# In[87]:


# Replace column names with descriptive names

df_new = df.rename(columns={'1':'stdAge', '2':'Sex', '3':'GradHigh', '4':'Schip', '5':'AddW',
                  '6':'ArtSprt', '7':'Partner', '8':'TotSal', '9':'Transp', '10':'Acc',
                   '11':'MotherEd', '12':'FatherEd', '13':'Sib', '14':'ParStat',
                   '15':'MotherOcc', '16':'FatherOcc', '17':'WeekStd', '18':'ReadFreqN',
                   '19':'ReadFreqS', '20':'AttdConf', '21':'ProjImp', '22':'AttdCls',
                   '23':'PrepMid1', '24':'PrepMid2', '25':'TakeNote', '26':'Listn',
                   '27':'Discn', '28':'FlipCl', '29':'CumGPA', '30':'ExpGPA',
                   'COURSE ID':'CID'
                  
                  })

df_new.head()


# In[88]:


df_new.drop(columns='CID', inplace=True)


# In[89]:


# All the values are categorical even though they are numeric
df_new.info() # Check general info


# In[90]:


df_new.drop(columns='STUDENT ID', inplace=True)


# In[91]:


df_new['GRADE'].unique()


# This dataset contains 145 rows and 33 columns. It contains all numeric values and has no missing values

# # Exploratory Data Analysis and Visualization

# In[92]:


sns.set_style('darkgrid')
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.facecolor'] = '#00000000'


# In[93]:


# Plotting the histogram

px.histogram(df_new, x='GRADE', nbins=8, title='Count of Student Grades')


# In[ ]:





# From the above histogram, most of the students have a GRADE of 1(DD) and only few failed (GRADE of 0)

# # Calculating Correlation betweent the categorical features
# 
# Comparing 2 categorical features, I will use Chi-Squared Test
# 
# I have these hypotheses:
# 
# H0 = There 2 categorical features are not correlated
# H1 = There 2 categorical features are correlated
# 
# Where p-value < 0.05
# 
# Chi-Squared Test result: The probability of H0 being True
# 
# I will test out all the features and see which of them are correlated with the GRADE
# 
# Then, I will know which features that influence the GRADE
# 
# 
# 

# In[94]:


# Creating a crosstab to see the count of the features against the GRADE


# First split features from the target GRADE
df_split = df_new.drop(columns='GRADE')
df_split


# In[95]:


# Store target GRADE in a new dataframe

df_grade = df['GRADE']
df_grade.head()


# In[96]:


# Create the crosstab

# This for loop creates a count of (for example) stdAges and how many of them have certain grades
for col in df_split:
    crt = pd.crosstab(df_split[col],df_grade, margins=True)
    print(crt)


# In[97]:


# Importing library to calculate Chi-Square

# Want to test the hypotheses

from scipy.stats import chi2_contingency


# In[98]:


# Calculate the p-value and accept or reject

def is_correlated(x, y):
    # Create a contingency table from the two columns in the DataFrame
    ct = pd.crosstab(index=df_new[x], columns=df_new[y])
    # Perform the Chi-Square test of independence on the contingency table
    chi_sq_result = chi2_contingency(ct,)
    # Extract the p-value from the Chi-Square test result
    # Determine if the columns are correlated based on the p-value (threshold of 0.05)
    p, x = chi_sq_result[1], 'correlated' if chi_sq_result[1] < 0.05 else 'not correlated'
    # Return the p-value and correlation status
    return p, x


# In[99]:


# Iterate over each column name in df_split
for col in df_split:
    # Call the function is_correlated to calculate correlation with 'GRADE'
    p_value, corr = is_correlated(col, 'GRADE')
    
    # Print the results indicating correlation status
    print(f'The p-value of {col} is {p_value}, hence it is {corr} with GRADE')


# In[100]:


# Iterate over each column name in df_split
for col in df_split:
    # Call the function is_correlated to calculate correlation with 'GRADE'
    p_value, corr = is_correlated(col, 'GRADE')
    # Print the results indicating correlation status
    print(f'The p-value of {col} is {p_value}, hence it is {corr} with GRADE')


# In[101]:


# Iterate over each column name in df_split
for col in df_split:
    # Call the function is_correlated to calculate correlation with 'GRADE'
    p_value, corr = is_correlated(col, 'GRADE')
    
    # Check if the p-value is less than 0.05
    if p_value < 0.05:
        # Print the results indicating significant correlation
        print(f'The p-value of {col} is {p_value}, hence it is {corr} with GRADE')
    else:
        # Skip to the next iteration if p-value >= 0.05
        continue


# # Interpreting results from the correlation above
# 
# There is a correlation between [Student Age; Sex; Scholarship; Mother Education; Project Impact; Cumulative GPA; Expected CGPA; and Course ID ] and GRADE of students
# 
# From the p-values:
# 
# 1. Cumulative GPA has the most impact with p-value (5.087929780266156e-05)
# 
# 2. If we were to consider other features apart from CGPA, then the most impactful would be Sex followed by Student Age then Scholarship
# 
# 3. The feature with least impact is (Expected GPA) is Mother Education
# 
# 4. Spouse and participating in arts and sports does not have any significant impact on the GRADE

# In[102]:


df_split.head()


# In[103]:


df_grade.dtypes


# In[104]:


#plt.scatter(df_grade, df_split, marker='*', color='blue')


# In[105]:


# Train Test and Split


# In[106]:


#from sklearn.model_selection import train_test_split


# In[107]:


#X_train, X_test, y_train, y_test = train_test_split(df_split, df_grade, test_size=0.20, random_state=42)


# In[108]:


# Training model


# In[109]:


#from sklearn.linear_model import LogisticRegression


# In[110]:


#lr = LogisticRegression()
#lr.fit(X_train, y_train)


# In[111]:


#lr.predict(X_test)


# In[112]:


#lr.score(X_test, y_test)


# # Model Tuning
# 
# This is a poor score. The model underfitted. I will try the following to tune the model:
# 
# 1. Standard Scaler
# 2. L1 or L2 regularization
#     

# In[113]:


# Feature Scaling using 
plt.figure(figsize=(10, 5))
sns.histplot(df_split, kde=True)
plt.title("Histogram with KDE")
plt.show()


# In[114]:


# The features do not follow Gaussian distribution, hence I will use Normalization for feature scaling
# Feature scaling is needed as the feature values for example male is 2 but female is 1 assumes male has more weight than female


# # Normalization of features
# 
# 

# In[115]:


from sklearn.preprocessing import MinMaxScaler


# In[116]:


scaler = MinMaxScaler()


# In[117]:


scaled_df = scaler.fit_transform(df_split)


# In[118]:


# Convert the scaled data back to a DataFrame
scaled_df = pd.DataFrame(scaled_df, columns=df_split.columns)
scaled_df.head()


# In[119]:


from sklearn.model_selection import train_test_split


# In[120]:


X_train, X_test, y_train, y_test = train_test_split(scaled_df, df_grade, test_size=0.20, random_state=42)


# In[121]:


from sklearn.linear_model import LogisticRegression


# In[122]:


normlr = LogisticRegression()
normlr.fit(X_train, y_train)


# In[146]:


y_pred = normlr.predict(X_test)
y_pred


# In[147]:


normlr.score(X_test, y_test)


# # Update
# 
# This is a little better. After Normalization, the score increased from 0.13 to 0.17
# 
# Let's try L2 regularization

# In[161]:


from sklearn.linear_model import Ridge
ridge_reg= Ridge(alpha=500, max_iter=100, tol=0.1)
ridge_reg.fit(X_train,y_train)


# In[162]:


ridge_reg.score(X_train,y_train)


# In[163]:


ridge_reg.score(X_test, y_test)


# In[156]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
r2


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




