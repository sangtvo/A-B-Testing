# A-B Testing on Web Forms
> Analyzing A/B test results using Python.

Table of Contents
---
1. [General Information](#general-information)
2. [Summary](#summary)
3. [Tech Stack](#tech-stack)
4. [Data Wrangling/Cleaning](#data-preprocessingcleaning)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [A/B Testing](#k-nearest-neighbors-knn)
    * [Hypothesis Testing](#)
7. [Solution](#solution)
8. [Key Takeaways](#key-takeaways)

<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#general-information"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#summary"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#tech-stack"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#data-preprocessingcleaning"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#exploratory-data-analysis"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#k-nearest-neighbors-knn"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#logistic-regression"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#random-forest"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#solution"/>
<a name="https://github.com/sangtvo/EDA-and-Hotel-Cancellation-Prediction#key-takeaways"/>

General Information
---
The capstone project is part of a graduate course in order to graduate at Western Governor's University. A completion of a capstone prospectus, executive summary, and a power point presentation is required to graduate, but will not be uploaded to my repository. 

**To expand the project even further (originally binary logistic regression), KNN and random forest analysis are added.**

Summary
---
The winning model is the **random forest** algorithm with an overall accuracy of 84.50% and precision of 87.43%. This means that the model will correctly predict hotel cancellation 84.50% of the time. In order for hotel management company to reduce their current hotel cancellation rate of ~37%, management should focus on requiring deposits because 80% of the data require no deposits. This can be mitigated if hotels require fees for cancellation or mandatory deposits. When hotels have stricter cancellation policies, guests are less inclined to cancel their reservation and hotel revenue will increase. In addition, lead time is another factor to be targeted. Guests who hold reservations for long periods of time are more likley to cancel. If management offers special offers or larger discount for on-site services when booking in advance, guests are less likely to cancel. 

Tech Stack
---
* Python
  * NumPy
  * Pandas
  * statsmodel
  * math
  * scipy.stats
* VS Code
* Jupyter Notebook

Data Wrangling/Cleaning
---
Merging two data sets (baseline and testing data)
```python
base_df = pd.read_csv('Baseline.csv')
treatment_df = pd.read_csv('Testing.csv')

df = pd.merge(left=base_df, right=treatment_df, how='left', left_on='prequal_id', right_on='prequal_id')
```
```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 190976 entries, 0 to 190975
Data columns (total 4 columns):
 #   Column             Non-Null Count   Dtype 
---  ------             --------------   ----- 
 0   prequal_id         190976 non-null  object
 1   prequal_date       190976 non-null  object
 2   completed_prequal  190976 non-null  int64 
 3   assignment_date    8609 non-null    object
dtypes: int64(1), object(3)
memory usage: 7.3+ MB
```

Converting dates into date time datatypes
```python
df['prequal_date'] = pd.to_datetime(df['prequal_date'])
df['assignment_date'] = pd.to_datetime(df['assignment_date'])
```

Filtering and assigning df with only June data
```python
df = df[df['prequal_date'].dt.month.isin([6])]
```

Checking for missing zeroes and filling them in.
```python
df.isna().sum()

df['assignment_date'] = df['assignment_date'].fillna(0)
```

Creating a group type column based on assignment_date
```python
df['group'] = np.where(df['assignment_date'] == 0, 'control', 'treatment')
```

Exploratory Data Analysis
---
Checking unique users
```python
if df["prequal_id"].count() == df["prequal_id"].nunique(): 
    print("There are NO duplicate prequal_id's.")
else:
    print("There are duplicate prequal_id's.")
```
```
There are NO duplicate prequal_id's.
```

Checking counts
```python
df['group'].value_counts()
```
```
control      28057
treatment     8609
Name: group, dtype: int64
```

Cross tabulation of percentages (non-conversion rate and conversion rate) by group
```python
pd.crosstab(df['group'], df['completed_prequal']).apply(lambda r: r/r.sum(), axis=1)
```
```
completed_prequal	0	1
group		
control	0.462416	0.537584
treatment	0.454524	0.545476
```

Looking at the relative differences of the conversion rates.
```python
ctrl = 0.537584
trt = 0.545476

non_relative_diff = trt - ctrl
print('Non-relative difference of conversion rate: {:.2%}'.format(non_relative_diff))

relative_diff = (trt - ctrl) / ctrl
print('Relative difference of conversion rate: {:.2%}'.format(relative_diff))
```
```
Non-relative difference of conversion rate: 0.79%
Relative difference of conversion rate: 1.47%
```
