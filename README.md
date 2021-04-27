# A-B Testing on Web Forms
> Analyzing A/B test results using Python.

Table of Contents
---
1. [General Information](#general-information)
2. [Summary](#summary)
3. [Tech Stack](#tech-stack)
4. [Data Wrangling/Cleaning](#data-wranglingcleaning)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [A/B Testing](#k-nearest-neighbors-knn)
    * [Hypothesis Testing](#)
7. [Solution](#solution)
8. [Key Takeaways](#key-takeaways)

<a name="https://github.com/sangtvo/A-B-Testing-on-Web-Forms#general-information"/>
<a name="https://github.com/sangtvo/A-B-Testing-on-Web-Forms#summary"/>
<a name="https://github.com/sangtvo/A-B-Testing-on-Web-Forms#tech-stack"/>
<a name="https://github.com/sangtvo/A-B-Testing-on-Web-Forms#data-wranglingcleaning"/>
<a name="https://github.com/sangtvo/A-B-Testing-on-Web-Forms#exploratory-data-analysis"/>
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
Merging two data sets (baseline and testing data).
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

Converting dates into date time datatypes.
```python
df['prequal_date'] = pd.to_datetime(df['prequal_date'])
df['assignment_date'] = pd.to_datetime(df['assignment_date'])
```

Filtering and assigning df with only June data.
```python
df = df[df['prequal_date'].dt.month.isin([6])]
```

Checking for missing zeroes and filling them in.
```python
df.isna().sum()

df['assignment_date'] = df['assignment_date'].fillna(0)
```

Creating a group type column based on assignment_date.
```python
df['group'] = np.where(df['assignment_date'] == 0, 'control', 'treatment')
```

Exploratory Data Analysis
---
Checking unique users.
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
control	        0.462416	0.537584
treatment	    0.454524	0.545476
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

A/B Testing
---
## Hypothesis Testing
Research question: Is there evidence that the probability of a 3-field form increase or decrease conversion rate compared to a 5-field form?

- **Null hypothesis**: The control and experiment groups have the same probability of completing the form.
- **Alternative hypothesis**: The control and experiment groups have a different probability of completing the form.

We can safely assume that this test checked the assumptions:
- independence
- random sample
- sample size (n) > 30

Next, let's subset the necessary columns and rename column for simplicity.
```python
df2 = df[['completed_prequal', 'group']]
df2 = df2.rename(columns={'completed_prequal': 'converted'})
```

Calculating the control & treatment group
```python
control_group = (df2['group'] == 'control')

control_conv = df2['converted'][control_group].sum()
control_total = df2['converted'][control_group].count()

treatment_group = (df2['group'] == 'treatment')

treatment_conv = df2['converted'][treatment_group].sum()
treatment_total = df2['converted'][treatment_group].count()

print('Percentage of control group: {:.2%}'.format(control_total / len(df2['converted'])))
print('Percentage of treatment group: {:.2%}'.format(treatment_total / len(df2['converted'])))

print('Number of control applicants who converted with 5-field form: {:,}'.format(control_conv))
print('Percentage of control applicants who converted: {:.2%}'.format(control_conv / control_total))

print('Number of treatment applicants who converted with 3-field form: {:,}'.format(treatment_conv))
print('Percentage of treatment applicants who converted: {:.2%}'.format(treatment_conv / treatment_total))
```
```
Percentage of control group: 76.52%
Percentage of treatment group: 23.48%

Number of control applicants who converted with 5-field form: 15,083
Percentage of control applicants who converted: 53.76%

Number of treatment applicants who converted with 3-field form: 4,696
Percentage of treatment applicants who converted: 54.55%
```

Now, let's set some parameters for the A/B test.

Calculating the baseline conversion--the control group.
```python
baseline = control_conv / control_total
baseline
```

Assigning practical significance (effect size)--subjective and user-defined. A 1% change in conversion probability can be large in real world.
```python
practical_sig = 0.01
```

Calculating the sample size with base and practical significance using statsmodel
```python
e_size = sms.proportion_effectsize(baseline, baseline + practical_sig)
e_size
```
```
-0.02007327798961067
```

Assign power (sensitivity) as 0.8 and alpha 0.05 (confidence level is 95%)
```python
sample_size = sms.NormalIndPower().solve_power(effect_size=e_size, power=0.8, alpha=0.05, ratio=1)

print('Sample size (n) for each group: {:,}'.format(round(sample_size)))
```
```
Sample size (n) for each group: 38,958
```

The test and control group assignment are not done correctly within this dataset. Ideally, it is best to split the control and the treatment 50/50 so that each group have the same exposure. The control group is 76.52% of the data while the treatment group is 23.48% of the data. This means that one group will risk less exposure to an inferior variant during the test. In addition, sample sizes affect the conversion rate and confidence interval calculation (as shown above) which will cause skewness in the distribution and inaccuracies between the groups due to unequal sample sizes.

It is apparent that there is an issue when calculating the required sample size for each group. Prior to the calculation of the A/B testing, we see that the required sample size is 38,958. However, there is only 8,609 in the treatment group which is 4.5x less than the required amount and therefore, the results are invalid. Moreover, the treatment group would not be able to do any classical t-test because it did not meet the required sample size to make inferences. In order to avoid this situation, it is best to calculate the required sample size per group prior to assigning control and treatment groups. Also the test should have continued until we have received 38,958 observations in the treatment group.

Despite these issues, let's continue A/B testing anyway with the given data for the purpose of this project.

```python
# Calculating the pooled probability of control and treatment groups--total number of users who converted divided by total number of users
pool_prob = (control_conv + treatment_conv) / (control_total + treatment_total)
pool_prob

# Calculating pooled standard error
pool_se = math.sqrt(pool_prob * ( 1 - pool_prob) * (1 / control_total + 1 / treatment_total))
pool_se

# Calculating z-score; 0.975 represents 95% confidence interval of a two-tailed test ( 1 - (0.05/2))
z_score = stats.norm.ppf(0.975)
z_score

# Calculcate margin of error
moe = z_score * pool_se
moe

# Calculate "d hat"--estimating the difference between probability of converted experiment and probability of converted control.
d_hat = (treatment_conv / treatment_total) - (control_conv / control_total)
d_hat
```

```python
# Testing the hypothesis and calculcating the confidence interval
lower_bound = d_hat - moe
upper_bound = d_hat + moe

if d_hat > upper_bound or d_hat < lower_bound or practical_sig < lower_bound:
    print('Reject the null hypothesis.')
else:
    print('Fail to reject the null hypothesis.')

print('The confidence interval is: [{}, {}]'.format(round(lower_bound, 4), round(upper_bound, 4)))
```
```
Fail to reject the null hypothesis.
The confidence interval is: [-0.0041, 0.0199]
```

Solution
---
Based on the testing, we fail to reject the null hypothesis. Therefore, this hypothesis test is not statistically significant and we are unable to provide enough evidence for the alternative hypothesis. It is statistically proven that there is no difference or the difference is too minimal that it's not worth implementing a 3 web form field from a 5 web form field. In fact, I am confident that there is not a practically significant change because it does not meet the 1% practical significance.

The confidence interval is between -0.0041 and 0.0199 at the 95% confidence level. It is probable that the conversion rate would change by at least 0.41%. If we were to also consider the practical significance, a 1% change (like most industry standard), it does not meet the 1% threshold. In addition, these results make sense because we also did not meet the required sample size for the test, and therefore, we should continue testing as these results are not reliable.

Key Takeaways
---
* The test does not meet the 1% practical significance.
* Sample size affects the conversion rate and the confidence interval calculation and in this test particular, the testing group did not meet the 38,958 sample size requirements.
* It seems that it is best to split the control and testing group 50/50 as much as possible to get better results.