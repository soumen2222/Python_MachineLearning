import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


df = pd.read_csv("automobileEDA.csv")
df.head()

dfcorr = df[['bore', 'stroke', 'compression-ratio', 'horsepower']]
print(dfcorr.corr())

"""
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
plt.show()
"""

print(df[['engine-size', 'highway-mpg', 'peak-rpm', 'stroke', 'price']].corr())

sns.boxplot(x="drive-wheels", y="price", data=df)
plt.ylim(0,)
plt.show()

# Descriptive Statistical Analysis

print(df.describe(include=['object']))

"""
Value-counts is a good way of understanding how many units of each
 characteristic/variable we have. We can apply the "value_counts"
  method on the column 'drive-wheels'. Don’t forget the method "value_counts"
   only works on Pandas series, not Pandas Dataframes. 
   As a result, we only include one bracket "df['drive-wheels']" 
   not two brackets "df[['drive-wheels']]".

"""
print(df['drive-wheels'].value_counts())

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(
    columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'


print(drive_wheels_counts.head())

# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(
    columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)


df_group_one = df[['drive-wheels', 'body-style', 'price']]
# grouping results
df_group_one = df_group_one.groupby(['drive-wheels'], as_index=False).mean()
print(df_group_one)
# grouping results
df_gptest = df[['drive-wheels', 'body-style', 'price']]
grouped_test1 = df_gptest.groupby(
    ['drive-wheels', 'body-style'], as_index=False).mean()
grouped_test1
print(grouped_test1)

"""
In this case, we will leave the drive-wheel variable as the rows of the table, 
and pivot body-style to become the columns of the table:
"""

grouped_pivot = grouped_test1.pivot(
    index='drive-wheels', columns='body-style').fillna(0)
print(grouped_pivot)

# Let's use a heat map to visualize the relationship between Body Style vs Price.
# use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

"""
The heatmap plots the target variable (price) proportional to colour with 
respect to the variables 'drive-wheel' and 'body-style' in the vertical 
and horizontal axis respectively. This allows us to visualize how the price
 is related to 'drive-wheel' and 'body-style'.
"""

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

# label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

# move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

# insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

# rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

"""
Correlation: a measure of the extent of interdependence between variables.

Causation: the relationship between cause and effect between two variables.

It is important to know the difference between these two and that correlation does not 
imply causation. Determining correlation is much simpler the determining causation as 
causation may require independent experimentation.

Pearson Correlation

The Pearson Correlation measures the linear dependence between two variables X and Y.

The resulting coefficient is a value between -1 and 1 inclusive, where:

1: Total positive linear correlation.
0: No linear correlation, the two variables most likely do not affect each other.
-1: Total negative linear correlation.
Pearson Correlation is the default method of the function "corr". Like before we can 
calculate the Pearson Correlation of the of the 'int64' or 'float64' variables.
"""

"""
P-value:

What is this P-value? The P-value is the probability value that the correlation 
between these two variables is statistically significant. Normally, we choose a 
significance level of 0.05, which means that we are 95% confident that the correlation
 between the variables is significant.

By convention, when the

p-value is  <  0.001: we say there is strong evidence that the correlation is significant.
the p-value is  <  0.05: there is moderate evidence that the correlation is significant.
the p-value is  <  0.1: there is weak evidence that the correlation is significant.
the p-value is  >  0.1: there is no evidence that the correlation is significant.

"""

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is",
      pearson_coef, " with a P-value of P =", p_value)

pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print("The Pearson Correlation Coefficient is",
      pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print("The Pearson Correlation Coefficient is",
      pearson_coef, " with a P-value of P = ", p_value)


"""
ANOVA: Analysis of Variance
The Analysis of Variance (ANOVA) is a statistical method used to test whether there are significant differences between the means of two or more groups. ANOVA returns two parameters:

F-test score: ANOVA assumes the means of all groups are the same, calculates how much the actual means deviate from the assumption, and reports it as the F-test score. A larger score means there is a larger difference between the means.

P-value: P-value tells how statistically significant is our calculated score value.

If our price variable is strongly correlated with the variable we are analyzing, expect ANOVA to return a sizeable F-test score and a small p-value.

Drive Wheels
Since ANOVA analyzes the difference between different groups of the same variable, the groupby function will come in handy. Because the ANOVA algorithm averages the data automatically, we do not need to take the average before hand.

Let's see if different types 'drive-wheels' impact 'price', we group the data.

Let's see if different types 'drive-wheels' impact 'price', we group the data.

"""
grouped_test2 = df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])

# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group(
    'rwd')['price'], grouped_test2.get_group('4wd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)

f_val, p_val = stats.f_oneway(grouped_test2.get_group(
    'fwd')['price'], grouped_test2.get_group('rwd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)
"""
Conclusion: Important Variables
We now have a better idea of what our data looks like and which variables are important to take into account when predicting the car price. We have narrowed it down to the following variables:

Continuous numerical variables:

Length
Width
Curb-weight
Engine-size
Horsepower
City-mpg
Highway-mpg
Wheel-base
Bore

Categorical variables:

Drive-wheels
As we now move into building machine learning models to automate our analysis,
 feeding the model with variables that meaningfully affect our target variable will
  improve our model's prediction performance.

"""
