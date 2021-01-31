# credit-approval-ml
Machine Learning using [Credit Approval Data Set](https://archive.ics.uci.edu/ml/datasets/credit+approval)

This file concerns credit card applications. All attribute names and values have been changed to meaningless symbols to protect the confidentiality of the data.
This will still suit our purposes as a demonstration dataset since we are not using the data to develop actual credit screening criteria. 

This dataset is interesting because there is a good mix of attributes -- continuous, nominal with small numbers of values, and nominal with larger numbers of values. There are also a few missing values.

## 1.  Attribute Information:

````
     A1:   b, a.
     A2:   continuous.
     A3:   continuous.
     A4:   u, y, l, t.
     A5:   g, p, gg.
     A6:   c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
     A7:   v, h, bb, j, n, z, dd, ff, o.
     A8:   continuous.
     A9:   t, f.
     A10:  t, f.
     A11:  continuous.
     A12:  t, f.
     A13:  g, p, s.
     A14:  continuous.
     A15:  continuous.
     A16:  +,-         (class attribute)
````

## 2.  Missing Attribute Values:

37 cases (5%) have one or more missing values.

The missing values from particular attributes are:

````
     A1:  12
     A2:  12
     A4:   6
     A5:   6
     A6:   9
     A7:   9
     A14: 13
````

## 3.  Class Distribution
   
````
     +: 307 (44.5%)
     -: 383 (55.5%)
````

We are going to use a mix of supervised and unsupervised learning models and figure out their impact on accuracy.


## 4. Steps expected are as follows:

    1. Load data and getting familiar with available data types.
    2. Handle missing data.
    3. Normalize data.
    4. Identify the right model to study data.

### Missing data handling:

    1. Remove rows with missing data.
    2. Fill continuous data with columns' mean/median values and categorical data with columns' most frequent category.
    3. Fill based on predictive models between mostly correlated columns.
    4. Fill using unsupervised learning.
    5. Fill using deep learning.
    6. Use Algorithms that support missing values.

#### 1. Remove rows with missing data.

* Pros:
  * A model trained with the removal of all missing values creates a robust model.
* Cons:
  * Loss of a lot of information.
  * Works poorly if the percentage of missing values is excessive in comparison to the complete dataset.

#### 2. Fill continuous data with columns' mean/median values and categorical data with columns' most frequent category.

* Pros:
  * Prevent data loss which results in deletion of rows or columns.
  * Works well with a small dataset and easy to implement.
* Cons:
  * Can cause data leakage.
  * Does not factor the covariance between features (for continuous ones).

#### 3. Fill based on predictive models between mostly correlated columns.

* Pros:
  * Gives a better result than earlier methods.
  * Takes into account the covariance between the missing value column and other columns.
* Cons:
  * Considered only as a proxy for the true values.

#### 4. Fill using unsupervised learning.

* Pros:
  * Support more non-linearity between data, as it doesn't need a strong correlation  
  * Takes into account the covariance between the missing value column and other columns.
* Cons:
  * Considered only as a proxy for the true values.

#### 5. Fill using deep learning.

* Pros:
  * Quite accurate compared to other methods.
  * It supports both CPUs and GPUs.
* Cons:
  * Still can be quite slow with large datasets.

#### 6. Use Algorithms that support missing values.

* Pros:
  * No need to handle missing values in each column as ML algorithms will handle it efficiently.
* Cons:
  * No implementation of these ML algorithms in the scikit-learn library.