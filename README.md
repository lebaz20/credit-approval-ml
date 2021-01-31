# credit-approval-ml
Machine Learning using [Credit Approval Data Set](https://archive.ics.uci.edu/ml/datasets/credit+approval)

This file concerns credit card applications. All attribute names and values have been changed to meaningless symbols to protect the confidentiality of the data.
This will still suit our purposes as a demonstration dataset since we are not using the data to develop actual credit screening criteria. 

This dataset is interesting because there is a good mix of attributes -- continuous, nominal with small numbers of values, and nominal with larger numbers of values. There are also a few missing values.

1.  Attribute Information:

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

2.  Missing Attribute Values:

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

3.  Class Distribution
   
````
     +: 307 (44.5%)
     -: 383 (55.5%)
````

We are going to use a mix of supervised and unsupervised learning models and figure out their impact on accuracy.

We are not 