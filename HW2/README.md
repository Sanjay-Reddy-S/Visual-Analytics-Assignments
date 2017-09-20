# Visual-Analytics-Assignments HW-2

Tested on the local system in Python2 and on Jupyter also. (Needs scikit-learn installed on Python 2.7 for ML regression models to run)
The rendered interactive .html files are also present in the repository.

There are mainly 2 types of methods to deal with missing-data:
* Interpolate Values: Statistical imputation
    * Mean: Replace the missing value with the mean of the whole column
    * Median: Replace the missing value with the median of the whole column
    * Random: Randomly pick a row and use it to fill the missing value
    * Nearby: Instead of taking average of the whole column, only nearby 6 values are used (Top 3 and Bottom 3)
    * Polynomial Fit: Using MSE as an optimization function, one can compute the best fit polynomial and try to interpolate. (Here considered only degree 1 to 10)

* Regression: Use Machine Learning models to fill the value
    * KNN: Using this clustering algorithm, find the most similar records & use values of those to fill
    * SVC: Use a C-support Vector classification algorithm
    * Linear Regression: Least MSE error Linear Regression
    * Logistic Regression: Yields surprisingly good results
    * SVR: Support Vector Machine Regression
    * Gradient Boosting Regressor: Generalization of boosting (an ensemble method)
   
   For regression, there are two variations I have implemented:
    * Use only Channel and Region as inputs for the model (Assumption being that the spending by customers depends purely on the type of shop and where shop is located and not on other features. As an example: Monetary units (m.u) spent on Grocery is independent of other columns like that spent on Frozen).
    * All columns are given as input to the models

Only line and scatter plots are used because using that felt natural and easily was giving insight how well the estimates are differing from actual values. The X-axis, represents 6 missing values (1-'Milk',2-'Grocery',2-'Detergents_paper',1-'Delicassen')
There are 2 types of graphs: In both of them, labels can be hid by clicking on the legend (allowing individual comparisons)
    * Directly plotting the values of the columns
    * Calculating the differences and then normalizing using the range (Max-Min) of each individual column with missing value.

It uses dataset from UCI: <a href="http://archive.ics.uci.edu/ml/datasets/Wholesale+customers">Wholesale customers dataset</a>

References:
1) <a href="https://bokeh.pydata.org/en/latest/docs/user_guide.html">Bokeh</a> 
2) <a href="https://stackoverflow.com/questions/19165259/python-numpy-scipy-curve-fitting">Stackoverflow answer</a> regarding curve fitting.
3) <a href="http://scikit-learn.org/stable/">Scikit-Learn</a>
4) <a href="https://docs.scipy.org/doc/numpy-dev/user/quickstart.html">Numpy</a>
