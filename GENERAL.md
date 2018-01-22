# Feature Engineering

- missing data imputation: http://www.stat.columbia.edu/~gelman/arm/missing.pdf

# L1 and L2 regularization

- L1 regularization helps perform feature selection in sparse feature spaces, and that is a good practical reason to use L1 in some situations.
- As a rule-of-thumb, you should always go for L2 in practice.
- Even in the case when you have a strong reason to use L1 given the number of features, I would recommend going for Elastic Nets instead. Granted this will only be a practical option if you are doing linear/logistic regression. But, in that case, Elastic Nets have proved to be (in theory and in practice) better than L1/Lasso. Elastic Nets combine L1 and L2 regularization at the "only" cost of introducing another hyperparameter to tune.

Source: https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization-How-does-it-solve-the-problem-of-overfitting-Which-regularizer-to-use-and-when

# ML Algorithms

## Regression

### Linear

 - **Loss Function** 
   - zero-one https://stats.stackexchange.com/questions/284028/0-1-loss-function-explanation
   - logistic (logistic regression), hinge (support vector machine). diagram: http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_loss_functions.html

 - **Outliers** least squares estimates for regression models are highly sensitive to (not robust against) outliers


