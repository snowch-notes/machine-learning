# Feature Engineering

- missing data imputation: http://www.stat.columbia.edu/~gelman/arm/missing.pdf



# ML Algorithms

## Terminology

 - **Convex function** There are many ways to rigorously define a convex function, but in loose terms it looks like the fabric of a trampoline being weighed down by a person standing on it. It means that a local minimum (see below) is also a global minimum, and the gradient at each point is pointing towards this minimum. You can get to the minimum by stepping along in the direction of these gradients, a technique called gradient descent. This is what you do to minimize your loss with respect to the parameters in your model. But you can’t do this with 0–1 loss because at each corner there is no gradient, and the dropoff from 0 to 1 has infinite slope. ([Source](https://www.quora.com/Why-is-the-0-1-indicator-function-a-poor-choice-for-loss-function))
 - **Local minimum** is a minimum relative to its immediately surrounding points, but it is not necessarily the **Global minimum**, which is the minimum of the entire function ([Source](https://frnsys.com/ai_notes/foundations/optimization.html)]
 - **L1 and L2 regularization** ([Source](https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization-How-does-it-solve-the-problem-of-overfitting-Which-regularizer-to-use-and-when))
    -  L1 regularization helps perform feature selection in sparse feature spaces, and that is a good practical reason to use L1 in some situations.
    - As a rule-of-thumb, you should always go for L2 in practice.
    - Even in the case when you have a strong reason to use L1 given the number of features, I would recommend going for Elastic Nets instead. Granted this will only be a practical option if you are doing linear/logistic regression. But, in that case, Elastic Nets have proved to be (in theory and in practice) better than L1/Lasso. Elastic Nets combine L1 and L2 regularization at the "only" cost of introducing another hyperparameter to tune.
 - **Link function** The link function provides the **relationship** between the **linear predictor** and the **mean of the distribution function**. There are many commonly used link functions, and their choice is informed by several considerations. ([Source](https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function))
 
## Regression

### Linear

 - **Loss Function** 
   - **zero-one / 0-1**
      - count the number of misclassified items.  Accuracy = count correctly classified / count total ([Source](https://stats.stackexchange.com/questions/284028/0-1-loss-function-explanation))
      - not used because it isn't convex (see [above](https://github.com/snowch-notes/machine-learning/blob/master/GENERAL.md#terminology)) and isn't differentiable ([Source](https://www.quora.com/Why-is-the-0-1-indicator-function-a-poor-choice-for-loss-function))
   - logistic (logistic regression), hinge (support vector machine). Diagram: http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_loss_functions.html

 - **Outliers** least squares estimates for regression models are highly sensitive to (not robust against) outliers


