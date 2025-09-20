#### 1. Which Linear Regression training algorithm can you use if you have a training set with millions of features?  
- 2 best algorithms are
  - Stochastic Gradient Descent
  - Mini-batch Gradient Descent
- Why not Least Square: runtime is O(n^3) where n is the number of features.
- Why not Batch GD: memory intensive
#### 2. Suppose the features in your training set have very different scales. Which algorithms might suffer from this, and how? What can you do about it?
- Gradient Descent-based Algorithms
- If features are on different scales, the cost function will have a long, narrow, and elongated shape.
This causes the gradient descent algorithm to take a long, zig-zagging path, making it converge much more slowly
- Moreover, regularized models may converge to a suboptimal solution if the features are not scaled: since regularization penalizes large weights,
features with smaller values will tend to be ignored compared to features with larger values.

<img width="497" height="453" alt="image" src="https://github.com/user-attachments/assets/4a496b23-67a8-4133-a42b-d97524be5943" />

- How to fix:
  - Min-Max scaling: x_scaled = (x-x_min)/(x_max-x_min)
  - Standardization: x_scaled = (x-μ)/σ
 
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### 3. Can Gradient Descent get stuck in a local minimum when training a Logistic Regression model?
No, Gradient Descent cannot get stuck in a local minimum when training a standard Logistic Regression model. 
This is because the cost function for Logistic Regression, which is the log-loss or binary cross-entropy, is a convex function. 
A convex function has a single, unique global minimum and no other local minima.

#### 4. Do all Gradient Descent algorithms lead to the same model, provided you let them run long enough?
No, algorithms like Stochastic Gradient Descent (SGD) and Mini-batch Gradient Descent (MGD) have randomness.

- SGD updates the model parameters using the gradient from a single, randomly chosen training instance.
- MGD uses a small, randomly chosen subset (a mini-batch) of the data.

#### 5. Suppose you use Batch Gradient Descent and you plot the validation error at every epoch. If you notice that the validation error consistently goes up, what is likely going on? How can you fix this?
- it's most likely because your learning rate is too high. This causes the algorithm to overshoot the optimal solution and diverge.
- 2nd root cause may be features have different scale. Unscaled features can lead overshooting even with reasonable learning rate.

#### 6. Suppose you are using Polynomial Regression. You plot the learning curves and you notice that there is a large gap between the training error and the validation error. What is happening? What are three ways to solve this?
- Overfitting
- 3 ways to resolve:
1. reduce the degree of polynomial model
2. regularize the model using Ridge or Lasso
3. increase traning set size

#### 7. Suppose you are using Ridge Regression and you notice that the training error and the validation error are almost equal and fairly high. Would you say that the model suffers from high bias or high variance? Should you increase the regularization hyperparameter α or reduce it?
- high bias or underfitting
- should reduce alpha because the model is penaltizing the weight too much.

#### 8. Why would you want to use:
##### a. Ridge Regression instead of plain Linear Regression (i.e., without any regularization)?
- when some features are highly correlated => Ridge Regression helps to distribute the weight evently.
##### b. Lasso instead of Ridge Regression?
- Lasso uses l1 panelty, which can reduce the weight to zero => can eliminate unrelevant features. If you are sure that only a few features matter, use Lasso.
##### c. Elastic Net instead of Lasso?
- Lasso can't deal with data with highly correlated features

#### 9. How do you interpret coefficient of a linear regression model
The coefficients in a linear regression model provide insights into the `strength and direction` of the relationship between the input features and the target variable.
It represents the average change in the target variable Y for a one-unit increase in the predictor X_i

#### 10. How do you measure the strength of a linear relationship between two variables?
- Pearson's product-moment correlation coefficient
- Value ranges from -1 to 1, with -1 indicating a perfect negative linear relationship, 0 indicating no linear relationship, and 1 indicating a perfect positive linear relationship.
  
<img width="315" height="72" alt="image" src="https://github.com/user-attachments/assets/8f6805c3-80e5-48c4-95b2-246a32be7f34" />

```
import pandas as pd
import plotly.express as px

file_path = '/content/drive/My Drive/Colab Notebooks/chicago_taxi_train.csv'
data = pd.read_csv(file_path)

training_df = data.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]
training_df.corr(numeric_only = True)
px.scatter_matrix(training_df, dimensions=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
```

