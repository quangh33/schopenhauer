#### 1. What are the main motivations for reducing a dataset’s dimensionality? What are the main drawbacks?
##### Motivation
- Improve performance and speed
- Reduce overfitting
- Data compression
##### Drawback
- Loss of information => might degrade the performance
- Complexity

#### 2. What is the curse of dimensionality?
Problems arise in high-dim space:
- data sparsity
- distance metrics lose meaning: all points are "far away" from each other => distance-based algo like knn might not work well.
- overfitting

Analogy: imagine trying to spread 100 grains of sand evenly across a 1-meter line. The grains would be quite close to each other. 
Now, try spreading those same 100 grains across a 1-meter by 1-meter square. They are much further apart.
Now, imagine trying to spread them across a 1-meter cube. The space is vast, and the grains are incredibly sparse.
The curse of dimensionality is this same principle applied to data with many features.

#### 3. Suppose you perform PCA on a 1,000-dimensional dataset, setting the explained variance ratio to 95%. How many dimensions will the resulting dataset have?
Depends.
- if dataset is composed of points that are almost perfectly aligned => 1 dim can still preserve 95% of variance
- if dataset is composed of perfectly random points => 950 dim are required to preserve 95% of variance

#### 4. How can you evaluate the performance of a dimensionality reduction algorithm on your dataset?
- Train a model (e.g., a classifier) on the original dataset. Note its performance (e.g., accuracy, F1-score).
- Apply the dimensionality reduction algorithm to the dataset.
- Train the same model on the reduced dataset. Note its performance.
