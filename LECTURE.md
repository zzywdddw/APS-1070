## Week1 - K-nearest neighbours classifier  
We use Euclidean distance to compute the distance  
![image](https://github.com/user-attachments/assets/99d94377-b659-47ee-af55-912ad170e799)  
<br>
Small K:  
- Good at capturing fine-grained patterns
- May overfit (excellent for training data, not that good for new data, too complex), i.e. be sensitive to random noise

Large K:  
- Makes stable prediction by averaging over lots of examples
- May underfit (not that good for  training data, not good for new data, too simple). i.e. fail to capture important regularities

kNN is a non-parametric, instance-based classification (or regression) algorithm, not a clustering algorithm.  
KNN sensitivity to the ranges of the input features can be addressed by normalization.  
K can be selected based on the best performance on the validation set.  
<br>
<br>
<br>
## Week2 - Asymptotic Complexity, Analysis of Algorithms, Monte Carlo Simulation  
Basic operations take constant time:  
- Arithmetic
- Assignment
- Access one array index
- Comparing two simple values (is x<3)

Other operations are summations or products
- Consecutive statements are summed
- Loops are (cost of loop body) * (number of loops)

Big-Oh Notation:  
Given two functions f(n) and g(n) for input n, we say f(n) is in O(g(n)) iff there exist positive constants c and n0 such that f(n) <= c g(n) for all n>=n0  
Eventually g(n) is always an upper bound on f(n) ignoring constants  
![image](https://github.com/user-attachments/assets/d33bedb6-99f1-43b1-a922-7712c237309a)  
<br>
Unsorted List  
- Inserting an element is O(1)
- The only way to find an element is to check each element, this takes O(n) for any reasonable implementation
- Also, deletion is O(n) because first we should find it and then other elements must be pushed to the left
<br>
Sorted List

- Inserting an element is harder because you have to find the right place and therefore it is o(n)
- Finding an element is easier because we can perform binary search O(logn)
- Deletion is O(n) because we need to puch other elements left
![image](https://github.com/user-attachments/assets/588b7e26-c4b2-4c25-b9c4-e40e2acb61a4)

Can we do better>???  
Hash function and Tables  
- Order irrelevant, aim for constant-time(i.e., O(1)) find, insert, and delete
- A hash table is an array of some fixed size: Maps keys -> indices which contains values
- Design so that two keys rarely hash to the same index - can lead to collisions
  
Collision: when two keys map to the same location in the hash table  

Resolution: Linear probing, if a key K hashes to a value h(k) that is already occupied, moves left one index at a time, wrapping around, until it finds an empty address  
Resolution2: double hashing, Use a second hash ∆(k) = t to move from h(k) to the left  by a fixed step size t, wrapping around, if necessary, until we find an empty address

Imagine there is n balls are thrown into m boxes, if n is much smaller than m, collisions will be few and most slots will be empty. If n is much larger than m, collisions will be many and no slots will be empty.  
Load factor:  Lambda = n/m  
Note that when the load factor is very small λ → 0, collisions are unlikely (for example Q(m,0) = 1 and Q(m,1) = 1).    
At the other extreme case of λ > 1, collisions are absolutely certain (i.e. Q(m, n) = 0)  
![image](https://github.com/user-attachments/assets/6affab86-2539-48aa-8432-55623c94f1c1)  
<br>
<br>
<br>
## Week3 - Decision Trees, Clustering, End-to-end Machine Learning  
Instance-Based:  
- system learns the examples by heart, then generalizes to new cases by using a similarity/distance measure to compare them to the learned examples.
  
Model-Based:
- Build a model of these examples and then use that model to make predictions.
  
Parametric:
- Have a fixed number of parameters
- Estimate of fixed parameters improve with more data
- Make strong assumption about the data

Nonparametric:  
- Number of parameters grow with #samples
- Size depends on #samples
- Complexity grows with #samples

 Trade-offs between parametric and non-parametric algorithms are in computational cost and accuracy
<br>
<br>  
#### Decision Tree  
- A rule-based supervised learning algorithm
- Can be applied to classification (discrete) and regression (continuous) task
- Highly interpretable

For the classification tree:  
- discrete output
- output node typically set to the most common value

For the regression tree:  
- continuous tree
- output node value typically set to the mean value in data

Managing overfitting:  
1. Add parameters to reduce potential for overfitting
2. Parameters include:
     - depth of tree
     - minimum number of samples

<br>
<br>   

#### Random Forests  
![image](https://github.com/user-attachments/assets/64b775e5-1cd6-4dee-b97e-2bb6191b1939)  

#### K-Means Clustering  
Distance-based unsupervised learning algorithm:  
1. Assign each sample/instance to its closest mean
2. update the means based on the assignment
3. repeat until convergence
![image](https://github.com/user-attachments/assets/72e01f3b-9b4f-48be-bb56-c16d97f458f8)
![image](https://github.com/user-attachments/assets/2985a050-12f8-4724-a433-351d31786bfe)
![image](https://github.com/user-attachments/assets/e86b44d4-9c90-4b88-a6cf-cf9beef23b5a)
![image](https://github.com/user-attachments/assets/c99b5a97-3e28-4ee7-96db-bc50fb671a1c)
Whenever an assignment is changed, the sum squared distances, J, of data points from their assigned cluster centres is reduced
Whenever a cluster is moved, J is reduced
Test for convergence: if the assignments do not change in the assignment step, we have converged (to at least a local minimum)
<br>
Option for avoiding local minimum:
- we could try many random starting points
- split a big cluster into two
- merge to nearby clusters
  
![image](https://github.com/user-attachments/assets/1252717c-cbb6-42f7-8f7f-1e2aa2d62d9c)  
![image](https://github.com/user-attachments/assets/f848217b-80f6-4964-b4cd-ad9000e41f55)  
![image](https://github.com/user-attachments/assets/9ad39677-0807-4d48-bffc-d03f0e248bbf)  

<br>  

K-Fold Cross-Validation:  
The dataset is divided into K equally sized folds. The model is trained on K-1 folds and tested on the remaining fold. This is repeated K times, with each fold serving as the test set exactly once.  
<br>Example:    
If you have a dataset of 100 samples and use 5-fold cross-validation (K = 5), the data is split into 5 subsets. The model is trained on 4 of the subsets (80% of the data) and tested on the remaining 1 (20%). This process is repeated 5 times, and the final result is the average of the 5 test performances.

<br>
<br>
<br>  

## Week4 - Probability Theory, Gaussian Distribution, Performance Metrics  
![image](https://github.com/user-attachments/assets/f613f0dd-b173-4682-a50c-aecfec2b0ac8)  
<br>  
Imagine that a coin we believe to be fiar is flipped three times and results in three heads.  
- Frequentist calculation: estimations come from experiments and experiments only. e.g, if we want to estimate how likely a six-sided die is to roll a 4, we should roll the die many times and observe how frequently 4 appears.  
- Bayesian: our prior knowledge that the coin is fair allows us to maintain some degree of belief that a tails is still possible  
<br>  
Sample space Ω is the set of all possible outcomes of an experiment.

Observation ω ∊ Ω are points in the space also called sample outcomes, realizations, or elements.  
Events E ⊂ Ω are subsets of the sample space.  

<br>

Axions of probability:  
1. Probablity of any event must be between 0 and 1
2. The sum of the probabilities of all events equals to 1
3. If two events are disjoint (mutually exclusive, which means cannot happen at the same time), the probability that either of the events happens is the sum of the probabilities that each happens. (If AB = {}, P(A ∪ B) = P(A) + P(B).)
<br>

### Discrete Probabilities
Probability mass function(PMF): PMF maps each value in the variable's sample sapce to a probability

A common discrete distribution is the Beronulli - A Bernoulli distribution specifies the probability for a random variable which can take on one of two values. (tail or head)
  - e.g., for a fair coin we have p = 0.5,
  - e.g., given the probability of rain is p = 0.2, we can infer the probability of no rain is 0.8.

![image](https://github.com/user-attachments/assets/652be369-af9d-4bee-8050-a34dd72c1aec)  
![image](https://github.com/user-attachments/assets/7e830ffe-30b8-40fb-a7e4-eb92e40c98d6)
Two variables x and y are independent if:  
    P(x,y) = P(x) * P(y)  
Two variables x and y are conditionally independent if:  
    P(x,y | z) = P(x|z) * P(y|z)  

### Continuous Probabilities  
Continuous random variables are described by probability density functions (PDF) which can be a bit more difficult to understand  
  - PDFs map an infinite sample space to relative likelihood values
Gaussian Distribution:
![image](https://github.com/user-attachments/assets/6f8a6943-1cd2-4871-8285-c81a5a79824a)
![image](https://github.com/user-attachments/assets/116b87db-3692-4254-83b6-a1ff08840ae8)
![image](https://github.com/user-attachments/assets/4db63e5e-625a-4dec-b5c1-5ed97c86278f)
![image](https://github.com/user-attachments/assets/4dbc3bcf-146e-490d-8a1c-6a61bcabc300)
![image](https://github.com/user-attachments/assets/168bf307-bed1-4424-863b-5dfff2d38dba)
In general, variance is a measure of how much a random variable varies from its mean.
![image](https://github.com/user-attachments/assets/06ba4dc5-dd0a-4cf0-8a9a-61fb1c706f84)
![image](https://github.com/user-attachments/assets/2e70e4c1-21b6-4126-b6fe-262d9295d9fb)  
![image](https://github.com/user-attachments/assets/8673279d-c5b4-452a-8393-596bded81fa4)
What is the precision (emails flagged as spam)

What is the recall (actual spam correctly classified)


<br>
<br>
<br>  

## Week5 - Linear Algebra, Analytical Geometry, Data Augmentation  
![image](https://github.com/user-attachments/assets/68a9e311-f00b-4eaf-90e7-4074b15a6830)  
![image](https://github.com/user-attachments/assets/1909bd95-25fc-42f9-88aa-59ff2035ff7a)  
<br>  

All column vectors are linearly independent iff all columns are pivot columns
![image](https://github.com/user-attachments/assets/0c1cf681-f1c9-4b94-b03c-64b1db39aa4d)

<br>  
Orthonormal = orthogonal and unit vectors  

![image](https://github.com/user-attachments/assets/b03cd600-d584-484b-a61e-1d9375904269)  

![image](https://github.com/user-attachments/assets/3329f15f-3cb1-4898-aec3-7c0eea920be1)  
<br>
<br>
<br>  
## Week6 - Reading week

## Week7 - Projections, Matrix Decompositions, Principal Component Analysis(PCA)  

Projection onto a plane instead of another vector  
![image](https://github.com/user-attachments/assets/f977e3f0-ab3b-491f-87f6-e82872c642cd)  
![image](https://github.com/user-attachments/assets/cb220456-07f0-43ea-8a40-281e64ae4726)  

If the determinant is zero (det(A)=0), the matrix is not invertible  
If the determinant is non-zero (det(A)=0), the matrix is invertible  
![image](https://github.com/user-attachments/assets/509886b9-300c-40b2-8fbd-ef85bc2487a9)  
<br>  
How to find eigenvalue and eigenvector?  - example 7.1.2
https://math.libretexts.org/Bookshelves/Linear_Algebra/A_First_Course_in_Linear_Algebra_(Kuttler)/07%3A_Spectral_Theory/7.01%3A_Eigenvalues_and_Eigenvectors_of_a_Matrix  
<br>  
Theorems:  
1. 入i is an eigenvalue if and only if 入i is a root of a characteristic polynomial.
2. eigenvectors of a square matrix (n x n) with n distinct eigenvalues are linearly independent and form a basis of Rn.
3. The determinant of a square matrix (n x n) is the product of its eigenvalues
4. The mean of the eigenvalues is the mean of the diagonal entries.  the sum of the eigenvalues is the sum of the diagonal entries
![image](https://github.com/user-attachments/assets/a6d76da2-233d-422e-8a7d-47bc0e237cb3)
![image](https://github.com/user-attachments/assets/909321cf-8a8a-4b6f-9db4-b09d9555056f)




















