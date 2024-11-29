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
## Week6 - Midterm

<br>
<br>
<br>  

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
<br>
<br>
Principle component analysis (PCA) is one example where eigendecomposition is used frequently

![image](https://github.com/user-attachments/assets/d19bb1ba-0eef-4c55-a0bf-97a652f4912f)  
![image](https://github.com/user-attachments/assets/5f67fbfb-1784-43e6-bceb-b715da9c3f9b)  
![image](https://github.com/user-attachments/assets/ea1be582-3980-4bc9-8308-9b55662b01ba)  




<br>
<br>
<br>  

## Week8 Reading week  

<br>
<br>
<br>  

## Week9 - PCA for clustering images, SVD, Application
Limaitations of PCA:  
PCA has its limitations. For example, when you have short (small n) and wide data (large p), PCA may not work well.  
We can speed up dimensionalitu reduction by exploiting the smaller value of n and p. To achieve this, we first need a way to decompose rectangular matrices...  
Singular Value Decommposition (SVD) to the rescue!  
<br>  
![image](https://github.com/user-attachments/assets/c3d4528b-4bd9-4b6d-ad0a-58c5ef0e5072)  
![image](https://github.com/user-attachments/assets/d9f0e767-84a9-4e2e-be7e-710c9d922ab8)  
![image](https://github.com/user-attachments/assets/73d7e84a-b7b5-4b5f-95cf-edacdfedeec9)  
![image](https://github.com/user-attachments/assets/a45e55ce-e4db-4cf9-9806-7a8a61f67397)  
![image](https://github.com/user-attachments/assets/90c02037-b073-4a8f-8bed-fd8fdd3bfab4)  
![image](https://github.com/user-attachments/assets/8c75dbdd-c07d-4941-a261-cb1f6b473cf7)

<br>
<br>
<br>  

## Week10 -Empirical Risk Minimization, Maximum Likelihood Estimation, Linear Regression, Polynomial Regression  
![image](https://github.com/user-attachments/assets/cf05434a-e695-408a-b6fe-943a16c7a5f8)  
![image](https://github.com/user-attachments/assets/b404aa8e-ff22-4198-ac41-4e04dd261cf1)  
![image](https://github.com/user-attachments/assets/6bd303d1-a4e5-4c25-850b-fb357b169e34)
![image](https://github.com/user-attachments/assets/b0abb742-b239-4f1b-9c9d-be624c06770b)  
![image](https://github.com/user-attachments/assets/6bfd207f-db57-49b6-80b9-272f005a7b74)  
![image](https://github.com/user-attachments/assets/004bd4e1-8902-4a22-990f-3b94e73420da)  
![image](https://github.com/user-attachments/assets/120a6c02-8d7a-4530-aec6-3435d1ca6a76)
![image](https://github.com/user-attachments/assets/ef507345-902a-4a6c-8e71-c819438754aa)  
![image](https://github.com/user-attachments/assets/8c721cfd-e15e-4a32-96f3-1a3721155938)  
![image](https://github.com/user-attachments/assets/d0755c87-fb81-4818-8f83-861f10986290)  
![image](https://github.com/user-attachments/assets/15cd362a-7cbc-4294-a443-f28debb34515)  
![image](https://github.com/user-attachments/assets/956dda9e-2574-49eb-b429-30b5d56c7e04)
![image](https://github.com/user-attachments/assets/6f2e4722-63e6-46a1-af75-7fbdea6fad8c)  
![image](https://github.com/user-attachments/assets/dea9ec64-fc6b-4727-b665-22ba16c75547)
![image](https://github.com/user-attachments/assets/e9e1ddca-b0cd-43d8-93dc-53b4d42fbc2a)  
![image](https://github.com/user-attachments/assets/5a7f41d5-5830-4a2e-b9cf-0e0492a80edc)  
Interpretation: TSS represents the total variation that could potentially be explained by the model. It serves as a baseline to compare how much of this total variability is accounted for by the model.  
Interpretation: RSS represents the "leftover" variability that the model fails to capture. A smaller SSR indicates a better-fitting model, as less unexplained variability remains.  
An R2 value closer to 1 indicates that a large proportion of the variability is explained by the model. For example, If R2 =0.85, it means 85% of the variability in house prices is explained by these factors.


<br>
<br>
<br>  

## Week11 - Classification, Neural Networks, Regularization  
![image](https://github.com/user-attachments/assets/56505866-4a55-45d7-bd6e-8346a00bde19)  
![image](https://github.com/user-attachments/assets/fed5a35a-a6f0-401e-bd04-5ed0f8939e54)  
![image](https://github.com/user-attachments/assets/01aecd7e-56a2-4ccb-9e1d-843b6c78f606)  
![image](https://github.com/user-attachments/assets/6d6ec42c-863e-4827-84dc-52d526a3f5ac)  
![image](https://github.com/user-attachments/assets/26ed6f11-1939-460d-b627-3989043f88a5)  
![image](https://github.com/user-attachments/assets/c28d622d-e649-4857-87f9-d93579a4a80b)  
![image](https://github.com/user-attachments/assets/da735273-5932-461f-b451-1939ab44945e)  
The L1 norm (or sum of absolute values) is regularizer that encourages weights to be exactly zero. When you expect that only a few features are important and want to perform feature selection.  
The L2 norm encourages smaller values for coefficients but does not force them to zero. Helps reduce the model's complexity without eliminating features. When all features are likely to contribute to the model but you want to reduce their individual importance to prevent overfitting.  


![image](https://github.com/user-attachments/assets/14ca401e-45c3-4bf4-bf90-ca01550f81c3)  
![image](https://github.com/user-attachments/assets/d9c417db-ef92-45ed-b902-75942e05142a)

<br>
<br>
<br>  

## Week12 - Vector Calculus, Automatic Differentiation, Deep Learning Architectures, Transfer Learning, Discrete Optimization  

![image](https://github.com/user-attachments/assets/7ad295d3-8dad-47b8-8b04-89be133c32a0)  
![image](https://github.com/user-attachments/assets/43db96fc-d31e-4d25-a5b0-7803ff6adb45)  
![image](https://github.com/user-attachments/assets/06cf2308-daf5-4860-b92a-f49b7dd42544)  
![image](https://github.com/user-attachments/assets/10d14849-5f79-47d5-8518-6e1cc55363f3)  
![image](https://github.com/user-attachments/assets/be1cc004-148c-4659-a034-bd895ec477ae)  
![image](https://github.com/user-attachments/assets/92019f5d-2f36-497a-b4b2-c2137855edef)  
![image](https://github.com/user-attachments/assets/6ebc0933-8b0d-4acc-bcd0-f3d342800605)  
![image](https://github.com/user-attachments/assets/99d4429a-7190-4b90-a582-777a11de6509)  
![image](https://github.com/user-attachments/assets/c4f2ba60-1639-4ab6-91f2-1b87b751e90e)  
![image](https://github.com/user-attachments/assets/b8e7fa45-4bde-4356-997b-30d765b9ddaa)  
![image](https://github.com/user-attachments/assets/5a1ad259-826f-4486-a170-8ea12935a7bc)  
![image](https://github.com/user-attachments/assets/3241ec91-48de-4aee-8c63-5e5dc78488be)  
![image](https://github.com/user-attachments/assets/14e43383-e23e-42ad-9ae4-9fdcd95f2387)  
![image](https://github.com/user-attachments/assets/b3da960e-1097-45b8-bf7b-818bfca7a8a4)























































