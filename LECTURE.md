### Week1 - K-nearest neighbours classifier  
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
### Week2 - Asymptotic Complexity, Analysis of Algorithms, Monte Carlo Simulation  
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




