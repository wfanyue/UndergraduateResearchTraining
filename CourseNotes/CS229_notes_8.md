## The k-means clustering algorithm

##### Intuition

<img src="note8_k-means algorithm.png" alt="note8_k-means algorithm" style="zoom:50%;" />

##### Mathematically:

Data: $\{x^{(1)},\cdots,x^{(m)} \}$

**Step 1:** Initialize cluster centroids $\mu_1,\cdots,\mu_k\in\mathbb{R}^n$ randomly.

The more common way to initialize these is just to pick $k$ training examples and set the cluster centroids to be at exactly the location of those examples.

**Step 2:** Repeat until convergence:

(a) Set $c^{(i)}:=\arg\min\limits_j||x^{(i)}-\mu_j||^2$ ("color the points")

(b) For $j=1,\cdots, k$, $\mu_j:=\dfrac{\sum\limits_{i=1}^m1\{c^{(i)}=j \}x^{(i)}}{\sum\limits_{i=1}^m1\{c^{(i)}=j \}}$;

##### Convergence

$J(c,\mu)=\sum\limits_{i=1}^m||x^{(i)}-\mu_{c^{(i)}}||^2$

"The distortion function J is a non-convex function, and so coordinate descent on J is not guaranteed to converge to the global minimum. In other words, k-means can be susceptible to local optima. Very often k-means will work fine and come up with very good clusterings despite this. But if you are worried about getting stuck in bad local minima, one common thing to do is run k-means many times (using different random initial values for the cluster centroids $μ_j$). Then, out of all the different clusterings found, pick the one that gives the lowest distortion $J(c, μ)$."

##### How to choose $k$?

Usually pick the number of the clusters $k$ either manually or looking at what you want to use $k$-means cluster for.

