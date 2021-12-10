## Decision Trees

Decision trees -- our first examples of a non-linear model

##### Canonical situation:

- No linear separation line
- Want to divide input space into "regions"
- Can do this by dividing input space into disjoint regions $R_i$

##### Method:

Greedy, Top-Down, Recursive, Partitioning

Specifically: Recursively splitting regions

- Parent region $R_p$
- Children regions $R_1$, $R_2$
- Split on feature $x_j$
- Looking for a split $S_p$

$S_p(j,t)=(R_1,R_2)$,

$R_1=\{x|x_j<t,x\in R_p\}$

$R_2=\{x|x_j\ge t,x\in R_p\}$

$x_j$: $j$-th feature; $t$: threshold;

<img src="decision tree splits.png" alt="decision tree splits" style="zoom:45%;" />

##### How to choose a split? How "good" is a split?

- Define $L(R)$: loss on region $R$;

- Loss of the parent region $L(R_p)$ must be higher than that of child regions $R_1$ and $R_2$

- When deciding which attribute to split on, pick the one which maximizes the "gain" in the loss

- $\max_\limits{j,t}L(R_p)-(L(R_1)+L(R_2))$

  - Greedy splitting

    $L(R_p)-\dfrac{|R_1|L(R_1)+|R_2|L(R_2)}{|R_1|+|R_2|}$

##### Entropy loss

- Given $C$ classes, define $\hat{p}_c$ to be the proportion of  examples in $R$ that are of class $C$.
- $L_{cross}(R)=0$ if all the data in region $R$ belongs to a single class.
- $L_{cross}(R)=-\sum\limits_c\hat{p}_c\log_2\hat{p}_c$

- The entropy loss is convex(凸)
- Under reasonable conditions, weighted average of children's loss is always less than parent's loss

<img src="entropy loss change.png" alt="entropy loss change" style="zoom:45%;" />

Common alternative: Gini impurity

$I_G(\hat{p})=\sum\limits_{i=1}^{c}\hat{p}_i\left(\sum\limits_{k\neq c}\hat{p}_k\right)=\sum\limits_{i=1}^{c}\hat{p}_i(1-\hat{p}_i)$

##### Regression Trees

Same growth process, but final prediction is the mean of all datapoints in region: $\hat{y}=\dfrac{\sum_{i\in R}y_i}{|R|}$

Use least squares loss to split: $L_{squared}(R)=\dfrac{\sum_{i\in R}(y_i-\hat{y})^2}{|R|}$

##### Regularization

Decision trees are highly prone to overfitting. 

- Minimum leaf size
- Maximum depth
- Maximum number of nodes

##### Runtime Complexity

$n$ examples, $f$ features and a tree of depth $d$

Test time complexity: $O(d)$ (balanced trees: $O(\log n)$)

Train time complexity: $O(nfd)$

##### Decision trees lack "additive" structure

<img src="additive structure of decision trees.png" alt="additive structure of decision trees" style="zoom:50%;" />

The left can be done, while the right can not.

## Ensembling(集成学习)

Take $x_i$'s which are random variable(RV) that are independent identically distributed(i.i.d.)

$Var(x_i)=\sigma^2$, $Var(\overline{x})=Var\left(\dfrac{1}{n}\sum\limits_ix_i\right)=\dfrac{\sigma^2}{n}$

Drop independence assumption, now $x_i$'s are identically distributed(i.d.).

$x_i$'s correlated by $\rho$

$Var(\overline{x})=\rho\sigma^2+\dfrac{1-p}{n}\sigma^2$

##### Ways to ensemble

- different algorithms
- different training sets
- Bagging (Random Forests)
- Boosting (AdaBoost, XGBoost)

##### Bagging -- Bootstrap Aggregation

Have a true population $P$

Training set $S$~$P$

Assume $P=S$

Bootstrap samples $Z$~$S$

Bootstrap samples $Z_1,\cdots,Z_n$

Train model $G_m$ on $Z_m$

$G(m)=\dfrac{\sum\limits_{m=1}^{M}G_m(x)}{M}$

$Var(\overline{x})=\rho\sigma^2+\dfrac{1-p}{M}\sigma^2$

Bootstrapping is driving down $\rho$

More $M$ -> less variance, bias slightly increased because of random subsampling

##### DTs+Bagging

DT are high variance low bias

Ideal fit for bagging

##### Random Forests

Each tree can use feature and sample bagging

- Randomly select a subset of the data to grow tree
- Randomly select a set of features
- Decreases the correlation between different trees int the forest

##### Boosting

- Iteratively add simple "weak" classifiers to improve classification performance
- After doing weak classifier, evaluate performance and reweight training samples
- Weak classifier can be decision tree of depth 1 (decision stump)
- Theoretically, can achieve zero training loss

