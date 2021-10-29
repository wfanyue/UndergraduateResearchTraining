# Part V Support Vector Machine

## Functional Margin

$h_{\theta}(x)=g(\theta^Tx)$

Predict "1" if $\theta^T\ge0$($h_{\theta}(x)=g(\theta^Tx)\ge0.5$;

​             "0" otherwise.

If $y{(i)}=1$, hope that $\theta^Tx^{(i)}\gg0$;

If $y{(i)}=0$, hope that $\theta^Tx^{(i)}\ll0$;

## Geometric Margin

<img src="geometric margin.png" alt="geometric margin" style="zoom:40%;" />

## Notation

Labels $y\in\{-1,+1\}$

Have $h$ output value in $\{-1,+1\}$
$$
g(z)=\left\{ \begin{align}

 & 1 & if\ & z\ge 0\\ 

 & -1 & if\ & z<0 \\ 

\end{align} \right.
$$
(an abrupt transition)

Previously: $h_{\theta}(x)=g(\theta^Tx)$, where $x\in\mathbb{R}^{n+1}0$, $x_0=1$.

Now: $h_{w,b}(x)=g(w^Tx+b)$, where $x\in\mathbb{R}^n$, $b\in\mathbb{R}$,

We drop $x_0=1$ convention. $\theta_0$ is new $b$, and $\theta_1,\cdots,\theta_n$ is new $w$.

##### Functional margin of hyperplane defined by $(w,b)$ with respect to $(x^{(i)},y^{(i)})$

$\hat{\gamma}^{(i)}=y^{(i)}(w^Tx^{(i)}+b)$

If $y^{(i)}=1$, want $w^Tx^{(i)}+b\gg0$;

If $y^{(i)}=-1$, want $w^Tx^{i}+b\ll0.$

want $\hat{\gamma}^{(i)}\gg0$.

##### Functional margin with respect to training set

$\hat{\gamma}=\min\limits_{i=1,\cdots,m}\hat{\gamma}^{(i)}$. (worse example)

Normalization: $||w||=1$, $(w,b)\rightarrow\left(\dfrac{w}{||w||},\dfrac{b}{||w||}\right)$.

##### Geometric margin of hyperplane defined by $(w,b)$ with respect to $(x^{(i)},y^{(i)})$

$\gamma^{(i)}=\dfrac{y^{(i)}(w^Tx^{(i)}+b)}{||w||}$

Relations: $\gamma^{(i)}=\dfrac{\hat\gamma}{||w||}$.

##### Geometric margin with respect to training set

$\gamma=\min\limits_{i=1,\cdots,m}\gamma^{(i)}$. (worse example)

## Optimal margin classifier (separable case)

Choose $w,b$ to maximize $\gamma$.

$\max\limits_{\gamma,w,b}\gamma$, s.t. $\dfrac{y^{(i)}(w^Tx^{(i)}+b)}{||w||}\ge\gamma$, $i=1,2,\cdots,m$. (maximize the worst case)

To simplify, to scale $||w||$ to be equal to $\dfrac{1}{\gamma}$.

This is equivalent to 

$\min\limits_{w,b}\dfrac{1}{2}||w||^2$, s.t. $y^{(i)}(w^Tx^{(i)}+b)\ge1$.

Suppose $w=\sum\limits_{i=1}^{m}\alpha_iy^{(i)}x^{(i)}$, where $y^{(i)}=\pm1$,

which means assuming $w$ is a linear combination of the training examples.

(**Representer Theorem**: we can make this assumption without losing any performance)

So this is equivalent to 

$\min\limits_{w,b}\dfrac{1}{2}||w||^2=\min\dfrac{1}{2}\left(\sum\limits_{i=1}^{m}\alpha_iy^{(i)}x^{(i)}\right)^T\left(\sum\limits_{i=1}^{m}\alpha_iy^{(i)}x^{(i)}\right)=\min\dfrac{1}{2}\sum\limits_i\sum\limits_j\alpha_i\alpha_jy^{(i)}y^{(j)}x^{(i)T}x^{(j)}$

$=\min\dfrac{1}{2}\sum\limits_i\sum\limits_j\alpha_i\alpha_jy^{(i)}y^{(j)}\left<x^{(i)},x^{(j)}\right>$,

s.t. $y^{(i)}\left(\left(\sum\limits_{j=1}^{m}\alpha_jy^{(j)}x^{(j)}\right)^Tx^{(i)}+b\right)=y^{(i)}\left(\sum\limits_{j}\alpha_jy^{(j)}\left< x^{(j)},x^{(i)}\right>+b\right)\ge1$.

## Kernel trick

Step 1: Write algorithm in terms of $\left<x^{(i)},x^{(j)}\right>$ (simplified: $\left<x,z\right>$);

Step 2: Let there be mapping from $x\mapsto\phi(x)$;

Step 3: Find a way to compute $K(x,z)=\phi(x)^T\phi(z)$ (Kernel Function);

Step 4: Replace $\left<x,z\right>$ in algorithm with $K(x,z)$;

e.g., $x=\left[\begin{matrix}x_1\\x_2\\x_3\end{matrix}\right]$,$\phi(x)=\left[\begin{matrix}x_1x_1\\x_1x_2\\x_1x_3\\x_2x_1\\x_2x_2\\x_2x_3\\x_3x_1\\x_3x_2\\x_3x_3\end{matrix}\right]$, then we have $K(x,z)=\phi(x)^T\phi(z)=(x^Tz)^2$,which can simplify $O(n^2)$ to $O(n)$.

e.g. If $K(x,z)=(x^Tz+c)^2$, $\phi$ is supposed to be modified to $\phi(x)=\left[\begin{matrix}x_1x_1\\x_1x_2\\x_1x_3\\x_2x_1\\x_2x_2\\x_2x_3\\x_3x_1\\x_3x_2\\x_3x_3\\\sqrt{2c}x_1\\\sqrt{2c}x_2\\\sqrt{2c}x_3\\c\\\end{matrix}\right]$.

## SVM=Optimal margin classifier+kernel tricks

<img src="SVW with polynomial kernal visualization.png" alt="SVW with polynomial kernal visualization" style="zoom:50%;" />

<center>SVM with polynomial kernal visualization (From class video)</center>

##### How to make kernels?

If $x,z$ are "similar" $K(x,z)=\phi(x)^T\phi(z)$ is "large";

If $x,z$ are "dissimilar" $K(x,z)=\phi(x)^T\phi(z)$ is "small";

e.g. $K(x,z)=\exp\left(-\dfrac{||x-z||^2}{2\sigma^2}\right)$ (Guassian kernel).

e.g. $K(x,z)=x^Tz$, where $\phi(x)=x$ (Linear kernel).

##### Does there exist $\phi$, s.t. $K(x,z)=\phi(x)^T\phi(z)$?

Let $\{x^{(1)},\cdots,x^{(d)}\}$ be $d$ points.

Let $K\in\mathbb{R}^{d\times d}$ to be "Kernel matrix", where $K_{ij}=K(x^{(i)},x^{(j)})$.

Given any vector $z$, we can prove that $z^TKz\ge0$, i.e., the kernel matrix $K$ is positive semi-definite(半正定).

**Theorem (Mercer):** $K$ is  a valid kernel function (i.e., there exist $\phi$, s.t. $K(x,z)=\phi(x)^T\phi(z)$) if and only if for any $d$ points $\{x^{(1)},\cdots,x^{(d)}\}$, the corresponding kernel matrix $K$ is positive semi-definite.

## $L_1$ norm soft margin SVM

$\min\limits_{w,b,\xi_i}\dfrac{1}{2}||w||^2+C\sum\limits_{i=1}^{m}\xi_i$, s.t. $y^{(i)}(w^Tx^{(i)}+b)\ge1-\xi_i$, $i=1,2,\cdots,m$.($\xi_i\ge0$)

This algorithm allows the SVM to still keep the decision boundary closer to the previous line, even when there's one outlier.

<img src="L_1 norm soft margin SVM.png" alt="L_1 norm soft margin SVM" style="zoom:40%;" />

<center>L_1 norm soft margin SVM (From class video)</center>

## Applications

MNIST

Protein sequence classifier

## Bias and Variance

In linear regression:

<img src="E:\罗之尧\070000 电子资料\470000 成电\476000 计算机科学与技术\476800 AI Science\CS229\notes_my\Bias and variance of linear regression.png" alt="Bias and variance of linear regression" style="zoom:40%;" />

In logistic regression:

<img src="Bias and variance of logistic regression.png" alt="Bias and variance of logistic regression" style="zoom:40%;" />

**Underfit:** not capturing the trend that is maybe semi-evident in the data;

**High bias:** having very strong preconceptions that the data could be fit by a specific model.

**Overfit:** being too precise on training sets;

**High variance:** maybe fitting some totally other varying model with sightly different training sets.

## Regularization

For linear regression: $\min\limits_{\theta}\dfrac{1}{2}\sum\limits_{i=1}^n||y^{(i)}-\theta^Tx^{(i)}||^2+\dfrac{\lambda}{2}||\theta||^2$.

<img src="Regularization terms.png" alt="Regularization terms" style="zoom:40%;" />

If $\lambda$ is too small, it may overfit;

If $\lambda$ is too large, it will force $\theta$ almost to $\bold{0}$, and we will get $h_{\theta}(x)\approx0$.

For logistic regression: $\max\limits_{\theta}\sum\limits_{i=1}^n\log p(y^{(i)}|x^{(i)};\theta)-\lambda||\theta||^2$.

## Generalized error

<img src="generalization error.png" alt="generalization error" style="zoom:40%;" />

## Data splits

Train/development/test sets

S -> S_trian, S_dev, S_test

##### Basic steps and methods:

- Train each model option for degree of polymonial on S_train;
- Measure error on S_dev; Pick model with lowest error on S_dev;
- Optional: Evaluate algorithm on test set(S_test) and report the error.

"Dev set" is sometimes called "cross validation set".（simple cross validation）

It is highly recommended to report the error on a very large test set when writing an academic paper.

##### How to split?

Historical: train:test=7:3, or train:dev:test=6:2:2 

Modern: It is recommended to choose the dev and test sets to be big enough when there are enough data to make meaningful comparisons between different algorithms.

##### k-fold Cross Validation (for small datasets)

For $i=1,\cdots,k$, (k is usually $10$)

​	Train on pieces other than $k$-th piece;

​	Test on the $k$-th piece;

Average

(for each degree)

Optional: Refit model on 100% of data.

##### Leave-one-out Cross Validation (for extremely small datasets)

When $m\le50$, we can choose $m$-fold cross validation.

## Feature selection

Start with $\mathcal{F}=\phi$;

Repeat:

​	Try adding each feature $i$ to $\mathcal{F}$, and see which single-feature addition most improve dev set performance;

​	Add that feature to $\mathcal{F}$;

