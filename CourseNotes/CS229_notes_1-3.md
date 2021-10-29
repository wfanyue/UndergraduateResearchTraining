# Part I Linear Regression

##### The process of Supervised learning

Training set -> Learing algorithm -> hypothesis

Input -> hypothesis ->Output

## Linear Regression

##### How to represent hypothesis $h$?

$h(x)=\theta_0+\theta_1 x$

More Inputs $x_i$?

$h(x)=\theta_0+\theta_1 x_1+\theta_2 x_2$

In order to make that notation a little bit more compact, we write

$h(x)=\sum\limits_{j=0}^{2}{\theta_j x_j}$, where $x_0=1$, which can be regraded as a dummy(虚拟的) feature that always takes on the value of $1$.

We can write as three-dimensional parameters
$$
\theta=\left[ \begin{align}

 & \theta_0 \\ 

 & \theta_1 \\ 

 & \theta_2 \\ 

\end{align} \right]
$$

$$
x=\left[ \begin{align}

 & x_0 \\ 

 & x_1 \\ 

 & x_2 \\ 

\end{align} \right]
$$

##### Some terminology:

$\theta$ is called the parameters of the learning algorithm, and the job of the learning algorithm is to choose parameters $\theta$; Choose parameters $\theta$ s.t. $h(x)\approx y$ for training examples.

$m$ is # training examples;

$x$ is "inputs" / features;

$y$ is "output" / target variable;

$(x,y)$ is one training example;

$(x^{(i)},y^{(i)})$ is $i$-th training example;

$n$ is # features (it is 2 above), so the vector is $n+1$ dimensions.

Sometimes we emphasize that $h$ depends both on parameteres $\theta$ and on the input features $x$, we're going to use $h_{\theta}(x)$.

##### In linear regression:

$J(\theta)=\dfrac{1}{2}\sum\limits_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$.

We want to minimize $J(\theta)$, which is called cost function.

## Gradient Descent

##### Steps:

Start with some $\theta$ (randomly or all zeros);

Keep changing $\theta$ to reduce $J(\theta)$;

##### Formally:

$\theta_j:=\theta_j-\alpha\dfrac{\partial}{\partial\theta_j}J(\theta)$, where $\alpha$ is called the learning rate.

$\dfrac{\partial}{\partial\theta_j}J(\theta)=\dfrac{1}{2}(h_{\theta}(x)-y)^2=(h_{\theta}(x)-y)\cdot\dfrac{\partial}{\partial\theta_j}(h_{\theta}(x)-y)$

$=(h_{\theta}(x)-y)\cdot\dfrac{\partial}{\partial\theta_j}(\theta_0x_0+\theta_1x_1+\cdots+\theta_nx_n-y)$, where we suppose $m=1$.

$=(h_{\theta}(x)-y)\cdot x_j$.

So, $\theta_j:=\theta_j-\alpha(h_{\theta}(x)-y)x_j$, which is called the LMS (least mean squares) update rule.

or, $\theta_j:=\theta_j-\alpha\sum\limits_{i=1}^{m}{(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}}$, for $j=1,2,\cdots ,n$;

Repeat until convergence.

##### Local minimun?

One property of gradient descent is that depend on where you initialize parameters, you can get to a local different points.

Actually, when you plot the cost function $J(\theta)$ for a linear regression model, it turns out that unlike the random diagram which has local optima, $J(\theta)$ turns out to be a quadratic function(二次函数), which will always look like a big bowl, where the only local optimal is also the global optimum.

##### How to determine $\alpha$?

Usually try on an exponential scale.

##### Batch gradient descent

We calculate this derivative by summing over the entire training set $m$. Sometimes this version of gradient descent has another name, which is batch gradient descent.

Main Disadvantages of batch gradient descent: If $m$ is very large, you need to scan through your entire database and sum them up.

##### Stochastic gradient descent

For $i=1$ to $m$, $\theta_j:=\theta_j-\alpha(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}$, for $j=1,2,\cdots ,n$.

Each time you improve parameters a little, but not going in the most direct direction downhill. But on average, it is headed toward the global minimum. Besides, stochastic gradient descent will never converge. Stop when looking like not descenting. But it allows your algorithm to make much faster progress, and it is used much more in practice. 

##### Can we combine the advantages of both? 

Another better algorithm is Mini-batch gradient descent, where you use a hundred examples at a time rather than one example at a time. 

## Normal Equation

This is not an iteration algorithm. But this algorithm is just fit for Linear Regression problems.

Suppose $A\in \mathbb{R}^{2\times 2}$, $f:\mathbb{R}^{2\times 2}\mapsto\mathbb{R}$
$$
\nabla_Af(A)=\left[ \begin{matrix}
   \dfrac{\partial f}{\partial {{A}_{11}}} & \dfrac{\partial f}{\partial {{A}_{12}}}  \\
   \dfrac{\partial f}{\partial {{A}_{21}}} & \dfrac{\partial f}{\partial {{A}_{22}}}  \\
\end{matrix} \right]
$$

##### Introduction of Trace

If $A$ is square, we define $\bold{tr}A=\sum_\limits{i}{A_{ii}}$.

Some properties of trace:

$\bold{tr}A=\bold{tr}A^T$;

$\bold{tr}AB=\bold{tr}BA$;

$\bold{tr}ABC=\bold{tr}CAB$;

If $f(A)=\bold{tr}AB$, $\nabla_Af(A)=B^T$;

$\nabla_A\bold{tr}AA^TC=CA+C^TA$;

##### Derivation of Normal Equation:

Let
$$
X=\left[ \begin{matrix}
    & — & {{({{x}^{(1)}})}^{T}} & — &    \\
    & — & {{({{x}^{(2)}})}^{T}} & — &    \\
   {} && \vdots  & {}  \\
    & — & {{({{x}^{(m)}})}^{T}} & — &    \\
\end{matrix} \right]
$$
$J(\theta)=\dfrac{1}{2}\sum\limits_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$.

Let $\nabla_{\theta}J(\theta)=\bold{0}$, we can have

Normal equation: $X^TX\theta=X^Ty$, from where $\theta=(X^TX)^{-1}X^Ty$.

$X$ is supposed to be invertible, or that usually means that you have redundant features, and that your features are linearly dependent.

## Locally Weighted Linear Regression

Actually, sometimes it is actually quite difficult to find features. Is it square root, log, ...?

##### "Non-parametric" learning algorithm:

parametric learning algorithm: fit fixed set of parameters($\theta_i$) to data;

Non-parametric leanign algorithm: amount of data/parameters you need to keep grows (linearly) with size of data;

Fit $\theta$ to minimize $\sum_\limits{i=1}^{m}{w^{(i)}(y^{(i)}-\theta^Tx^{(i)})^2}$, 

where $w^{(i)}$ is a weighting function, $w^{(i)}=\exp\left(-\dfrac{(x^{(i)}-x)^2}{2}\right)$,

which means if $x^{(i)}$ is close to $x$, $w^{(i)}$ is close to $1$. Conversely, $w^{(i)}\approx 0$. We just pay more attention to the points around $x$.

##### About Width?

$\tau$ = "bandwidth"

The formula is modified to $w^{(i)}=\exp\left(-\dfrac{(x^{(i)}-x)^2}{2\tau^2}\right)$

The choosing of $\tau$ is a bit complicated.

## Probabilitistic intepretation

##### Why least squares?

Assume $y^{(i)}=\theta^Tx^{(i)}+\epsilon^{(i)}$, where $\epsilon$ is "error" (e.g. modelled effects, random noise),

and $\epsilon^{(i)}\sim N(0,\sigma^2)$.

$P(\epsilon^{(i)})=\dfrac{1}{\sqrt{2\pi}\sigma}\exp{\left(-\dfrac{(\epsilon^{(i)})^2}{2\sigma^2}\right)}$, which does integrate to $1$.

IID: Independently and Identically Distributed. These $\epsilon$ here are IID.

This implies $P(y^{(i)}\big|x^{(i)};\theta)=\dfrac{1}{\sqrt{2\pi}\sigma}\exp{\left(-\dfrac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}\right)}$,

i.e. $y^{(i)}\big|x^{(i)};\theta\sim N(\theta^Tx^{(i)},\sigma^2)$.

Note: $\theta$ is not a random variable, just parametrized by $\theta$.

"likelihood of $\theta$": $\mathcal{L}(\theta)=p(\bold{y}\big|x;\theta)=\prod\limits_{i=1}^{m}{p(y^{(i)}\big|x;\theta)}=\prod\limits_{i=1}^{m}{\dfrac{1}{\sqrt{2\pi}\sigma}\exp{\left(-\dfrac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}\right)}}$.

The likelihood of parameters is exactly the same thing as the probability of data.

"log likelihood" $l(\theta)=\log{\mathcal{L}(\theta)}=m\log{\dfrac{1}{\sqrt{2\pi}\sigma}}+\sum\limits_{i=1}^{m}{-\dfrac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}}$.

MLE: Maximum Likelihood Estimation

Choose $\theta$ to maximize $\mathcal{L}(\theta)$, i.e. minimize $\dfrac{1}{2}\sum\limits_{i=1}^{m}{(y^{(i)}-\theta^Tx^{(i)})^2}$, which is $J(\theta)$, the cost function for linear regression. So we use least squares. 

# Part II Classification and Logistic Regression

$y\in \{0,1\}$, which is binary classification. 

Want $h_{\theta}(x)\in [0,1]$.

## Logistic regression

$h_{\theta}(x)=g(\theta^Tx)=\dfrac{1}{1+e^{-\theta^Tx}}$, where $g(z)=\dfrac{1}{1+e^{-z}}$, which is called "sigmoid" or "logistic" function.

<img src="E:\罗之尧\070000 电子资料\470000 成电\476000 计算机科学与技术\476800 AI Science\CS229\notes_my\logistic g(z).png" alt="logistic g(z)" style="zoom:40%;" />

$P(y=1\big|x;\theta)=h_{\theta}(x)$,$P(y=0\big|x;\theta)=1-h_{\theta}(x)$.

Or $P(y\big|x;\theta)=(h(x))^y(1-h(x))^{1-y}$.

$\mathcal{L}(\theta)=p(\bold{y}\big|x;\theta)=\prod\limits_{i=1}^{m}{{p(y^{(i)}\big|x;\theta)}=\prod\limits_{i=1}^{m}(h(x))^y(1-h(x))^{1-y}}$.

$l(\theta)=\log{\mathcal{L}(\theta)}=\sum\limits_{i=1}^{m}{y^{(i)}\log{h_{\theta}x^{(i)}}+(1-y^{(i)})\log(1-h_{\theta}(x^{i}))}$.

Choose $\theta$ to maximize $\mathcal{L}(\theta)$.

Batch gradient ascent: $\theta_j:=\theta_j+\alpha\dfrac{\partial}{\partial\theta_j}l(\theta)$.

We can get $\theta_j:=\theta_j+\alpha\sum\limits_{i=1}^{m}{(y^{(i)}-h_{\theta}(x^{(i)}))x_j^{(i)}}$.

## Perception learning algorithm

Let
$$
g(z)=\left\{ \begin{align}

 & 1 & if\ & z\ge 0\\ 

 & 0 & if\ & z<0 \\ 

\end{align} \right.
$$
If we then let $h_{\theta}(x)=g(\theta^Tx)$ as before, and if we use the update rule $\theta_j:=\theta_j+\alpha\sum\limits_{i=1}^{m}{(y^{(i)}-h_{\theta}(x^{(i)}))x_j^{(i)}}$, then we have the perceptron learning algorithm.

## Newton's methods

Want to minimize $l(\theta)$, i.e. want $l'(\theta)=0$.

##### So how to find $\theta$, s.t. $f(\theta)=0$? -- Newton's method

Repeat: $\theta^{(t+1)}:=\theta^{(t)}-\dfrac{f(\theta^{(t)})}{f'(\theta^{(t)})}$.

Let $f(\theta)=l'(\theta)$, so we have $\theta^{(t+1)}:=\theta^{(t)}-\dfrac{l'(\theta^{(t)})}{l''(\theta^{(t)})}$.

Newton's method is fast, which enjoys a property called quadratic convergence.

(0.01->0.0001->0.00000001)

When $\theta$ is a vector:

$\theta^{(t+1)}:=\theta^{(t)}-H^{-1}\nabla_{\theta}l$, where $H$ is the Hessian matrix, and $H_{ij}\equiv\dfrac{\partial^2l}{\partial\theta_i\partial\theta_j}$.

##### Advantages and Disadvantages?

The advantages of Newton's method is that it requires less iterations,

but the disadvantage is that in high-dimensional(>1000) problems, each step is much more expensive.

# Part III Generalized Linear Models

## Exponential Family

PDF: Probability density function;

$P(y;\eta)=b(y)\exp[\eta^TT(y)-a(\eta)]$,

$y$=data

$\eta$=natural parameter

$T(y)=y$=sufficient statistic

$b(y)$=Base measure

$a(\eta)$=log-partition

##### Examples:

##### Bernoulli distributions

$\phi$=probability

$P(y;\phi)=\phi^y(1-\phi)^{(1-y)}=\exp[\log(\dfrac{\phi}{1-\phi})y+\log(1-\phi)]$

##### Gaussian distributions (with fixed variance)

Assume $\sigma^2=1$

$P(y;\eta)=\dfrac{1}{\sqrt{2\pi}\exp\left(-\dfrac{(y-\mu)^2}{2}\right)}=\dfrac{1}{\sqrt{2\pi}}e^{-\frac{y^2}{2}}\exp(\mu y-\dfrac{1}{2}\mu^2)$

##### Others

Real--Gaussian;

Binary--Bernouli;

Count--Poisson;

$\mathbb{R}^+$--Gamma,Exponential;

Distributions--Beta,Dilichlet;(Bayesian)

##### Properties:

MLE(Maximum Likelihood Estimate) with respect to $\eta$ is concave;

NLL(Negative Log-Likelihood) is convex;

$E[y;\eta]=\dfrac{\partial}{\partial\eta}{a(\eta)}$

$Var[y;\eta]=\dfrac{\partial^2}{\partial\eta^2}a(\eta)$

## Generalized Linear Models

Assumptions/ Design choices

(i)$y\big|x;\theta\sim$Exponential Family($\eta$)

(ii)$\eta=\theta^Tx$,where $\theta,x\in\mathbb{R}^n$

(iii)Test time: output $E[y\big|x;\theta]\Rightarrow h_{\theta}(x)=E[y\big|x;\theta]$

x->Linear Model($\theta^Tx$)->Exponential Family($\eta$)->$E[y;\eta]=E[y;\theta^Tx]=h_{\theta}(x)$

Training time: $\max\limits_{\theta}\log P(y^{(i)};\theta^Tx^{(i)})$. We are doing gradient ascent by taking gradients on $\theta$.

##### GLM training:

The learning update Rule: $\theta_j:=\theta_j+\alpha(y^{(i)}-h_{\theta}(x^{(i)}))x_j^{(i)}$.

$\eta$=natural parameter

$E[y;\eta]=g(\eta)$=canonical response function

$\eta=g^{-1}(\mu)$=canonical link function

$g(\eta)=\dfrac{\partial}{\partial\eta}a(\eta)$.

##### Parametizations:

Model param $\theta$: the only thing we are going to learn;

Natural param $\eta$: $\theta$-Design choice($\theta^Tx$)->$\eta$;

Canonical Param $\phi$--Bor, $\mu\sigma^2$--Gamma, $\lambda$--Poisson: $\eta$-g->canonical param;

##### For logistic regression:

$h_{\theta}(x)=E[y\big|x;\theta]=\phi=\dfrac{1}{1+e^{-\eta}}=\dfrac{1}{1+e^{-\theta^Tx}}$

##### For softmax regression:

For Multiple classification problem, that is, $y\in\{1,\cdots,k\}$

$k$=#classes

$x^{(i)}\in\mathbb{R}^n$

Label $y=[\{0,1\}^k]$

$\theta_{class}\in\mathbb{R}^n$

In this case, it's outputting a probability distribution over all the classes.

Our hypothesis will output

$h_{\theta}(x)=E[T(y)\big|x;\theta]$
$$
=\left[ \begin{matrix}
\phi_1 \\ 
\phi_2 \\ 
\vdots \\ 
\phi_{k-1} \\ 
\end{matrix} \right]
$$

$$
=\left[ \begin{matrix}
\frac{\exp(\theta_1^Tx)}{\sum\limits_{j=1}^{k}\exp(\theta_j^Tx)} \\ 
\frac{\exp(\theta_2^Tx)}{\sum\limits_{j=1}^{k}\exp(\theta_j^Tx)} \\ 
\vdots \\ 
\frac{\exp(\theta_{k-1}^Tx)}{\sum\limits_{j=1}^{k}\exp(\theta_j^Tx)} \\ 
\end{matrix} \right]
$$

# Part IV Generative Learning Algorithms

##### Discriminative learning algorithm(判别式学习算法):

Learn $P(y\big|x)$

(or learn $h_{\theta}(x)=0\ or\ 1$ directly)

## Generative learning algorithm:

Learn $P(x\big|y)$

That means given the classes, learn what will the features be like.

$P(y)$: class prior;

##### Using Bayes rule:

$P(y=1|x)=\dfrac{P(x|y=1)P(y=1)}{P(x)}$

$P(x)=P(x|y=1)P(y=1)+P(x|y=0)P(y=0)$

## Gaussian Discriminant Analysis (GDA)

##### Multivariate Gaussian distribution:

Suppose $x\in\mathbb{R}^n$ (drop $x_0=1$ convention)

Assume $P(x|y)$ is Gaussian.

$z\sim N(\mu,\Sigma)$, $z\in\mathbb{R}^n$, where $\mu\in\mathbb{R}^n$, $\Sigma\in\mathbb{R}^{n\times n}$.

$E[z]=\mu$

$Cov(z)=E[(z-\mu)(z-\mu)^T]=Ezz^T-(Ez)(Ez)^T$.

$p(z)=\dfrac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp(-\dfrac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$.

$\mu$ is Mean Vector; $\Sigma$ is Co-variance Matrix (Symmetric).

When $\mu=\bold{0}$, and $\Sigma=I$, it is the standard form.

##### GDA Model:

$P(x|y=0)=\dfrac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp(-\dfrac{1}{2}(x-\mu_0)^T\Sigma^{-1}(x-\mu_0))$

$P(x|y=1)=\dfrac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp(-\dfrac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1))$

$P(y)=\phi^y(1-\phi)^{(1-y)}$, where $P(y=1)=\phi$

Parameters: $\mu_0$,$\mu_1$,$\Sigma$,$\phi$.

Training set: $\{x^{(i)},y^{(i)}\}_{i=1}^m$

log-likelihood:

$l(\phi,\mu_0,\mu_1,\Sigma)=\log\prod\limits_{i=1}^{m}p(x^{(i)},y^{(i)};\phi,\mu_0,\mu_1,\Sigma)$

$=\log\prod\limits_{i=1}^{m}p(x^{(i)}|y^{(i)};\mu_0,\mu_1,\Sigma)p(y^{(i)};\phi)$.

By maximizing $l$ with respect to the parameters, we find the maximum likelihood estimate of the parameters to be:

$\phi=\dfrac{1}{m}\sum\limits_{i=1}^{m}1\{y^{(i)}=1\}$

$\mu_0=\dfrac{\sum\limits_{i=1}^{m}1\{y^{(i)}=0\}x^{(i)}}{\sum\limits_{i=1}^{m}1\{y^{(i)}=0\}}$(which means $\dfrac{\text{sum of feature vectors for examples with }y=0}{\text{#examples with }y=0}$)

$\mu_0=\dfrac{\sum\limits_{i=1}^{m}1\{y^{(i)}=1\}x^{(i)}}{\sum\limits_{i=1}^{m}1\{y^{(i)}=1\}}$

$\Sigma=\dfrac{1}{m}\sum\limits_{i=1}^m(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T$

where $1\{true\}=1$, and $1\{false\}=0$, which is an indicator factor.

Prediction:

$\arg\max\limits_yP(y|x)=\arg\max\limits_y\dfrac{P(x|y)P(y)}{P(x)}=\arg\max\limits_yP(x|y)P(y)$.

## GDA and Logistic regression

GDA makes stronger modeling assumptions, and is more data efficient (i.e., requires less training data to learn well) when the modeling assumptions are correct or at least approximately correct. Logistic regression makes weaker assumptions, and is signicantly more robust to deviations from modeling assumptions. Specically, when the data is indeed non-Gaussian, then in the limit of large datasets, logistic regression will almost always do better than GDA. For this reason, in practice logistic regression is used more often than GDA. [lecture notes]

## Naive Bayes

##### In text classification, how to map the text to a  feature vector $X$?

Create a dictionary with $n$ words.

Vector $X$: a $n$-dimensional binary feature vector,

i.e., $x\in\{0,1\}^n$, $x_i=1\{\text{word }i\text{ appear in email}\}$

##### Assumption:

Assume $x_i$'s are conditionally independent given $y$.

$P(x_1,\cdots,x_n|y)=P(x_1|y)P(x_2|x_1,y)\cdots P(x_n|x_1,\cdots,x_{n-1},y)$

(Assume)$=P(x_1|y)P(x_2|y)\cdots P(x_n|y)=\prod\limits_{i=1}^{n}{P(x_i|y)}$.

##### Parameters:

$\phi_{j|y=1}=P(x_j=1|y=1)$

$\phi_{j|y=0}=P(x_j=1|y=0)$

$\phi_y=P(y=1)$

##### Joint likelihood:

$L(\phi_y,\phi_{j|y}=\prod\limits_{i=1}^{m}{P(x^{(i)},y^{(i)};\phi_y,\phi_{j|y}})$

##### MLE:

$\phi_y=\dfrac{\sum\limits_{i=1}^{m}1\{y^{(i)}=1\}}{m}$



$\phi_{j|y=1}=\dfrac{\sum\limits_{i=1}^{m}1\{x^{(i)}_j=1\and y^{(i)}=1\}}{\sum\limits_{i=1}^{m}1\{y^{(i)}=1\}}$

$\phi_{j|y=0}=\dfrac{\sum\limits_{i=1}^{m}1\{x^{(i)}_j=1\and y^{(i)}=0\}}{\sum\limits_{i=1}^{m}1\{y^{(i)}=0\}}$

## Laplace Smoothing

##### What if it hasn't seem a word before?

We may obtain $P(y=1|x)=\dfrac{0}{0}$.

We can replace the above estimate with $\phi_j=\dfrac{\sum\limits_{i=1}^{m}1\{z^{(i)}=j\}+1}{m+k}$.

$\phi_{j|y=1}=\dfrac{\sum\limits_{i=1}^{m}1\{x^{(i)}_j=1\and y^{(i)}=1\}+1}{\sum\limits_{i=1}^{m}1\{y^{(i)}=1\}+2}$

$\phi_{j|y=0}=\dfrac{\sum\limits_{i=1}^{m}1\{x^{(i)}_j=1\and y^{(i)}=0\}+1}{\sum\limits_{i=1}^{m}1\{y^{(i)}=0\}+2}$

## Multinomial Event Model

What we talked above is called Multi-variate Bernulli Event Model.

##### Multinomial Event Model:

$x_j$: the prior of the $j$-th word in the dictionary.

$x_j\in\{1,\cdots,|V|\}$, where $|V|$ is the size of dictionary.

$x\in\mathbb{R}^n$, where $n$ is the length of email.

$P(x_1,\cdots,x_n,y)=P(x_1,\cdots,x_n|y)P(y)=P(x_1|y)P(x_2|x_1,y)\cdots P(x_n|x_1,\cdots,x_{n-1},y)P(y)$

(Assume)$=P(x_1|y)P(x_2|y)\cdots P(x_n|y)P(y)=\prod\limits_{i=1}^{n}{P(x_i|y)}P(y)$.

##### Parameters:

$\phi_y=P(y=1)$

$\phi_{k|y=0}=P(x_j=k|y=0)$, which means chance of word $j$ being $k$ if $y=0$.

$\phi_{k|y=1}=P(x_j=k|y=1)$

# Part V Support Vector Machine

## Functional Margin

$h_{\theta}(x)=g(\theta^Tx)$

Predict "1" if $\theta^T\ge0$($h_{\theta}(x)=g(\theta^Tx)\ge0.5$;

​             "0" otherwise.

If $y{(i)}=1$, hope that $\theta^Tx^{(i)}\gg0$;

If $y{(i)}=0$, hope that $\theta^Tx^{(i)}\ll0$;

## Geometric Margin

<img src="E:\罗之尧\070000 电子资料\470000 成电\476000 计算机科学与技术\476800 AI Science\CS229\notes_my\geometric margin.png" alt="geometric margin" style="zoom:40%;" />

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

##### Optimal margin classfier (separable case)

Choose $w,b$ to maximize $\gamma$.

$\max\limits_{\gamma,w,b}\gamma$, s.t. $\dfrac{y^{(i)}(w^Tx^{i}+b)}{||w||}\ge\gamma$, $i=1,2,\cdots,m$.

This is equivlant to 

$\min\limits_{w,b}||w||^2$, s.t. $y^{(i)}(w^Tx^{(i)}+b)\ge1$.



