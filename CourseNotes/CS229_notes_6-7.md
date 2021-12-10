# Deep learning

## Logistic Regression

e.g. A 64*64-pixel RGB picture of Cats

##### Goal: Find cats in images: 1/0 (presence or not)

$\hat{y}=\sigma(\theta^Tx)=\sigma(wx+b)$, where $x$ is 12288\*1, $w$ is 1\*12288, b is 1\*1, y is 1\*1, $\sigma$ is sigmoid function here.

Steps:

i) Initiate $w$, $b$, where $w$ is weights, $b$ is bias;

ii) Find the optimal $w$, $b$;

iii) Use $\hat{y}=\sigma(wx+b)$ to predict;

(Eq1) neuron = linear + activation

e.g. logistic regression = $wx+b$ + $\sigma$

(Eq2) model = architecture + parameters

##### Goal 2.0: Find cat/ lion/ iguana in images

<img src="neural network lead-in.png" alt="neural network lead-in" style="zoom:50%;" />

Square bracket "[1]" means the layer.

12288\*3 parameters in total.

The label of picture is a three-dimensional binary vector.

**Loss three neurons:** 

$L_{3N}=-\sum\limits_{k=1}^3\left[y_k\log\hat{y}_k+(1-y_k)\log(1-\hat{y}_k)\right]$;

(Just doing exactly three times the loss for each of the neurons, and summing them up)

$w:=w-\alpha\dfrac{\partial L_{3N}}{\partial w_j^{[i]}}$;

$b:=b-\alpha\dfrac{\partial L_{3N}}{\partial b_j^{[i]}}$;

Exactly the same complexity during the derivation.

##### Goal 3.0: Add constraint: unique animal on an image

<img src="neural network lead-in2.png" alt="neural network lead-in2" style="zoom:40%;" />

$z_j^{[i]}$ stands for $w_j^{[i]}x+b_j^{[i]}$.

Softmax multi-class regression. The label of picture is vector which has only one "1".

**Cross-entropy loss:**

$L_{CE}=-\sum\limits_{k=1}^3y_k\log\hat{y}_k$.

The derivations with respect to $w$ and $b$ are a bit more complicated.

## Neural Networks

##### Goal: Find cats in images: 1/0 (presence or not)

<img src="neural network layers.png" alt="neural network layers" style="zoom:40%;" />

**Layer:** neurons that are not connected to each other. We call this cluster of neurons a layer.

This has 3 layers. Input layer - Hidden layers - Output layer.

**Fully connected layers:** all the neurons from one layer to another are connected with each other.

**End-to-end learning (blackbox models):** just training based on the input and output. 

House price network example:

<img src="neural network full layers.png" alt="neural network full layers" style="zoom:40%;" />

## Propagation equation

$z^{[1]}=w^{[1]}x+b^{[1]}$, where $z^{[1]}$ is 3\*1, $x$ is $n$\*1, $b^{[1]}$ is 3\*1, so $w^{[1]}$ is 3\*$n$;

$a^{[1]}=\sigma(z^{[1]})$, where $a^{[1]}$ is 3\*1;

$z^{[2]}=w^{[2]}a^{[1]}+b^{[2]}$, where $z^{[2]}$ is 2\*1, $b^{[2]}$ is 2\*1, so $w^{[2]}$ is 2\*3;

$a^{[2]}=\sigma(z^{[2]})$, where $a^{[2]}$ is 2\*1;​

$z^{[3]}=w^{[3]}a^{[2]}+b^{[3]}$, where $z^{[3]}$ is 1\*1, $b^{[3]}$ is 1\*1, so $w^{[3]}$ is 1\*2;

$\hat{y}=a^{[3]}=\sigma(z^{[3]})$, where $a^{[3]}$ is 1\*1;

##### What happens for an input batch of $m$ example?

$$
X=\left( \begin{matrix}
   \text{ }\!\!|\!\!\text{ } & \text{ }\!\!|\!\!\text{ } & {} & \text{ }\!\!|\!\!\text{ }  \\
   {{x}^{(1)}} & {{x}^{(2)}} & \cdots  & {{x}^{(m)}}  \\
   \text{ }\!\!|\!\!\text{ } & \text{ }\!\!|\!\!\text{ } & {} & \text{ }\!\!|\!\!\text{ }  \\
\end{matrix} \right)
$$

Round bracket "(1)" refers to the id of the example.

$Z^{[i]}=w^{[i]}X+b^{[i]}$, where $X$ is $n$\*$m$, $Z^{[i]}$ is 3\*$m$, $\tilde{b}^{[i]}$ is 3\*$m$;
$$
\tilde{b}^{[i]}=\left( \begin{matrix}
   \text{ }\!\!|\!\!\text{ } & \text{ }\!\!|\!\!\text{ } & {} & \text{ }\!\!|\!\!\text{ }  \\
   {{b}^{[i]}} & {{b}^{[i]}} & \cdots  & {{b}^{[i]}}  \\
   \text{ }\!\!|\!\!\text{ } & \text{ }\!\!|\!\!\text{ } & {} & \text{ }\!\!|\!\!\text{ }  \\
\end{matrix} \right)(m\text{ columns})
$$
This technique is called broadcasting, which can make codes paralleled.

##### Optimizing $w^{[1]},w^{[2]},w^{[3]},b^{[1]},b^{[2]},b^{[3]}$

Define loss/cost function:

$J(\hat{y},y)=\dfrac{1}{m}\sum\limits_{i=1}^{m}L^{(i)}$

with $L^{(i)}=-\left[y^{(i)}\log \hat{y}^{(i)}+(1-y^{(i)})\log(1-\hat{y}^{(i)}) \right]$ (cross entropy)

##### Backward propagation

What we want to do ultimately is 

$\forall l=1,\cdots,3: w^{[l]}:=w^{[l]}-\alpha\dfrac{\partial{J}}{\partial{w^{[l]}}}$

​                            $b^{[l]}:=b^{[l]}-\alpha\dfrac{\partial{J}}{\partial{b^{[l]}}}$

$\dfrac{\partial{J}}{\partial{w^{[3]}}}=\dfrac{\partial{J}}{\partial z^{[3]}}\dfrac{\partial{z^{[3]}}}{\partial w^{[3]}}=\dfrac{\partial{J}}{\partial a^{[3]}}\dfrac{\partial{a^{[3]}}}{\partial z^{[3]}}\dfrac{\partial{z^{[3]}}}{\partial w^{[3]}}$;

$\dfrac{\partial J}{\partial w^{[2]}}=\dfrac{\partial{J}}{\partial z^{[2]}}\dfrac{\partial{z^{[2]}}}{\partial w^{[2]}}=\dfrac{\partial{J}}{\partial z^{[3]}}\dfrac{\partial{z^{[3]}}}{\partial a^{[2]}}\dfrac{\partial{a^{[2]}}}{\partial z^{[2]}}\dfrac{\partial{z^{[2]}}}{\partial w^{[2]}}$;

$\dfrac{\partial J}{\partial w^{[1]}}=\dfrac{\partial{J}}{\partial z^{[1]}}\dfrac{\partial{z^{[1]}}}{\partial w^{[1]}}=\dfrac{\partial{J}}{\partial z^{[2]}}\dfrac{\partial{z^{[2]}}}{\partial a^{[1]}}\dfrac{\partial{a^{[1]}}}{\partial z^{[1]}}\dfrac{\partial{z^{[1]}}}{\partial w^{[1]}}$;

We won't need to redo the work we did. We just store the right values while back-propagating and continue to derivative.

Since derivative is linear, we just calculate $\dfrac{\partial L}{\partial w^{[3]}}$ instead of $\dfrac{\partial J}{\partial w^{[3]}}$.

Suppose $\sigma$ is sigmoid function,

$\dfrac{\partial L}{\partial w^{[3]}}=-\left[y^{(i)}\dfrac{\partial}{\partial w^{[3]}}\log\sigma(w^{[3]}a^{[2]}+b^{[3]})+(1-y^{(i)})\dfrac{\partial}{\partial w^{[3]}}\log(1-\sigma(w^{[3]}a^{[2]}+b^{[3]})) \right]$



$=-\left[y^{(i)}\dfrac{1}{a^{[3]}}a^{[3]}(1-a^{[3]})a^{[2]T}+(1-y^{(i)})\dfrac{1}{(1-a^{[3]})}(-1)a^{[3]}(1-a^{[3]})a^{[2]T} \right]$

$=-\left[y^{(i)}(1-a^{[3]})a^{[2]T}-(1-y^{(i)})a^{[3]}a^{[2]T} \right]$

$=-(y^{(i)}-a^{[3]})a^{[2]T}$.

$\dfrac{\partial L}{\partial w^{[2]}}=\dfrac{\partial{L}}{\partial z^{[3]}}\dfrac{\partial{z^{[3]}}}{\partial a^{[2]}}\dfrac{\partial{a^{[2]}}}{\partial z^{[2]}}\dfrac{\partial{z^{[2]}}}{\partial w^{[2]}}$

$=(a^{[3]}-y)w^{[3]T}a^{[2]}(1-a^{[2]})a^{[1]T}$(WRONG, Matrix Shape Problem!).

$=w^{[3]T}*a^{[2]}*(1-a^{[2]})(a^{[3]}-y)a^{[1]T}$, where $*$ denotes element wise product.

$\dfrac{\partial L}{\partial w^{[1]}}=\dfrac{\partial{L}}{\partial z^{[2]}}\dfrac{\partial{z^{[2]}}}{\partial a^{[1]}}\dfrac{\partial{a^{[1]}}}{\partial z^{[1]}}\dfrac{\partial{z^{[1]}}}{\partial w^{[1]}}$

$=w^{[2]T}*a^{[1]}*(1-a^{[1]})(a^{[2]}-y)x^{T}$.

## Improving Neural Networks

#### A. Activation function

<img src="ReLU.png" alt="ReLU" style="zoom:50%;" />

<img src="Sigmoid.png" alt="Sigmoid" style="zoom:50%;" />

<img src="tanh.png" alt="tanh" style="zoom:50%;" />

**Why do we need activation function?**

Supposed activation is identity function, i.e. $\sigma(z)=z$,

$\hat{y}=(w^{[3]}w^{[2]}w^{[1]})x+(w^{[3]}w^{[2]}b^{[1]}+w^{[3]}b^{[2]}+b^{[3]})$.

So, if we don't choose activation functions, no matter how deep our network is, it's going to be equivalent to linear regression.

#### B. Normalization methods and Initialization methods

**i) Normalization methods**

$\mu=\dfrac{1}{m}\sum\limits_{i=1}^mx^{(i)}$,

Let $\tilde{x}=x-\mu$;

$\sigma^2=\dfrac{1}{m}\sum\limits_{i=1}^m(x^{(i)})^2$,

Let $\tilde{\tilde{x}}=\dfrac{\tilde{x}}{\sigma}$.

**ii) Initialization methods**

If $z=w_1x_1+\cdots+w_nx_n$,

large $n$ $\rightarrow$ small $w_i$, i.e. $w_i\sim\dfrac{1}{n}$.

Otherwise, there will be vanishing/ exploding gradients.

Examples:

for sigmoid: $w^{[l]}=$`np.random.randn(shape)*`$\sqrt{\dfrac{1}{n^{[l-1]}}}$;

for ReLU: $w^{[l]}=$`np.random.randn(shape)*`$\sqrt{\dfrac{2}{n^{[l-1]}}}$;

for tanh: Xavien Initialization: $w^{[l]}\sim\sqrt{\dfrac{1}{n^{[l-1]}}}$; He Initialization: $w^{[l]}\sim\sqrt{\dfrac{2}{n^{[l]}+n^{[l-1]}}}$.

The reason we have a random function is to avoid some problems called the symmetry problem, where every neuron is going to learn kind of the same thing. 

#### C. Optimization

**i) Mini-batch gradient descent**

Have mentioned in Part I (Linear Regression).

**ii) Gradient descent Momentum algorithm**

<img src="momentum algorithm.png" alt="momentum algorithm" style="zoom:40%;" />
$$
\left\{ \begin{align}
&v:=\beta v+(1-\beta)\dfrac{\partial L}{\partial w}\\ 
&w:=w-\alpha v\\ 
\end{align} \right.
$$
