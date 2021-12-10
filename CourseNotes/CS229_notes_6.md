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

e.g. logistic gegression = $wx+b$ + $\sigma$

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

##### Goal 3.0: Add constriant: unique animal on an image

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

with $L^{(i)}=-\left[y^{(i)}\log \hat{y}^{(i)}+(1-y^{(i)})\log(1-\hat{y}^{(i)}) \right]$

##### Backward propagation

What we want to do ultimately is 

$\forall l=1,\cdots,3: w^{[l]}:=w^{[l]}-\alpha\dfrac{\partial{J}}{\partial{w^{[l]}}}$

​                            $b^{[l]}:=b^{[l]}-\alpha\dfrac{\partial{J}}{\partial{b^{[l]}}}$

$\dfrac{\partial{J}}{\partial{w^{[3]}}}=\dfrac{\partial{J}}{\partial z^{[3]}}\dfrac{\partial{z^{[3]}}}{\partial w^{[3]}}=\dfrac{\partial{J}}{\partial a^{[3]}}\dfrac{\partial{a^{[3]}}}{\partial z^{[3]}}\dfrac{\partial{z^{[3]}}}{\partial w^{[3]}}$;

$\dfrac{\partial J}{\partial w^{[2]}}=\dfrac{\partial{J}}{\partial z^{[2]}}\dfrac{\partial{z^{[2]}}}{\partial w^{[2]}}=\dfrac{\partial{J}}{\partial z^{[3]}}\dfrac{\partial{z^{[3]}}}{\partial a^{[2]}}\dfrac{\partial{a^{[2]}}}{\partial z^{[2]}}\dfrac{\partial{z^{[2]}}}{\partial w^{[2]}}$;

$\dfrac{\partial J}{\partial w^{[1]}}=\dfrac{\partial{J}}{\partial z^{[1]}}\dfrac{\partial{z^{[1]}}}{\partial w^{[1]}}=\dfrac{\partial{J}}{\partial z^{[2]}}\dfrac{\partial{z^{[2]}}}{\partial a^{[1]}}\dfrac{\partial{a^{[1]}}}{\partial z^{[1]}}\dfrac{\partial{z^{[1]}}}{\partial w^{[1]}}$;

We won't need to redo the work we did. We just store the right values while back-propagating and continue to derivate.

