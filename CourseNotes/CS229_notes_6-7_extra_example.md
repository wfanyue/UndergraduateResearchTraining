## e.g. A Simple Neural Network

For the dataset in Figure 1, we want to classify them into 2 categories.

<img src="note7_plot of dataset.png" alt="note7_plot of dataset" style="zoom:55%;" />

The examples in class 1 are marked as "$\times$" and examples in class 0 are marked as "$\circ$". We want
to perform binary classication using a simple neural network with the architecture shown in
Figure 2.

<img src="note7_architecture.png" alt="note7_architecture" style="zoom:55%;" />

##### Notation:

Input: $x$ (2\*1)

Hidden layer: $h$ (3\*1)

Output: $o$ (1\*1); $y$: given output of dataset

Weights: from $x$ to $h$: $w^{[1]}$ (3\*2)

​                from $h$ to $o$: $w^{[2]}$ (1\*3)

Bias: $w_0$

Loss function: $L^{(i)}=(o^{(i)}-y^{(i)})^2$

$J=\dfrac{1}{m}\sum\limits_{i=1}^mL^{(i)}$

##### Q1: Suppose we use the sigmoid function as the activation function. Write the gradient descent update to $w^{[1]}$ in terms of $x$, $o$, $y$, $w$, $\alpha$.

Recall the following equations:

$$
\begin{align*}
z^{[1]} & = w^{[1]} x + w_0^{[1]} \\
h & = \sigma (z^{[1]}) \\
z^{[2]} & = w^{[2]} h + w_0^{[2]} \\
o & = \sigma (z^{[2]}) \\
J & = \frac{1}{m} \sum_{i = 1}^{m} (o^{(i)} - y^{(i)})^2 = \frac{1}{m} \sum_{i = 1}^{m} L^{(i)}
\end{align*}
$$
For a single training example,

$$
\begin{align*}
\frac{\partial L}{\partial w^{[1]}} & = \frac{\partial L}{\partial z^{[2]}} \frac{\partial z^{[2]}}{\partial h} \frac{\partial h}{\partial z^{[1]}} \frac{\partial z^{[1]}}{\partial w^{[1]}} \\
                                          & = \text{(roughly) }2 (o - y) \cdot o (1 - o) \cdot w^{[2]} \cdot h (1 - h) \cdot x\\
                                          & =w^{[2]T}*h*(1-h)2(o-y)o(1-o)x^T
\end{align*}
$$
where $h = w^{[1]} x_1 + w^{[1]} x_2 + w^{[1]}$.

Therefore, the gradient descent update rule for $w^{[1]}$ is

$w^{[1]} := w^{[1]} - \alpha \dfrac{2}{m} \sum\limits_{i = 1}^{m} w^{[2]T}*h*(1-h)2(o-y)o(1-o)x^T$

where $h = w^{[1]} x_1 + w^{[1]} x_2 + w^{[1]}$.

##### Q2: Now, suppose we instead use the step function $f(x)$ as the activation function, defined as

$$
f(x)=\left\{ \begin{align}
  & 1,x\ge 0 \\ 
 & 0,x<0 \\ 
\end{align} \right.
$$

**Is it possible to have a set of weights that allow the neural network to classify this dataset with 100% accuracy? If possible, provide $w$.**

It is possible. The three neurons can be treated as three independent linear classifiers. The three decision boundaries form a triangle that classifies the outside data into class 1, and the inside ones into class 0.

Decision boundaries: Line $x_1=0.5$, $x_2=0.5$, $x_1+x_2=4$.

$w_{0,1}^{[1]}=0.5$, $w_{1,1}^{[1]}=-1$, $w_{2,1}^{[1]}=0$;

$w_{0,2}^{[1]}=0.5$, $w_{1,2}^{[1]}=0$, $w_{2,2}^{[1]}=-1$;

$w_{0,3}^{[1]}=-4$, $w_{1,3}^{[1]}=1$, $w_{2,3}^{[1]}=1$;

$w_0^{[2]}=-0.5$, $w_1^{[2]}=1$, $w_2^{[2]}=1$, $w_3^{[2]}=1$.

It can be easily shown that only if the point is in the triangle surrounded by three decision boundaries, $h_1$, $h_2$, $h_3$ will be all $0$, when $o$ will be $0$. Otherwise, $o$ will be $1$. So this neural network is able to classify this dataset with 100% accuracy.

##### Q3: Now, suppose we instead use the function $f(x)=x$ as the activation function. Is it possible to have a set of weights that allow the neural network to classify this dataset with 100% accuracy? If possible, provide $w$.

No, it is not possible to achieve 100% accuracy using identity function as the activation functions for $h_1$, $h_2$ and $h_3$. Because

$$
\begin{align*}
o & = \sigma (z^{[2]}) \\
  & = \sigma (w^{[2]} h + w_0^{[2]}) \\
  & = \sigma (w^{[2]} (w^{[1]} x + w_0^{[1]}) + w_0^{[2]}) \\
  & = \sigma (w^{[2]} w^{[1]} x + w^{[2]} w_0^{[1]} + w_0^{[2]}) \\
  & = \sigma (\tilde{W} x + \tilde{W_0})
\end{align*}
$$
where $\tilde{W} = w^{[2]} w^{[1]}$ and $\tilde{W_0} = w^{[2]} w_0^{[1]} + w_0^{[2]}$.

We can see that the resulting classifier is still linear, and it is not able to classify datasets that are not linearly separable with 100% accuracy.