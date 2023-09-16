### 介绍

**动机**. 在本节中，我们将深入理解**反向传播backpropagation**的直观含义，它是通过递归应用**链式法则chain rule**来计算表达式的梯度的方法。理解这一过程及其细微之处对于您理解、有效开发、设计和调试神经网络至关重要。

**问题描述**. 本节研究的核心问题如下：我们给定了某个函数 $f(x)$ ，其中 $x$ 是输入向量，我们想计算 $f$ 在 $x$ 处的梯度（即 $\nabla f(x)$ ）。

**动机**. 请记住，我们对此问题感兴趣的主要原因是，在神经网络的特定情况下，$f$ 将对应于损失函数（ $L$ ），而输入 $x$ 将由训练数据和神经网络权重组成。例如，损失可能是SVM损失函数，输入既包括训练数据 $(x_i,y_i), i=1 \ldots N$ ，也包括权重和偏差 $W,b$。 请注意（如机器学习中通常的情况），我们认为训练数据是给定和固定的，权重是我们可以控制的变量。因此，尽管我们可以很容易地使用反向传播来计算输入示例 $x_i$ 的梯度，但实际上我们通常只计算参数（例如 $W,b$）的梯度，以便我们可以使用它进行参数更新。然而，正如我们稍后在课堂上将看到的，$x_i$ 上的梯度有时仍然很有用，例如用于可视化和解释神经网络可能在做什么。

如果你参加这个课程，并且对使用链式法则推导梯度感到很舒适，我们仍然希望鼓励你至少浏览这一部分，因为它从实值电路的反向流中提供了一个关于反向传播的不常见视角，你从中获得的任何洞察都可能在整个课程中对你有所帮助。

### 梯度的简单表达和解释

让我们从简单的情况开始，以便我们可以建立更复杂表达式的符号和约定。考虑一个简单的两个数字相乘的函数 $f(x,y) = x y$。通过简单的微积分，可以得出对于任一输入的偏导数：



$$
f(x,y) = x y \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = y \hspace{0.5in} \frac{\partial f}{\partial y} = x
$$

**解释**：记住导数告诉你什么：它们指示了一个函数在某一点附近无限小区域内相对于该变量的变化速率：



$$
\frac{df(x)}{dx} = \lim_{h\ \to 0} \frac{f(x + h) - f(x)}{h}
$$

技术上要注意的是，左边的除号与右边的不同，它不表示除法。相反，这个符号表示运算符 $\frac{d}{dx}$ 应用于函数 $f$，并返回一个不同的函数（导数）。一个很好的理解上述表达式的方式是，当 $h$ 非常小的时候，函数可以很好地近似为一条直线，而导数就是它的斜率。换句话说，每个变量的导数告诉你整个表达式对它的值的敏感度。例如，如果 $x = 4, y = -3$，那么 $f(x,y) = -12$，并且对于 $x$ 的导数 $\frac{\partial f}{\partial x} = -3$。这告诉我们，如果我们将这个变量的值增加一个微小的量，由于负号的作用，整个表达式的效果将减小（减小了三倍），这可以通过重新排列上面的方程 ($f(x + h) = f(x) + h \frac{df(x)}{dx}$ ) 来看出。类似地，由于 $\frac{\partial f}{\partial y} = 4$，我们期望通过一些非常小的量 $h$ 增加 $y$ 的值也会增加函数的输出（由于正号），增加的量为 $4h$。

> **每个变量的导数告诉你整个表达式对其值的敏感度。**

正如前面提到的，梯度 $\nabla f$ 是偏导数的向量，因此我们有 $\nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}] = [y, x]$。尽管梯度在技术上是一个向量，但为了简单起见，我们通常会使用诸如 *"x 上的梯度"* 这样的术语，而不是技术上正确的 *"x 上的偏导数"*。

我们还可以推导出加法运算的导数：



$$
f(x,y) = x + y \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = 1 \hspace{0.5in} \frac{\partial f}{\partial y} = 1
$$

也就是说，无论 $x,y$ 的值是什么，$x$ 和 $y$ 的导数都是一。这是有道理的，因为增加 $x$ 或 $y$ 中的任何一个都会增加 $f$ 的输出，并且这种增加的速度不受 $x,y$ 的实际值的影响（与上面的乘法的情况不同）。我们在课程中会经常使用的最后一个函数是 *最大值max* 运算：



$$
f(x,y) = \max(x, y) \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = \mathbb{1}(x >= y) \hspace{0.5in} \frac{\partial f}{\partial y} = \mathbb{1}(y >= x)
$$

也就是说，(子)梯度在较大的输入上为1，在另一个输入上为0。直观地说，如果输入是 $x = 4, y = 2$，那么最大值是4，函数对于 $y$ 的设置不敏感。也就是说，如果我们将其增加一个微小的量 $h$，函数将继续输出4，因此梯度为零：没有影响。当然，如果我们将 $y$ 增加一个大量（例如大于2），那么 $f$ 的值会发生变化，但导数不告诉我们关于这种大幅度变化对函数输入的影响。它们只提供了关于输入的微小、无限小变化的信息，正如在其定义中所示，由 $\lim_{h \rightarrow 0}$ 表示。

### 使用链式法则的复合表达式

现在让我们考虑涉及多个组合函数的更复杂表达式，比如 $f(x, y, z) = (x + y)z$。这个表达式仍然足够简单，可以直接进行求导，但我们将采取一种特定的方法，有助于理解反向传播背后的直觉。特别要注意，这个表达式可以分解为两个表达式：$q = x + y$ 和 $f = qz$。此外，我们知道如何分别计算这两个表达式的导数，就像在前一节中看到的那样。$f$ 只是 $q$ 和 $z$ 的乘法，所以 $\frac{\partial f}{\partial q} = z, \frac{\partial f}{\partial z} = q$，而 $q$ 是 $x$ 和 $y$ 的加法，所以 $\frac{\partial q}{\partial x} = 1, \frac{\partial q}{\partial y} = 1$。然而，我们不一定关心中间值 $q$ 的梯度 —— $\frac{\partial f}{\partial q}$ 的值是无用的。相反，我们最终关心的是 $f$ 相对于其输入 $x, y, z$ 的梯度。**链式法则** 告诉我们正确将这些梯度表达式"链"在一起的方法是通过乘法。例如，$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \frac{\partial q}{\partial x}$。在实践中，这只是两个保存两个梯度的数的乘法。让我们通过一个示例来看看这一点：

```python
# set some inputs
x = -2; y = 5; z = -4

# perform the forward pass
q = x + y # q becomes 3
f = q * z # f becomes -12

# perform the backward pass (backpropagation) in reverse order:
# first backprop through f = q * z
dfdz = q # df/dz = q, so gradient on z becomes 3
dfdq = z # df/dq = z, so gradient on q becomes -4
dqdx = 1.0
dqdy = 1.0
# now backprop through q = x + y
dfdx = dfdq * dqdx  # The multiplication here is the chain rule!
dfdy = dfdq * dqdy  
```

我们最终得到了变量 `[dfdx, dfdy, dfdz]` 中的梯度，这告诉我们变量 `x, y, z` 对 `f` 的敏感度！这是反向传播的最简单示例。在接下来的内容中，我们将使用更简洁的符号，省略 `df` 前缀。例如，我们将简单地写 `dq` 而不是 `dfdq`，并始终假设梯度是针对最终输出进行计算的。

这个计算也可以通过电路图circuit diagram很好地进行可视化：

<div class="fig figleft fighighlight">
<svg style="max-width: 420px" viewbox="0 0 420 220"><defs><marker id="arrowhead" refX="6" refY="2" markerWidth="6" markerHeight="4" orient="auto"><path d="M 0,0 V 4 L6,2 Z"></path></marker></defs><line x1="40" y1="30" x2="110" y2="30" stroke="black" stroke-width="1"></line><text x="45" y="24" font-size="16" fill="green">-2</text><text x="45" y="47" font-size="16" fill="red">-4</text><text x="35" y="24" font-size="16" text-anchor="end" fill="black">x</text><line x1="40" y1="100" x2="110" y2="100" stroke="black" stroke-width="1"></line><text x="45" y="94" font-size="16" fill="green">5</text><text x="45" y="117" font-size="16" fill="red">-4</text><text x="35" y="94" font-size="16" text-anchor="end" fill="black">y</text><line x1="40" y1="170" x2="110" y2="170" stroke="black" stroke-width="1"></line><text x="45" y="164" font-size="16" fill="green">-4</text><text x="45" y="187" font-size="16" fill="red">3</text><text x="35" y="164" font-size="16" text-anchor="end" fill="black">z</text><line x1="210" y1="65" x2="280" y2="65" stroke="black" stroke-width="1"></line><text x="215" y="59" font-size="16" fill="green">3</text><text x="215" y="82" font-size="16" fill="red">-4</text><text x="205" y="59" font-size="16" text-anchor="end" fill="black">q</text><circle cx="170" cy="65" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="170" y="70" font-size="20" fill="black" text-anchor="middle">+</text><line x1="110" y1="30" x2="150" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="110" y1="100" x2="150" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="190" y1="65" x2="210" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="380" y1="117" x2="450" y2="117" stroke="black" stroke-width="1"></line><text x="385" y="111" font-size="16" fill="green">-12</text><text x="385" y="134" font-size="16" fill="red">1</text><text x="375" y="111" font-size="16" text-anchor="end" fill="black">f</text><circle cx="340" cy="117" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="340" y="127" font-size="20" fill="black" text-anchor="middle">*</text><line x1="280" y1="65" x2="320" y2="117" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="110" y1="170" x2="320" y2="117" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="360" y1="117" x2="380" y2="117" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line></svg>

<div style="clear:both;"></div>

</div>

>实值real-valued的<i>"电路"</i>在左侧展示了计算的可视化表示。<b>正向传播forward pass</b>从输入到输出计算值（用绿色显示）。然后<b>反向传播backward pass</b>从末端开始，并递归地应用链式法则来计算梯度（用红色显示），直到电路的输入。可以将梯度看作是通过电路反向流动的。

### 反向传播的直观理解

注意，反向传播是一个美妙的局部过程。电路图中的每个门都接收一些输入，并可以立即计算两件事：1. 它的输出值和 2. 其输出相对于其输入的*局部*梯度。注意，门可以完全独立地做到这一点，而不需要了解它们所嵌入的整个电路的任何细节。然而，一旦正向传播结束，在反向传播过程中，该门最终会了解到其输出值对整个电路最终输出的梯度。链式法则告诉我们，门应该将该梯度乘以它通常为所有输入计算的每个梯度。

> 由于链式法则导致的这种额外乘法（对于每个输入）可以将一个单一且相对无用的门变成一个复杂电路（如整个神经网络）中的一个齿轮cog。

让我们再次参考例子，以直观地了解这是如何工作的。加法门接收输入 [-2, 5] 并计算输出 3。由于门正在进行加法运算，其对其两个输入的局部梯度都是 +1。电路的其余部分计算出最终值，即 -12。在应用链式法则递归反向通过电路的反向传播过程中，加法门（是乘法门的一个输入）了解到其输出的梯度是 -4。如果我们将电路拟人化anthropomorphize为希望输出更高的值（这有助于直观理解），那么我们可以认为电路“希望”加法门的输出更低（由于负号），并且有 *4 的力量*。为了继续递归并链接梯度，加法门取得该梯度并将其乘以其输入的所有局部梯度（使 **x** 和 **y** 上的梯度都变为 1 * -4 = -4）。注意到这有预期的效果：如果 **x, y** 减少（响应于他们的负梯度），那么加法门的输出就会减少，这反过来又使乘法门的输出增加。

因此，反向传播可以被认为是门通过梯度信号彼此通信，它们是否希望其输出增加或减少（以及多强烈），以使最终输出值更高。

### 模块化：Sigmoid示例

我们上面介绍的门是相对随意的。任何可微分的函数都可以作为门，我们可以将多个门组合成一个单一的门，或者在方便的时候将一个函数分解成多个门。让我们看看另一个表达式，以说明这一点：



$$
f(w,x) = \frac{1}{1+e^{-(w_0x_0 + w_1x_1 + w_2)}}
$$

正如我们将在课堂上看到的，这个表达式描述了一个使用 *sigmoid激活* 函数的二维神经元（具有输入 **x** 和权重 **w**）。但现在，让我们简单地将其视为从输入 *w,x* 到一个单一数字的函数。这个函数由多个门组成。除了上面描述的那些门（add、mul、max）之外，还有四个门：



$$
f(x) = \frac{1}{x} 
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = -1/x^2 
\\\\
f_c(x) = c + x
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = 1 
\\\\
f(x) = e^x
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = e^x
\\\\
f_a(x) = ax
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = a
$$

其中函数 $f_c, f_a$ 分别通过常数 $c$ 平移输入和通过常数 $a$ 缩放输入。从技术上讲，它们是加法和乘法的特殊情况，但我们在这里将它们引入为（新的）一元门，因为我们不需要常数 $c,a$ 的梯度。完整的电路如下所示：

<div class="fig figleft fighighlight">
<svg style="max-width: 799px" viewbox="0 0 799 306"><g transform="scale(0.8)"><defs><marker id="arrowhead" refX="6" refY="2" markerWidth="6" markerHeight="4" orient="auto"><path d="M 0,0 V 4 L6,2 Z"></path></marker></defs><line x1="50" y1="30" x2="90" y2="30" stroke="black" stroke-width="1"></line><text x="55" y="24" font-size="16" fill="green">2.00</text><text x="55" y="47" font-size="16" fill="red">-0.20</text><text x="45" y="24" font-size="16" text-anchor="end" fill="black">w0</text><line x1="50" y1="100" x2="90" y2="100" stroke="black" stroke-width="1"></line><text x="55" y="94" font-size="16" fill="green">-1.00</text><text x="55" y="117" font-size="16" fill="red">0.39</text><text x="45" y="94" font-size="16" text-anchor="end" fill="black">x0</text><line x1="50" y1="170" x2="90" y2="170" stroke="black" stroke-width="1"></line><text x="55" y="164" font-size="16" fill="green">-3.00</text><text x="55" y="187" font-size="16" fill="red">-0.39</text><text x="45" y="164" font-size="16" text-anchor="end" fill="black">w1</text><line x1="50" y1="240" x2="90" y2="240" stroke="black" stroke-width="1"></line><text x="55" y="234" font-size="16" fill="green">-2.00</text><text x="55" y="257" font-size="16" fill="red">-0.59</text><text x="45" y="234" font-size="16" text-anchor="end" fill="black">x1</text><line x1="50" y1="310" x2="90" y2="310" stroke="black" stroke-width="1"></line><text x="55" y="304" font-size="16" fill="green">-3.00</text><text x="55" y="327" font-size="16" fill="red">0.20</text><text x="45" y="304" font-size="16" text-anchor="end" fill="black">w2</text><line x1="170" y1="65" x2="210" y2="65" stroke="black" stroke-width="1"></line><text x="175" y="59" font-size="16" fill="green">-2.00</text><text x="175" y="82" font-size="16" fill="red">0.20</text><circle cx="130" cy="65" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="75" font-size="20" fill="black" text-anchor="middle">*</text><line x1="90" y1="30" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="100" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="65" x2="170" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="170" y1="205" x2="210" y2="205" stroke="black" stroke-width="1"></line><text x="175" y="199" font-size="16" fill="green">6.00</text><text x="175" y="222" font-size="16" fill="red">0.20</text><circle cx="130" cy="205" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="215" font-size="20" fill="black" text-anchor="middle">*</text><line x1="90" y1="170" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="240" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="205" x2="170" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="290" y1="135" x2="330" y2="135" stroke="black" stroke-width="1"></line><text x="295" y="129" font-size="16" fill="green">4.00</text><text x="295" y="152" font-size="16" fill="red">0.20</text><circle cx="250" cy="135" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="250" y="140" font-size="20" fill="black" text-anchor="middle">+</text><line x1="210" y1="65" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="210" y1="205" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="270" y1="135" x2="290" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="410" y1="222" x2="450" y2="222" stroke="black" stroke-width="1"></line><text x="415" y="216" font-size="16" fill="green">1.00</text><text x="415" y="239" font-size="16" fill="red">0.20</text><circle cx="370" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="370" y="227" font-size="20" fill="black" text-anchor="middle">+</text><line x1="330" y1="135" x2="350" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="310" x2="350" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="390" y1="222" x2="410" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="530" y1="222" x2="570" y2="222" stroke="black" stroke-width="1"></line><text x="535" y="216" font-size="16" fill="green">-1.00</text><text x="535" y="239" font-size="16" fill="red">-0.20</text><circle cx="490" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="490" y="227" font-size="20" fill="black" text-anchor="middle">*-1</text><line x1="450" y1="222" x2="470" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="510" y1="222" x2="530" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="650" y1="222" x2="690" y2="222" stroke="black" stroke-width="1"></line><text x="655" y="216" font-size="16" fill="green">0.37</text><text x="655" y="239" font-size="16" fill="red">-0.53</text><circle cx="610" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="610" y="227" font-size="20" fill="black" text-anchor="middle">exp</text><line x1="570" y1="222" x2="590" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="630" y1="222" x2="650" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="770" y1="222" x2="810" y2="222" stroke="black" stroke-width="1"></line><text x="775" y="216" font-size="16" fill="green">1.37</text><text x="775" y="239" font-size="16" fill="red">-0.53</text><circle cx="730" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="730" y="227" font-size="20" fill="black" text-anchor="middle">+1</text><line x1="690" y1="222" x2="710" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="750" y1="222" x2="770" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="890" y1="222" x2="930" y2="222" stroke="black" stroke-width="1"></line><text x="895" y="216" font-size="16" fill="green">0.73</text><text x="895" y="239" font-size="16" fill="red">1.00</text><circle cx="850" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="850" y="227" font-size="20" fill="black" text-anchor="middle">1/x</text><line x1="810" y1="222" x2="830" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="870" y1="222" x2="890" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line></g></svg>
<div style="clear:both;"></div>
</div>

> 带有 sigmoid 激活函数的二维神经元的示例电路。输入是 [x0,x1]，神经元的（可学习的）权重是 [w0,w1,w2]。正如我们将在后面看到的，神经元计算输入的点积，然后通过 sigmoid 函数将其激活，使其范围在 0 到 1 之间。

在上面的示例中，我们看到了一长串的函数应用，它们作用于 **w,x** 的点积的结果。这些操作实现的函数称为 *sigmoid 函数* $\sigma(x)$。事实证明，sigmoid 函数关于其输入的导数在进行导数计算时（在分子中添加和减去 1 的有趣部分之后）会简化：



$$
\sigma(x) = \frac{1}{1+e^{-x}} \\\\
\rightarrow \hspace{0.3in} \frac{d\sigma(x)}{dx} = \frac{e^{-x}}{(1+e^{-x})^2} = \left( \frac{1 + e^{-x} - 1}{1 + e^{-x}} \right) \left( \frac{1}{1+e^{-x}} \right) 
= \left( 1 - \sigma(x) \right) \sigma(x)
$$

正如我们所看到的，梯度的计算变得非常简化并且变得非常简单。例如，sigmoid 表达式接收输入 1.0 并在前向传播过程中计算输出 0.73。上面的推导显示，*局部* 梯度只需 (1 - 0.73) * 0.73 ~= 0.2，就像电路之前计算的一样（见上面的图像），只不过这样可以用单个简单而高效的表达式完成（并且减少了数值问题）。因此，在任何实际应用中，将这些操作组合成一个单一的门将非常有用。让我们看看这个神经元的反向传播代码：

```python
w = [2,-3,-3] # assume some random weights and data
x = [-1, -2]

# forward pass
dot = w[0]*x[0] + w[1]*x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot)) # sigmoid function

# backward pass through the neuron (backpropagation)
ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
dx = [w[0] * ddot, w[1] * ddot] # backprop into x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # backprop into w
# we're done! we have the gradients on the inputs to the circuit
```

**实施技巧：分阶段反向传播**。如上面的代码所示，在实践中，将前向传播分解成容易进行反向传播的阶段总是有用的。例如，在这里我们创建了一个中间变量`dot`，该变量保存了`w`和`x`之间点积的输出。在反向传播过程中，我们便依次计算（按相反的顺序）相应的变量（例如`ddot`，最终是`dw, dx`），这些变量保存了这些变量的梯度。

本节的重点是，如何进行反向传播的细节，以及我们将前向函数的哪些部分视为“门”（gates），都是为了方便。了解表达式的哪些部分具有容易计算的局部梯度有助于用最少的代码和努力将它们链接在一起。

### 实践中的反向传播：分阶段计算

让我们通过另一个例子来看这一点。假设我们有一个形式如下的函数：



$$
f(x,y) = \frac{x + \sigma(y)}{\sigma(x) + (x+y)^2}
$$

明确地说，这个函数完全没有用，也不清楚为什么你会想要计算它的梯度，除了它是一个反向传播在实践中的好例子。非常重要的一点是强调，如果你试图对$x$或$y$进行微分，你将会得到非常大和复杂的表达式。然而，事实证明，这样做是完全不必要的，因为我们不需要有一个显式地计算梯度的函数。我们只需要知道如何计算它。以下是如何构建这样的表达式的前向传播结构：

```python
x = 3 # example values
y = -4

# forward pass
sigy = 1.0 / (1 + math.exp(-y)) # sigmoid in numerator   #(1)
num = x + sigy # numerator                               #(2)
sigx = 1.0 / (1 + math.exp(-x)) # sigmoid in denominator #(3)
xpy = x + y                                              #(4)
xpysqr = xpy**2                                          #(5)
den = sigx + xpysqr # denominator                        #(6)
invden = 1.0 / den                                       #(7)
f = num * invden # done!                                 #(8)
```

哎呀，在表达式的最后，我们已经计算了前向传递。请注意，我们以一种结构化的方式编写了代码，其中包含多个中间变量，每个变量都只是已知局部梯度的简单表达式。因此，计算反向传播非常容易：我们将反向传递，对于前向传递中的每个变量（`sigy, num, sigx, xpy, xpysqr, den, invden`），我们将有相同的变量，但以`d`开头，它将保存电路输出相对于该变量的梯度。此外，还要注意，我们反向传播中的每个部分都涉及计算该表达式的局部梯度，并将其与该表达式的梯度相乘进行链接。对于每一行，我们还会突出显示它所指的前向传递的哪一部分：

```python
# backprop f = num * invden
dnum = invden # gradient on numerator                             #(8)
dinvden = num                                                     #(8)
# backprop invden = 1.0 / den 
dden = (-1.0 / (den**2)) * dinvden                                #(7)
# backprop den = sigx + xpysqr
dsigx = (1) * dden                                                #(6)
dxpysqr = (1) * dden                                              #(6)
# backprop xpysqr = xpy**2
dxpy = (2 * xpy) * dxpysqr                                        #(5)
# backprop xpy = x + y
dx = (1) * dxpy                                                   #(4)
dy = (1) * dxpy                                                   #(4)
# backprop sigx = 1.0 / (1 + math.exp(-x))
dx += ((1 - sigx) * sigx) * dsigx # Notice += !! See notes below  #(3)
# backprop num = x + sigy
dx += (1) * dnum                                                  #(2)
dsigy = (1) * dnum                                                #(2)
# backprop sigy = 1.0 / (1 + math.exp(-y))
dy += ((1 - sigy) * sigy) * dsigy                                 #(1)
# done! phew
```

请注意以下几点：

**缓存前向传递变量**。为了计算反向传递，缓存在前向传递中使用的一些变量非常有帮助。在实际操作中，您希望构造您的代码以便缓存这些变量，并且它们在反向传播期间是可用的。如果这太困难，也可以重新计算它们，但这会浪费时间。

**梯度在分叉forks点相加**。前向表达式多次涉及变量**x**和**y**，因此在执行反向传播时，我们必须小心使用`+=`而不是`=`来累积这些变量上的梯度（否则会覆盖它）。这遵循了微积分中的*多变量链式法则multivariable chain rule*，该法则规定，如果一个变量分支到电路的不同部分，那么流回该变量的梯度将相加。

### 反向传播中的模式

有趣的是，许多情况下，反向传播的梯度可以直观地解释。例如，在神经网络中三个最常用的门（*加法、乘法、最大值*）在反向传播期间都有非常简单的解释。考虑这个示例电路：

<div class="fig figleft fighighlight">
<svg style="max-width: 460px" viewbox="0 0 460 290"><g transform="scale(1)"><defs><marker id="arrowhead" refX="6" refY="2" markerWidth="6" markerHeight="4" orient="auto"><path d="M 0,0 V 4 L6,2 Z"></path></marker></defs><line x1="50" y1="30" x2="90" y2="30" stroke="black" stroke-width="1"></line><text x="55" y="24" font-size="16" fill="green">3.00</text><text x="55" y="47" font-size="16" fill="red">-8.00</text><text x="45" y="24" font-size="16" text-anchor="end" fill="black">x</text><line x1="50" y1="100" x2="90" y2="100" stroke="black" stroke-width="1"></line><text x="55" y="94" font-size="16" fill="green">-4.00</text><text x="55" y="117" font-size="16" fill="red">6.00</text><text x="45" y="94" font-size="16" text-anchor="end" fill="black">y</text><line x1="50" y1="170" x2="90" y2="170" stroke="black" stroke-width="1"></line><text x="55" y="164" font-size="16" fill="green">2.00</text><text x="55" y="187" font-size="16" fill="red">2.00</text><text x="45" y="164" font-size="16" text-anchor="end" fill="black">z</text><line x1="50" y1="240" x2="90" y2="240" stroke="black" stroke-width="1"></line><text x="55" y="234" font-size="16" fill="green">-1.00</text><text x="55" y="257" font-size="16" fill="red">0.00</text><text x="45" y="234" font-size="16" text-anchor="end" fill="black">w</text><line x1="170" y1="65" x2="210" y2="65" stroke="black" stroke-width="1"></line><text x="175" y="59" font-size="16" fill="green">-12.00</text><text x="175" y="82" font-size="16" fill="red">2.00</text><circle cx="130" cy="65" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="75" font-size="20" fill="black" text-anchor="middle">*</text><line x1="90" y1="30" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="100" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="65" x2="170" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="170" y1="205" x2="210" y2="205" stroke="black" stroke-width="1"></line><text x="175" y="199" font-size="16" fill="green">2.00</text><text x="175" y="222" font-size="16" fill="red">2.00</text><circle cx="130" cy="205" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="210" font-size="20" fill="black" text-anchor="middle">max</text><line x1="90" y1="170" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="240" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="205" x2="170" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="290" y1="135" x2="330" y2="135" stroke="black" stroke-width="1"></line><text x="295" y="129" font-size="16" fill="green">-10.00</text><text x="295" y="152" font-size="16" fill="red">2.00</text><circle cx="250" cy="135" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="250" y="140" font-size="20" fill="black" text-anchor="middle">+</text><line x1="210" y1="65" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="210" y1="205" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="270" y1="135" x2="290" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="410" y1="135" x2="450" y2="135" stroke="black" stroke-width="1"></line><text x="415" y="129" font-size="16" fill="green">-20.00</text><text x="415" y="152" font-size="16" fill="red">1.00</text><circle cx="370" cy="135" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="370" y="140" font-size="20" fill="black" text-anchor="middle">*2</text><line x1="330" y1="135" x2="350" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="390" y1="135" x2="410" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line></g></svg>
<div style="clear:both;"></div>
</div>

>一个示例电路可以展示反向传播在反向传递过程中执行的操作，以计算输入上的梯度。求和操作将梯度均匀分配到所有输入。最大值操作将梯度路由route到较高的输入。乘法门获取输入激活，交换它们并乘以它的梯度。

通过上面的示例电路，我们可以看到以下情况：

**加法门** 总是将梯度分配到其所有输入中，不管它们在前向传递期间的值是什么。这是因为加法操作的局部梯度仅为+1.0，因此所有输入上的梯度将完全等于输出上的梯度，因为它将乘以1.0（保持不变）。在上面的示例电路中，请注意加法门将梯度2.00均匀分配给了其两个输入，并保持不变。

**最大值门** 路由梯度。与将梯度不变地分配给所有输入的加法门不同，最大值门将梯度（不变地）分配给其一个输入（在前向传递期间具有最高值的输入）。这是因为最大值门的局部梯度对于最高值为1.0，对于所有其他值为0.0。在上面的示例电路中，最大值操作将梯度2.00路由到了变量**z**上，因为**z**的值比**w**高，而**w**上的梯度保持为零。

**乘法门** 稍微难以解释一些。它的局部梯度是输入值（除了交换），并且这个梯度会在链式法则中与其输出上的梯度相乘。在上面的示例中，**x**上的梯度为-8.00，这是-4.00 x 2.00。

*不直观的效果及其影响*：请注意，如果乘法门的一个输入非常小，而另一个非常大，那么乘法门会执行一些略微不直观的操作：它将为小输入分配相对较大的梯度，并为大输入分配微小的梯度。请注意，在线性分类器中，权重与输入进行点积 $w^Tx_i$（相乘），这意味着数据的规模会影响权重的梯度大小。例如，如果在预处理过程中将所有输入数据示例 $x_i$ 乘以1000，那么权重上的梯度将大1000倍，您需要将学习率降低相同的因子来补偿。这就是为什么预处理非常重要，有时是以微妙的方式！并且对梯度流动方式有直观的理解可以帮助您调试其中的一些情况。

### 矢量化操作的梯度

上面的部分涉及单个变量，但所有概念都以直观的方式扩展到矩阵和向量操作。但是，必须更加关注维度和转置操作。

**矩阵-矩阵乘法梯度**。可能最棘手的操作之一是矩阵-矩阵乘法（通用化了所有矩阵-向量和向量-向量乘法操作）：

```python
# forward pass
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)
```

*提示：使用维度分析！* 请注意，您不需要记住 `dW` 和 `dX` 的表达式，因为它们可以基于维度轻松重新推导。例如，我们知道权重 `dW` 在计算后必须与 `W` 的大小相同，并且它必须依赖于 `X` 和 `dD` 的矩阵乘法（当 `X,W` 都是单个数字而不是矩阵时就是这种情况）。总是有一种确切的方法可以实现这一点，以使维度配合起来。例如，如果 `X` 的大小为 [10 x 3]，`dD` 的大小为 [5 x 3]，那么如果我们想要 `dW`，而 `W` 的形状为 [5 x 10]，那么实现这一点的唯一方法就是 `dD.dot(X.T)`，如上所示。

**使用小而明确的示例**。一些人可能会发现首次为一些矢量化表达式推导梯度更新有些困难。我们建议首先明确地编写一个最小的矢量化示例，然后在纸上推导梯度，然后将模式推广到其高效的矢量化形式。

Erik Learned-Miller 还编写了一份与矩阵/向量导数有关的更长的相关文档，您可能会发现它很有帮助。[在这里找到](http://cs231n.stanford.edu/vecDerivs.pdf)。

### 总结

- 我们对梯度的含义有了直观认识，了解了它们在电路中如何反向传播backwards ，以及它们如何传达电路的哪个部分应该增加或减少，以及以何种力量来使最终输出变高。
- 我们讨论了**分阶段计算**在反向传播的实际实现中的重要性。您始终希望将函数分解为可以轻松推导出局部梯度的模块，然后使用链式法则连接它们。关键是，您几乎永远不希望在纸上写出这些表达式，并以符号方式完全不同地进行微分，因为您永远不需要输入变量的显式数学方程。因此，请将您的表达式分解成阶段，以便可以独立地对每个阶段进行微分（这些阶段将是矩阵向量乘法、最大操作、求和操作等），然后一次性反向传播变量。

在下一节中，我们将开始定义神经网络，而反向传播将允许我们有效地计算损失函数相对于其参数的梯度。换句话说，我们现在已经准备好训练神经网络，这门课程中最具概念性挑战的部分已经过去了！然后，卷积神经网络就近在眼前了。

### 参考资料

- [Automatic differentiation in machine learning: a survey](http://arxiv.org/abs/1502.05767)