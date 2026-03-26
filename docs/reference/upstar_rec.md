# UPSTAR 复现对齐笔记（方法与实验重点节选）

> 本文档整理自前面对论文与附录的精读内容，重点覆盖：
>
> 4. **3.1 Purchase Motivation Identification**  
> 5. **3.2 In and Cross-Session Item Representation Learning**  
> 6. **3.1.4 + 3.3：三路预测与联合训练**  
> 7. **实现细节（正文与附录中明确给出的）**  
> 8. **实验协议（必须优先对齐）**  
> 9. **Ta-Feng 数据集对齐要点**  
> 10. **论文结果中与复现最相关的参考值**  
> 11. **关键消融（复现时应尽量支持）**

---

# 4. 3.1 Purchase Motivation Identification

## 4.1 这一节在做什么

这一节是全文最核心的部分，因为作者真正的创新不在“用了 LSTM 还是 GNN”，而在于：

> **把购买动机从抽象心理概念，转成了可以在 item level 上识别的信号。**

作者先基于前文理论框架说明：  
在识别 stable preference 和 exploratory intent 之前，要先将那些**纯粹由 instrumental incentives 驱动**的购买排除掉，也就是那种主要因为折扣、优惠券、促销赠品等外在激励而发生的购买。

做完这一步之后，才在“高内在兴趣”的购买中进一步区分：

- **Stable Preference**
- **Exploratory Intent**

作者对二者的核心判断是：

- **Stable Preference** 对 `timing` 和 `copurchased items` 的依赖通常较低
- **Exploratory Intent** 对这两者的依赖通常较高

所以后面整个技术设计，实际上都在围绕两个可观测代理变量展开：

- `timing dependence`
- `copurchase dependence`

---

## 4.2 3.1.1 Attributed Item-Time Graph Construction

为了识别购买动机，作者构造了一个 **attributed item-time graph**。

### 图结构

这是一个对**每个用户**构建的二部图，包含两类节点：

- `item nodes`
- `time nodes`

若某 item 在某时刻被购买，则在该 item 节点和对应的 time 节点之间连边。

### 图中编码的信息

这个图天然编码了两种和动机高度相关的行为模式：

#### 1. 一跳邻居：timing dependence
某 item 的一跳 time 邻居，表示它在哪些时间点被买过，因此可反映该 item 对时间的依赖。

#### 2. 两跳邻居：copurchase dependence
某 item 通过时间节点与其他 item 两跳相连，因此可反映它与其他商品的共购关系。

### 节点特征

图记为：

\[
S=(A,X)
\]

其中：

- \(A\) 是 item-time 二部图邻接矩阵
- \(X\) 是节点特征矩阵

具体地：

\[
A=
\begin{bmatrix}
0_{I,I} & E \\
E^\top & 0_{T,T}
\end{bmatrix}
\]

\[
X=[X_{item}, 0_{T,D}]
\]

其中：

- `X_item` 来自 3.2 学到的 item representation
- time node 用零向量表示

### 这一步的意义

作者不是手工构造几个统计量，而是把：

- 商品买在什么时候
- 商品和什么一起买
- 商品本身是什么

统一放进一个图结构中，再让模型自动学习动机信号。

---

## 4.3 3.1.2 Proposed Measure STB

这一部分提出了全文最核心的概念：**STB**。

作者指出：

- exploratory item 往往在 item-time graph 上有很强的 timing / copurchase 模式，因此相对容易识别
- stable item 不能只根据一个静态观察到的图去判断，因为“稳定”意味着它应当在不同可能情境下仍然稳定

于是作者提出：

> **Stable Preference = 对 timing 与 copurchase 上下文扰动具有鲁棒性。**

基于这个思想，定义：

\[
STB_{n,t}=\inf_{S'_t\in B_t} 1-\Pr[h([e(S'_t)]_n)\neq StablePreference]
\]

其中：

- \(S_t\)：当前时间窗口的 attributed item-time graph
- \(B_t\)：对 \(S_t\) 做合理扰动得到的一组图
- \(e(\cdot)\)：GNN encoder
- \(h(\cdot)\)：分类器

### 直观含义

STB 衡量的是：

> **在所有可能扰动下，一个 item 在最坏情况下还能保持 stable 的程度。**

因此：

- `STB 高` → 更像 stable preference
- `STB 低` → 更像 exploratory intent

### 论文的重要思想

stable 不是“出现次数多”或者“复购次数高”，而是：

> **换一个时间、换一个搭配上下文，它仍然稳。**

---

## 4.4 3.1.3 Approximate Solution

STB 的定义很好，但直接求很难。困难主要有两个：

1. 没有真实的 item-level 动机标签
2. 扰动对象既有连续特征 \(X\)，也有离散图结构 \(A\)

因此作者做了近似求解。

### 第一步：Theorem 1，把问题转成互信息上界

论文通过 Theorem 1 将“被识别为 stable 的概率”上界到 mutual information：

\[
1-\Pr[h(e(S))\neq stable]
\le
\frac{I(S;e(S))+\log 2}{\log |Y|}
\]

附录 C 给出了证明，核心依赖：

- Fano’s inequality
- data processing inequality

于是，原问题进一步近似为：

\[
\min_{S' \in B} I(S'; e(S'))
\]

也就是：

> 寻找一个最坏扰动图，使图和其表示之间保留的信息尽量少。

### 第二步：互信息估计

由于 \(S=(A,X)\) 同时包含离散和连续部分，互信息难以直接闭式计算。  
因此作者采用了类似 **noise-contrastive / mutual information neural estimation** 的做法，用可训练判别器去近似 \(I(S;e(S))\)。

### 第三步：GNN encoder

作者使用 GNN 对 attributed item-time graph 编码。  
其作用是把：

- item 自身特征
- time / copurchase 邻域信息

一起映射到表示空间中。

### 第四步：最坏扰动搜索

扰动对象包括：

#### 1. 对特征 \(X\) 的扰动
使用 PGD 风格方法，在预算约束下寻找对 stable 识别最不利的扰动。

#### 2. 对结构 \(A\) 的扰动
由于 \(A\) 是布尔图结构，作者先把它 relax 到连续空间，再做 projected gradient style topology attack。

### 这一节最该抓住的点

3.1.3 的本质不是“互信息本身”，而是：

> **作者把 stable preference 的识别问题，转成了一个受约束的最坏扰动鲁棒性估计问题。**

---

## 4.5 3.1.4 Next Item Prediction（与 3.3 连接处）

有了 STB 之后，作者对原始行为序列进行拆分：

- stable preference subsequence
- exploratory intent subsequence
- entire original sequence

其中：

- stable / exploratory 来自 STB 分类结果
- uncategorized items 不会丢掉，而是保留在 entire sequence 中，由 O-model 处理

这是后面三路时序建模的输入基础。

---

# 5. 3.2 In and Cross-Session Item Representation Learning

## 5.1 这一节的目标

3.2 不是单纯“提升推荐精度”的辅助模块，而是给：

- 3.1 的动机识别
- 3.1.4 / 3.3 的下一商品预测

提供更好的商品表示底座。

作者认为，仅靠单一 session 内的共现关系来学习 item representation 是不够的，因为：

1. 许多相关 item 不一定在同一次 session 中共同出现
2. 单个 session 可能太稀疏，无法学到稳定表示

因此作者提出：

> **同时利用 in-session 和 cross-session 的商品关系，学习更可靠的 item representation。**

---

## 5.2 In and Cross-Session Item Graph

作者构建一个有向 item graph：

- 节点：item
- 若某用户在任意 session / 行为序列中先买了 \(i_n\)，再买了 \(i_m\)，则建边 \(i_n \to i_m\)

这不是普通的无向共现图，而是一个保留顺序关系的图。

### 这张图编码了什么

它编码的是：

- 商品之间的顺序关联
- in-session 与 cross-session 的广义转移关系
- 哪些 item 常常作为前置触发项，导向另一些 item

---

## 5.3 Item-GNN

作者在 item graph 上使用 **Item-GNN** 学习商品表示。

### Message Passing

作者显式区分：

- `in-neighbors`
- `out-neighbors`

并分别用不同参数矩阵处理：

\[
h_n^{g(k)} \leftarrow ReLU\Big(
W_{in}^{(k)} \cdot \sum h_p^{(k-1)}
+
W_{out}^{(k)} \cdot \sum h_q^{(k-1)}
\Big)
\]

### 为什么区分 in / out

因为推荐任务中，前驱 item 和后继 item 并不对称。  
区分 in/out，有助于模型学习更精细的商品转移语义。

---

## 5.4 Remember Gate

仅靠图聚合还不够，因为 item 自身属性也很重要。  
因此作者加入 **remember gate**：

\[
z_n^{(k)}=\sigma(\text{Linear}(h_n^{g(k)}, h_n^{(k-1)}))
\]

\[
h_n^{(k)}=(1-z_n^{(k)})\odot h_n^{(k-1)}+z_n^{(k)}\odot h_n^{g(k)}
\]

### 作用

remember gate 用来平衡：

- item 节点自身表示
- 邻域传播过来的信息

最终避免“图传播把节点自身 identity 冲淡”。

---

## 5.5 输出与作用

最终 Item-GNN 输出：

\[
X_{item}
\]

这组 item representations 会进一步进入 3.1 attributed item-time graph，作为 item node features。

### 这一节最重要的理解

3.2 不是独立支线，而是 3.1 的前置增强器。

没有 3.2，3.1 里的 STB 更容易过度依赖稀疏行为模式；  
有了 3.2，STB 才能同时看：

- item 是什么
- item 在什么情境下被买

---

# 6. 3.1.4 + 3.3：三路预测与联合训练

## 6.1 三路序列建模

根据 STB 结果，作者把原始行为序列拆成三条：

- stable subsequence
- exploratory subsequence
- entire original sequence

分别进入三个模型：

- **S-model**：建模 stable preference
- **E-model**：建模 exploratory intent
- **O-model**：建模 other / uncategorized / complex motivations

论文中三者都采用 LSTM。

### 各路输出

- `z_stab`, `y_hat_stab`
- `z_expl`, `y_hat_expl`
- `z_other`, `y_hat_other`

---

## 6.2 Global Fusion

作者进一步引入一个 global fusion gate，对三路表示进行自适应加权融合，得到：

- `z_global`
- `y_hat_global`

这一步的意义是：

> 不同样本对 stable / exploratory / other 的依赖程度不同，因此不能简单平均，而要动态决定三路权重。

---

## 6.3 3.3 Dual Teacher–Student Training

3.3 解决的问题是：

> 在完成动机分解后，如何通过联合训练让三条路径既分工又协作，从而真正提升 next-item prediction。

论文中的训练目标包括四类。

---

### 6.3.1 Global Loss

对最终融合输出 `y_hat_global` 做 next-item cross-entropy：

\[
L_{global}
\]

作用：

- 保证整个系统的最终目标仍然是“下一商品预测准确”

---

### 6.3.2 Branch Loss

对三条分支输出分别加监督：

\[
L_{S\&E\&O}
\]

即分别监督：

- `y_hat_stab`
- `y_hat_expl`
- `y_hat_other`

作用：

- 让 S/E/O 三条链都具备独立的预测能力
- 避免三条路径只是 global fusion 的中间特征提取器

---

### 6.3.3 Orthogonality Loss

由于 O-model 的 entire sequence 本身包含 stable 和 exploratory 的信息，  
如果不加限制，O-model 很容易重复学习前两条分支的内容。

因此引入：

\[
L_{orth}=\tau_s z_{other}^{\top} z_{stab}+\tau_e z_{other}^{\top} z_{expl}
\]

作用：

- 让 `z_other` 尽量与 `z_stab`、`z_expl` 正交
- 逼 O-model 学 stable/exploratory 之外的“剩余复杂动机”

---

### 6.3.4 Dual Teacher–Student

这是 3.3 最有特色的部分。

作者观察到：

- 当真实 next item 属于 stable preference 时，S-model 更擅长
- 当真实 next item 属于 exploratory intent 时，E-model 更擅长

因此设计动态蒸馏机制：

#### Stable 样本
若真实 next item 属于 stable：
- 固定 S-model
- 用 S 的 soft prediction 监督 E

\[
L_{S \to E}=s_{next} D_{KL}(\hat y_{stab}\Vert \hat y_{expl})
\]

#### Exploratory 样本
若真实 next item 属于 exploratory：
- 固定 E-model
- 用 E 的 soft prediction 监督 S

\[
L_{E \to S}=e_{next} D_{KL}(\hat y_{expl}\Vert \hat y_{stab})
\]

### 核心思想

不是固定一个 teacher，而是依据样本动机类型动态切换：

- stable 样本：S teaches E
- exploratory 样本：E teaches S

这使得 S 和 E 在各自不擅长的样本上，也能借助对方知识做补课。

---

# 7. 实现细节（正文与附录中明确给出的）

以下是论文明确写出的、复现时应优先对齐的参数与设置。

## 7.1 动机识别相关

- `ρ = 50`
- `β = 40`
- item-time graph 的 time node 采用 **day-level**
- STB 的 GNN encoder：**1 layer**
- STB encoder hidden size：**512**
- STB optimizer：**Adam**
- STB learning rate：**1e-3**
- 扰动相关参数：
  - `α = 0.4`
  - `ε = 0.1`
  - `ε_x = 0.1`
  - `ε_a = 0.1`

---

## 7.2 商品表示学习相关

- Item-GNN：**1 layer**
- item representation dimension：**128**

---

## 7.3 推荐模型相关

- S/E/O 三路序列模型：使用 **LSTM**
- hidden size：**128**
- num_layers：**4**

---

## 7.4 联合训练相关

- optimizer：**Adam**
- learning rate：**3e-4**
- batch size：**64**
- dual teacher–student 参数：
  - `λ = 0.7`
  - `τ_s = 0.5`
  - `τ_e = 0.5`

---

## 7.5 附录补充理解点

### 附录 B
STB 不只是由行为模式决定，也依赖 item intrinsic attributes。  
因此：

- rare purchase ≠ exploratory
- item attribute 表示也参与 STB 判断

### 附录 C
Theorem 1 是一种**上界放松**，不是严格等价变换。  
也就是说：

- STB 中“被判为 stable 的概率”
- 与 mutual information 之间是上界关系

### 附录 D
复杂度分析显示：

- Item representation learning 相对较轻
- STB 模块更重
- dual teacher–student 复杂度可控

### 附录 E
若未来引入购买之外的行为（如 visit / like / add-to-cart），可扩展为多 edge type item graph。

---

# 8. 实验协议（必须优先对齐）

## 8.1 评估指标

论文使用：

- `Precision@k`
- `NDCG@k`
- `MRR@k`

其中：

\[
k \in \{1,5,10,15,20,50\}
\]

### 正文主表通常展示
- `P@5`, `P@20`
- `NDCG@5`, `NDCG@20`
- `MRR@5`, `MRR@20`

### 附录会展示 full table
- k in {1,5,10,15,20,50}

---

## 8.2 实验组织方式

论文使用：

- **10-fold cross-validation**

结果输出形式：

- `mean ± std`
- 百分比形式

这点必须优先对齐。  
在 baseline 还没按 10-fold 对齐前，不应贸然比较 UPSTAR 和论文结果。

---

# 9. Ta-Feng 数据集对齐要点

论文对 Ta-Feng 的处理方式是：

1. 将**同一用户购买的所有 item 拼成一个 session**
2. 删除长度小于 3 的 session

处理后统计量为：

- `29,142 sessions`
- `23,782 items`
- 平均 session 长度 `25.34`

这些统计量是复现时最重要的对齐基线之一。

## 需要重点核对的地方

- 是否真的按用户级长序列构造 session
- 是否删除了长度 < 3 的 session
- 最终 item 数是否接近 23,782
- 平均 session 长度是否接近 25.34

如果这里对不齐，后续 baseline 和 UPSTAR 的结果都可能严重偏离论文。

---

# 10. 论文结果中与复现最相关的参考值

## 10.1 Ta-Feng 上的 LSTM 参考值

论文主表中，Ta-Feng 上 LSTM 结果约为：

- `P@5 = 8.23 ± 0.48`
- `P@20 = 16.48 ± 0.64`
- `NDCG@5 = 5.75 ± 0.39`
- `NDCG@20 = 8.13 ± 0.25`
- `MRR@5 = 4.92 ± 0.33`
- `MRR@20 = 5.77 ± 0.44`

这些值是你检查 baseline 是否接近论文量级的核心参照。

---

## 10.2 Ta-Feng 上的 UPSTAR 参考值

论文主表中，Ta-Feng 上 UPSTAR 结果约为：

- `P@5 = 16.24 ± 0.22`
- `P@20 = 25.98 ± 0.40`
- `NDCG@5 = 12.31 ± 0.12`
- `NDCG@20 = 15.20 ± 0.19`
- `MRR@5 = 11.00 ± 0.29`
- `MRR@20 = 12.07 ± 0.33`

这组数值代表了“完整模型”应达到的大致量级。

---

# 11. 关键消融（复现时应尽量支持）

论文消融表明，以下模块去掉后通常都会带来性能下降，因此复现时尽量要支持这些 ablation：

## 11.1 不做动机拆分
- 不区分 stable / exploratory
- 只使用 entire sequence

## 11.2 去掉 O-model
- 不建模 other / uncategorized motivations

## 11.3 去掉 uncategorized handling
- 不保留中间模糊区间
- 或不让 entire sequence 吸收其影响

## 11.4 去掉 orthogonality loss
- 不加 `L_orth`

## 11.5 去掉 dual teacher–student
- 不加 KL distillation

## 11.6 去掉 cross-session item graph
- item representation 不再使用 in and cross-session 图增强

## 11.7 Full UPSTAR
- 作为完整对照组

### 为什么这些消融重要

因为它们可以帮助判断：

- stable / exploratory 分工是否真的有效
- O-model 是否确实在吸收剩余复杂动机
- dual teacher–student 是否真的在补短板
- cross-session item graph 是否提升了 item representation 质量

---

# 12. 一句话总结

UPSTAR 的核心不是“更复杂的推荐网络”，而是：

> 先学习商品表示，再通过 item-time graph 和 STB 识别“为什么会买”，再把行为拆成 stable / exploratory / other 三条路径分别建模，最后通过融合、正交约束和双向 teacher–student 联合训练，完成更准确且更可解释的 next-item recommendation。