# 我们首先对比两篇论文： SaMIA and SimMIA

SaMIA 是“第一代可落地的 text-only black-box MIA”，而 SimMIA 是在同一攻击设定上，针对 SaMIA 的两个核心弱点做系统升级。

## 解决相同的问题 Text-only LLM：

两篇都在研究 LLM 的 membership inference attack (MIA)：给定一段文本，判断它是否出现在模型训练数据里。两篇都特别关心 严格 black-box / text-only 场景，也就是攻击者拿不到 logits、likelihood、loss，只能通过 API 看模型生成的文本。这一点是 SaMIA 的出发点，也是 SimMIA 反复强调的设定。

所以从任务层面说，SaMIA：先证明“不看 likelihood，只靠采样文本也能做 MIA”；SimMIA：再进一步证明“只靠采样文本不但能做，而且还能做得更稳、更强、适配更新的 proprietary LLM”
footnote: proprietary LLM = "A proprietary Large Language Model (LLM) is an AI model developed, owned, and controlled by a specific organization, with its code, architecture, and training data kept private. These models are typically accessed via _APIs or subscriptions_, offering high performance for specialized tasks but often with higher costs and limited transparency compared to open-source alternatives."

## SaMIA

关键思想：把待测文本切成 prefix 和 reference suffix，然后把 prefix 喂给目标 LLM，采样出多条 continuation；如果这些 continuation 和真实 suffix 的 n-gram overlap 很高，就说明模型很可能“记住了”这段文本，于是判为 member。论文里具体用 ROUGE-N 来度量这种表面重合度，并把它当成一种 sampling-based pseudo-likelihood (SPL) 近似。
直觉：“如果一段文本在训练集中，模型从前半段续写时，更容易续出和后半段表面上相似的内容。” 它把原来依赖 likelihood 的 MIA，改成了依赖 sampled continuations + lexical overlap 的 MIA
贡献：

* 提出一个 likelihood-free 的 black-box MIA；
* 证明采样文本和 reference 的 n-gram match 可以作为 pseudo-likelihood 近似；
* 实验上说明即使没有 likelihood，也能接近甚至达到已有 likelihood-based 方法的效果。


## SimMIA 认为 SaMIA 的问题 以及 Motivation：

1. distribution drift
   SaMIA 采样的是“整段 continuation”。SimMIA 认为，这种长程生成会越来越依赖模型自己刚生成出来的 token，而不是只依赖原始 prefix，所以后面位置的采样分布会偏离真实文本的条件分布。论文把这叫做 self-conditioning 带来的分布漂移

2. Signal Sparsity
   SaMIA 主要看 exact n-gram overlap。SimMIA 认为这太“硬”了：哪怕模型没有逐字复现，只是生成了语义很接近的词，SaMIA 也很难利用这些信号，因此跨 domain 时会不稳定。SimMIA 用“exact N-gram matching fails to capture semantic retention”来概括这一点。SaMIA的信号是“表面复现”，SimMIA 想抓的是“局部条件下的语义一致性”。

## SimMIA的方法改进：

解决问题1: 采样方式：从“整段续写”改成“逐词采样” word-by-word sampling：
SaMIA 是固定一个 prefix，然后采多条完整 continuation，再和 suffix 做 ROUGE。
SimMIA 则是把文本分成词序列，从第二个词开始，对每个位置 $x_i$ 都只采样“下一个词”。也就是说，它不是一次性生成整段，而是对每个 prefix $x_{<i}$ 单独估计当前词的局部生成行为。作者认为这样可以避免 long-form generation 的分布漂移，同时不增加采样成本。
这个从“模型能不能把后半段整段接出来？”变化到“在每一个局部位置上，模型是不是持续偏向真实词或其语义近邻？”

解决问题2: 打分方式 word-level scoring：从“ROUGE lexical match”改成“embedding semantic score”
SaMIA 的打分本质上是 ROUGE-N，偏 lexical / surface-form。作者自己也在分析里比较过 ROUGE-1 和 ROUGE-2，并指出 unigram overlap 更有效。（SaMIA：基于 surface overlap）
SimMIA 先给出一个硬匹配版本 SimMIA*：统计采样词里和真实词完全相同的频率；但作者进一步说这仍然太 sparse，于是引入正式版 SimMIA：对真实词和采样词计算 embedding cosine similarity，并取期望作为 word-level score。也就是说，只要生成的是语义接近的词，也能贡献分数。（SimMIA：基于 local semantic consistency）

新痛点：聚合方式：SimMIA 多了 relative aggregation
SaMIA 本质上是把多个 continuation 与 reference 的相似度聚合起来做判别。
SimMIA 除了逐词打分，还额外引入了 relative aggregation：把正常 prefix 下的分数，和加上 non-member prefix 扰动后的分数做比值，再在全文上平均。作者借鉴了相对条件 likelihood 的思路，认为 member 文本在被非成员上下文扰动时，分数下降模式与 non-member 不同。
这一步是 SimMIA 比 SaMIA 更“像现代 MIA”的地方，因为它不只看原始分数，而看 相对分数在扰动前后的变化。

贡献：
明确诊断 SaMIA 类方法的问题：distribution drift 和 signal sparsity；
提出 word-by-word sampling + semantic scoring + relative aggregation；
引入 WikiMIA-25，把评测扩展到更新、更贴近现实的 proprietary / modern LLM。

## SaMIA 对比 SimMIA：

SaMIA 像是在看：“你给模型文章前半段，它能不能把后半段几乎原样续出来？”
SimMIA 像是在看：“文章每往后走一步，模型对下一个词是不是持续表现出偏向真实语义的倾向；而且这种倾向在被错误上下文扰动后，是否仍然表现出 member/non-member 的差异？”

前者更依赖“显式记忆复现”，后者更依赖“局部条件偏好 + 语义接近 + 扰动鲁棒性”。因此在文本风格变化更大、模型不逐字背诵的情况下，SimMIA 的信号理论上更不容易丢。这个判断和 SimMIA 作者自己给出的“signal sparsity / semantic retention”论证是一致的。

总结：
SaMIA = sampled continuations + ROUGE overlap ≈ pseudo-likelihood。
SimMIA = word-by-word sampling + semantic scoring + relative aggregation。
SimMIA 本质上是在修 SaMIA 的两处痛点：长程采样漂移、硬匹配太稀疏

| 维度         | SaMIA                                     | SimMIA                                    |
| ------------ | ----------------------------------------- | ----------------------------------------- |
| 攻击设定     | strict black-box / text-only              | strict black-box / text-only              |
| 核心问题     | 没有 likelihood 时如何做 MIA              | 如何让 text-only MIA 更稳、更强           |
| 采样单位     | 整段 continuation                         | 每个 prefix 下逐词采样                    |
| 核心信号     | ROUGE-N / n-gram overlap                  | embedding-based semantic similarity       |
| 聚合         | continuation-level similarity aggregation | word-level scoring + relative aggregation |
| 主要批评对象 | likelihood-dependent MIA 不可用           | SaMIA 类方法有 drift 和 sparse signal     |
| 典型优势     | 简单、直接、适合 verbatim memorization    | 更强、更鲁棒、适配更新 proprietary LLM    |
| 实验定位     | 与 likelihood-based 方法接近              | 超过 SaMIA，并逼近部分 gray-box 方法      |


SimMIA算法的基本流程：

* 把待检测文本切成词序列；
* 对每个位置 i，在真实 prefix x 下采样 N 个下一个词；
* 用 embedding similarity 算该位置的 soft score；
* 再在加了 non-member prefix 的扰动上下文下重复一次，得到相对分数；
* 把所有位置的相对分数聚合成一个整体 membership score；
* 分数越高，越倾向判为 member。


# Experimental Plan

## Compare

### 现有方法复现

我们已经成功复现了原始 SimMIA 方法，包括 word-by-word sampling, embedding-based word-level scoring 以及 relative aggregation。原先版本的缓存文件包括 `full_dataset.jsonl`, `prefix_data.json`, 和 `records.jsonl`。

原先版本中，三个缓存的职责是：

* `full_dataset.jsonl`: 保存真正进入评估的待检测 sequence。每一行包含 `input`, `label`, 以及一个语义较模糊的 `prefix` 字段；这里的 `prefix` 实际上是 fixed non-member prefix。
* `prefix_data.json`: 保存 prefix bank，即抽出的 `member` 和 `nonmember` sequence。原始 SimMIA 主流程只会把 `nonmember` 拼起来作为扰动 prefix。
* `records.jsonl`: 保存最耗时的 target LLM sampling 结果。原先每行主要包含：

```text
input                  # 待检测文本
label                  # 真实答案，1 是 member，0 是 non-member
prefix                 # fixed non-member prefix
label_results          # 真实 next word 序列
sample_results         # 原始上下文下的 word-level sampling 统计
prefix_sample_results  # fixed non-member prefix 下的 word-level sampling 统计
```

其中 `sample_results` 和 `prefix_sample_results` 是原始 SimMIA / SimMIA* 真正用于 postprocess 和 inference 的两路采样结果。`label_results` 提供每个位置的真实 next word。

### 当前缓存改动

为了之后测试 member-prefix contrastive 方法，我们已经把 `records.jsonl` 的 sampling 缓存改成更显式的三路结构。新的 `full_dataset.jsonl` 每行包含：

```text
input             # 待检测文本
label             # 真实答案
nonmember_prefix  # fixed non-member prefix
member_prefix     # fixed member prefix
```

新的 `records.jsonl` 每行包含：

```text
input                              # 待检测文本
label                              # 真实答案
nonmember_prefix                   # fixed non-member prefix
member_prefix                      # fixed member prefix
label_results                      # 真实 next word 序列
sample_results                     # raw / 原始上下文下的 word-level sampling 统计
nonmember_prefix_sample_results    # non-member prefix 下的 word-level sampling 统计
member_prefix_sample_results       # member prefix 下的 word-level sampling 统计
```

这个改动不改变当前 SimMIA / SimMIA* 的复现逻辑：现有 `process_relative_word_data` 仍只处理 `sample_results` 和 `nonmember_prefix_sample_results`，然后交给 `relative_semantic_ratio` 或 `relative_label_ratio` 计算原方法分数。新增的 `member_prefix_sample_results` 暂时只作为 raw cache 保存，方便后续实验直接读取。

### 改进方案设计

为了提升 SimMIA 的稳定性和统计可解释性，我们提出以下逐步改进方案。新的顺序应该先从 member prefix 开始，因为现在 member-prefix sampling 已经被提前写进 `records.jsonl`；在已有缓存上测试它只需要新增后处理/打分逻辑，不需要立刻重新调用目标 LLM。之后再考虑改进 word-level score，最后才做外层 prefix 扰动。

#### Step 1: 引入 member prefix

* **目标**: 利用新的 `member_prefix_sample_results` 和 `nonmember_prefix_sample_results` 形成 member/non-member prefix 对照，计算 contrastive membership score。

* **公式**:
  $$
  r_i = \frac{\widehat{S}(x_i \mid P_{nm}+x_{<i}) - \widehat{S}(x_i \mid P_m+x_{<i})}{\widehat{S}(x_i \mid x_{<i})}
  $$

  这里的 $\widehat{S}$ 可以先沿用 SimMIA 当前的 word-level score，也就是 embedding similarity expectation；hard 版本也可以对应使用真实 label word 的采样频率。

* **sequence-level membership score**:
  $$
  \text{membership score} = - \frac{1}{L} \sum_{i=1}^{L} r_i
  $$

* **实现细节**:

  * 不新增 target LLM sampling，直接读取新的 `records.jsonl`
  * 使用 `sample_results` 作为 raw/original baseline
  * 使用 `nonmember_prefix_sample_results` 作为 non-member perturbation
  * 使用 `member_prefix_sample_results` 作为 member perturbation
  * 新建独立 postprocess / inference 文件，避免影响原始 SimMIA 复现

#### Step 2: 替换或校准 word-level score

* **目标**: 在 Step 1 的 member-prefix contrastive 框架基础上，测试更稳定、更接近概率密度解释的 word-level score，同时保持对 SimMIA 框架的最小改动。

* **动机**: 原始 SimMIA 的 soft score 是 semantic similarity expectation。直接把 cosine similarity softmax 成 likelihood proxy 可能会放大候选集合内部的相对差异，因此更适合作为 kernelized semantic density score，而不是严格的 token likelihood。

* **候选公式**:
  $$
  S_i(P) = \sum_j f_{i,j}(P)\exp(\text{cosine}(x_i, \hat{x}_{i,j}) / \tau)
  $$

  或使用 log-ratio 形式：
  $$
  r_i = \log S_i(P_{nm}+x_{<i}) - \log S_i(P_m+x_{<i})
  $$

* **实现细节**:

  * 内层采样 M 次已经存在于新的 `records.jsonl`
  * 只需要替换 postprocess / inference 中的 word-level score 计算
  * $\tau$ 为温度参数，需要做 sweep
  * 与原始 cosine expectation 做 ablation

#### Step 3: 外层 prefix 扰动（最大规模实验）

* **目标**: 增加统计稳定性和鲁棒性，验证 prefix 组成变化对 Step 1 / Step 2 的影响。

* **扰动类型**:

  1. 顺序打乱 prefix
  2. 删除 1-2 条 sequence
  3. paraphrase prefix

* **公式**:
  $$
  \tilde{r}_i = \frac{1}{|\Delta|} \sum_{\delta \in \Delta} 
  \frac{\widehat{S}(x_i \mid P_{nm}^{\delta}+x_{<i}) - \widehat{S}(x_i \mid P_m+x_{<i})}
  {\widehat{S}(x_i \mid x_{<i})}
  $$

  $$
  \text{membership score} = - \frac{1}{L} \sum_{i=1}^{L} \tilde{r}_i
  $$

* **实现细节**:

  * outer loop N 个扰动 prefix
  * 每个扰动 prefix 都需要新的 word-by-word sampling
  * 这是最大规模实验，应放在 Step 1 / Step 2 有正向结果之后

#### Step 4: 评估 & ablation

* ROC-AUC, TPR@FPR, FDR
* Ablation studies:

  1. 无 member prefix vs 引入 member prefix
  2. 原始 word-level score vs kernelized semantic density score
  3. 固定 prefix vs prefix perturbation ensemble
  4. 内层采样 M vs 外层扰动 N

### 实验顺序与 rationale

1. **Step 1: 引入 member prefix**。新的 `records.jsonl` 已经包含 raw, non-member prefix, member prefix 三路 sampling，是当前成本最低、最直接的改进方向。
2. **Step 2: 替换或校准 word-level score**。在 Step 1 框架有效后，再测试不同 semantic score / density score 是否能进一步提升稳定性。
3. **Step 3: 外层 prefix 扰动**。这是最大规模采样实验，主要用于验证鲁棒性和 prefix sensitivity，应最后进行。

> 该顺序从“完全复用现有缓存”开始，再进入“只改 postprocess/inference”，最后才进入“重新大规模调用目标 LLM”。这样可以最大化利用已有缓存，并逐步降低实验不确定性。

### 对于word-level scoring的thinking

最初提出的 word-level scoring 方向不是错误的，但它不够 strong 的地方在于：它把 embedding similarity 的相对排序包装成了 likelihood / probability，而这两者之间缺少真正的概率校准。

如果使用类似下面的形式：

```text
p(x_i | P + x_<i) ≈ softmax(cosine(real_word, sampled_word) / tau)
```

这个 softmax 实际回答的是：

```text
在这些 sampled words 里面，哪个和真实词最像？
```

但 MIA 更需要回答的是：

```text
模型在这个上下文下，到底有多倾向真实词或真实语义？
```

这两个问题并不等价。比如真实词是 `Conference`，如果采样词是 `tournament`, `championship`, `league`，它们整体都比较相关，分数高是合理的。但如果采样词是 `banana`, `window`, `quietly`，所有词都和真实词很远，softmax 仍然会在这些差候选里强行选出一个“相对最好”的词，并给它较高权重。这样会掩盖“整体都不像真实词”这个重要信号。

所以这个 scoring 的主要问题是：

```text
softmax 只保留候选内部的相对差异，削弱了绝对语义接近程度。
```

原始 SimMIA 的 score 反而更直接：

```text
score = sum frequency(sample_word) * similarity(real_word, sample_word)
```

它保留了“整体像不像”的绝对强度。如果所有 sampled words 都不像真实词，score 就自然低。

因此，如果要改进 word-level score，更合理的表述可能是 kernelized semantic density score，例如：

```text
S_i(P) = sum_j freq_j * exp(sim(real_word, sampled_word_j) / tau)
```

这个形式不强行把候选集合归一化成 1，而是保留“当前上下文下采样分布整体靠近真实词语义”的强弱。

一句话总结：

```text
原方案的问题是把“候选词之间谁更像真实词”当成了“模型有多支持真实词”；但 MIA 需要的是后者。
```

因此，在当前阶段，member-prefix contrastive 更值得优先测试。它不是只重新包装同一批 similarity，而是引入了一个新的对照条件，可能产生真正新的判别信号。
