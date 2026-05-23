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

### 长 prefix / 大模型 sampling 的 KV cache 稳定性问题

在生成新的三路 sampling cache 时，我们遇到过两个和 sampling 执行稳定性相关的问题，主要出现在 LLaMA-13B / OPT-6.7B / GPT-NeoX-20B 这类较大模型、较长 WikiMIA length 或较长 fixed prefix 的组合上：

1. **CUDA OOM**: 原始实现默认一次 `generate(num_return_sequences=num_samples)`，当 `num_samples=100` 且 prompt/prefix 较长时，单次 generation batch 太大，容易显存不足。
2. **HF KV cache length mismatch**: 部分模型在长 prefix + batched generation + `past_key_values` 复用时会报 `Key and Value must have the same sequence length`。这不是 SimMIA / SimMIA* 的 scoring 逻辑问题，而是 HuggingFace generation cache 在失败或长上下文下可能进入不一致状态。

为了解决这个问题，我们对 `src/simmia/sampling/word_by_word.py` 做了最小稳定性改动：

* 新增可选运行参数 `sampling_batch_size`，仍然通过已有 `--params` 入口传入，例如：

```bash
--params sampling_batch_size:25
```

* `sampling_batch_size` 只控制单次 `generate` 的 chunk 大小，不改变总采样数。也就是说，`--num_samples 100` 仍然会为每个 word 采满 100 个样本，只是可能拆成多轮 `25 + 25 + 25 + 25` 或更小的 batch。
* 默认不传 `sampling_batch_size` 时保持旧行为：初始 batch size 等于 `num_samples`。
* 将以下错误视为 retryable：

```text
CUDA out of memory
Key and Value must have the same sequence length
```

* 当 retryable error 出现时，代码会：
  * 清理 CUDA cache；
  * 将 batch size 减半，例如 `100 -> 50 -> 25 -> 12 -> 6 -> 3 -> 1`；
  * 重建 fresh `transformers.DynamicCache()` 和 generation kwargs，避免复用可能已经损坏的 `past_key_values`；
  * 如果 batch size 已经降到 1 仍失败，则继续抛错，说明该组合需要更短 prefix 或更小上下文。

这个修复只影响 sampling 的执行方式和稳定性，不改变 SimMIA / SimMIA* 的算法定义。hard SimMIA* 仍使用 `relative_label_ratio`，soft SimMIA 仍使用 `relative_semantic_ratio`；当前复现仍只消费 `sample_results` 和 `nonmember_prefix_sample_results`，`member_prefix_sample_results` 仍作为后续 contrastive 方法的 raw cache。

迁移到其他机器继续跑 cache 时，建议按模型和数据集选择初始 `sampling_batch_size`：

* WikiMIA-25 的 Pythia-6.9B / Qwen3-8B-Base：`num_samples=10`，可以用 `sampling_batch_size:100`，实际会被截断成 10。
* WikiMIA 的 GPT-NeoX-20B：建议保守一些。length32/64 可以先用 `sampling_batch_size:2`，length128 建议直接用 `sampling_batch_size:1`。
* LLaMA-13B / 长 WikiMIA length：如果日志频繁出现 OOM 降 batch，重启时直接用稳定后的 batch，例如 length128 常见需要 3 左右。

### 改进方案设计

这一阶段的思路是：先尽量复用已经生成好的三路 `records.jsonl` cache，不急着重新调用 target LLM。原因很直接：member-prefix sampling 已经提前写进了 cache，所以现在最低成本的方向，是只新增 postprocess / inference，用已有 sampling 结果重新打分。

当前阶段只保留两个核心动作：

1. 引入 member prefix，并使用已经跑好的 SimMIA / SimMIA* 三路 sampling cache；
2. 基于已有 cache 重新计算 semantic-kernel likelihood score。

暂时不做 prefix perturbation。也就是说，这一阶段不考虑 prefix 顺序打乱、deletion、paraphrase 或 ensemble。

#### Step 1: 构建三路 sampling cache

对于待检测文本 \(x=(x_1,\ldots,x_L)\)，在每个 word position \(i\)，我们考虑三种 context：

$$
c_i^0=x_{<i},\quad
c_i^{nm}=P_{nm}\oplus x_{<i},\quad
c_i^{m}=P_m\oplus x_{<i}.
$$

这里 \(P_{nm}\) 是由 non-member sequences 构成的 prefix，\(P_m\) 是由 member sequences 构成的 prefix，\(x_{<i}\) 是真实 word \(x_i\) 前面的文本。

现在 `records.jsonl` 已经保存了这三路 sampling result：

* `sample_results`: 对应 \(c_i^0\)
* `nonmember_prefix_sample_results`: 对应 \(c_i^{nm}\)
* `member_prefix_sample_results`: 对应 \(c_i^m\)
* `label_results`: 真实 target words \(x_i\)

所以后续实验不需要重新调用 target LLM，只需要读取已有 `records.jsonl` 做 post-processing。

#### Step 2: Semantic-kernel likelihood score

原始 gray-box log likelihood 可以写成：

$$
LL(x)=\sum_{i=1}^{L}\log p(x_i\mid x_{<i}).
$$

在 black-box setting 里，我们拿不到真实 token probability，所以这里用 sampling 结果估计一个 word-level semantic probability，再取 log，最后聚合成 text-level pseudo log-likelihood。

对于每个 context type \(q\in\{0,nm,m\}\)，设第 \(j\) 次 sampling 得到的 word 是 \(\hat{x}_{i,j}^{q}\)。用 semantic kernel 估计真实 word \(x_i\) 附近的 probability mass：

$$
\widehat{p}_i^q
=
\frac{1}{M}
\sum_{j=1}^{M}
K_\tau(x_i,\hat{x}_{i,j}^{q}).
$$

本阶段只使用下面这个 semantic kernel：

$$
K_\tau(x_i,\hat{x})
=
\exp
\left(
\frac{\cos(e(x_i),e(\hat{x}))-1}{\tau}
\right).
$$

其中 \(e(\cdot)\) 是 word embedding，\(\tau\) 是 temperature，\(M\) 是每个 word 的 sampling 次数。

然后对每个 word-level probability estimate 取 log：

$$
\widehat{\ell}_i^q
=
\log(\epsilon+\widehat{p}_i^q).
$$

这里 \(\epsilon\) 是 smoothing constant，用来避免数值为 0。

接着把 word-level score 聚合成三种 text-level log-likelihood：

$$
\widehat{L}^{0}(x)
=
\sum_{i=1}^{L}\widehat{\ell}_i^0,\quad
\widehat{L}^{nm}(x)
=
\sum_{i=1}^{L}\widehat{\ell}_i^{nm},\quad
\widehat{L}^{m}(x)
=
\sum_{i=1}^{L}\widehat{\ell}_i^{m}.
$$

\(\widehat{L}^{0}(x)\) 是原始 context 下的 log-likelihood，\(\widehat{L}^{nm}(x)\) 是 non-member prefix 下的 conditional log-likelihood，\(\widehat{L}^{m}(x)\) 是 member prefix 下的 conditional log-likelihood。

最后使用 sequence-level likelihood 计算 membership score：

$$
S_{\mathrm{WPMIA}}(x)
=
\frac{
\widehat{L}^{nm}(x)
-
\gamma \widehat{L}^{m}(x)
}{
\widehat{L}^{0}(x)
}.
$$

默认 \(\gamma=1\)，也就是：

$$
S_{\mathrm{WPMIA}}(x)
=
\frac{
\widehat{L}^{nm}(x)
-
\widehat{L}^{m}(x)
}{
\widehat{L}^{0}(x)
}.
$$

这里暂时不使用 per-word ratio：

$$
\frac{1}{L}
\sum_{i=1}^{L}
\frac{
\widehat{\ell}_i^{nm}
-
\gamma \widehat{\ell}_i^{m}
}{
\widehat{\ell}_i^{0}
}.
$$

原因是 per-word ratio 不稳定，并且会破坏 log-likelihood 的 additive structure。因此这一版只在 text-level 做 normalization。

#### Implementation

对每条 record，流程是：

1. 读取 `label_results` 作为 target words。
2. 读取三路 samples：`sample_results`、`nonmember_prefix_sample_results`、`member_prefix_sample_results`。
3. 对每个 word 和每个 context type 计算 \(\widehat{p}_i^q\)。
4. 计算 \(\widehat{\ell}_i^q=\log(\epsilon+\widehat{p}_i^q)\)。
5. 聚合得到 \(\widehat{L}^{0}(x)\)、\(\widehat{L}^{nm}(x)\)、\(\widehat{L}^{m}(x)\)。
6. 计算 \(S_{\mathrm{WPMIA}}(x)\)。
7. 基于 score 计算 AUC 和 TPR@5%FPR。

#### Hyperparameters

默认设置：

* \(\tau=0.1\)
* \(\epsilon=10^{-8}\)
* \(\gamma=1.0\)

可选 sweep：

$$
\tau\in\{0.03,0.05,0.1,0.2,0.5\},
$$

$$
\gamma\in\{0,0.25,0.5,0.75,1.0\}.
$$

#### Evaluation

主要汇报两个指标：

* AUC
* TPR@5%FPR

#### Ablation

所有需要新的 `records.jsonl` 的 ablation 结果，都放在新的 log 文件夹下。本阶段只做以下 ablation。

1. SimMIA vs WPMIA

比较原始 SimMIA score 和新的 likelihood-based score。

原始 SimMIA 使用 raw semantic similarity ratio。

WPMIA 使用：

$$
S_{\mathrm{WPMIA}}(x)
=
\frac{
\widehat{L}^{nm}(x)
-
\gamma \widehat{L}^{m}(x)
}{
\widehat{L}^{0}(x)
}.
$$

2. Without member prefix vs With member prefix

不使用 member prefix：

$$
S_{\mathrm{nm}}(x)
=
\frac{
\widehat{L}^{nm}(x)
}{
\widehat{L}^{0}(x)
}.
$$

使用 member prefix：

$$
S_{\mathrm{con}}(x)
=
\frac{
\widehat{L}^{nm}(x)
-
\gamma \widehat{L}^{m}(x)
}{
\widehat{L}^{0}(x)
}.
$$

3. Gamma sweep

固定 \(\tau\)，改变：

$$
\gamma\in\{0,0.25,0.5,0.75,1.0\}.
$$

汇报 AUC 和 TPR@5%FPR。

4. Tau sweep

固定 \(\gamma\)，改变：

$$
\tau\in\{0.03,0.05,0.1,0.2,0.5\}.
$$

汇报 AUC 和 TPR@5%FPR。

#### Current Scope

本阶段只验证一个核心问题：

> Can semantic-kernel pseudo log-likelihood and member/non-member prefix contrast improve SimMIA under the same black-box sampling cache?
