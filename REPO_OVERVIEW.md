# SimMIA 仓库快速入门

这份文档给合作者和 Codex 快速了解本仓库用。它不替代论文或 README，只说明这个仓库的基础功能、怎么运行、几个脚本分别做什么，以及最常见的实验参数。

## 这个仓库是做什么的

SimMIA 是一个针对大语言模型的 membership inference attack 工具。简单说，它想判断一段文本更像是模型训练集中出现过的 member，还是没有出现过的 non-member。

这个仓库重点支持 black-box / text-only 场景：攻击者不需要模型 logits、loss 或内部状态，只使用模型生成出来的文本。

主入口命令是：

```bash
simmia.benchmark
```

这个命令由 `setup.py` 注册，实际运行逻辑在 `src/simmia/run.py`。

## 最常用运行方式

运行正式 SimMIA 的一个例子：

```bash
simmia.benchmark \
  --gpu_ids 0 1 2 3 4 5 6 7 \
  --model_name_or_path EleutherAI/pythia-6.9b \
  --sampling relative_word_by_word \
  --postprocess process_relative_word_data \
  --inference relative_semantic_ratio \
  --output_dir simmia_out \
  --num_samples 100 \
  --data SimMIA/WikiMIA-25 \
  --sub_dataset paper_subset \
  --num_shots 7 \
  --prefix_ratio 0.0 \
  --top_k 20
```

也可以直接用仓库里的脚本：

```bash
# 正式 SimMIA，也就是 soft semantic 版本
bash scripts/run_simmia_soft.sh EleutherAI/pythia-6.9b SimMIA/WikiMIA-25 paper_subset "0 1 2 3 4 5 6 7"

# SimMIA*，也就是 hard exact-label 版本
bash scripts/run_simmia_hard.sh EleutherAI/pythia-6.9b SimMIA/WikiMIA-25 paper_subset "0 1 2 3 4 5 6 7"

# SaMIA baseline
bash scripts/run_samia.sh EleutherAI/pythia-6.9b SimMIA/WikiMIA-25 paper_subset "0 1 2 3 4 5 6 7"
```

如果用闭源 API 模型，模型名要加 `api:` 前缀，例如：

```bash
--model_name_or_path api:google/gemini-2.5-flash
```

API 模型可以用 `--concurrency` 控制并发请求数量。

## 三个脚本的区别

### `scripts/run_simmia_soft.sh`

这是论文里的正式 SimMIA 配置。

它使用：

```bash
--sampling relative_word_by_word
--postprocess process_relative_word_data
--inference relative_semantic_ratio
```

它的核心想法是：逐词采样模型接下来会生成什么，然后用 embedding 语义相似度判断生成词和真实词有多接近。最后再比较正常上下文和 non-member prefix 扰动上下文下的相对变化。

这是最推荐先看的 SimMIA 实现。

### `scripts/run_simmia_hard.sh`

这是 SimMIA*，也可以理解成 hard matching 版本。

它同样使用 word-by-word sampling，但 inference 换成：

```bash
--inference relative_label_ratio
--smoothing
```

它不使用 embedding 语义相似度，而是看真实词本身在采样结果里出现的频率。`--smoothing` 会做简单平滑，减少真实词没有出现时的极端情况。

### `scripts/run_samia.sh`

这是 SaMIA baseline。

它使用：

```bash
--sampling generate_all_remaining
--inference rouge_n
--prefix_ratio 0.5
```

它不是逐词采样，而是把文本前 50% 当作 prefix，让模型生成后半段 continuation，再用 ROUGE-1 比较生成后缀和真实后缀是否相似。

## 核心实现流程

整个流程在 `src/simmia/run.py` 里串起来。以 `SimMIA/WikiMIA-25` 的 `paper_subset` 为例，它原始有 240 条 sequence，包含 120 条 member 和 120 条 non-member。

### 1. 数据和 fixed prefix

`src/simmia/benchmark.py` 会先把 member 和 non-member 分开，然后随机打乱。

如果设置：

```bash
--num_shots 7
```

代码会抽出：

```text
7 条 member sequence
7 条 non-member sequence
```

这 14 条 sequence 不再进入最终评估集。剩下：

```text
113 条 member
113 条 non-member
总共 226 条待检测 sequence
```

当前 SimMIA 实际会把 7 条 non-member sequence 拼成一个固定 prefix：

```text
fixed_prefix = nonmember_seq_1 + nonmember_seq_2 + ... + nonmember_seq_7
```

之后所有 226 条待检测 sequence 都共用这个 fixed non-member prefix。代码也会保存抽出来的 member prefix，但主流程里真正加到样本前面的，是 non-member prefix。

这个 prefix 一旦生成并写入 `prefix_data.json`，内容和拼接顺序都会固定。后续不加 `--overwrite`、不删除缓存、也不换输出目录时，代码会复用同一组 non-member sequence，并按同样顺序拼接。

论文里分析 prefix 随机性时，主要比较两种情况：

```text
Fixed Prefix   # 固定同一组 non-member prefix，只看 sampling seed 带来的波动
Random Prefix  # 每次重新随机选 non-member prefix，看 prefix 组成带来的波动
```

表中的均值和下标标准差通常来自 5 次完整实验的 AUC。例如随机选 5 组不同 prefix，各跑出一个 AUC，再计算这 5 个 AUC 的 mean 和 standard deviation。直观结论是：sampling seed 的影响较小，随机换 prefix 的影响更大。

### 2. 对每条 sequence 做两遍采样

正式 SimMIA 使用：

```bash
--sampling relative_word_by_word
```

对每条待检测 sequence，它会跑两遍 word-by-word sampling：

```text
原始上下文：sequence 自己
扰动上下文：fixed non-member prefix + sequence 自己
```

假设当前 sequence 是：

```text
The 2025 Mid-American Conference baseball tournament ...
```

代码会从第一个词开始，逐个位置预测真实的 next word。比如当前 prefix 是：

```text
The 2025 Mid-American
```

真实 next word 是：

```text
Conference
```

模型会在这个 prefix 下采样 `--num_samples` 次 short continuation。HF 模型实现里每次 continuation 最多生成 3 个 tokenizer tokens，然后后处理阶段取每个 continuation 的第一个 word。

如果设置：

```bash
--num_samples 100
--top_k 20
```

意思是：每个 word position 采样 100 次；每一步 token sampling 只从概率最高的 20 个 tokenizer token 里抽样。注意 `top_k` 限制的是 token，不是最终 word，所以 100 次采样得到的 first word 不一定最多只有 20 种，但通常会有很多重复。

采样结果会先写入 `records.jsonl`，主要包含：

```text
label_results          # 真实 next word 序列
sample_results         # 原始上下文下的采样统计
prefix_sample_results  # 加 fixed prefix 后的采样统计
```

### 3. Postprocess：把采样结果变成 word-level 统计

正式 SimMIA 使用：

```bash
--postprocess process_relative_word_data
```

这一步在 `src/simmia/postprocess/process_word.py`。它不会重新调用目标 LLM，而是读取 sampling 的结果，对每个 word position 做统计：

```text
真实 word
采样 first word 的频率分布
真实 word 和每个采样 word 的 embedding 语义相似度
```

soft SimMIA 会得到：

```text
sem_target / freq_target
prefix_sem_target / prefix_freq_target
```

hard SimMIA 会更关注：

```text
label_freq_target / prefix_label_freq_target
```

### 4. Inference：计算 membership score

正式 SimMIA 使用：

```bash
--inference relative_semantic_ratio
```

这一步在 `src/simmia/inference/relative.py`。它使用 postprocess 得到的 word-level 统计来算每个位置的分数：

```text
orig_score = semantic_similarity · sample_frequency
prefix_score = prefix_semantic_similarity · prefix_sample_frequency
ratio = prefix_score / orig_score
```

最后对一整条 sequence 的所有 word-level ratio 取平均：

```text
membership_score = -mean(ratio)
```

所有 sequence 都得到一个 score 后，`src/simmia/utils/eval.py` 会根据真实 label 计算 AUC、Accuracy、TPR@1%FPR、TPR@5%FPR、TPR@10%FPR，并保存 ROC 图。

### 5. 输出和缓存

输出目录通常长这样：

```text
simmia_out/WikiMIA-25/paper_subset/pythia-6.9b/7/
```

里面常见文件：

`full_dataset.jsonl`

本次实验真正用于评估的样本缓存。它不是原始 Hugging Face 数据全集，而是预处理后的待检测 sequence。以 `paper_subset` 和 `num_shots=7` 为例，这里会有 226 行；每一行包含待检测文本、真实 label，以及固定 non-member prefix：

```text
input   # 待检测文本
label   # 真实答案，1 是 member，0 是 non-member
prefix  # 固定的 non-member prefix
```

`prefix_data.json`

本次实验抽出来做 prefix bank 的样本，通常包含：

```text
member     # 抽出的 member sequence
nonmember  # 抽出的 non-member sequence
```

当前 SimMIA 主流程实际会把 `nonmember` 里的 sequence 拼成 fixed prefix。保存这个文件是为了下次运行时复用同一组 prefix，不重新随机抽。

`records.jsonl`

最重要、也最耗时生成的 sampling 缓存。每一行对应 `full_dataset.jsonl` 里的一条样本，并额外保存目标模型的采样统计：

```text
label_results          # 真实 next word 序列
sample_results         # 原始上下文下的 word-level 采样统计
prefix_sample_results  # 加 fixed prefix 后的 word-level 采样统计
```

后续的 postprocess 和 inference 都依赖这个文件；如果它已经完整存在，就不需要重新跑昂贵的 sampling。

`roc_tpr_at_5_fpr.png`

最终评估图。程序根据每条样本的 membership score 和真实 label 画 ROC 曲线，并标出 TPR@5%FPR。

这些文件会被复用。不加 `--overwrite` 时，代码会优先读取已有缓存；如果 `records.jsonl` 只完成了一部分，重新运行同一命令通常会从未完成的位置继续。

## 必须知道的超参数

`--model_name_or_path`

目标模型。Hugging Face 模型直接写模型名或本地路径；API 模型使用 `api:openai/...`、`api:google/...`、`api:anthropic/...`。

`--data`

选择数据集。当前支持：

```text
swj0419/WikiMIA
SimMIA/WikiMIA-25
iamgroot42/mimir
```

`--sub_dataset`

选择数据子集。例如 WikiMIA-25 可以用 `paper_subset`、`WikiMIA_length32`、`WikiMIA_length64`、`WikiMIA_length128`。

`--gpu_ids`

指定使用哪些 GPU，同时也决定本地模型采样和 embedding 后处理的并行 worker 数量。

`--sampling`

采样方法。正式 SimMIA 用 `relative_word_by_word`，SaMIA 用 `generate_all_remaining`。

`--postprocess`

后处理方法。正式 SimMIA 用 `process_relative_word_data`。SaMIA 不需要这个参数。

`--inference`

membership score 的计算方法：

```text
relative_semantic_ratio   # 正式 SimMIA soft 版本
relative_label_ratio      # SimMIA* hard 版本
rouge_n                   # SaMIA baseline
```

`--num_samples`

每个位置采样多少次。值越大结果通常越稳，但速度更慢、显存压力更大。

`--top_k`

控制采样时从 top-k token 中采样。SimMIA 脚本默认是 20，SaMIA 脚本默认是 50。

`--num_shots`

prefix bank 使用多少条样本。WikiMIA / WikiMIA-25 脚本默认 7，MIMIR 脚本默认 10。

`--prefix_ratio`

SaMIA 用它决定文本前多少比例作为 prefix，脚本默认 0.5。正式 SimMIA 通常设为 0.0。

`--embedding_model`

soft SimMIA 用来计算词语语义相似度的 sentence-transformers 模型，默认是：

```text
sentence-transformers/all-MiniLM-L6-v2
```

`--smoothing`

hard SimMIA 常用，对真实词频做平滑。

`--exact_match_number`

遇到数字任务时可以打开。脚本里对 MIMIR 的 `dm_mathematics` 会自动打开它。

`--overwrite`

是否覆盖已有缓存。这个仓库会复用 `records.jsonl`、`full_dataset.jsonl`、`prefix_data.json` 等缓存；如果改了关键参数但想重新跑，建议加 `--overwrite`。

## 给合作者的快速判断

如果想复现实验，优先看 `scripts/` 下面三个 bash。

如果想理解 SimMIA 方法本身，优先看：

```text
src/simmia/sampling/word_by_word.py
src/simmia/postprocess/process_word.py
src/simmia/inference/relative.py
```

如果想理解数据怎么被加载和缓存，优先看：

```text
src/simmia/benchmark.py
```

如果想理解命令行参数和主流程，优先看：

```text
src/simmia/options.py
src/simmia/run.py
```
