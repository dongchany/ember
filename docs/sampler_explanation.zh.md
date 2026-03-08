# LLM 采样器 (Sampler) 完全解析

## 一、为什么需要采样？

### 1.1 LLM 的输出是什么？

语言模型的最后一层输出的不是"下一个词"，而是一个 **logits 向量**：

```
logits = [2.1, -0.5, 5.3, 1.2, ...]  // 长度 = vocab_size (比如 151936)
```

每个位置对应词表中的一个 token，数值表示模型认为该 token 作为下一个词的"合适程度"。

**关键问题**：logits 只是原始分数，不是概率。我们需要：
1. 把 logits 转成概率分布
2. 从概率分布中选出下一个 token

这就是 Sampler 的工作。

### 1.2 最简单的方案：Greedy（贪心）

直接选 logits 最大的那个：

```cpp
int argmax(const std::vector<float>& logits) {
    return std::max_element(logits.begin(), logits.end()) - logits.begin();
}
```

**问题**：输出完全确定，没有创造性，容易陷入重复。

### 1.3 更好的方案：随机采样

把 logits 转成概率，然后按概率随机选。这就引出了完整的采样流程。

---

## 二、完整采样流程

你的代码实现的流程是：

```
logits
   │
   ▼
┌─────────────────────────────┐
│  1. 惩罚机制 (Pre-softmax)   │  ← 抑制重复 token
│     - Repetition Penalty    │
│     - Presence Penalty      │
│     - Frequency Penalty     │
│     - No-repeat N-gram      │
└─────────────────────────────┘
   │
   ▼
┌─────────────────────────────┐
│  2. Temperature Scaling     │  ← 控制分布的"尖锐度"
└─────────────────────────────┘
   │
   ▼
┌─────────────────────────────┐
│  3. Softmax                 │  ← logits → 概率
└─────────────────────────────┘
   │
   ▼
┌─────────────────────────────┐
│  4. Top-K 过滤              │  ← 只保留 K 个最高概率
└─────────────────────────────┘
   │
   ▼
┌─────────────────────────────┐
│  5. Top-P 过滤              │  ← 只保留累积概率达 P 的
└─────────────────────────────┘
   │
   ▼
┌─────────────────────────────┐
│  6. 分类采样                │  ← 按概率随机选一个
└─────────────────────────────┘
   │
   ▼
token_id
```

下面逐个详解。

---

## 三、Softmax：从 logits 到概率

### 3.1 数学定义

$$P(i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

其中 $z_i$ 是第 $i$ 个 logit。

**性质**：
- 所有输出都在 (0, 1) 之间
- 所有输出之和 = 1（合法的概率分布）
- 保持相对顺序（logit 越大，概率越高）

### 3.2 数值稳定性问题

直接计算 $e^{z_i}$ 会溢出。比如 $e^{1000} = \infty$。

**解决方案**：减去最大值

$$P(i) = \frac{e^{z_i - z_{max}}}{\sum_{j=1}^{n} e^{z_j - z_{max}}}$$

数学上完全等价（分子分母同乘 $e^{-z_{max}}$），但数值稳定。

### 3.3 代码实现

```cpp
static void softmax(std::vector<float>& x) {
    // 找最大值，防止溢出
    float max_val = *std::max_element(x.begin(), x.end());
    
    float sum = 0;
    for (float& v : x) {
        v = std::exp(v - max_val);  // 减去 max，数值稳定
        sum += v;
    }
    
    // 归一化
    for (float& v : x) {
        v /= sum;
    }
}
```

### 3.4 直观理解

假设 logits = [1.0, 2.0, 3.0]：

```
e^1 ≈ 2.72
e^2 ≈ 7.39
e^3 ≈ 20.09
sum ≈ 30.20

P = [0.09, 0.24, 0.67]
```

logit 差 1，概率差约 2.7 倍（因为 $e^1 \approx 2.7$）。这个"放大效应"是 softmax 的核心特性。

---

## 四、Temperature：控制分布的尖锐度

### 4.1 数学定义

在 softmax 之前，把 logits 除以 temperature $T$：

$$P(i) = \frac{e^{z_i / T}}{\sum_{j=1}^{n} e^{z_j / T}}$$

### 4.2 直观理解

| Temperature | 效果 | 用途 |
|-------------|------|------|
| T → 0 | 概率集中在最大值，几乎等于 greedy | 确定性输出、事实问答 |
| T = 1 | 原始分布 | 默认值 |
| T > 1 | 分布变平，更随机 | 创意写作、脑暴 |

**数学解释**：

假设 logits = [1, 2, 3]：

```
T = 0.5 时: logits/T = [2, 4, 6]
            P ≈ [0.02, 0.12, 0.86]  ← 更尖锐

T = 1.0 时: logits/T = [1, 2, 3]
            P ≈ [0.09, 0.24, 0.67]  ← 原始

T = 2.0 时: logits/T = [0.5, 1, 1.5]
            P ≈ [0.19, 0.31, 0.50]  ← 更平坦
```

### 4.3 代码实现

```cpp
// 应用 temperature（在 softmax 之前）
if (config.temperature > 0 && config.temperature != 1.0f) {
    for (float& p : probs) {
        p /= config.temperature;
    }
}
```

### 4.4 为什么 T=0 是 Greedy？

当 $T \to 0$：

$$\lim_{T \to 0} \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}} = \begin{cases} 1 & \text{if } z_i = \max(z) \\ 0 & \text{otherwise} \end{cases}$$

所有概率都集中到最大值上。

---

## 五、Top-K 采样

### 5.1 动机

即使用了 temperature，词表中仍有几万个 token。很多低概率 token 虽然概率小，但累积起来可能被选中，导致输出"跑偏"。

**解决方案**：只考虑概率最高的 K 个 token。

### 5.2 算法

1. 找到第 K 大的概率值作为阈值
2. 把小于阈值的概率置 0
3. 重新归一化

### 5.3 代码实现

```cpp
static void top_k_filter(std::vector<float>& probs, int k) {
    if (k >= static_cast<int>(probs.size())) return;
    if (k <= 0) return;
    
    // 找到第 k 大的值（用 nth_element，O(n) 复杂度）
    std::vector<float> sorted = probs;
    std::nth_element(sorted.begin(), sorted.begin() + (k - 1), 
                     sorted.end(), std::greater<float>());
    float threshold = sorted[k - 1];
    
    // 过滤
    for (float& p : probs) {
        if (p < threshold) p = 0;
    }
    
    // 重新归一化（概率之和 = 1）
    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (sum > 0) {
        for (float& p : probs) p /= sum;
    }
}
```

### 5.4 nth_element 的妙用

`std::nth_element` 是部分排序算法：
- 把第 n 大的元素放到位置 n
- 左边的都 >= 它，右边的都 <= 它
- 平均 O(n)，比完整排序的 O(n log n) 快

### 5.5 示例

```
原始概率: [0.05, 0.30, 0.10, 0.40, 0.15]
K = 3

Top-3 是索引 1, 3, 4（概率 0.30, 0.40, 0.15）
阈值 = 0.15

过滤后: [0, 0.30, 0, 0.40, 0.15]
归一化: [0, 0.35, 0, 0.47, 0.18]
```

---

## 六、Top-P (Nucleus) 采样

### 6.1 Top-K 的问题

K 是固定的。但不同情况下，模型的"确定程度"不同：

- 情况 A："The capital of France is" → 模型很确定是 "Paris"，P(Paris) = 0.95
- 情况 B："I like to eat" → 很多词都可以，分布很平

用固定的 K：
- 情况 A：K=50 会引入太多不相关词
- 情况 B：K=50 可能还不够

### 6.2 Top-P 的思想

**动态选择**：选择最少的 token，使它们的累积概率 ≥ P。

这个 token 集合叫 **nucleus**（核），所以也叫 nucleus sampling。

### 6.3 算法

1. 按概率降序排列
2. 从高到低累加，直到累积概率 ≥ P
3. 保留这些 token，其余置 0
4. 重新归一化

### 6.4 代码实现

```cpp
static void top_p_filter(std::vector<float>& probs, float p) {
    // 创建 (索引, 概率) 对
    std::vector<std::pair<int, float>> indexed(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        indexed[i] = {static_cast<int>(i), probs[i]};
    }
    
    // 按概率降序排列
    std::sort(indexed.begin(), indexed.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // 累积概率，找截断点
    float cumsum = 0;
    size_t cutoff = indexed.size();
    for (size_t i = 0; i < indexed.size(); ++i) {
        cumsum += indexed[i].second;
        if (cumsum >= p) {
            cutoff = i + 1;  // 包含当前这个
            break;
        }
    }
    
    // 构建新的概率分布
    std::vector<float> new_probs(probs.size(), 0);
    for (size_t i = 0; i < cutoff; ++i) {
        new_probs[indexed[i].first] = indexed[i].second;
    }
    
    // 归一化
    float sum = std::accumulate(new_probs.begin(), new_probs.end(), 0.0f);
    if (sum > 0) {
        for (float& p : new_probs) p /= sum;
    }
    
    probs = std::move(new_probs);
}
```

### 6.5 示例

```
概率: [0.05, 0.30, 0.10, 0.40, 0.15]
P = 0.9

排序后: [(3, 0.40), (1, 0.30), (4, 0.15), (2, 0.10), (0, 0.05)]

累积:
  0.40 (< 0.9, 继续)
  0.40 + 0.30 = 0.70 (< 0.9, 继续)
  0.70 + 0.15 = 0.85 (< 0.9, 继续)
  0.85 + 0.10 = 0.95 (≥ 0.9, 停止)

保留索引: 3, 1, 4, 2
过滤后: [0, 0.30, 0.10, 0.40, 0.15]
归一化: [0, 0.316, 0.105, 0.421, 0.158]
```

### 6.6 Top-K vs Top-P

| 特性 | Top-K | Top-P |
|------|-------|-------|
| 参数 | 固定数量 K | 累积概率阈值 P |
| 自适应 | ❌ | ✅ |
| 典型值 | K = 40~100 | P = 0.9~0.95 |

**实践中常常同时使用**：先 Top-K 粗筛，再 Top-P 细筛。

---

## 七、惩罚机制：抑制重复

LLM 有个通病：喜欢重复自己说过的话。惩罚机制就是解决这个问题的。

**重要**：惩罚是在 **softmax 之前** 应用到 logits 上的。

### 7.1 Repetition Penalty（重复惩罚）

**论文来源**：[CTRL: A Conditional Transformer Language Model](https://arxiv.org/abs/1909.05858)

**数学**：

$$z_i' = \begin{cases} z_i / \alpha & \text{if } z_i > 0 \text{ and } i \in \text{history} \\ z_i \times \alpha & \text{if } z_i < 0 \text{ and } i \in \text{history} \\ z_i & \text{otherwise} \end{cases}$$

其中 $\alpha > 1$ 是惩罚系数。

**直觉**：
- 正 logit（模型想选的）→ 除以 α → 变小 → 概率降低
- 负 logit（模型不想选的）→ 乘以 α → 更负 → 概率更低

**效果**：出现过的 token 被抑制，惩罚系数越大抑制越强。

### 7.2 代码实现

```cpp
if (config.repetition_penalty > 1.0f) {
    for (const auto& item : counts) {
        float& logit = probs[item.first];
        if (logit < 0.0f) {
            logit *= config.repetition_penalty;  // 负数乘以 >1 的数，变得更负
        } else {
            logit /= config.repetition_penalty;  // 正数除以 >1 的数，变小
        }
    }
}
```

### 7.3 Presence Penalty（存在惩罚）

**来源**：OpenAI API

**数学**：

$$z_i' = z_i - \beta \cdot \mathbf{1}[i \in \text{history}]$$

其中 $\beta$ 是惩罚值，$\mathbf{1}[\cdot]$ 是指示函数（出现过为 1，否则为 0）。

**特点**：只要出现过，就减去固定值，不管出现几次。

### 7.4 Frequency Penalty（频率惩罚）

**数学**：

$$z_i' = z_i - \gamma \cdot \text{count}(i)$$

其中 $\gamma$ 是惩罚系数，$\text{count}(i)$ 是 token $i$ 在历史中出现的次数。

**特点**：出现次数越多，惩罚越重。

### 7.5 代码实现

```cpp
// 先统计每个 token 出现的次数
std::unordered_map<int, int> counts;
for (int token : history) {
    ++counts[token];
}

// 应用惩罚
for (const auto& item : counts) {
    float& logit = probs[item.first];
    
    // Presence: 出现过就减
    if (config.presence_penalty != 0.0f) {
        logit -= config.presence_penalty;
    }
    
    // Frequency: 按次数减
    if (config.frequency_penalty != 0.0f) {
        logit -= config.frequency_penalty * static_cast<float>(item.second);
    }
}
```

### 7.6 三种惩罚的对比

| 惩罚类型 | 公式 | 特点 | 典型值 |
|----------|------|------|--------|
| Repetition | 乘/除 | 相对调整，自适应 | 1.0~1.5 |
| Presence | 减固定值 | 只看有没有 | 0~2.0 |
| Frequency | 减 count×系数 | 出现越多惩罚越重 | 0~2.0 |

---

## 八、No-Repeat N-gram：禁止重复片段

### 8.1 动机

有时候惩罚机制不够，模型还是会重复整个短语，比如：

> "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat."

N-gram 禁止可以强制避免这种情况。

### 8.2 算法思想

**N-gram** = 连续 N 个 token 组成的片段。

如果设置 `no_repeat_ngram_size = 3`，我们要禁止任何 3-gram 重复。

**方法**：
1. 看历史中的最后 (N-1) 个 token 作为"前缀"
2. 扫描历史，找所有匹配这个前缀的位置
3. 把这些位置后面紧跟的 token 禁掉（设为极小值）

### 8.3 示例

```
历史: [5, 6, 7, 6, 7]
N = 3

最后 2 个 token（前缀）: [6, 7]

扫描历史，找 [6, 7] 出现的位置:
  位置 1-2: history[1]=6, history[2]=7 ✓
  这个 2-gram 后面跟着 history[3]=6

所以 token 6 被禁止（因为 [6,7,6] 会重复 [6,7,6]）
```

### 8.4 代码实现

```cpp
int ngram = config.no_repeat_ngram_size;
if (ngram > 1 && history.size() >= static_cast<size_t>(ngram)) {
    // 前缀的起始位置：最后 (ngram-1) 个 token
    size_t prefix_start = history.size() - static_cast<size_t>(ngram - 1);
    
    // 扫描历史中所有可能的 ngram 起始位置
    for (size_t i = 0; i + static_cast<size_t>(ngram) <= history.size(); ++i) {
        // 检查前 (ngram-1) 个是否匹配
        bool match = true;
        for (int j = 0; j < ngram - 1; ++j) {
            if (history[i + j] != history[prefix_start + j]) {
                match = false;
                break;
            }
        }
        
        // 如果匹配，禁止这个 ngram 的最后一个 token
        if (match) {
            int banned = history[i + ngram - 1];
            probs[banned] = -1e9f;  // 设为极小值，softmax 后约等于 0
        }
    }
}
```

### 8.5 为什么用 -1e9 而不是 0？

因为这是在 softmax **之前** 应用的，还是 logits 阶段。

- 如果设为 0：$e^0 = 1$，还是有概率
- 如果设为 -1e9：$e^{-10^9} \approx 0$，概率几乎为 0

---

## 九、分类采样（Categorical Sampling）

### 9.1 问题

现在我们有了概率分布 $P = [p_1, p_2, ..., p_n]$，如何按概率随机选一个？

### 9.2 算法：逆变换采样

1. 生成 [0, 1) 均匀随机数 r
2. 计算累积分布函数（CDF）
3. 找到第一个 CDF ≥ r 的位置

### 9.3 数学原理

累积分布函数：$F(i) = \sum_{j=1}^{i} p_j$

选择规则：选最小的 $i$ 使得 $F(i) \geq r$

**为什么正确？**

$P(\text{选中 } i) = P(F(i-1) < r \leq F(i)) = F(i) - F(i-1) = p_i$ ✓

### 9.4 代码实现

```cpp
int categorical_sample(const std::vector<float>& probs) {
    // 生成 [0, 1) 均匀随机数
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    
    // 累积并查找
    float cumsum = 0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (r < cumsum) {
            return static_cast<int>(i);
        }
    }
    
    // 数值误差兜底：找最后一个非零概率
    for (int i = static_cast<int>(probs.size()) - 1; i >= 0; --i) {
        if (probs[i] > 0) return i;
    }
    return 0;
}
```

### 9.5 图解

```
probs = [0.1, 0.3, 0.4, 0.2]

CDF:     0.1   0.4   0.8   1.0
         |     |     |     |
    0----+-----+-----+-----+----1
         ^     ^     ^     ^
     选0   选1   选2   选3

如果 r = 0.35，落在 [0.1, 0.4)，选 token 1
如果 r = 0.75，落在 [0.4, 0.8)，选 token 2
```

---

## 十、整体代码流程回顾

```cpp
int sample(const std::vector<float>& logits,
           const RuntimeConfig& config,
           const std::vector<int>& history) {
    
    std::vector<float> probs = logits;  // 复制一份，不修改原始数据
    
    // ========== 阶段 1: 惩罚 (Pre-softmax) ==========
    if (!history.empty()) {
        // 统计历史 token 出现次数
        std::unordered_map<int, int> counts;
        for (int token : history) {
            ++counts[token];
        }
        
        // Repetition penalty
        // Presence/Frequency penalty
        // No-repeat n-gram
    }
    
    // ========== 阶段 2: Temperature ==========
    if (config.temperature > 0 && config.temperature != 1.0f) {
        for (float& p : probs) {
            p /= config.temperature;
        }
    }
    
    // ========== 阶段 3: Softmax ==========
    softmax(probs);  // logits → 概率
    
    // ========== 阶段 4: Top-K ==========
    if (config.top_k > 0) {
        top_k_filter(probs, config.top_k);
    }
    
    // ========== 阶段 5: Top-P ==========
    if (config.top_p < 1.0f) {
        top_p_filter(probs, config.top_p);
    }
    
    // ========== 阶段 6: 采样 ==========
    if (config.temperature <= 0) {
        return argmax(probs);  // Greedy
    } else {
        return categorical_sample(probs);  // 随机
    }
}
```

---

## 十一、常见参数组合

### 11.1 确定性输出（代码生成、数学）

```cpp
temperature = 0.0  // 或很小如 0.1
top_k = 1          // 等价于 greedy
top_p = 1.0
```

### 11.2 平衡输出（通用对话）

```cpp
temperature = 0.7
top_k = 40
top_p = 0.9
repetition_penalty = 1.1
```

### 11.3 创意输出（故事、诗歌）

```cpp
temperature = 1.0~1.2
top_k = 100
top_p = 0.95
presence_penalty = 0.5  // 鼓励新词
```

### 11.4 避免重复的对话

```cpp
temperature = 0.8
top_p = 0.9
no_repeat_ngram_size = 3
frequency_penalty = 0.5
```

---

## 十二、你的实现可以优化的地方

### 12.1 Top-K 的边界情况

当有多个 token 概率相同且恰好在阈值上时，当前实现可能保留超过 K 个。

```cpp
// 改进：记录索引，精确保留 K 个
```

### 12.2 Top-P + Top-K 的顺序

你的实现是先 Top-K 再 Top-P，这是对的。因为：
- Top-K 先粗筛（快）
- Top-P 再细调（慢但精确）

### 12.3 性能优化

对于大词表（150k+），可以考虑：
- 使用堆（heap）做 Top-K，O(n log k) 代替 O(n)
- 避免多次遍历

---

## 十三、总结

| 步骤 | 作用 | 数学本质 |
|------|------|----------|
| Repetition Penalty | 抑制重复 token | 乘除 logits |
| Presence/Frequency | 抑制重复 | 减 logits |
| No-repeat N-gram | 禁止重复短语 | 设为 -∞ |
| Temperature | 控制随机性 | 缩放 logits |
| Softmax | logits → 概率 | 指数归一化 |
| Top-K | 粗筛 | 保留最大 K 个 |
| Top-P | 细筛 | 累积概率截断 |
| Categorical | 按概率选 | 逆变换采样 |

理解了这些，你就完全掌握了 LLM 文本生成的核心机制。
