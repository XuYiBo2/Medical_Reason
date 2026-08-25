# MedReason-RLVR — SPEC

> **Offline Difficulty-Aware GRPO with a DAPO-Inspired Objective for Medical Reasoning**

| Item | Specification |
|---|---|
| Version | **5.0.0 — FINAL** |
| Project type | Personal resume / research-oriented LLM post-training project |
| Base model | `Qwen/Qwen3-8B-Base` |
| TRL baseline | `trl==1.10.0` |
| Pipeline | QLoRA SFT → GRPO Smoke → Protocol Freeze → Difficulty Scan → Random/Difficulty GRPO → Eval |
| Primary benchmark | MedQA USMLE 4-options |
| Hardware target | 1 × NVIDIA GPU, 32 GB VRAM |
| Seed | `42` |
| Status | **Frozen development baseline** |

优先级：

$$
\boxed{
\text{Algorithm Correctness}
>
\text{Experiment Fairness}
>
\text{Observable Evidence}
>
\text{Reproducibility}
>
\text{Code Simplicity}
}
$$

本项目不建设企业级训练平台。除非本 SPEC 明确要求，否则不新增算法、基础设施或消融实验。

---

## 1. Project Definition

### 1.1 Goal

```text
MedMCQA
   ↓
4-bit QLoRA SFT
   ↓
GRPO Smoke
   ↓
Protocol Freeze
   ↓
MedQA Offline Difficulty Scan
   ↓
Informative Pool / Random Control
   ↓
E2 Random GRPO  vs  E3 Difficulty-Aware GRPO
   ↓
Persistence Diagnostic
   ↓
MedQA / MedMCQA / MMLU-Pro Health Evaluation
```

实现要求：

1. 用 MedMCQA 专家解释数据完成 SFT 冷启动；
2. 正式 scan 前跑通 GRPO，并冻结 rollout / reward / training protocol；
3. 用冻结的 SFT policy 对 MedQA train 多次 rollout，估计 initial-policy pass rate；
4. 构造 informative pool 与等规模 random control；
5. E2/E3 从完全相同的 SFT checkpoint 独立训练；
6. 比较 zero-variance group density 与 downstream performance；
7. 用 RL-held-out probe 诊断 initial informativeness 的 persistence。

所有流程必须通过配置与脚本执行，不依赖 Notebook 或手工修改中间结果。

### 1.2 Research Questions

主问题：

> **Can policy-local prompt informativeness estimated before RL predict which prompts will continue to yield informative task-success groups during subsequent GRPO updates?**

第二层问题：

> 在相同 GRPO protocol 下，offline informative-prompt selection 能否相对 random sampling 提高 informative-group density，并进一步改善医学 MCQA 表现？

### 1.3 Hypotheses

**H1 — Persistence**

$$
P(I_t=1\mid I_0=1)
>
P(I_t=1\mid I_0=0).
$$

**H2 — Informative-Group Yield**

E3 相对 E2 降低：

```text
frac_reward_zero_std
```

**H3 — Optimization Effect**

更高 informative-group density 应首先提高有效 RL signal 利用率；是否提高 final performance 由实验决定。

### 1.4 Policy-Local Difficulty

$$
p_\theta(x)
=
P_{\pi_\theta}(r_{\mathrm{task}}=1\mid x),
$$

$$
\hat p_\theta(x)
=
\frac{1}{G}
\sum_{j=1}^{G}r_{\mathrm{task},j}.
$$

本项目中的 difficulty 仅表示：

> **policy-local empirical task-success difficulty**

不是题目的固有医学难度。

使用 binary task-success reward：

$$
r_{\mathrm{task}}\in\{0,1\}.
$$

若 rollout 近似为 Bernoulli samples：

$$
P_{\mathrm{zero}}(p)=p^G+(1-p)^G,
$$

$$
P_{\mathrm{info}}(p)=1-p^G-(1-p)^G.
$$

因此筛选目标是 frozen initial policy 下产生 non-degenerate task-success outcomes 的 prompts。

### 1.5 Method Boundary

```text
Offline Difficulty-Aware GRPO
│
├── Binary Verifiable Task-Success Reward
├── Initial-Policy Informativeness Prior
│   └── offline prompt selection
└── DAPO-Inspired Objective
    ├── token-level loss normalization
    ├── Clip-Higher
    └── truncated-completion masking
```

本项目**不是完整 DAPO**，不实现 online Dynamic Sampling。E2/E3 使用相同 DAPO-inspired objective，主要变量只有 prompt selection。

不得声称：

- Offline Scan 等价于 DAPO Dynamic Sampling；
- initial difficulty 是静态 ground truth；
- zero-variance group 减少必然提高 final performance；
- binary verifier 能验证 reasoning 文本本身；
- 未计入 scan/probe overhead 时 E3 具有更低 end-to-end compute cost。

### 1.6 Non-Goals

不包含：开放式临床诊断、真实患者数据、RAG、Agent/Tool Use、PPO/Value Model、Reward Model、LLM Judge、OPD、online Dynamic Sampling、多卡训练、Web/API、MLflow、CI/CD、GPU CI、复杂 CLI、大规模近似去重。

---

## 2. Experiment Design

### 2.1 Models

| ID | Model | Prompt Selection | RL Objective |
|---|---|---|---|
| E0 | Base | — | — |
| E1 | SFT-only | — | — |
| E2 | Random GRPO | Uniform random | DAPO-inspired |
| E3 | Difficulty-Aware GRPO | Initial informative pool | DAPO-inspired |

### 2.2 Controlled Comparison

E2 vs E3 必须保持：

- 相同 immutable SFT initialization；
- 相同 trainable SFT LoRA continuation；
- 相同 pool size；
- 相同 optimizer steps；
- 相同 fresh prompt-group / rollout budget；
- 相同 reward；
- 相同 `num_generations` / `num_iterations`；
- 相同 loss / clipping / truncation masking；
- 相同 LoRA / optimizer / learning rate；
- 相同 generation config；
- 相同 checkpoint / dev-selection protocol。

唯一主要变量：

```text
prompt selection strategy
```

Scan trajectories 不得复用进 GRPO。

### 2.3 Persistence Diagnostic

Persistence Probe 是 secondary diagnostic，不改变训练过程。Probe prompts 参与 initial scan，但从不参与 E2/E3 gradient updates。

---

## 3. Data, Prompt and Reward Contract

### 3.1 Dataset Roles

| Dataset | Split | Purpose |
|---|---|---|
| MedMCQA | train | SFT |
| MedMCQA | validation A | SFT dev |
| MedMCQA | validation B | final retention eval |
| MedQA | train | scan + probe source + E2/E3 pool source |
| MedQA | dev | RL checkpoint selection |
| MedQA | test | final main eval |
| MMLU-Pro | Health test | final OOD eval |

Default sources：

```text
openlifescienceai/medmcqa
jind11/MedQA
TIGER-Lab/MMLU-Pro
```

`configs/data.yaml` 必须记录 dataset ID、revision、split。Raw question banks 不提交 Git。

### 3.2 Internal Schema

```json
{
  "id": "string",
  "source": "string",
  "split": "string",
  "question": "string",
  "options": {"A": "text", "B": "text"},
  "answer": "A",
  "explanation": null,
  "subject": null
}
```

要求：ID 唯一；question/options 合法；answer 属于 option labels；SFT explanation 非空；MMLU-Pro 支持 A-J；train 与 final eval 做 exact `question + options` dedup。仅实现 exact dedup。

### 3.3 MedMCQA SFT Set

保留：

- `choice_type == "single"`；
- 四个选项完整；
- gold 可映射 A-D；
- explanation 非空且 `<=160` tokens；
- `prompt + completion + EOS <=1024` tokens。

仅删除 explanation 开头明确 gold leakage：

```text
Ans. is A
Answer: B
Correct answer is C
The correct option is D
```

不得重写医学内容。

```yaml
sft_train_samples: 12000
sft_dev_samples: 512
medmcqa_eval_samples: 1000
seed: 42
```

SFT train 按 subject 分层采样；dev 与 final MedMCQA eval 不重叠。

### 3.4 MedQA Scan Universe

```yaml
initial_scan_samples: 1000
scan_expand_step: 500
seed: 42
```

- 只使用 MedQA train；
- dev/test 不进入 scan、probe、training pools；
- 不足时每次追加 500 个未扫描 prompts；
- 全部扩展 scan 使用同一 frozen protocol。

正式 pool size：

$$
N_{\mathrm{pool}}
=
\max\left(256,N_{\mathrm{fresh,prompt}}^{\mathrm{planned}}\right).
$$

`planned_fresh_prompt_groups` 必须在 GRPO smoke / freeze 阶段根据 locked TRL sampler semantics 解析，不得仅依据 `max_steps` 猜测。

### 3.5 Shared Renderer

所有阶段调用：

```python
render_prompt(sample) -> str
```

冻结：

```yaml
prompt_serialization: plain_text_v1
use_chat_template: false
decode_skip_special_tokens: true
```

SFT / Scan / GRPO / Probe / Eval 禁止使用不同 serialization。

### 3.6 Prompt

```text
You are solving a medical multiple-choice exam question.
Reason briefly and choose the single best option.

Question:
{question}

Options:
{rendered_options}

End the response with exactly one final answer tag:
<answer>X</answer>
```

保持原始 option 顺序；MedQA/MedMCQA 支持 A-D，MMLU-Pro 支持 A-J。

### 3.7 SFT Completion / Length Contract

```text
{cleaned_explanation}

<answer>{gold_label}</answer>
```

completion 末尾追加 tokenizer `eos_token`。

要求：

- 不使用 `<reasoning>` tag；
- 预处理时用实际 tokenizer 计算长度；
- `prompt + completion + EOS <=1024`；
- 不依赖 trainer truncation 截掉 answer tag；
- decode 使用 `skip_special_tokens=True`；
- Scan/GRPO/Probe/Eval 满足 `prompt_tokens + frozen_max_completion_length <= context_limit`；
- 禁止 silent prompt truncation。

### 3.8 Parser / Format Gate

```python
parse_final_answer(
    text: str,
    allowed_labels: set[str],
) -> str | None
```

合法输出当且仅当：

1. 恰好一个完整 `<answer>...</answer>`；
2. tag 内只有一个 allowed label；
3. answer tag 是最后一个非空内容。

**不要求 answer tag 前存在文本。**

```text
Valid:   Reasoning... <answer>B</answer>
Valid:   <answer>B</answer>
Invalid: B
Invalid: Answer: B
Invalid: <answer>B</answer> extra text
Invalid: <answer>B</answer><answer>B</answer>
```

### 3.9 Binary Verifiable Task-Success Reward

$$
r_{\mathrm{task}}(x,y)
=
\begin{cases}
1,&\operatorname{parse}(y)=y^*,\\
0,&\text{otherwise}.
\end{cases}
$$

Reward=1 表示：

```text
valid answer contract AND correct gold option
```

禁止：standalone format reward、duplicate penalty、reasoning-length reward、minimum-reasoning gate、truncation reward penalty、subject-dependent reward、LLM Judge、KL penalty as reward。

### 3.10 Diagnostics / Evaluator Consistency

只做诊断：

```text
pre_answer_tokens_mean
direct_answer_rate
```

$$
\mathrm{direct\_answer\_rate}
=
\Pr(\mathrm{pre\_answer\_tokens}<8).
$$

`<answer>B</answer>` 可合法 reward=1，同时记为 direct-answer case。

以下阶段必须共用 renderer、parser、`task_success()`：

```text
Formal Scan
E2/E3 Training
Persistence Probe
Checkpoint Evaluation
Final Evaluation
```

统一指标：

```text
format_rate = fraction(parser succeeds)
strict_task_success_accuracy = mean(r_task)
```

主结果不得使用另一套 lenient parser。

---

## 4. SFT

### 4.1 Model / Quantization / LoRA

```yaml
model: Qwen/Qwen3-8B-Base

load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
bnb_4bit_compute_dtype: bfloat16

lora_r: 16
lora_alpha: 32
lora_dropout: 0.0
bias: none
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

Base weights 冻结。

### 4.2 Training

```yaml
max_length: 1024
completion_only_loss: true
packing: false

per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-4
num_train_epochs: 1
warmup_ratio: 0.03
lr_scheduler_type: cosine
weight_decay: 0.0

bf16: true
gradient_checkpointing: true
max_grad_norm: 1.0
seed: 42
```

Loss mask：

```text
prompt / padding → ignore
completion / EOS → train
```

Smoke 中抽查 token labels，确认 prompt positions 为 `-100`，answer tag 与 EOS 未被 mask。

### 4.3 SFT Smoke / Full

Smoke：

```yaml
samples: 32
max_steps: 5
```

验证：no OOM、finite loss、LoRA updated、generation works、adapter save/reload works。

Full：12,000 samples，1 epoch。

Outputs：

```text
outputs/sft/adapter/
outputs/sft/train_metrics.json
outputs/sft/dev_predictions.jsonl
outputs/sft/dev_metrics.json
```

Gate：Full 正常结束；immutable adapter 可重载；`format_rate >=95%`；未访问 MedQA test。

若 format 不达标，仅排查 loss mask、prompt/completion、sequence truncation、generation termination；不切换 Instruct fallback model。

---

## 5. GRPO Protocol and Freeze

### 5.1 Initialization

使用 locked `TRL GRPOTrainer`，不自行重写完整 trainer。

E2/E3 都执行：

```text
quantized base
→ load outputs/sft/adapter/
→ is_trainable=True
→ continue updating the same SFT LoRA
```

禁止 fresh RL LoRA；禁止 E2/E3 相互初始化。

### 5.2 Candidate Config Before Freeze

```yaml
learning_rate: 1.0e-6
max_steps: 300

per_device_train_batch_size: 1
gradient_accumulation_steps: 4
steps_per_generation: 4
num_generations: 4
num_iterations: 2

max_completion_length: 256
temperature: 0.8
top_p: 0.95

loss_type: dapo
epsilon: 0.20
epsilon_high: 0.28
beta: 0.0
scale_rewards: group
mask_truncated_completions: true

use_vllm: false
bf16: true
gradient_checkpointing: true
max_grad_norm: 1.0
save_strategy: steps
save_steps: 100
save_total_limit: 3
seed: 42
```

### 5.3 Objective / KL / Truncation

必须保持：

```yaml
loss_type: dapo
epsilon: 0.20
epsilon_high: 0.28
num_iterations: 2
beta: 0.0
mask_truncated_completions: true
```

含义：token-level loss normalization + Clip-Higher + repeated policy iteration + truncated-completion masking。

KL 边界：

```text
Task Reward != Policy Regularization
```

主实验 `beta=0.0`，不加载 reference model；KL 不进入 reward；不做非零 KL MVP ablation。

Truncated completion：reward 仍记录；truncation 状态记录；不参与 policy loss；不加 reward penalty。

### 5.4 Generation Semantics

单进程默认：

$$
B_{\mathrm{gen}}
=1\times1\times4=4.
$$

默认 `G=4`：

```text
fresh prompt groups / generation batch = 1
fresh rollouts / generation batch = 4
```

由于 `num_iterations=2`：

```text
optimizer_steps != fresh_prompt_groups != fresh_rollouts
```

必须从真实 generation events 记录：

```text
optimizer_steps
fresh_generation_batches
fresh_prompt_groups
fresh_rollouts
fresh_generated_completion_tokens
```

不得用 `max_steps` 推算 fresh rollout 数。

### 5.5 GRPO Smoke

```yaml
samples: 32
max_steps: 5
```

从 immutable SFT adapter 的 disposable copy 开始，写入 `outputs/smoke/`，不得覆盖 SFT adapter。

验证：

- renderer / group shape / generation count；
- reward ∈ `{0,1}`；
- direct-answer valid case；
- finite loss / gradients；
- same SFT LoRA updated；
- no fresh LoRA；
- DAPO loss / clipping / truncation mask；
- fresh counters；
- no OOM；
- format / truncation 无明显异常。

资源调整仅在 freeze 前允许：

```text
max_completion_length:
256
├── truncation high + memory available → 320
└── OOM → 192 → 160

G:
4 → 2 only if necessary after length reduction

learning_rate:
1e-6 → 3e-6 / 5e-6 only if updates are negligible
```

任何调整后重新 smoke。通过后丢弃 smoke adapter，重新加载 immutable $\theta_0$。

### 5.6 Protocol Freeze

生成：

```text
outputs/protocol/frozen_protocol.yaml
```

至少冻结：

```text
base / tokenizer revision
quantization config
SFT adapter identity
prompt serialization / prompt version
EOS / decode / length contract
parser / reward version
G / num_generations
temperature / top_p
max_completion_length
LoRA config
learning_rate / optimizer
steps_per_generation / num_iterations
loss_type / epsilon / epsilon_high / beta / scale_rewards
mask_truncated_completions
max_steps
planned_fresh_prompt_groups
planned_fresh_rollouts
checkpoint / dev-selection protocol
```

`planned_fresh_*` 必须根据 locked TRL sampler semantics 与 smoke 解析为 numeric values。

### 5.7 Invalidation Rules

**A. Rollout-affecting change**：$\theta_0$、prompt、parser/reward、G、temperature/top-p、completion cap、model/tokenizer revision、generation-affecting quantization。

```text
invalidate Scan + p0 + Probe + Pools + E2 + E3
→ smoke
→ freeze
→ rescan
→ rebuild probe/pools
→ rerun E2/E3 from θ0
```

若同时影响 SFT contract/data/tokenization，则先重新 SFT。

**B. Training-only change**：learning rate/optimizer、loss/clipping/beta/scale_rewards、num_iterations、steps_per_generation、max_steps/budget。

```text
Scan/p0 may remain valid
E2 + E3 must both rerun from θ0
```

若该变化导致 `planned_fresh_prompt_groups / planned_fresh_rollouts` 变化，则必须重新计算 `N_pool` 并重建 `D_info / D_random`；已有 Formal Scan 覆盖不足时，仅按原 frozen rollout/reward protocol 扩大 Scan，无需重算已有 `p_0`。

LoRA rank / alpha / target_modules 等 SFT LoRA architecture 变化不属于 training-only change；必须重新 SFT，并按 Rollout-affecting change 处理。

**C. Evaluation-only change**：evaluation metric/implementation bug。

```text
no retraining
re-evaluate all affected E0/E1/E2/E3
```

任何 frozen-protocol 修复不得 branch-local patch。

---

## 6. Offline Difficulty Scan and Pools

### 6.1 Formal Scan

Formal Scan 只能在 Protocol Freeze 后运行。

Input：

```text
Frozen SFT θ0
+
Frozen rollout/reward protocol
+
MedQA train
```

读取：

```yaml
num_rollouts: <frozen G>
temperature: <frozen temperature>
top_p: <frozen top_p>
max_new_tokens: <frozen max_completion_length>
```

### 6.2 Initial Pass Rate

$$
\hat p_0(x)
=
\frac{1}{G}\sum_{j=1}^{G}r_{\mathrm{task},j},
$$

$$
I_0(x)=\mathbf 1[0<\hat p_0(x)<1].
$$

$\hat p_0$ 唯一来自 Formal Scan。不得使用 smoke、Persistence step-0 re-rollout 或 Full GRPO first batch 替代。

默认 `G=4`：

```text
p_hat ∈ {0, 0.25, 0.50, 0.75, 1.00}
```

### 6.3 Persistence Probe Sampling

Formal Scan 获得 $I_0$ 后：

```yaml
persistence_probe_target: 256
probe_per_stratum_target: 128
probe_seed: 42
```

默认从：

```text
I0 = 1 → 128
I0 = 0 → 128
```

uniform sample。

要求：probe IDs 冻结；E2/E3 使用相同 IDs；probe 从 training pools 完全排除；step 0 直接读取 scan 结果；probe 不执行梯度。

任一 stratum 不足 128 时继续扩 scan。完整 MedQA train 后仍不足：两组对称取可用最小值；若每组 `<64`，Persistence 标记 exploratory / low-support。

### 6.4 Informative Pool

Eligible set：

$$
\mathcal D_{\mathrm{eligible}}
=
\mathcal D_{\mathrm{scan}}\setminus\mathcal D_{\mathrm{probe}}.
$$

Informative candidates：

$$
\mathcal D_{\mathrm{info,cand}}
=
\{x\in\mathcal D_{\mathrm{eligible}}:0<\hat p_0(x)<1\}.
$$

最终：

$$
|\mathcal D_{\mathrm{info}}|=N_{\mathrm{pool}}.
$$

多于 `N_pool` 时固定 seed uniform sample；不足则扩 scan。不使用 `4p(1-p)` 或其他 difficulty weighting。

### 6.5 Random Control

从同一 eligible set：

$$
|\mathcal D_{\mathrm{random}}|
=
|\mathcal D_{\mathrm{info}}|
=
N_{\mathrm{pool}}.
$$

Uniform random，固定 seed，不依据 $\hat p_0$ 过滤；允许与 `D_info` 自然重叠。

记录：

```text
pool_overlap_count
pool_overlap_ratio
```

### 6.6 Pool Gate

Scan 扩展直到同时满足：

```text
probe strata satisfy Section 6.3
after probe removal: |D_info,cand| >= N_pool
```

禁止通过提高 temperature、改 G、放宽 `0<p_hat<1`、混入 MedMCQA、使用 probe prompts 来补 pool。

完整 train 后仍不足：先检查 parser/reward/truncation/serialization；无 bug 后允许停止 Full GRPO 并保留负结果。

### 6.7 Trajectory Boundary / Outputs

Scan trajectories 只用于 $p_0$、$I_0$、probe/pools 和 diagnostics；不得作为 GRPO training trajectories。

Outputs：

```text
outputs/difficulty/scan.jsonl
outputs/difficulty/probe_ids.json
outputs/difficulty/informative_pool.jsonl
outputs/difficulty/random_pool.jsonl
outputs/difficulty/summary.json
```

`scan.jsonl` 至少保存：

```text
sample_id
num_rollouts
correct_count
pass_rate
reward_std_population (ddof=0)
is_informative
format_rate
truncation_rate
mean_completion_tokens
```

`summary.json` 保存 frozen protocol identity、scan counts/tokens、pool sizes、overlap statistics。

---

## 7. Full GRPO Controlled Experiment

### 7.1 Branches / Config

```text
E2 = Random GRPO + DAPO-inspired objective
E3 = Difficulty-Aware GRPO + same DAPO-inspired objective
```

两者从同一 immutable SFT adapter 重新加载并设置 trainable，直接读取 `frozen_protocol.yaml`。

默认候选 frozen config：

```yaml
learning_rate: 1.0e-6
max_steps: 300
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
steps_per_generation: 4
num_generations: 4
num_iterations: 2
max_completion_length: 256
temperature: 0.8
top_p: 0.95
loss_type: dapo
epsilon: 0.20
epsilon_high: 0.28
beta: 0.0
scale_rewards: group
mask_truncated_completions: true
use_vllm: false
bf16: true
gradient_checkpointing: true
max_grad_norm: 1.0
save_steps: 100
seed: 42
```

如 frozen values 不同，正式 run 使用 frozen values。

### 7.2 Budget Symmetry

E2/E3 必须具有相同：

```text
optimizer_steps
fresh_prompt_groups
fresh_rollouts
```

并记录 generated tokens。预算不匹配时先排查 trainer/incomplete run，不直接比较 efficiency。

### 7.3 Checkpoints / Metrics

保存并评测：

```text
step 100
step 200
step 300
```

每个 checkpoint 用同一 MedQA dev deterministic evaluation；E3 同时运行 Persistence Probe。

训练日志：

```text
reward_mean
reward_std
frac_reward_zero_std
format_rate
pre_answer_tokens_mean
direct_answer_rate
completion_length_mean
truncation_rate
clip_ratio_low
clip_ratio_high
loss
gradient_norm
learning_rate
optimizer_steps
fresh_generation_batches
fresh_prompt_groups
fresh_rollouts
fresh_generated_completion_tokens
```

$$
\mathrm{frac\_reward\_zero\_std}
=
\frac{
\#\{\text{fresh groups with std}(r_{\mathrm{task}})=0\}
}{
\#\{\text{fresh prompt groups}\}
}.
$$

只使用 binary task reward 计算。

### 7.4 Failure Symmetry

停止条件：NaN/Inf、OOM、LoRA 不更新、持续 `format_rate<90%`、持续 `truncation_rate>25%`。

短期 dev performance 下降不 early-stop。任何需要修改 frozen protocol 的修复都按 Section 5.7 对称处理 E2/E3。

---

## 8. Persistence Probe — Secondary Diagnostic

### 8.1 Checkpoints / Protocol

E3：

```text
step 0
step 100
step 200
step 300
```

- step 0 复用 Formal Scan；
- 100/200/300 使用对应 E3 checkpoint；
- E2 persistence 可选，不属于 MVP。

Probe generation 只读取 frozen stochastic config：

```yaml
num_rollouts: <frozen G>
temperature: <frozen temperature>
top_p: <frozen top_p>
max_new_tokens: <frozen max_completion_length>
prompt_serialization: plain_text_v1
```

### 8.2 Metrics

$$
\hat p_t(x)
=
\frac{1}{G}\sum_{j=1}^{G}r_{\mathrm{task},t,j},
\qquad
I_t(x)=\mathbf 1[0<\hat p_t(x)<1].
$$

Primary：

$$
P(I_t=1\mid I_0=1),
\qquad
P(I_t=1\mid I_0=0),
$$

$$
\Delta I_t
=
P(I_t=1\mid I_0=1)
-
P(I_t=1\mid I_0=0).
$$

Auxiliary：

$$
\rho_t=\operatorname{Spearman}(\hat p_0,\hat p_t),
$$

$$
R_t=\frac{|\mathcal I_0\cap\mathcal I_t|}{|\mathcal I_0|},
\qquad
J_t=\frac{|\mathcal I_0\cap\mathcal I_t|}{|\mathcal I_0\cup\mathcal I_t|}.
$$

默认 `G=4` ties 很多，因此 Spearman 仅作 auxiliary diagnostic。Persistence gap 弱、归零或反转均允许，不修改主算法。

Outputs：

```text
outputs/persistence/probe_step_100.jsonl
outputs/persistence/probe_step_200.jsonl
outputs/persistence/probe_step_300.jsonl
outputs/persistence/summary.json
```

---

## 9. Evaluation and Statistics

### 9.1 Checkpoint Selection

候选：step 100 / 200 / 300。只使用 MedQA dev。

优先级：

1. higher `strict_task_success_accuracy`；
2. higher `format_rate`；
3. lower `truncation_rate`；
4. lower `mean_completion_tokens`。

不得使用 MedQA test、MedMCQA final eval、MMLU-Pro Health 选择 checkpoint。

### 9.2 Final Models / Benchmarks

```text
E0 Base
E1 SFT
E2 Random GRPO
E3 Difficulty-Aware GRPO
```

```text
MedQA test
MedMCQA held-out eval
MMLU-Pro Health
```

### 9.3 Unified Evaluation Protocol

所有模型使用相同 base/tokenizer revision、renderer、4-bit NF4 loading、dtype、adapter loading、unmerged PEFT state、parser 和 deterministic generation。

```text
E0: quantized Base
E1: Base + SFT adapter
E2: Base + selected Random-GRPO adapter
E3: Base + selected Difficulty-GRPO adapter
```

```yaml
do_sample: false
num_beams: 1
max_new_tokens: <frozen max_completion_length>
```

所有 decode 使用 `skip_special_tokens=True`。

Final Eval 与 stochastic scan 的 sampling mode 不同，但共享 renderer、answer contract、parser、completion cap 与 loading conventions。

### 9.4 Metrics

Primary：

```text
strict_task_success_accuracy = mean(r_task)
```

同时报告：

```text
format_rate
pre_answer_tokens_mean
direct_answer_rate
mean / median completion tokens
truncation_rate
```

不得把 strict metric 与使用不同 parser/contract 的外部 benchmark accuracy 直接横比。

E2 vs E3 additionally：

```text
frac_reward_zero_std
reward_mean
fresh_generation_batches
fresh_prompt_groups
fresh_rollouts
fresh_generated_completion_tokens
MedQA dev strict metric @ 0/100/200/300
selected-checkpoint MedQA test strict metric
```

`informative-group efficiency` 仅指 RL training phase 的 non-zero reward-variance fresh-group ratio，不等价于 lower total compute。

### 9.5 Compute Accounting

分别记录：

```text
Scan: prompts / rollouts / generated tokens
E2/E3: fresh groups / rollouts / generated tokens / optimizer steps
Probe: rollouts / generated tokens
```

只有计入 scan/probe overhead 后才能讨论 end-to-end generation cost。

### 9.6 Paired Bootstrap / Error Analysis

对 `E3-E1`、`E3-E2`：

```yaml
resamples: 5000
confidence_level: 0.95
seed: 42
```

报告 strict-task-success delta 与 95% CI，并标注：

> **single-training-seed conditional CI**

它不等价于跨 training seeds 的算法级显著性。

最多人工检查 30 个 MedQA test samples：E1 wrong/E3 correct、E1 correct/E3 wrong、E2/E3 disagreement、format/truncation abnormal。

---

## 10. Repository, Tests and Dependencies

### 10.1 Repository

```text
medreason-rlvr/
├── SPEC.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── configs/
│   ├── data.yaml
│   ├── sft.yaml
│   ├── grpo.yaml
│   └── eval.yaml
├── src/medreason/
│   ├── __init__.py
│   ├── data.py
│   ├── prompt.py
│   ├── reward.py
│   ├── protocol.py
│   ├── train_sft.py
│   ├── scan_difficulty.py
│   ├── train_grpo.py
│   ├── persistence.py
│   ├── evaluate.py
│   └── analyze.py
├── tests/
│   └── test_core.py
├── scripts/
│   ├── prepare_data.sh
│   ├── train_sft.sh
│   ├── grpo_smoke.sh
│   ├── freeze_protocol.sh
│   ├── scan_difficulty.sh
│   ├── train_random_grpo.sh
│   ├── train_difficulty_grpo.sh
│   ├── probe_persistence.sh
│   ├── evaluate.sh
│   └── analyze.sh
└── outputs/
    ├── protocol/
    ├── sft/
    ├── smoke/
    ├── difficulty/
    ├── random_grpo/
    ├── difficulty_grpo/
    ├── persistence/
    └── eval/
```

Shell scripts 只负责 orchestration；算法逻辑在 `src/medreason/`；不建设复杂统一 CLI。

### 10.2 Unit Tests

只覆盖 silent algorithm/data bugs：

- schema / label mapping / exact dedup；
- renderer / EOS / no-silent-truncation contract；
- parser / Format Gate / direct-answer valid case；
- binary reward / reasoning diagnostics；
- pass-rate / informative filtering；
- stratified probe sampling；
- `D_info` excludes probe；
- equal-size pools；
- persistence metrics；
- fresh generation counters；
- frozen protocol field validation。

不设置 coverage threshold。GPU path 由 SFT smoke + GRPO smoke 验证。

### 10.3 Runtime

```text
Python 3.11
uv
trl==1.10.0
```

Core dependencies：

```text
torch
transformers
trl==1.10.0
peft
bitsandbytes
accelerate
datasets
safetensors
pyyaml
numpy
pandas
scipy
pytest
```

`pyproject.toml` 声明依赖，`uv.lock` 锁定实际版本。不强制 wandb/mlflow/vllm/deepspeed。

### 10.4 TRL API Gate

Phase 0 / GRPO smoke 前验证：

```text
SFTConfig.max_length
SFTConfig.completion_only_loss
GRPOConfig.loss_type
GRPOConfig.epsilon
GRPOConfig.epsilon_high
GRPOConfig.beta
GRPOConfig.scale_rewards
GRPOConfig.num_iterations
GRPOConfig.num_generations
GRPOConfig.max_completion_length
GRPOConfig.steps_per_generation
GRPOConfig.generation_batch_size
GRPOConfig.mask_truncated_completions
```

要求：使用 `steps_per_generation` 时不同时设置 `generation_batch_size`；generation batch 与 group structure 兼容；`num_iterations>1` 时 generation 与 optimizer updates 不一一对应；`beta=0` 不加载 ref model；DAPO/Clip-Higher/truncation mask 通过真实 trainer behavior 验证。

同一正式实验不得静默升级 TRL。

---

## 11. Development Phases

| Phase | Main Work | Gate |
|---|---|---|
| 0 | Environment / 4-bit Base / LoRA / TRL API | forward/backward/save-reload/no OOM |
| 1 | Data / renderer / parser / reward / tests | deterministic core tests pass |
| 2 | SFT smoke + full | immutable adapter reload; format ≥95% |
| 3 | GRPO smoke + protocol freeze | RL path/counters/objective/resources pass |
| 4 | Formal Scan + probe + pools | p0/pools/probe isolation valid |
| 5 | E2 / E3 Full GRPO | same frozen protocol and fresh budgets |
| 6 | Persistence + final evaluation | unified eval; no test-driven tuning |
| 7 | README / figures / commands | only real results reported |

README 至少包含：problem/pipeline、reward、policy-local informativeness、Offline Scan、Random-vs-Difficulty control、DAPO-inspired objective、zero-variance comparison、persistence、final benchmarks、compute accounting、limitations、reproducible commands。

Core figures：

```text
1. E0/E1/E2/E3 strict_task_success_accuracy
2. E2 vs E3 frac_reward_zero_std
3. persistence predictive gap
4. optional training curve
```

---

## 12. Final Definition of Done

- [ ] 4-bit Base + LoRA path 可运行，TRL 1.10.0 已锁定。
- [ ] Data roles、splits、dedup、renderer、length contract 正确。
- [ ] Parser / binary reward / direct-answer valid case tests 通过。
- [ ] SFT smoke/full 完成；immutable adapter 可重载；format ≥95%。
- [ ] GRPO smoke 用 disposable SFT copy 完成；Protocol 在 Formal Scan 前冻结。
- [ ] $p_0$ 唯一来自 Formal Scan。
- [ ] Probe 按 $I_0$ 分层，完全排除于 E2/E3 pools。
- [ ] `D_info` / `D_random` 等规模；Random 允许自然 overlap。
- [ ] Scan trajectories 未进入 GRPO training。
- [ ] E2/E3 从同一 SFT LoRA 独立初始化并继续优化同一 adapter。
- [ ] `loss_type=dapo`、`epsilon=0.20`、`epsilon_high=0.28`、`num_iterations=2`。
- [ ] `beta=0.0`；KL 未进入 reward；未加载 reference model。
- [ ] `mask_truncated_completions=true`。
- [ ] E2/E3 actual fresh prompt-group / rollout budgets 匹配。
- [ ] step 100/200/300 checkpoints 与 MedQA dev evaluations 完成。
- [ ] `frac_reward_zero_std` 比较完成。
- [ ] Persistence primary / auxiliary metrics 完成。
- [ ] E0/E1/E2/E3 使用统一 final evaluation protocol。
- [ ] Primary metric 为 `strict_task_success_accuracy`，同时报告 `format_rate`。
- [ ] Paired bootstrap 完成并标注 single-training-seed conditional CI。
- [ ] Scan/probe overhead 与 RL training cost 分开记录。
- [ ] README / resume 只使用真实结果。

Required artifacts：

```text
SPEC.md
README.md
pyproject.toml
uv.lock
configs/ src/ tests/ scripts/
outputs/protocol/frozen_protocol.yaml
outputs/sft/adapter/
outputs/smoke/
outputs/difficulty/{scan.jsonl,probe_ids.json,informative_pool.jsonl,random_pool.jsonl,summary.json}
outputs/random_grpo/
outputs/difficulty_grpo/
outputs/persistence/
outputs/eval/
```

---

## 13. Technical References

- Qwen3-8B-Base — <https://huggingface.co/Qwen/Qwen3-8B-Base>
- TRL 1.10.0 GRPOTrainer — <https://huggingface.co/docs/trl/v1.10.0/en/grpo_trainer>
- TRL 1.10.0 SFTTrainer — <https://huggingface.co/docs/trl/v1.10.0/en/sft_trainer>
- TRL PEFT Integration — <https://huggingface.co/docs/trl/en/peft_integration>
- DAPO — <https://arxiv.org/abs/2503.14476>
- verl DAPO Recipe — <https://github.com/verl-project/verl-recipe/tree/main/dapo>
- MedMCQA — <https://huggingface.co/datasets/openlifescienceai/medmcqa>
- MedQA — <https://github.com/jind11/MedQA>
- MMLU-Pro — <https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro>

---

## 14. Final Development Principle

新增任何功能、测试、fallback、ablation 或抽象前，只问一个问题：

> **它是否直接降低核心训练链路失败风险，或直接提高 Random-vs-Difficulty controlled comparison 的可信度？**

若否，不进入项目。

```text
no extra algorithms
no extra infrastructure
no branch-specific fixes
no result-driven protocol changes
```
