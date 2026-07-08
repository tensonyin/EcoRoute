比赛要求：
Track 1
⭐ Beginner Friendly · AI Agent Track
Hybrid Token-Efficient Routing Agent
Build an AI agent that gets the job done using the least tokens possible.

Build an AI agent that completes a fixed set of tasks autonomously, deciding in real time which Fireworks AI model is the cheapest one that can still answer accurately. The goal: use the fewest tokens possible, without falling below the accuracy threshold.

Every submission is scored on a standardized environment. You can develop and test on any hardware, but final scoring runs on this standardized environment only. Only inference routed through Fireworks AI counts toward your score, so routing intelligence — picking the cheapest sufficient Fireworks model for each task — wins, not raw compute power.

💡
Local models are optional and only useful for development/testing. Inference run locally is not tracked and does not count toward your score — only calls through Fireworks AI (FIREWORKS_BASE_URL, using a model from ALLOWED_MODELS) are scored.
We recommend running a local eval step to check your output quality before submitting.

Want to fine-tune your router? Go for it. Prompt-based and fine-tuned approaches are scored exactly the same way: token count and output accuracy.

💡 Build Ideas
Model Router / Cost Optimizer
A routing layer that reads each query and instantly picks the cheapest, best-suited model from the available endpoints.
Level
Beginner
Judging
Token count and output accuracy
Compute
Fireworks AI API (local models optional, for dev/testing only)


Track 1: General-Purpose AI Agent Build an agent that handles tasks across 8 categories (factual Q&A, math reasoning, sentiment, summarization, NER, code debugging, logic puzzles, code generation) using Fireworks AI models. Submit a Docker image. Scored on an accuracy gate, then ranked by token efficiency. Allowed models (Track 1):
minimax-m3
kimi-k2p7-code
gemma-4-31b-it
gemma-4-26b-a4b-it
gemma-4-31b-it-nvfp4
Keep in mind (all tracks):
Docker images (Tracks 1 & 2) must be publicly pullable and include a linux/amd64 manifest
Image size capped at 10GB
No hardcoded/cached answers: evaluation uses unseen variants
Submissions are rate-limited, so test locally before repeated submits GPU Access:
Use this link to get access to GPUs: https://notebooks.amd.com/hackathon
Read the full guide for exact I/O formats, environment variables, and scoring details.


Notebook 有2个可选环境：
1、ROCm7.2 + vLLM 0.16.0 + PyTorch 2.9
2、Unsloth + llama.cpp for Radeon


## 📅 一、 项目全流程里程碑规划 (Project Milestones)

整个项目从开发到最终提交，分为四个阶段，确保在截止日前交付最稳健的容器：

```
[阶段 1: 架构与 I/O 稳固] ──> [阶段 2: 纯代码智能路由] ──> [阶段 3: 极致 Token 压缩] ──> [阶段 4: 平台提交通关]

```

### 阶段 1：基础架构与标准 I/O 稳固 (15% 进度)

* 
**核心任务**：构建最轻量化的 Linux 基础环境，配置官方要求的标准 I/O 读写管线 。


* **确保合规**：
* 容器启动时，必须毫秒级读取 `/input/tasks.json` 。


* 在 10 分钟最大运行时限内，必须稳定将结果写入 `/output/results.json` 。


* 捕获所有运行时环境注入的变量（`FIREWORKS_API_KEY`、`FIREWORKS_BASE_URL`、`ALLOWED_MODELS`），绝不硬编码 。


* 
**夺冠保障**：整个容器内部设置全局异常拦截（Catch-All），即使单次网络请求超时或报错，也必须返回合法的 JSON 格式并以 `Exit Code 0` 正常退出，坚决防范因格式错误直接被计零分的风险 。





### 阶段 2：基于“零成本”纯代码的智能分流路由器 (40% 进度)

* 
**核心任务**：编写高性能、无模型（Model-less）的确定性前置过滤逻辑。官方明确指出，本地运行的推理不被跟踪、不计入分数，只有走远程 API 的才会算入 Token 总量 。


* 
**夺冠保障**：拒绝在 Docker 镜像中塞入任何本地大模型（防止纯 CPU 评测环境导致的严重超时崩溃 ）。采用 **Python 纯代码（高级正则表达式、精确关键词矩阵）**，在 0 毫秒、0 Token 消耗的前提下，将输入的盲盒 Prompt 精准剥离分流到 8 大任务领域中 。



### 阶段 3：极致 Token 压榨与防御性调优 (30% 进度)

* **核心任务**：在 AMD 开发环境（Jupyter Notebook）中，使用官方样例进行本地评估（Local Eval Step），针对分流矩阵进行极限微调。
* **夺冠保障**：
* **输入端压缩**：通过纯代码算法，自动对输入的 Prompt 进行冗余字符清洗（如剔除连续换行符、无意义空格）。
* **提示词缩减**：将全局 System Prompt 缩减到极限（如缩减到仅 3-5 个单词），因为系统提示词的字符同样在每次请求中被全面计费。
* **输出端硬拦截**：针对不同的任务，动态调整 `max_tokens` 拦截阈值。对于文本分类、命名实体识别等任务，强行切断其“说废话”的可能，从根本上杜绝模型在回答后附加解释性 Token。



### 阶段 4：官方平台提交通关与多媒体包装 (15% 进度)

* 
**核心任务**：根据官方表单（Step 1 到 Step 3）进行高标准的内容交付，完成 Docker 编译、代码开源和视频录制 。


* 
**确保合规**：使用专用交叉编译命令，确保最终的 Docker 镜像包含 `linux/amd64` 清单，且压缩总体积远低于 10GB 。



---

## 🎯 二、 夺冠核心：5 大官方模型精细化分流矩阵 (Routing Matrix)

由于排行榜最终比拼的是 **Token 效率（越少越好）** ，全部调用最聪明的模型会导致账单爆炸 ；全部调用最便宜的模型则会导致准确率跌破门槛被直接除名 。

因此，最可能夺冠的方案是针对官方允许的 5 大模型，建立**极度严苛的阶梯分流矩阵**：

| 任务难度等级 | 涵盖的官方任务领域（共8类） 

 | 指派的官方线上模型 | 夺冠控费逻辑 |
| --- | --- | --- | --- |
| **经济型分流** | 3. 情绪分类 (Sentiment)<br>

<br>4. 文本摘要 (Summarization)<br>

<br>5. 命名实体识别 (NER) 

 | **`gemma-4-26b-a4b-it`** | **极限刷分**：这三类任务属于标准的结构化文本提取，无须强大的逻辑。调用列表中最便宜的 26B 开源量化模型，并将 `max_tokens` 严控在 30~150 之间，以最低的 Token 成本拿下这部分基础分。 |
| **进阶型分流** | 2. 数学推理 (Math Reasoning)<br>

<br>7. 逻辑谜题 (Logic Puzzles) 

 | **`gemma-4-31b-it`** 或<br>

<br>**`gemma-4-31b-it-nvfp4`** | <br>**性价比平衡**：涉及多步算术和约束条件匹配，需要一定的推理深度 。使用 31B 密集型模型，保持逻辑严密性的同时，避免调用顶级闭源模型带来的巨额 Token 开销。

 |
| **专家型分流** | 6. 代码 Debug (Code Debugging)<br>

<br>8. 代码 generation (Code Generation) 

 | **`kimi-k2p7-code`** | **垂类精准指派**：代码任务如果找错模型会疯狂出错。直接分流给专门优化过的 Kimi 代码专家模型。由于其对代码语法的超高理解力，可以用极简的 Prompt 换取正确的代码输出，省去多次重试的 Token 成本。 |
| **最高防线兜底** | 1. 事实知识问答 (Factual Q&A) 

<br>

<br>以及**无法通过代码识别的未知变体 Prompt** 

 | **`minimax-m3`** | <br>**保命跨过准确率门槛**：对于无法用纯代码特征归类的通用盲盒问答，直接指派给全能闭源旗舰 MiniMax M3，确保通过硬性准确率大关，防止因回答错误被踢出排行榜 。

 |

---

## 🏆 三、 最终平台提交通关指南 (Submission Checklist)

在填写提交表单时，人工评委（Human Judges）会极度看重工程的规范性与商业落地价值。按照以下方案填写，可使你的项目在非技术维度同样拿到高分：

### 1. 基础信息填写 (Step 1)

* **Submission Title (项目名称)**：起一个具有商业化、具有成本控制美感的名称。如：`EcoRoute: Zero-Token Deterministic Multi-Model Routing Agent`。
* **Short Description (短描述)**：一句话点明核心。如：`A lightweight, zero-token deterministic routing agent that dynamically optimizes multi-domain task distribution across Fireworks AI models to guarantee accuracy while minimizing token expenses.`
* **Long Description (长描述)**：至少 100 词。采用企业级文案风格，阐述混合多模型调度在现实企业中控制大模型（LLM）算力成本的巨大商业价值。清晰罗列出对 8 大分类任务的按需分流策略，并着重强调该项目不仅效率极高，且由于未打包大模型，镜像体积仅有一百多兆（极轻量、极健壮、高韧性）。

### 2. 多媒体演示包装 (Step 2)

* **Slide Presentation (幻灯片 PDF)**：制作一份 5-8 页、具有 AMD 企业色调风格的商业级 PPT。包含：团队（team-546）分工介绍、0-Token 前置路由层架构图、8 大任务与 5 大模型的性价比矩阵分析图、以及异常降级兜底机制说明。
* **Video Presentation (视频展示)**：录制一段 3 分钟的黄金讲解视频。**绝对不要大篇幅去机械地读代码**。先用 30 秒展示无脑调用旗舰大模型导致的昂贵 Token 账单，再用 1 分钟可视化展示你们的 EcoRoute 路由技术如何对任务进行精细化降本分流，最后 1 分钟演示容器在 Linux 评测环境下秒级启动、秒级通过测试集并生成完美结果 JSON 的实录。

### 3. 代码仓库与镜像发布 (Step 3)

* 
**GitHub Repository**：保持项目目录极其干净规范。编写高质量的 `README.md`，包含清晰的项目架构图、本地评估指南（Local Eval），以及如何利用 `.env` 配置文件进行本地测试的说明 。


* 
**Docker Image**：在最终提交前，务必在干净的无登录态终端中测试运行 `docker pull <你的公共镜像地址>`，确保评测沙盒能 100% 公开拉取，且架构清单完美支持 `linux/amd64` 。