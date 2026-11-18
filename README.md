# CS/NLP Learning Assistant

### 面向专业知识学习场景的 Agentic RAG 智能学习助手

<p align="center">
  <b>RAG · Agent · Hybrid Search · Citation Verification · LLM Evaluation</b>
</p>

<p align="center">
  基于大语言模型构建的智能学习助手，支持专业资料检索、知识问答、多轮对话、工具调用与 PDF 原文溯源。
</p>

<p align="center">
  Python · LangGraph · FastAPI · MySQL · Redis · DeepSeek
</p>

---

## 1. 项目简介

### 1.1 项目背景

在学习计算机科学、自然语言处理等专业知识时，学习者通常需要阅读大量教材、论文和技术文档。传统学习方式以及通用大模型问答存在一些问题：

- **知识查找效率低：** 专业教材内容庞大，定位特定知识点和相关解释需要频繁翻阅文档。
- **复杂知识理解困难：** 专业概念、数学公式和定理证明通常需要结合上下文进行连续理解。
- **AI 回答缺乏可靠依据：** 通用大模型的回答未必严格基于指定教材，用户难以确认答案来源。
- **学习上下文容易丢失：** 围绕同一个知识点连续追问时，需要系统理解前序对话。
- **答案验证成本较高：** 用户需要返回原始文档手动查找相关内容。

基于自身专业学习过程中的实际需求，本项目尝试构建一个结合专业知识库和大语言模型的智能学习助手，降低专业资料查找与知识理解成本，提高 AI 问答的可验证性和连续性。

### 1.2 产品定位

面向计算机科学、自然语言处理及相关专业学习者的 AI 辅助学习工具。

通过 RAG 实现基于专业资料的知识问答，通过 Agent 工作流实现问题分诊、工具调用与多轮交互，并通过引用校验机制提高答案的可验证性。

### 1.3 项目职责

**个人独立项目**

负责项目需求分析、功能规划、技术方案设计、系统开发及效果评估，独立完成从问题发现到产品实现的完整过程。

目前主要用于个人学习场景，尚未开展正式的外部用户调研和规模化用户验证。

---

## 2. 核心功能

| 功能模块 | 功能说明 | 用户价值 |
|---|---|---|
| 专业知识问答 | 基于专业知识库检索并生成回答 | 降低知识查找成本 |
| PDF 引用溯源 | 展示引用页码，支持点击定位原文 | 提高答案可验证性 |
| Agent 智能调度 | 根据问题类型选择不同工具 | 支持复杂任务处理 |
| 多轮对话记忆 | 管理会话上下文与用户长期偏好 | 提高连续学习体验 |
| 数学计算工具 | 调用计算工具处理特定数学及模型计算任务 | 辅助专业问题求解 |
| 答案质量控制 | 结合引用校验及拒答机制约束回答 | 降低无依据回答风险 |

### 2.1 专业知识问答

基于用户提供的专业资料，通过 RAG 检索相关知识片段，并利用大语言模型生成回答。

### 2.2 PDF 引用溯源

在回答中展示相关文档的页码引用，支持用户点击引用跳转至 PDF 对应页面，方便核对原始资料。

### 2.3 Agent 工具调用

基于 LangGraph 构建 Agent 工作流，根据问题类型选择知识检索或数学计算工具，实现多工具协同。

### 2.4 多轮对话记忆

结合会话状态管理与长期偏好存储，在连续交互中召回相关上下文，改善多轮学习体验。

### 2.5 答案可靠性控制

结合检索上下文、引用校验及资料不足时的拒答策略，减少无依据回答，并使用 LLM-as-a-Judge 评估生成质量。

---

## 3. 产品交互展示

以下位置建议放置项目实际运行截图。

### 3.1 智能问答

<!-- 将实际截图放入 screenshots/chat.png 后启用下一行 -->
<!-- ![智能问答界面](screenshots/chat.png) -->

展示用户输入问题、Agent 处理及答案生成过程。

### 3.2 PDF 引用溯源

<!-- 将实际截图放入 screenshots/citation.png 后启用下一行 -->
<!-- ![PDF 引用溯源](screenshots/citation.png) -->

展示答案引用、文档页码及 PDF 原文定位。

### 3.3 Agent 工具调用

<!-- 将实际截图放入 screenshots/agent.png 后启用下一行 -->
<!-- ![Agent 工具调用](screenshots/agent.png) -->

展示 Agent 根据任务类型调用知识检索和数学计算工具的过程。

---

## 4. 核心业务流程

```text
用户输入问题
      |
      v
问题理解与任务分诊
      |
      v
历史记忆召回（按需）
      |
      v
判断任务类型
      |
      +----------------------+
      |                      |
      v                      v
   知识问答               数学计算
      |                      |
      v                      v
  混合检索                计算工具
      |
      v
  RRF 排序融合
      |
      v
  Reranker 精排
      |
      v
  大模型生成答案
      |
      v
  引用信息校验
      |
      v
  返回答案及引用
```

---

## 5. 技术架构

### 5.1 系统架构

```text
                 User
                   |
          +--------+--------+
          |                 |
      Native Web         Streamlit
          |                 |
          +--------+--------+
                   |
                FastAPI
                   |
             LangGraph Agent
                   |
       +-----------+-----------+
       |           |           |
    Memory       Triage      Agent
       |                       |
 SQLite / Redis           Tool Selection
                               |
                    +----------+----------+
                    |                     |
                 Search                  Calc
                    |
              Hybrid Retrieval
                    |
                 RRF
                    |
                Reranker
                    |
               DeepSeek LLM
                    |
            Answer Generation
                    |
          Citation Verification
                    |
              Final Response
```

### 5.2 Agent 工作流设计

基于 LangGraph 构建 Agent 状态机，主要包含：

| 节点 | 主要作用 |
|---|---|
| `memory_retrieval` | 召回历史对话及长期记忆 |
| `triage` | 判断问题类型和处理方式 |
| `agent` | 根据当前任务决定下一步操作 |
| `tools` | 执行知识检索或数学计算 |

通过状态管理和工具调用，实现多步骤任务处理。

### 5.3 混合检索与排序优化

针对专业教材中术语、公式及语义表达并存的特点，采用 Dense + Sparse 混合检索。

**检索流程：**

```text
                  User Query
                      |
            +---------+---------+
            |                   |
       Dense Search         BM25 Search
            |                   |
            +---------+---------+
                      |
                RRF Fusion
                      |
                  Reranker
                      |
                  Top-K
                      |
                 LLM Answer
```

核心组件：

- **Dense Retrieval：** 使用 `bge-small-zh` Embedding 进行语义检索。
- **Sparse Retrieval：** 使用 BM25 进行关键词检索。
- **RRF：** 融合不同检索通道的排序结果。
- **Reranker：** 使用 `BAAI/bge-reranker-v2-m3` 对候选文档进行精排。

### 5.4 记忆管理

- 使用 LangGraph Checkpointer 管理会话状态。
- 使用 `AsyncSqliteSaver` 持久化会话检查点。
- 使用 Redis Hash 存储用户长期偏好。
- 在多轮交互中根据需要召回相关记忆。

### 5.5 工程架构优化

- **异步服务：** 使用 FastAPI、`aiomysql` 和 `aioredis` 实现异步服务与数据库访问。
- **流式交互：** 使用 SSE 实现前端实时接收模型生成内容。
- **计算任务隔离：** 使用 `asyncio.to_thread` 将 CPU 密集型任务放入后台线程执行。
- **缓存机制：** 使用 Redis Cache-Aside 策略减少重复请求。
- **并发保护：** 使用运行中请求状态管理，减少重复请求造成的资源浪费。

### 5.6 技术栈

| 类别 | 技术 |
|---|---|
| 编程语言 | Python |
| Agent 框架 | LangGraph |
| 后端服务 | FastAPI |
| 数据库 | MySQL |
| 缓存 | Redis |
| 会话状态 | SQLite / LangGraph Checkpointer |
| 检索方法 | Dense Embedding / BM25 / RRF |
| Embedding | bge-small-zh |
| Reranker | bge-reranker-v2-m3 |
| 大语言模型 | DeepSeek |
| 前端 | HTML / CSS / JavaScript |
| 调试界面 | Streamlit |
| 流式传输 | SSE |

---

## 6. 效果评估

为了评估不同检索方案的表现，构建了包含 59 条跨章节问题的测试集，对不同检索策略进行对比。

### 6.1 检索效果对比

| 版本 | 优化策略 | Hit@1_any | Hit@3_all | MRR_all |
|---|---|---:|---:|---:|
| V1 | 基础向量检索 | 0.71 | 0.55 | 0.51 |
| V2 | + BM25 Hybrid | 0.86 | 0.58 | 0.54 |
| V3 | + Multi-Query | 0.83 | 0.63 | 0.53 |
| V4 | + Reranker | **0.91** | 0.61 | 0.48 |

**实验分析：**

- 引入 BM25 后，Hit@1_any 从 0.71 提升至 0.86，说明关键词检索能够补充语义检索的召回能力。
- 引入 Multi-Query 后，Hit@3_all 达到 0.63，说明多查询策略对前三条结果的覆盖有一定帮助。
- 引入 Reranker 后，Hit@1_any 达到 0.91，较基础版本提升 20 个百分点。
- V4 的 MRR_all 为 0.48，低于 V1 的 0.51，说明不同检索指标之间存在权衡，整体排序质量仍有优化空间。

### 6.2 答案质量评估

采用 LLM-as-a-Judge 对生成答案进行分类评估。

| 评估类别 | 占比 |
|---|---:|
| Correct | 72.88% |
| Partial | 13.56% |
| Wrong | 5.08% |
| Abstain | 8.47% |

**实验分析：**

- 系统能够在一定程度上生成正确且有依据的答案。
- 部分正确回答说明复杂专业问题仍存在信息遗漏或推理不完整的情况。
- 拒答机制能够在证据不足时限制无依据回答。
- 后续需要结合人工抽样评估，进一步检查 LLM-as-a-Judge 的评价偏差。

> 注：以上结果来自项目现有离线评测。测试集规模为 59 条，评估结果不代表真实用户场景下的整体准确率，也不能直接证明学习效率提升。

---

## 7. 项目复盘与后续规划

### 7.1 当前不足

1. 测试集规模较小，尚不能充分覆盖复杂专业知识问答场景。
2. 不同检索指标之间存在权衡，仍需要优化整体排序效果。
3. LLM-as-a-Judge 可能存在评价偏差，需要人工抽样复核。
4. 当前主要基于个人学习场景开发，缺乏外部用户反馈。
5. 尚未建立完整的端到端用户体验评估体系。

### 7.2 后续迭代规划

| 优先级 | 迭代方向 | 预期目标 |
|---|---|---|
| P0 | 扩充测试集与人工标注 | 提高评估可靠性 |
| P0 | 优化引用准确性与拒答机制 | 提高答案可信度 |
| P1 | 邀请同专业学生体验 | 验证真实用户需求 |
| P1 | 优化多轮记忆管理 | 改善连续学习体验 |
| P2 | 个性化学习计划 | 拓展学习辅助场景 |
| P2 | 知识点关联与错题整理 | 提供更完整的学习支持 |

---

## 8. 项目结构

```text
cs_nlp_learning_assistant/
│
├── eval/                 # 评估相关代码与数据
├── src/                  # 核心业务代码
├── static/               # Web 前端静态资源
│
├── app.py                # Streamlit 调试入口
├── db.py                 # 数据库相关逻辑
├── demo_agent.py         # Agent 示例
├── generate_questions.py # 测试问题生成
├── main.py               # FastAPI 应用入口
│
└── README.md
```

---

## 9. 本地运行

### 9.1 环境要求

- Python >= 3.9
- MySQL
- Redis
- 可用的大语言模型 API

### 9.2 安装依赖

请根据项目实际依赖文件安装所需 Python 包，并配置相应的模型 API、数据库和 Redis 参数。

```bash
git clone https://github.com/kongweijia86-beep/cs_nlp_learning_assistant.git

cd cs_nlp_learning_assistant
```

### 9.3 配置说明

运行前需要完成：

1. 配置大语言模型 API。
2. 配置 MySQL 连接信息。
3. 配置 Redis 连接信息。
4. 配置 Embedding 与 Reranker 模型。
5. 根据实际项目配置初始化数据库。

请勿将 API Key、数据库密码等敏感信息提交到公开仓库。

### 9.4 启动入口

- FastAPI 服务入口：`main.py`,打开后<img width="692" height="392" alt="image" src="https://github.com/user-attachments/assets/cf1d06aa-6011-41b7-802b-ea3aca0421cc" /><img width="691" height="388" alt="image" src="https://github.com/user-attachments/assets/85088b1e-3b52-4aa6-9a52-1065b83eedef" />


- Streamlit 调试入口：`app.py`

具体启动参数及命令以项目实际配置为准。

---

## 10. 项目总结

本项目从个人专业学习场景出发，探索大语言模型在专业知识学习辅助中的应用。

在项目实践中，完成了从需求发现、功能规划、RAG 与 Agent 方案设计，到系统开发和离线效果评估的完整过程。

通过混合检索、RRF 排序融合和 Reranker 精排，改善了测试集上的 Top-1 检索表现；通过引用校验、记忆管理和工具调用，探索提高专业知识问答可靠性与交互连续性的产品方案。

这也是一次将实际学习问题转化为 AI 产品，并通过实验评估持续优化的实践。

---

## 项目链接

GitHub：https://github.com/kongweijia86-beep/cs_nlp_learning_assistant

**项目类型：** 个人独立项目

**主要职责：** 需求分析、功能规划、技术方案设计、系统开发、效果评估。
