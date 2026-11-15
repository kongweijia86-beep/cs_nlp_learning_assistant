# CS/NLP Learning Assistant (Agentic RAG)

基于大语言模型（LLM）与 RAG 架构的智能学习助手。系统不仅具备高精度的混合检索与防幻觉能力，还通过 **LangGraph** 引入了智能体运行框架（Agent Harness），支持多工具调用、多步推理、长短期记忆与端到端流式交互。

**核心亮点**：项目提供**双端架构**——原生 HTML/JS (FastAPI) 提供生产级的高并发流式交互；Streamlit 提供轻量级的内部测试看板。后端采用 **FastAPI + aiomysql + aioredis** 完成全链路异步化改造。

## 界面与运行展示

<img width="692" height="392" alt="image" src="https://github.com/user-attachments/assets/549e4698-0c04-4bb0-ba98-dabf07439092" />


<img width="691" height="388" alt="image" src="https://github.com/user-attachments/assets/c5c4235e-2ad0-4e54-b8d2-5d3a9e68d5b2" />

*图示：Web 前端中，Agent 正在自主调用计算工具与知识库，输出带 `[Page X]` 引用校验的精准答案，并支持点击跳转 PDF。*

## 核心特性

### 1. Agentic RAG (智能体调度)
- **LangGraph 状态机**：构建含 `memory_retrieval`(记忆召回)、`triage`(问题分诊)、`agent`(决策)、`tools`(执行) 的复杂工作流图。
- **动态工具调用**：内建数学计算工具（Attention FLOPs/参数量计算）与教材检索工具，支持事实交叉验证。
- **长短期记忆融合**：利用 `AsyncSqliteSaver` 作为 Checkpointer 维持会话状态；结合 Redis Hash 存储用户长期偏好特征，在多轮对话中自动注入背景提示。

### 2. 混合检索与防幻觉机制
- **混合检索 (Hybrid Search)**：Dense (`bge-small-zh`) + Sparse (`BM25`) 双路召回，采用 **RRF（倒数排名融合）** 算法合并。
- **Reranker 精排**：引入 `BAAI/bge-reranker-v2-m3` 交叉编码器，大幅提升 Top-1 命中率与 NDCG。
- **可靠性机制**：强约束 Prompt + 强制引用校验（Citation Verification），在评测集中将幻觉率（Wrong Rate）压制在 **5.08%**。

### 3. 工业级全链路异步架构 & 双端体验
- **原生 Web 前端 (`static/`)**：零刷新 SSE 流式渲染；原生 JS 正则解析 `[Page X]` 并动态重建 `<iframe>`，实现点击引用一键跳转 PDF。
- **Streamlit 调试端 (`app.py`)**：保留轻量级可视化看板，便于快速进行本地 Prompt 调优与 Badcase 排查。
- **纯异步 IO**：后端全面拥抱 `aiomysql` 连接池与 `aioredis`，配合 FastAPI 实现极致并发。
- **CPU 密集任务隔离**：利用 `asyncio.to_thread` 将本地 Embedding 编码与检索抛入后台线程池，杜绝 Event Loop 阻塞。
- **高并发保护**：Redis 实现 Cache-Aside 旁路缓存，配合 `_running_queries` 字典防连点击穿大模型。

## 系统架构

`User -> Web (原生/Streamlit) -> FastAPI 异步网关 -> LangGraph Harness -> Tools(Search/Calc) -> RAG(Hybrid+RRF+Rerank) -> DeepSeek LLM -> 异步写入 MySQL/Redis`

## 评估结果（基于 59 条跨章节测试集）

| 版本 | 优化策略 | Hit@1_any | Hit@3_all | MRR_all |
|---|---|---|---|---|
| V1 | 基础向量检索 | 0.71 | 0.55 | 0.51 |
| V2 | + BM25 Hybrid | 0.86 | 0.58 | 0.54 |
| V3 | + Multi-Query | 0.83 | 0.63 | 0.53 |
| V4 | + Reranker | **0.91** | 0.61 | 0.48 |

> **生成质量评估 (LLM-as-a-Judge)**：Correct 72.88%，Partial 13.56%，**Wrong 5.08%**，Abstain 8.47%。

## 快速开始

### 1. 安装依赖
确保你的 Python 环境版本 >= 3.9，然后运行：
```bash
pip install -r requirements.txt
