# TraceScholar

[简体中文](README.md) | [English](README.en.md)

**当前版本：0.1.0**

TraceScholar 是一个面向科研论文调研的证据优先助手。它将研究问题组织成可复现的调研流程，并让综述草稿中的事实陈述能够追溯到具体论文、PDF 版本、页码和原文证据。

## v0.1 可以做什么

- **研究规划与检索**：把研究问题整理为结构化 ResearchPlan，并基于计划生成检索式。
- **多源论文发现**：通过 OpenAlex 和 Crossref 检索，合并重复论文并保留来源记录。
- **两阶段筛选**：先筛标题/摘要，再结合全文相关段落复核。
- **开放全文获取与解析**：仅获取合法开放获取全文；解析 PDF 并生成带页码、章节和定位信息的 Chunk。
- **全文检索**：通过云端 Embedding 和 PostgreSQL pgvector，在当前研究范围内检索相关段落。
- **研究版本归一化**：关联预印本和正式发表版本，按独立 Study 进行证据统计。
- **Evidence Ledger**：将 Claim 与带原文、立场、页码和来源链的 EvidenceSpan 关联。
- **Grounded Review**：生成与 Claim、Evidence 对应的结构化综述草稿。
- **三层审计**：逐句检查引用链、语义支持情况，以及是否遗漏同一 Claim 下的重要证据。
- **可复现与可恢复**：保存 Run Manifest；workflow 按顺序执行，可从失败阶段继续。
- **确定性导出**：导出带引用和审计状态的 Markdown 与 JSON，不在导出时调用 LLM。

## 限制与人工复核

- 当前不支持 OCR；纯图片 PDF 可能无法解析。
- 只尝试获取合法开放获取全文，不绕过付费墙或登录限制；不能保证每篇论文都能获取全文。
- Semantic Audit 可能给出 `revise` 或 `reject`。修订建议不会自动覆盖原始草稿，需由用户复核。
- TraceScholar 不能替代系统综述规范流程或 Meta-analysis。
- 自动化规划、筛选、证据抽取与综述可能调用外部模型，并消耗服务额度或产生费用；结果需要人工检查。

## 环境要求

- Python 3.12 或更高版本
- Docker 和 Docker Compose（用于 PostgreSQL 16 + pgvector）
- 可访问的 OpenAI 兼容 LLM 服务及独立的 Embedding 服务
- 网络连接（论文检索和开放全文获取）

## 安装

在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

验证命令行工具：

```bash
tracescholar --version
```

## 配置

复制环境变量模板：

```bash
cp .env.example .env
```

至少在 `.env` 中配置 LLM 服务：

```dotenv
TRACESCHOLAR_LLM_BASE_URL=https://你的服务地址/v1
TRACESCHOLAR_LLM_API_KEY=你的密钥
TRACESCHOLAR_LLM_MODEL=你的模型名
```

Embedding 使用单独的兼容服务和密钥：

```dotenv
TRACESCHOLAR_EMBEDDING_BASE_URL=https://你的Embedding服务地址/v1
TRACESCHOLAR_EMBEDDING_API_KEY=你的Embedding密钥
TRACESCHOLAR_EMBEDDING_MODEL=你的Embedding模型名
TRACESCHOLAR_EMBEDDING_DIMENSIONS=1024
```

请根据服务商提供的信息填写地址、模型和向量维度。OpenAlex API Key 和 Crossref 邮箱为可选配置；常规使用建议设置 OpenAlex API Key。**不要把真实密钥写入仓库或提交 `.env`**。所有配置统一由 `tracescholar.config.Settings` 管理。

## 启动数据库

```bash
docker compose up -d postgres
alembic upgrade head
```

数据库结构由 Alembic migration 管理。

## 创建并运行调研

创建一个 ResearchRun：

```bash
tracescholar research \
  "RAG 中 query rewriting 对多跳问答的效果如何？" \
  --scope-json '{"year_from": 2023, "languages": ["en"]}'
```

命令会返回 `RUN_ID`。运行尚未完成的 workflow：

```bash
tracescholar run <RUN_ID>
tracescholar workflow-status <RUN_ID>
```

`run` 会按顺序执行可运行的阶段，直到完成、遇到失败或被前置条件阻塞。再次运行同一命令可以从未完成阶段继续；已完成的阶段会按幂等逻辑复用，不会无意义地重复执行。

完整过程通常包括：

```text
研究问题
  → ResearchPlan
  → OpenAlex / Crossref 检索与去重
  → 标题/摘要筛选
  → 合法开放全文获取与页码感知解析
  → Embedding 与全文检索
  → 全文筛选与 Study 版本归一化
  → Claim / Evidence Ledger
  → Grounded Review 与逐句审计
  → Run Manifest
  → Markdown / JSON 导出
```

## 查看 Manifest 与导出报告

workflow 完成相应阶段后，生成并查看 Run Manifest：

```bash
tracescholar manifest <RUN_ID>
tracescholar show-manifest <MANIFEST_ID>
```

导出已保存的草稿和审计状态：

```bash
tracescholar export <RUN_ID> --output-dir ./review-export
```

导出是确定性的，不会重新执行研究或调用 LLM。语义审计为 `revise` 的句子会保留原文并标注状态；`reject` 不会被无提示地当作普通事实句输出。

## 常用命令

```bash
tracescholar plan <RUN_ID>                  # 生成并保存 ResearchPlan
tracescholar discover <RUN_ID> --limit 5    # 根据冻结计划执行自动检索
tracescholar screen <RUN_ID>                # 标题/摘要筛选
tracescholar acquire <RUN_ID>               # 尝试获取开放全文
tracescholar parse <RUN_ID>                 # 解析 PDF 并生成页码级 Chunk
tracescholar embed <RUN_ID>                 # 生成并保存 Chunk Embedding
tracescholar retrieve <RUN_ID> "检索问题" --top-k 10  # 在该 Run 范围内检索全文
tracescholar workflow-status <RUN_ID>        # 查看 workflow 状态
tracescholar run <RUN_ID>                   # 连续执行未完成阶段
```

更多分阶段命令、参数、数据结构与操作说明见[英文完整指南](README.en.md)。

## 运行测试

```bash
python -m unittest discover -s tests
```

## 许可

TraceScholar 采用 Apache License 2.0，详见 [LICENSE](LICENSE)。
