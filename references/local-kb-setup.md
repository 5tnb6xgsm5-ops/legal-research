# 本地知识库自建指引（可选增强）

> 面向：想把「本地知识库」作为检索第一优先级的新用户。
> **这是可选增强，不是运行前提**——没建库时引擎自动从北大法宝 MCP / WebSearch 起步，主流程不受影响。

---

## 一、本地知识库是什么、为什么要建

legal-research 的检索是**六级降级链**：

```
本地知识库 → 北大法宝 MCP → 元典 MCP → ima 知识库 → 元宝 WSA → WebSearch
```

本地知识库排第一，因为它装的是**你自己的东西**：自办案件、你收集的法院公众号文章、法条条文组。命中时标注 `[本地KB·已核实]`，带 doc_id 出处，可回溯、可核验——这是 retrieve_first 检索增强的完整闭环。

**不建会怎样**：前置条件检查检测到本地库不可用，自动回退到北大法宝 MCP，照常跑，只是少了"私有语料"这一层。

---

## 二、建库两条路线（任选其一）

本包 `scripts/` 已附带全部脚本。依赖：`pip install pymupdf jieba`（jieba 缺了会自动退化为二元切分，检索仍可用、精度略降）。

### 路线 A：tag_documents（推荐，PDF/MD/TXT 通吃）

自动提取文本 + 打标（案由/法院/关键词）+ PII 脱敏，入库 `tags.db`。

```bash
python3 scripts/tag_documents.py --scan 你的文档目录 --db tags.db --source-type 公众号
```

### 路线 B：ingest + index_to_sqlite（PDF 判决书专用）

写入 `legal_docs.db`，走全文检索。

```bash
python3 scripts/ingest.py 你的PDF目录 --db legal_docs.db
```

> ⚠️ 扫描件（纯图片 PDF）本包**不含 OCR**，请先用自己的 OCR 工具转成带文字层的 PDF 再摄入。

---

## 三、检索验证

```bash
# 路线 A 的库（tags.db）——双路召回：精确 + 语义，返回带 doc_id 出处的片段
python3 scripts/rag_ask.py "你的问题" --db tags.db --json

# 路线 B 的库（legal_docs.db）——全文检索
python3 scripts/index_to_sqlite.py search "关键词" -d legal_docs.db
```

把返回的 context 交给 AI，让它**只基于片段作答并标注 doc_id**。

> 可选增强：下载 bge-m3 向量模型后执行 `python3 scripts/embed_index.py build`，开启真·语义向量召回（方法见脚本内部说明）。

---

## 四、建好后告诉引擎「库在哪」

本包用 `config/search_targets/registry.json` 声明可用数据源。建库后把 `local_kb` 的 `status` 改成 `connected`，并把 `capabilities.rag` 的工具路径指向你的 `rag_ask.py`。引擎前置检查会自动探测，命中即用。

没建库时保持 `status: disconnected`，引擎自动跳过本地库、从北大法宝起步——不需要改任何别的。

---

## 五、常见问题

| 问题 | 解法 |
|------|------|
| pymupdf 装不上 | 换 `pip install pymupdf --user`，或降级 Python 到 3.9–3.11 |
| 扫描件摄入为空 | 扫描件需先 OCR 转成带文字层的 PDF |
| 检索结果排序乱 | 以向量通道结果为准；未装 bge-m3 时以 tag_documents 路线的 FTS 结果为准 |
| 数据库文件建在哪 | `--db` 参数指定路径；不指定时默认在 scripts/ 上一级目录生成 |
