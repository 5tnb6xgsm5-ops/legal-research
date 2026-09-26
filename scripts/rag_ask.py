#!/usr/bin/env python3
"""
RAG 问答检索层（P2 MVP）
rag_ask.py

职责：问题 → 双路召回（精确+语义）→ 拼装带出处的原文片段 context。
不自己调 LLM：把召回结果交给对话模型基于出处作答（避免 MVP 耦合外部 API）。

输出格式（出处优先）：
  1. 原文片段列表（doc_id + 标题 + 来源 + 片段）
  2. 拼接好的 context 文本（供对话模型引用）
  3. 明确提示：答案须基于以下片段，标注出处 doc_id

用法：
    python3 rag_ask.py "民间借贷利息怎么算" --json
    python3 rag_ask.py "砍头息认定" 
"""

import os
import sqlite3
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(SCRIPT_DIR, "..", "tags.db")


def _fetch_doc(conn, doc_id: str) -> dict:
    r = conn.execute(
        """SELECT doc_id, title, source_type, case_types, dispute_focus,
                  keywords, court, judgment_date, text_preview, full_text
           FROM documents WHERE doc_id=?""",
        (doc_id,)
    ).fetchone()
    if not r:
        return {}
    keys = ["doc_id", "title", "source_type", "case_types", "dispute_focus",
            "keywords", "court", "judgment_date", "text_preview", "full_text"]
    d = dict(zip(keys, r))
    for k in ("case_types", "dispute_focus", "keywords"):
        try:
            d[k] = json.loads(d[k]) if d[k] else []
        except Exception:
            d[k] = []
    return d


def ask(query: str, db_path: str = DB, top_k: int = 5, use_hybrid: bool = True) -> dict:
    # 召回：向量(若 bge-m3 已建) + 同义词扩展 + FTS5（retrieve_first 默认双路）
    from embed_index import hybrid_search, neural_available
    hits = hybrid_search(query, db_path, top_k=top_k)
    ranked = [(h['doc_id'], h.get('score', 0.15)) for h in hits]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    docs = []
    context_parts = []
    for doc_id, score in ranked:
        d = _fetch_doc(conn, doc_id)
        if not d:
            continue
        docs.append(d)
        # 原文片段（出处优先）：标题+来源+案由+法院+全文片段
        snippet = f"【{d['title']}】（{d['source_type']}，doc_id={d['doc_id']}）\n"
        if d['case_types']:
            snippet += f"案由：{', '.join(d['case_types'])}\n"
        if d['court']:
            snippet += f"法院：{d['court']}\n"
        # 优先用全文，退回预览
        body = d.get('full_text') or d.get('text_preview') or ''
        if body:
            # 判决书按段落切分，取前 2 段作为片段（约 400 字）
            paras = [p for p in body.split('\n') if p.strip()]
            frag = '\n'.join(paras[:2])[:400]
            snippet += f"片段：{frag}\n"
        context_parts.append(snippet)
    conn.close()

    context = "\n---\n".join(context_parts)
    return {
        "query": query,
        "hits": docs,
        "context": context,
        "neural": neural_available(),
        "note": "以下片段来自本地知识库 tags.db，答案须基于这些片段并标注 doc_id 出处。新摄入的文档含全文片段；早期公众号文章仅含元数据+预览。"
    }


def main():
    parser = argparse.ArgumentParser(description='RAG 问答检索层（P2 MVP）')
    parser.add_argument('query', help='问题')
    parser.add_argument('--db', default=DB)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--fts-only', action='store_true', help='仅用 FTS5，不调向量')
    parser.add_argument('--json', action='store_true', help='JSON 输出')
    args = parser.parse_args()

    res = ask(args.query, args.db, args.top_k, use_hybrid=not args.fts_only)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    print(f"\n🔍 问题：「{res['query']}」— 召回 {len(res['hits'])} 篇\n")
    for i, d in enumerate(res['hits']):
        print(f"{i+1}. 📄 {d['title']}  [{d['source_type']}]")
        print(f"   doc_id: {d['doc_id']}")
        if d['case_types']:
            print(f"   案由: {', '.join(d['case_types'])}")
        if d['court']:
            print(f"   法院: {d['court']}")
        if d['text_preview']:
            print(f"   片段: {d['text_preview'][:200]}...")
        print()
    print("=" * 50)
    print("📋 拼装 context（交给对话模型基于出处作答）：")
    print(res['context'][:1500])


if __name__ == '__main__':
    main()
