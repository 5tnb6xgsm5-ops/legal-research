#!/usr/bin/env python3
"""
向量索引 / 语义检索（P1 + 神经层）
embed_index.py

设计（第一性原理）：
    RAG 的生成 = f(用户问题, 检索到的上下文)
    检索不是可选项，而是生成的架构性前置条件 → 默认 retrieve_first。
    本地知识库（tags.db）是首选检索源；IMA 仅用于拉取公众号，不存本地案卷。

神经层（bge-m3）：
    - 环境实测：HuggingFace 官方源在沙箱不可达（Connection reset）；
      hf-mirror 的 bge-m3 返回 404；modelscope resolve 端点对 2.3GB 文件返回错误。
    - 因此权重须由用户在【Mac 本机】下载（本机有网络/VPN），落入
      ~/.cache/huggingface/hub/models--BAAI--bge-m3/，沙箱可直接读取。
    - 本脚本自动探测该路径；命中即构建 doc_vectors，否则降级到同义词+FTS5。

依赖：transformers + torch 已装（无需 sentence_transformers）。
      bge-m3 加载：AutoModel + AutoTokenizer，mean-pooling + L2 归一化。

用法：
    python3 embed_index.py build            # 探测 bge-m3；命中则建向量，否则报告降级
    python3 embed_index.py search "砍头息"   # 同义词扩展 + FTS5 召回
    python3 embed_index.py hybrid "砍头息"   # 向量(若可用) + 同义词 + FTS5 双路
    python3 embed_index.py status            # 仅报告神经层可用性
"""

import os
import re
import sqlite3
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(SCRIPT_DIR, "..", "tags.db")

# ---------------------------------------------------------------------------
# 本地模型探测
# ---------------------------------------------------------------------------
def _discover_bge_m3():
    """在 HF cache 中查找 bge-m3 权重，返回 snapshot 目录或 None。"""
    hub_root = os.path.expanduser("~/.cache/huggingface/hub")
    base = os.path.join(hub_root, "models--BAAI--bge-m3")
    if not os.path.isdir(base):
        return None
    snap = os.path.join(base, "snapshots")
    if not os.path.isdir(snap):
        return None
    for rev in os.listdir(snap):
        for fname in ("model.safetensors", "pytorch_model.bin"):
            cand = os.path.join(snap, rev, fname)
            if os.path.isfile(cand) and os.path.getsize(cand) > 1_000_000:
                return os.path.join(snap, rev)
    return None


_BGE_PATH = _discover_bge_m3()
_MODEL = None
_MODEL_DIM = None

def _load_bge_m3():
    global _MODEL, _MODEL_DIM
    if _MODEL is not None or _BGE_PATH is None:
        return _MODEL
    from transformers import AutoModel, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(_BGE_PATH)
    mdl = AutoModel.from_pretrained(_BGE_PATH)
    mdl.eval()
    _MODEL = (mdl, tok)
    # bge-m3 隐藏层维度
    _MODEL_DIM = mdl.config.hidden_size
    return _MODEL


def neural_available():
    return _discover_bge_m3() is not None


# ---------------------------------------------------------------------------
# 同义词 / 近义词扩展（MVP 阶段零依赖语义召回，始终可用）
# ---------------------------------------------------------------------------
SYNONYM_GROUPS = [
    ["砍头息", "预先扣除利息", "息除本金", "变相高息", "抽头"],
    ["民间借贷", "借款合同纠纷", "借贷纠纷", "借钱", "放款"],
    ["优先受偿权", "建设工程价款优先受偿", "建工优先权", "工程款优先权"],
    ["违约金", "违约责任", "罚金", "赔偿金"],
    ["合同无效", "合同效力", "合同解除", "可撤销合同"],
    ["抵押", "抵押权", "质押", "担保物权"],
    ["不当得利", "无因管理"],
    ["诉讼时效", "除斥期间", "起诉期限"],
    ["夫妻共同财产", "婚前财产", "财产分割"],
    ["交通事故", "交通肇事", "交通事故责任"],
    ["工伤认定", "工伤保险", "劳动能力鉴定"],
    ["公司决议", "股东会决议", "董事会决议", "决议效力"],
    ["股权转让", "股东退出", "股份转让"],
    ["知识产权", "商标侵权", "专利侵权", "著作权"],
    ["破产", "破产清算", "破产重整", "破产债权"],
    ["执行", "强制执行", "申请执行", "终本"],
    ["管辖", "管辖权", "受案范围", "诉讼管辖"],
]

_SYN_MAP = {}
for _grp in SYNONYM_GROUPS:
    for _w in _grp:
        _SYN_MAP[_w] = _grp


def _tokenize(text: str) -> list:
    """中文分词：优先 jieba；未安装时退化为子串匹配 + 二元切分。"""
    t = (text or '').strip()
    if not t:
        return []
    try:
        import jieba
        return [w.strip() for w in jieba.cut(t) if w.strip()]
    except ImportError:
        pass
    # 退化方案：同义词表整词命中 + 原文二元切分（保证 FTS 通道可用）
    tokens = set()
    for k in _SYN_MAP:
        if k in t:
            tokens.add(k)
    flat = re.sub(r'\s+', '', t)
    for i in range(len(flat) - 1):
        tokens.add(flat[i:i + 2])
    return list(tokens)


def expand_query(query: str) -> list:
    tokens = _tokenize(query)
    expanded = set(tokens)
    for t in tokens:
        if t in _SYN_MAP:
            expanded.update(_SYN_MAP[t])
        for k, grp in _SYN_MAP.items():
            if k in t or t in k:
                expanded.update(grp)
    return list(expanded)


def fts_search(expanded_terms: list, db_path: str = DB, top_k: int = 20):
    conn = sqlite3.connect(db_path)
    results = {}
    for term in expanded_terms:
        try:
            rows = conn.execute(
                "SELECT doc_id, title FROM docs_fts WHERE docs_fts MATCH ? LIMIT ?",
                (term, top_k)
            ).fetchall()
        except Exception:
            continue
        for r in rows:
            if r[0] not in results:
                results[r[0]] = {"doc_id": r[0], "title": r[1]}
    conn.close()
    return list(results.values())


# ---------------------------------------------------------------------------
# 神经向量检索（bge-m3）
# ---------------------------------------------------------------------------
def _embed_texts(texts: list) -> "np.ndarray":
    """对一批文本做 mean-pooling + L2 归一化，返回 (n, dim) 数组。"""
    import torch, numpy as np
    mdl, tok = _load_bge_m3()
    out = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=512,
                      padding=True)
            o = mdl(**ids)
            h = o.last_hidden_state[0]
            mask = ids["attention_mask"][0].unsqueeze(-1)
            vec = (h * mask).sum(0) / mask.sum(0)
            vec = vec / (vec.norm() + 1e-9)
            out.append(vec.float().numpy())
    return np.vstack(out)


def _assemble_embed_text(title: str, body: str) -> str:
    """为 bge-m3 向量编码组装文本：优先提取裁判理由段。

    中国判决书结构：当事人→原告诉称→被告辩称→审理查明→本院认为→判决主文。
    旧策略 text[:1500] 截断在「审理查明」后，丢失最关键的「本院认为」裁判理由。
    新策略：找「本院认为」→ 取该段；无则头尾部策略（元数据+尾部结论）。
    """
    import re
    t = (title or "").strip()
    b = (body or "").strip()
    if not b:
        return t

    # Try to locate the court opinion section
    # Patterns: 本院认为, 本院再审认为, 本院经审理认为, etc.
    opinion_patterns = [
        r'本院\s*(?:经审理|再审)?\s*认为',
        r'本(?:院|庭)\s*(?:经审理|审查)?\s*(?:认为|查明)',
        r'裁决如下',
    ]
    best_start = -1
    for pat in opinion_patterns:
        m = re.search(pat, b)
        if m:
            idx = m.start()
            if best_start == -1 or idx < best_start:
                best_start = idx

    if best_start >= 0:
        # Found: title + metadata snippet + court opinion
        meta = b[:min(300, b.rfind('\n', 0, best_start))] if best_start > 300 else b[:best_start]
        opinion = b[best_start:best_start + 1200]
        return f"{t}\n{meta}\n{opinion}"

    # No court opinion section found — head+tail strategy
    # Head: first 600 chars (case metadata, claims)
    # Tail: last 800 chars (conclusion/judgment)
    head = b[:600]
    tail = b[-800:] if len(b) > 800 else ""
    return f"{t}\n{head}\n{tail}"


def build_vectors(db_path: str = DB, force: bool = False):
    """用 bge-m3 对全部文档编码并写入 doc_vectors。无权重则降级报告。"""
    if not neural_available():
        print("⚠️  神经层不可用（沙箱无法下载 bge-m3）。")
        print("    请在 Mac 本机执行下载后重试（见下方 status 中的命令）。")
        print("    当前语义召回由「同义词扩展 + FTS5」承载（已验证可用）。")
        return False
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS doc_vectors (
        doc_id TEXT PRIMARY KEY, model TEXT, dim INT, vec BLOB, built_at TEXT)""")
    rows = conn.execute(
        "SELECT doc_id, title, COALESCE(full_text,text_preview) FROM documents"
    ).fetchall()
    import datetime, numpy as np
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    for doc_id, title, body in rows:
        if (not force) and conn.execute(
                "SELECT 1 FROM doc_vectors WHERE doc_id=?", (doc_id,)).fetchone():
            continue
        # Smart text assembly: prioritize 裁判理由 (court opinion)
        # Old behaviour: text[:1500] cuts off before 本院认为, losing the key reasoning.
        text = _assemble_embed_text(title, body)
        vec = _embed_texts([text])[0]
        conn.execute(
            "INSERT OR REPLACE INTO doc_vectors(doc_id,model,dim,vec,built_at) "
            "VALUES(?,?,?,?,?)",
            (doc_id, "bge-m3", vec.shape[0], vec.astype("<f4").tobytes(), ts))
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM doc_vectors").fetchone()[0]
    conn.close()
    print(f"✅ bge-m3 向量已构建：{n} 篇，维度 {_MODEL_DIM}")
    return True


def vector_search(query: str, db_path: str = DB, top_k: int = 10):
    """纯向量召回（bge-m3）。无权重/无向量表返回空。"""
    if not neural_available():
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT v.doc_id, d.title, v.vec FROM doc_vectors v "
            "JOIN documents d ON v.doc_id = d.doc_id"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []  # doc_vectors 未构建（未跑过 embed_index.py build）
    finally:
        conn.close()
    if not rows:
        return []
    import numpy as np
    q = _embed_texts([query])[0].astype("<f4")
    scored = []
    for doc_id, title, blob in rows:
        v = np.frombuffer(blob, dtype="<f4")
        sim = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
        scored.append((sim, doc_id, title))
    scored.sort(reverse=True)
    return [{"doc_id": d, "title": t, "score": s}
            for s, d, t in scored[:top_k]]


# ---------------------------------------------------------------------------
# 组合检索
# ---------------------------------------------------------------------------
def semantic_search(query: str, db_path: str = DB, top_k: int = 10):
    """同义词扩展 + FTS5（始终可用）。"""
    expanded = expand_query(query)
    return fts_search(expanded, db_path, top_k * 2)[:top_k]


def _get_doc_authority(doc_id: str, db_path: str) -> int:
    """获取文档权威性权重（B-05 修复：法条优先于公众号文章）。
    
    权重规则（5 分制）：
    - 5 分：法条条文组（source_type 含「法条」或「法规」）
    - 4 分：指导案例/入库案例（source_type 含「指导案例」或「入库案例」）
    - 3 分：公报案例/典型案例（source_type 含「公报」或「典型」）
    - 2 分：普通裁判文书（source_type 含「判决」或「裁定」或「裁判文书」）
    - 1 分：公众号文章（source_type 含「公众号」或「文章」）
    - 0 分：未知类型
    """
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT source_type FROM documents WHERE doc_id=?",
            (doc_id,)
        ).fetchone()
        if not row:
            return 0
        st = (row[0] or '').lower()
        if any(k in st for k in ['法条', '法规', '法律']):
            return 5
        if any(k in st for k in ['指导案例', '入库案例']):
            return 4
        if any(k in st for k in ['公报', '典型案例']):
            return 3
        if any(k in st for k in ['判决', '裁定', '裁判文书', '裁决']):
            return 2
        if any(k in st for k in ['公众号', '文章', '推送']):
            return 1
        return 0
    except Exception:
        return 0
    finally:
        conn.close()


def hybrid_search(query: str, db_path: str = DB, top_k: int = 10):
    """双路：向量(若可用) + 同义词扩展 FTS5，合并去重并按权威性加权排序。
    
    B-05 修复：不再固定 lex 通道 score 0.15 垫底，而是按文档权威性加权：
    - 法条/法规类文档 score 加权提升，避免被公众号文章压制
    - 向量通道和 FTS 通道都按 doc_authority 调整最终排序
    """
    dense = vector_search(query, db_path, top_k)
    dense_ids = {d["doc_id"] for d in dense}
    lex = semantic_search(query, db_path, top_k)
    
    # 合并去重
    merged = list(dense)
    for h in lex:
        if h["doc_id"] not in dense_ids:
            merged.append({"doc_id": h["doc_id"], "title": h["title"],
                           "score": 0.15})
    
    # B-05：按文档权威性加权排序（法条优先于公众号文章）
    for item in merged:
        authority = _get_doc_authority(item["doc_id"], db_path)
        # 权威性加成：法条+0.5，指导案例+0.4，公报/典型+0.3，裁判文书+0.2，公众号+0
        item["authority"] = authority
        item["score"] = item.get("score", 0.15) + authority * 0.1
    
    # 按加权后 score 降序排序
    merged.sort(key=lambda x: x.get("score", 0), reverse=True)
    return merged[:top_k]


def status(db_path: str = DB):
    print("=== 神经层状态 ===")
    if neural_available():
        print(f"✅ bge-m3 已就绪：{_BGE_PATH}")
        conn = sqlite3.connect(db_path)
        n = conn.execute("SELECT COUNT(*) FROM doc_vectors").fetchone()[0]
        conn.close()
        print(f"   doc_vectors 已建：{n} 篇")
    else:
        print("⚠️  bge-m3 未在本机 HF cache 中找到。")
        print("   在【Mac 本机】终端执行（本机有网络/VPN）：")
        print("     pip install -U huggingface_hub")
        print("     hf download BAAI/bge-m3 --local-dir ~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/local")
        print("   或：")
        print("     python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3')\"")
        print("   下载完成后本脚本会自动探测并 build 向量，RAG 即具备真语义检索。")
    print(f"\n=== 同义词层 ===")
    print(f"   {len(_SYN_MAP)} 词 / {len(SYNONYM_GROUPS)} 组（始终可用）")


def build(db_path: str = DB):
    """入口：优先建向量；无权重则确认同义词层就绪。"""
    if build_vectors(db_path):
        return
    conn = sqlite3.connect(db_path)
    n = conn.execute("SELECT COUNT(*) FROM docs_fts").fetchone()[0]
    conn.close()
    print(f"✅ 同义词扩展索引就绪（{len(_SYN_MAP)} 词 / {len(SYNONYM_GROUPS)} 组）")
    print(f"   FTS5 索引文档数: {n}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='语义检索 / 向量索引')
    parser.add_argument('action', choices=['build', 'search', 'hybrid', 'status'],
                        help='build=建索引 / search=扩展检索 / hybrid=双路 / status=状态')
    parser.add_argument('query', nargs='?', default='')
    parser.add_argument('--db', default=DB)
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()

    if args.action == 'status':
        status(args.db)
        return
    if args.action == 'build':
        build(args.db)
        return
    if not args.query:
        print("❌ 需要 query")
        return

    if args.action == 'search':
        hits = semantic_search(args.query, args.db, args.top_k)
    else:
        hits = hybrid_search(args.query, args.db, args.top_k)

    if not hits:
        print("📭 无结果")
        return
    print(f"\n🔍 「{args.query}」— 召回 {len(hits)} 篇")
    for h in hits:
        sc = f" (sim={h['score']:.3f})" if 'score' in h and h['score'] != 0.15 else ""
        print(f"  📄 {h['title']}{sc}")


if __name__ == '__main__':
    main()
