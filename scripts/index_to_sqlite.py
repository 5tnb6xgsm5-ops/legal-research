#!/usr/bin/env python3
"""
法律文档 SQLite 索引引擎
index_to_sqlite.py

将 parse_legal_doc.py 产出的结构化 JSON 写入 SQLite 数据库，
支持：全文检索（FTS5）、元数据过滤、条目级查询。

用法：
    python3 index_to_sqlite.py ingest judgement_output.json -d legal.db
    python3 index_to_sqlite.py search "建设工程 合同纠纷" -d legal.db
    python3 index_to_sqlite.py list -d legal.db --type judgement
    python3 index_to_sqlite.py stats -d legal.db

数据库表结构：
    documents       - 文档主表
    sections        - 章节表（每个 section 一条记录）
    entities        - 实体表（案号/法院/当事人/金额等）
    documents_fts   - 全文搜索索引（FTS5）
"""

import argparse
import json
import os
import sqlite3
import sys
import re
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


# ════════════════════════════════════════════
# 数据库 Schema
# ════════════════════════════════════════════

SCHEMA_SQL = """
-- 文档主表
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT UNIQUE NOT NULL,
    doc_type TEXT NOT NULL DEFAULT 'unknown',
    source_file TEXT,
    body_text TEXT NOT NULL,
    raw_length INTEGER,
    clean_length INTEGER,
    quality_score REAL,
    parsed_at TEXT,
    indexed_at TEXT NOT NULL DEFAULT (datetime('now')),
    tags TEXT DEFAULT '[]'    -- JSON array of custom tags
);

-- 章节表
CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    section_type TEXT NOT NULL,
    category TEXT,
    title TEXT,
    level INTEGER DEFAULT 0,
    content TEXT,
    content_length INTEGER DEFAULT 0,
    section_order INTEGER DEFAULT 0     -- 在原文档中的顺序
);

-- 实体表
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    entity_type TEXT NOT NULL,           -- 'case_number' / 'court' / 'party' / 'amount' / 'date' / 'cause'
    entity_key TEXT,                     -- 'plaintiff' / 'defendant' / etc
    entity_value TEXT NOT NULL,
    entity_detail TEXT                   -- JSON extra data
);

-- 全文搜索（FTS5）
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    doc_id UNINDEXED,
    title,
    body,
    content=documents,
    content_rowid=id
);

-- 全文搜索索引触发器
CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, doc_id, title, body)
    VALUES (new.id, new.doc_id,
        (SELECT group_concat(title, ' | ') FROM sections WHERE doc_id = new.doc_id),
        new.body_text
    );
END;

CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, doc_id, title, body)
    VALUES ('delete', old.id, old.doc_id, '', '');
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, doc_id, title, body)
    VALUES ('delete', old.id, old.doc_id, '', '');
    INSERT INTO documents_fts(rowid, doc_id, title, body)
    VALUES (new.id, new.doc_id,
        (SELECT group_concat(title, ' | ') FROM sections WHERE doc_id = new.doc_id),
        new.body_text
    );
END;

-- 索引
CREATE INDEX IF NOT EXISTS idx_sections_doc_id ON sections(doc_id);
CREATE INDEX IF NOT EXISTS idx_sections_type ON sections(section_type);
CREATE INDEX IF NOT EXISTS idx_entities_doc_id ON entities(doc_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_value ON entities(entity_value);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_quality ON documents(quality_score);
"""


# ════════════════════════════════════════════
# 数据库引擎
# ════════════════════════════════════════════

class LegalIndex:
    """法律文档 SQLite 索引"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # --- 写入 ---

    def ingest(self, doc: Dict) -> str:
        """将解析后的文档写入索引，返回 doc_id"""
        doc_id = doc['doc_id']
        doc_type = doc.get('doc_type', 'unknown')
        body_text = doc.get('body_text', '')

        # 1. 写入主表（UPSERT）
        self.conn.execute("""
            INSERT INTO documents (doc_id, doc_type, source_file, body_text,
                raw_length, clean_length, quality_score, parsed_at, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(doc_id) DO UPDATE SET
                doc_type=excluded.doc_type,
                source_file=excluded.source_file,
                body_text=excluded.body_text,
                raw_length=excluded.raw_length,
                clean_length=excluded.clean_length,
                quality_score=excluded.quality_score,
                parsed_at=excluded.parsed_at,
                indexed_at=datetime('now')
        """, (
            doc_id,
            doc_type,
            doc.get('source_file', ''),
            body_text,
            doc.get('metadata', {}).get('raw_length', 0),
            doc.get('metadata', {}).get('clean_length', 0),
            doc.get('quality', {}).get('overall_score', 0),
            doc.get('parsed_at', ''),
        ))

        # 2. 写入章节
        self.conn.execute("DELETE FROM sections WHERE doc_id = ?", (doc_id,))
        for i, section in enumerate(doc.get('structure', {}).get('sections', [])):
            self.conn.execute("""
                INSERT INTO sections (doc_id, section_type, category, title, level,
                    content, content_length, section_order)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                section.get('type', 'unknown'),
                section.get('category', ''),
                section.get('title', ''),
                section.get('level', 0),
                section.get('content', ''),
                section.get('content_length', 0),
                i,
            ))

        # 3. 写入实体
        self.conn.execute("DELETE FROM entities WHERE doc_id = ?", (doc_id,))
        entities = doc.get('entities', {})

        # 案号
        if entities.get('case_number'):
            self._insert_entity(doc_id, 'case_number', '', entities['case_number'])

        # 法院
        if entities.get('court'):
            self._insert_entity(doc_id, 'court', '', entities['court'])

        # 案由
        if entities.get('cause_of_action'):
            self._insert_entity(doc_id, 'cause_of_action', '', entities['cause_of_action'])

        # 当事人
        for role, name in entities.get('parties', {}).items():
            self._insert_entity(doc_id, 'party', role, name)

        # 日期
        for d in entities.get('dates', []):
            self._insert_entity(doc_id, 'date', '', d)

        # 金额
        for amt in entities.get('amounts', []):
            self._insert_entity(doc_id, 'amount', '',
                f"{amt.get('value', 0)}{amt.get('unit', '元')}",
                json.dumps(amt, ensure_ascii=False))

        # 律师
        for lawyer in entities.get('lawyers', []):
            self._insert_entity(doc_id, 'lawyer', lawyer.get('name', ''),
                lawyer.get('firm', ''),
                json.dumps(lawyer, ensure_ascii=False))

        self.conn.commit()
        return doc_id

    def _insert_entity(self, doc_id, entity_type, entity_key, entity_value, detail=None):
        self.conn.execute("""
            INSERT INTO entities (doc_id, entity_type, entity_key, entity_value, entity_detail)
            VALUES (?, ?, ?, ?, ?)
        """, (doc_id, entity_type, entity_key, entity_value, detail))

    def ingest_file(self, json_path: str) -> str:
        """从 JSON 文件导入文档"""
        with open(json_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        return self.ingest(doc)

    # --- 查询 ---

    def search(self, query: str, doc_type: Optional[str] = None,
               limit: int = 20, offset: int = 0) -> List[Dict]:
        """全文搜索：jieba 分词 + SQLite LIKE 组合"""
        results = []

        # 策略1: jieba 分词后多关键词 LIKE 搜索
        if JIEBA_AVAILABLE:
            tokens = list(jieba.cut_for_search(query))
            keywords = [t for t in tokens if len(t) >= 2 and not t.isspace()]
            if keywords:
                conditions = " AND ".join(["d.body_text LIKE ?" for _ in keywords])
                sql = f"""
                    SELECT d.doc_id, d.doc_type, d.quality_score, d.body_text
                    FROM documents d
                    WHERE {conditions}
                """
                params = [f'%{kw}%' for kw in keywords[:10]]
                if doc_type:
                    sql += " AND d.doc_type = ?"
                    params.append(doc_type)
                sql += " ORDER BY d.quality_score DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                rows = self.conn.execute(sql, params).fetchall()
                results = [dict(r) for r in rows]

        # 策略2: 如果 jieba 不可用或无结果，用 LIKE 原始查询
        if not results:
            sql = """
                SELECT d.doc_id, d.doc_type, d.quality_score, d.body_text
                FROM documents d
                WHERE d.body_text LIKE ?
            """
            params = [f'%{query}%']
            if doc_type:
                sql += " AND d.doc_type = ?"
                params.append(doc_type)
            sql += " ORDER BY d.quality_score DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = self.conn.execute(sql, params).fetchall()
            results = [dict(r) for r in rows]

        # 添加 snippet（截取匹配上下文）
        for r in results:
            body = r.get('body_text', '')
            # 在正文中找第一个匹配词的位置
            pos = body.find(query) if query in body else 0
            start = max(0, pos - 30)
            end = min(len(body), pos + len(query) + 50)
            snippet = body[start:end]
            if start > 0:
                snippet = '...' + snippet
            if end < len(body):
                snippet = snippet + '...'
            # 高亮
            snippet = snippet.replace(query, f'<mark>{query}</mark>')
            r['snippet'] = snippet

        return results

    def list_documents(self, doc_type: Optional[str] = None,
                       min_score: Optional[float] = None,
                       limit: int = 50, offset: int = 0) -> List[Dict]:
        """列出文档"""
        sql = """
            SELECT doc_id, doc_type, quality_score, source_file,
                   clean_length, parsed_at, indexed_at
            FROM documents WHERE 1=1
        """
        params = []

        if doc_type:
            sql += " AND doc_type = ?"
            params.append(doc_type)
        if min_score is not None:
            sql += " AND quality_score >= ?"
            params.append(min_score)

        sql += " ORDER BY indexed_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """获取单个文档完整信息"""
        row = self.conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not row:
            return None

        doc = dict(row)

        # 加载章节
        sections = self.conn.execute(
            "SELECT * FROM sections WHERE doc_id = ? ORDER BY section_order",
            (doc_id,)
        ).fetchall()
        doc['sections'] = [dict(s) for s in sections]

        # 加载实体
        entities = self.conn.execute(
            "SELECT * FROM entities WHERE doc_id = ?",
            (doc_id,)
        ).fetchall()
        doc['entities'] = [dict(e) for e in entities]

        return doc

    def filter_by_entity(self, entity_type: str, entity_value: str,
                         limit: int = 20) -> List[Dict]:
        """按实体过滤（如按法院名、案由过滤）"""
        rows = self.conn.execute("""
            SELECT DISTINCT d.doc_id, d.doc_type, d.quality_score, d.source_file
            FROM entities e
            JOIN documents d ON d.doc_id = e.doc_id
            WHERE e.entity_type = ? AND e.entity_value LIKE ?
            LIMIT ?
        """, (entity_type, f'%{entity_value}%', limit)).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> Dict:
        """统计信息"""
        return {
            'total_documents': self.conn.execute(
                "SELECT COUNT(*) FROM documents").fetchone()[0],
            'by_type': {
                r['doc_type']: r['cnt'] for r in
                self.conn.execute(
                    "SELECT doc_type, COUNT(*) as cnt FROM documents GROUP BY doc_type"
                ).fetchall()
            },
            'by_section_type': {
                r['section_type']: r['cnt'] for r in
                self.conn.execute(
                    "SELECT section_type, COUNT(*) as cnt FROM sections GROUP BY section_type"
                ).fetchall()
            },
            'total_entities': self.conn.execute(
                "SELECT COUNT(*) FROM entities").fetchone()[0],
            'avg_quality': self.conn.execute(
                "SELECT AVG(quality_score) FROM documents").fetchone()[0] or 0,
            'top_courts': [
                dict(r) for r in self.conn.execute(
                    "SELECT entity_value, COUNT(*) as cnt FROM entities "
                    "WHERE entity_type='court' GROUP BY entity_value "
                    "ORDER BY cnt DESC LIMIT 10"
                ).fetchall()
            ],
            'top_causes': [
                dict(r) for r in self.conn.execute(
                    "SELECT entity_value, COUNT(*) as cnt FROM entities "
                    "WHERE entity_type='cause_of_action' GROUP BY entity_value "
                    "ORDER BY cnt DESC LIMIT 10"
                ).fetchall()
            ],
        }

    def export_jsonl(self, output_path: str, doc_type: Optional[str] = None):
        """导出为 JSONL 格式（便于批量处理）"""
        sql = "SELECT doc_id, doc_type, body_text, quality_score FROM documents"
        params = []
        if doc_type:
            sql += " WHERE doc_type = ?"
            params.append(doc_type)

        rows = self.conn.execute(sql, params).fetchall()
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in rows:
                obj = {
                    'doc_id': row['doc_id'],
                    'doc_type': row['doc_type'],
                    'body_text': row['body_text'],
                    'quality_score': row['quality_score'],
                }
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')

        print(f"✅ 导出 {len(rows)} 条记录到 {output_path}")


# ════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='法律文档 SQLite 索引引擎',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('action', choices=['ingest', 'search', 'list', 'get', 'filter',
                                           'stats', 'export'],
                       help='操作类型')
    parser.add_argument('target', nargs='?', help='操作目标（文件路径/搜索词/doc_id）')
    parser.add_argument('-d', '--db', default='legal_docs.db', help='数据库路径 (默认: legal_docs.db)')
    parser.add_argument('-t', '--type', dest='doc_type', help='按文档类型过滤')
    parser.add_argument('--min-score', type=float, help='最低质量分')
    parser.add_argument('-n', '--limit', type=int, default=20, help='返回数量 (默认20)')
    parser.add_argument('-o', '--output', help='导出文件路径')
    parser.add_argument('--entity-type', help='实体类型 (用于 filter 命令)')
    parser.add_argument('--entity-value', help='实体值 (用于 filter 命令)')

    args = parser.parse_args()

    index = LegalIndex(args.db)

    try:
        if args.action == 'ingest':
            if not args.target:
                print("❌ ingest 需要指定 JSON 文件路径", file=sys.stderr)
                sys.exit(1)
            doc_id = index.ingest_file(args.target)
            print(f"✅ 已索引: {doc_id}")

        elif args.action == 'search':
            if not args.target:
                print("❌ search 需要搜索词", file=sys.stderr)
                sys.exit(1)
            results = index.search(args.target, doc_type=args.doc_type, limit=args.limit)
            for r in results:
                print(f"\n📄 {r['doc_id']} [{r['doc_type']}] 评分:{r['quality_score']}")
                print(f"   {r['snippet']}")

        elif args.action == 'list':
            results = index.list_documents(
                doc_type=args.doc_type,
                min_score=args.min_score,
                limit=args.limit)
            for r in results:
                print(f"  {r['doc_id']:30s} [{r['doc_type']:10s}] "
                      f"评分:{r['quality_score'] or '-':>5} "
                      f"{r['clean_length']:>6}字 "
                      f"{r['indexed_at'][:19] if r['indexed_at'] else ''}")

        elif args.action == 'get':
            if not args.target:
                print("❌ get 需要 doc_id", file=sys.stderr)
                sys.exit(1)
            doc = index.get_document(args.target)
            if doc:
                print(json.dumps({
                    'doc_id': doc['doc_id'],
                    'doc_type': doc['doc_type'],
                    'quality_score': doc['quality_score'],
                    'sections': [{'type': s['section_type'], 'title': s['title'], 'content_len': s['content_length']} for s in doc['sections']],
                    'entities': [{'type': e['entity_type'], 'key': e['entity_key'], 'value': e['entity_value']} for e in doc['entities']],
                }, ensure_ascii=False, indent=2))
            else:
                print(f"❌ 未找到: {args.target}")

        elif args.action == 'filter':
            if not args.entity_type or not args.entity_value:
                print("❌ filter 需要 --entity-type 和 --entity-value", file=sys.stderr)
                sys.exit(1)
            results = index.filter_by_entity(args.entity_type, args.entity_value, limit=args.limit)
            for r in results:
                print(f"  {r['doc_id']:30s} [{r['doc_type']:10s}] 评分:{r['quality_score']}")

        elif args.action == 'stats':
            stats = index.stats()
            print(json.dumps(stats, ensure_ascii=False, indent=2))

        elif args.action == 'export':
            index.export_jsonl(args.output or 'export.jsonl', doc_type=args.doc_type)

    finally:
        index.close()


if __name__ == '__main__':
    main()
