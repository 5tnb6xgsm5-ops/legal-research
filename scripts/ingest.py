#!/usr/bin/env python3
"""
批量文档导入管线
ingest.py

扫目录 → 判断类型 → 提取文本 → 解析 → 入库 SQLite

用法：
    python3 ingest.py /path/to/pdfs/                    # 扫描目录
    python3 ingest.py /path/to/pdfs/ --db legal.db      # 指定数据库
    python3 ingest.py /path/to/pdfs/ --force             # 强制重新处理所有文件
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    import fitz  # pymupdf
except ImportError:
    print('❌ 缺少依赖 pymupdf，请先执行: pip install pymupdf')
    sys.exit(2)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_SCRIPT = os.path.join(SCRIPT_DIR, 'index_to_sqlite.py')
PYTHON = sys.executable


def _contains_local_case_material(directory: str, db_path: str) -> bool:
    """合规保护：判断待摄入目录是否含本地案件材料（自办案件）。

    判定规则：目录路径落在本地案件归档树（归档-01-案件与项目）下，
    即视为本地案件材料，禁止推送 IMA。IMA 仅用于采集公众号文章。
    返回 True 表示应被拦截。
    """
    # 规则 1：路径落在案件归档树
    case_root_markers = ['归档-01-案件与项目', '案件与项目', '自办案件']
    norm = os.path.normpath(directory)
    for marker in case_root_markers:
        if marker in norm:
            return True
    # 规则 2：目录内已有文档在 tags.db 中标记为自办案件
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT DISTINCT source_type FROM documents"
            ).fetchall()
            types = [r[0] for r in rows]
            if any('自办案件' in t for t in types):
                # 仅当本目录文件已对应自办案件时拦截
                for f in os.listdir(directory):
                    if f.lower().endswith('.pdf'):
                        doc_id = os.path.splitext(f)[0]
                        hit = conn.execute(
                            "SELECT 1 FROM documents WHERE doc_id=? AND source_type LIKE '%自办案件%'",
                            (doc_id,)
                        ).fetchone()
                        if hit:
                            conn.close()
                            return True
        except Exception:
            pass
        finally:
            conn.close()
    return False


def detect_pdf_type(pdf_path: str, password: str = None) -> str:
    """判断 PDF 是数字文档（有文字层）还是扫描件（纯图片）"""
    doc = fitz.open(pdf_path)
    if doc.needs_pass:
        if password:
            doc.authenticate(password)
        else:
            doc.close()
            return 'scanned'  # 加密无密码→假定为扫描件
    text_chars = 0
    total_pages = len(doc)
    for page in doc:
        text_chars += len(page.get_text().strip())
    doc.close()

    avg_chars = text_chars / max(total_pages, 1)
    if avg_chars > 100:
        return 'digital'
    elif avg_chars > 10:
        return 'mixed'
    return 'scanned'


def already_indexed(pdf_path: str, db_path: str) -> bool:
    """检查文件是否已入库（按 doc_id 查）"""
    if not os.path.exists(db_path):
        return False
    conn = sqlite3.connect(db_path)
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    cursor = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE source_file = ? OR doc_id = ?",
        (os.path.basename(pdf_path), doc_id)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def extract_digital(pdf_path: str, password: str = None) -> str:
    """从数字 PDF 直接提取文本"""
    doc = fitz.open(pdf_path)
    if doc.needs_pass and password:
        doc.authenticate(password)
    texts = []
    for page in doc:
        t = page.get_text().strip()
        if t:
            texts.append(t)
    doc.close()
    return '\n\n'.join(texts)


def extract_scanned(pdf_path: str, password: str = None, dpi: int = 300) -> str:
    """扫描件文本提取。

    完整版包不含 OCR 脚本（scan_pdf_ocr.py 属内部增强件）。
    扫描件无文字层，直接提取会得到空文本，由主流程按失败处理并给出提示。
    """
    print('  ⚠️ 扫描件需要 OCR 能力（完整版包未附 OCR 脚本），本次跳过。')
    print('     处理方式：先用你自己的 OCR 工具把扫描件转成带文字层的 PDF，再重新跑 ingest。')
    return ""


def parse_document(text: str, doc_id: str, output_dir: str = '/tmp') -> Optional[str]:
    """提取的全文 → 最小结构化 JSON（供 index_to_sqlite.py ingest 使用）。

    完整版包不含 parse_legal_doc.py（章节切分/实体抽取属内部增强件），
    此处生成 ingest 兼容的最小 JSON，保证资料库链路开箱可用。
    """
    if len(text.strip()) < 50:
        return None

    json_path = os.path.join(output_dir, f'{doc_id}_parsed.json')
    doc = {
        'doc_id': doc_id,
        'doc_type': 'judgement',
        'source_file': f'{doc_id}.pdf',
        'body_text': text,
        'metadata': {'raw_length': len(text), 'clean_length': len(text)},
        'quality': {'overall_score': 0.5},
        'parsed_at': datetime.now().isoformat(timespec='seconds'),
        'structure': {'sections': []},
        'entities': {},
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    return json_path


def index_document(json_path: str, db_path: str) -> bool:
    """将解析结果入库"""
    result = subprocess.run(
        [PYTHON, INDEX_SCRIPT, 'ingest', json_path, '-d', db_path],
        capture_output=True, text=True, timeout=30
    )
    return result.returncode == 0


def scan_directory(directory: str, extensions: List[str] = None) -> List[str]:
    """递归扫描目录中的 PDF 文件"""
    if extensions is None:
        extensions = ['.pdf']
    files = []
    for root, _, filenames in os.walk(directory):
        for fn in filenames:
            if any(fn.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(root, fn))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description='批量文档导入管线')
    parser.add_argument('directory', help='PDF 目录路径')
    parser.add_argument('--db', default='legal_docs.db', help='SQLite 数据库路径')
    parser.add_argument('--password', help='加密 PDF 密码')
    parser.add_argument('--force', action='store_true', help='强制重新处理已入库文件')
    parser.add_argument('--dpi', type=int, default=300, help='扫描件 OCR DPI')
    parser.add_argument('--push-ima', help='入库后推送到 IMA 知识库（知识库 ID）。注意：本地案件材料（source_type 含「自办案件」）禁止推送，仅限公众号等已公开素材')
    args = parser.parse_args()

    # 合规保护：本地案件材料（自办案件）一律禁止上云
    # 设计原则：IMA 仅作公众号文章采集入口，本地材料不推 IMA（用户 2026-07-06 决定）
    if args.push_ima:
        blocked = _contains_local_case_material(args.directory, args.db)
        if blocked:
            print('❌ 合规拦截：检测到本地案件材料（自办案件），禁止推送到 IMA。')
            print('   IMA 仅用于采集公众号文章，本地材料请保留在 tags.db（本地真相源）。')
            sys.exit(2)

    pdf_files = scan_directory(args.directory)
    if not pdf_files:
        print(f'❌ 目录中没有 PDF 文件: {args.directory}')
        sys.exit(1)

    print(f'📂 找到 {len(pdf_files)} 个 PDF 文件\n')

    stats = {'total': len(pdf_files), 'skipped': 0, 'digital': 0,
             'scanned': 0, 'parsed': 0, 'indexed': 0, 'failed': 0}

    start_time = time.time()

    for i, pdf_path in enumerate(pdf_files):
        fname = os.path.basename(pdf_path)
        print(f'[{i+1}/{len(pdf_files)}] {fname}', flush=True)

        # 检查是否已入库
        if not args.force and already_indexed(pdf_path, args.db):
            print(f'  ⏭️  已入库，跳过')
            stats['skipped'] += 1
            continue

        # 判断类型
        pdf_type = detect_pdf_type(pdf_path, args.password)
        doc_id = os.path.splitext(fname)[0]

        # 提取文本
        t0 = time.time()
        if pdf_type == 'digital':
            text = extract_digital(pdf_path, args.password)
            stats['digital'] += 1
            print(f'  📄 数字 PDF → {len(text)} 字 ({time.time()-t0:.1f}s)')
        elif pdf_type == 'scanned':
            text = extract_scanned(pdf_path, args.password, args.dpi)
            stats['scanned'] += 1
            if text:
                print(f'  🖼️  扫描件 OCR → {len(text)} 字 ({time.time()-t0:.1f}s)')
            else:
                print(f'  ❌ OCR 失败')
                stats['failed'] += 1
                continue
        else:
            text = extract_digital(pdf_path, args.password)
            stats['digital'] += 1
            print(f'  ⚠️  混合型 → {len(text)} 字 ({time.time()-t0:.1f}s)')

        # 文本太少，跳过
        if len(text.strip()) < 50:
            print(f'  ❌ 文本太少 ({len(text)} 字)，无法解析')
            stats['failed'] += 1
            continue

        # 解析
        t0 = time.time()
        json_path = parse_document(text, doc_id)
        if json_path:
            stats['parsed'] += 1
            print(f'  🔧 解析完成 ({time.time()-t0:.1f}s)')
        else:
            print(f'  ❌ 解析失败')
            stats['failed'] += 1
            continue

        # 入库
        t0 = time.time()
        if index_document(json_path, args.db):
            stats['indexed'] += 1
            print(f'  ✅ 入库完成 ({time.time()-t0:.1f}s)')
            os.unlink(json_path)

            # IMA 自动推送（需自行实现 push_to_ima，完整版包不含该模块）
            if args.push_ima:
                try:
                    from push_to_ima import push_to_ima
                    push_to_ima(args.db, args.push_ima, doc_ids=[doc_id], dry_run=False)
                except ImportError:
                    print('  ⚠️ 未找到 push_to_ima 模块，跳过 IMA 推送（不影响本地入库）')
                except Exception as e:
                    print(f'  ⚠️ IMA推送失败: {e}')
        else:
            print(f'  ❌ 入库失败')
            stats['failed'] += 1

    elapsed = time.time() - start_time

    # 汇总报告
    print(f'\n{"="*60}')
    print(f'📊 导入报告')
    print(f'   总计: {stats["total"]} 文件')
    print(f'   跳过(已入库): {stats["skipped"]}')
    print(f'   数字 PDF: {stats["digital"]}')
    print(f'   扫描件: {stats["scanned"]}')
    print(f'   解析成功: {stats["parsed"]}')
    print(f'   入库成功: {stats["indexed"]}')
    print(f'   失败: {stats["failed"]}')
    print(f'   总耗时: {elapsed:.0f}s')
    print(f'{"="*60}')

    # 索引统计
    if os.path.exists(args.db) and stats['indexed'] > 0:
        print(f'\n📈 数据库统计:')
        subprocess.run([PYTHON, INDEX_SCRIPT, 'stats', '-d', args.db])


if __name__ == '__main__':
    main()
