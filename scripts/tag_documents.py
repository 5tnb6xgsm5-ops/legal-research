#!/usr/bin/env python3
"""
元数据标签引擎 v2
tag_documents.py

从文档文本提取案由/法院/关键词等元数据，写入本地 tags.db。
支持批量处理和单文档处理。

结合 schema: index_to_sqlite.py 中的 tags 表
"""

import json, os, re, sqlite3, sys, time, hashlib
from datetime import datetime
from typing import Dict, List, Optional

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tags.db")


def tokenize(text: str) -> str:
    """中文分词：优先 jieba；未安装时退化为二元切分，保证 FTS5 检索可用。"""
    t = (text or '').strip()
    if not t:
        return ''
    if JIEBA_AVAILABLE:
        return ' '.join(w.strip() for w in jieba.cut(t) if w.strip())
    flat = re.sub(r'\s+', '', t)
    if len(flat) <= 2:
        return flat
    return ' '.join(flat[i:i + 2] for i in range(len(flat) - 1))


def init_db(db_path: str = None):
    """初始化 tags.db 表结构"""
    db = db_path or DEFAULT_DB
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            source_type TEXT DEFAULT '',
            content_type TEXT DEFAULT '',
            doc_authority INTEGER DEFAULT 3,
            kb_id TEXT DEFAULT '',
            ima_media_id TEXT DEFAULT '',
            
            case_types TEXT DEFAULT '[]',
            dispute_focus TEXT DEFAULT '[]',
            procedure_type TEXT DEFAULT '',
            
            court TEXT DEFAULT '',
            court_level TEXT DEFAULT '',
            court_path TEXT DEFAULT '',
            
            keywords TEXT DEFAULT '[]',
            judgment_date TEXT DEFAULT '',
            parties TEXT DEFAULT '[]',
            amount TEXT DEFAULT '',
            
            content_hash TEXT UNIQUE,
            text_preview TEXT DEFAULT '',
            full_text TEXT DEFAULT '',
            ingested_at TEXT DEFAULT (datetime('now'))
        );
        
        CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
            doc_id UNINDEXED, title, case_types, dispute_focus, keywords, court,
            content
        );
    """)
    conn.commit()
    conn.close()
    return db


def content_hash(text: str) -> str:
    # 归一化后再计算 hash：全角→半角、统一标点、去空白差异
    import unicodedata
    norm = text.replace('\r\n', '\n').replace('\u3000', ' ')
    norm = unicodedata.normalize('NFKC', norm)
    return hashlib.sha256(norm.encode('utf-8')).hexdigest()[:16]


# ---------------------------------------------------------------------------
# P0-1: 刑事案卷拦截（路径 + 文件名 + 文本 三级过滤）
# ---------------------------------------------------------------------------
CRIMINAL_PATH_KEYWORDS = [
    '刑事', '故意伤害', '非法采矿', '抽逃出资', '强迫卖淫', '容留卖淫',
    '诉讼文书卷', '证据卷', '侦查卷', '公安卷', '检察卷', '起诉意见书',
]
CRIMINAL_TEXT_KEYWORDS = [
    '讯问笔录', '刑事拘留', '逮捕', '侦查机关', '犯罪嫌疑人',
    '提起公诉', '审查起诉', '监视居住', '取保候审',
]


def contains_criminal_material(filepath: str = '', text: str = '') -> tuple:
    """检查是否包含刑事案卷材料。返回 (是否刑案, 命中理由)。

    路径/文件名级 + 文本级双重过滤。
    依据：律师法第38条、刑诉法第54条——刑案卷宗属于国家秘密/侦查秘密范畴，
    不得入库为全文可检索的知识条目。
    """
    reasons = []

    # Level 1: 路径/文件名检测
    fp = (filepath or '').lower()
    for kw in CRIMINAL_PATH_KEYWORDS:
        if kw in fp:
            reasons.append(f'路径命中「{kw}」')

    # Level 2: 文本内容关键词检测
    txt = (text or '')
    for kw in CRIMINAL_TEXT_KEYWORDS:
        if kw in txt:
            reasons.append(f'文本命中「{kw}」')

    # Level 3: 案号模式检测（刑初/刑终 字样的判决书）
    if re.search(r'[（(]\d{4}[）)]\s*\S{0,4}刑[初终]\w*\d+\s*号', txt):
        reasons.append('案号含「刑初/刑终」')

    return (len(reasons) > 0, reasons)


# ---------------------------------------------------------------------------
# P0-2: PII 脱敏过滤器
# ---------------------------------------------------------------------------
def anonymize_pii(text: str) -> str:
    """对文本中的个人信息做脱敏处理。按顺序执行，避免误匹配。

    - 身份证号（18/15位）：优先脱敏，防止尾部段落被手机/银行卡误匹配
    - 手机号：11位
    - 银行卡号：16-19位（排除已脱敏身份证号位置）
    - 住址：掩码为 [住址已脱敏]
    """
    if not text:
        return text
    t = text

    # 1. 身份证号 18 位（优先，防止被手机/银行卡误匹配）
    t = re.sub(
        r'(?<!\d)(\d{6})(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}([\dXx])(?!\d)',
        lambda m: m.group(1) + '****' + '****' + m.group(5), t)

    # 2. 身份证号 15 位
    t = re.sub(
        r'(?<!\d)(\d{6})\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}(?!\d)',
        lambda m: m.group(1) + '****' + '**', t)

    # 3. 手机号（11位，1开头）—— 身份证已脱敏，尾段误匹配风险消除
    t = re.sub(r'(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)',
               r'\1****\3', t)

    # 4. 银行卡号（16-19位）—— 身份证已脱敏，不再误匹配
    t = re.sub(r'(?<!\d)(\d{4})\d{8,11}(\d{4})(?!\d)',
               r'\1****\2', t)

    # 5. 住址（非贪婪，只匹配到第一个句号/逗号/换行）
    t = re.sub(r'(住址|住所地|户籍地)[：:]\s*[^。,\n]{1,80}',
               r'\1：[住址已脱敏]', t)

    # 6. 身份证号/银行卡的标签写法
    t = re.sub(r'(身份证号|身份证号码|公民身份号码)[：:]\s*\*+\d+[\dXx*]*',
               r'\1：[身份证号已脱敏]', t)

    return t


# ---------------------------------------------------------------------------
def extract_court_info(court_name: str) -> dict:
    """从法院全名提取层级和路径"""
    if not court_name:
        return {'court': '', 'court_level': '', 'court_path': ''}
    
    level_map = {
        '最高人民法院': '最高',
        '高级人民法院': '高级',
        '中级人民法院': '中级',
        '人民法院': '基层',
    }
    level = ''
    path = court_name
    
    for k, v in level_map.items():
        if k in court_name:
            level = v
            break
    
    # Build path from province-city-court
    parts = []
    province_match = re.match(r'(北京市|上海市|天津市|重庆市|[\u4e00-\u9fff]{2,3}省|[\u4e00-\u9fff]+自治区)', court_name)
    if province_match:
        parts.append(province_match.group(1))
    
    return {'court': court_name, 'court_level': level, 'court_path': '/'.join(parts) if parts else court_name}


def extract_metadata(text: str, title: str = "", source_type: str = "自办案件") -> dict:
    """从文本提取法律元数据"""
    meta = {
        'source_type': source_type,
        'content_type': '',
        'doc_authority': 3,
        'case_types': [],
        'dispute_focus': [],
        'procedure_type': '',
        'court': '',
        'court_level': '',
        'court_path': '',
        'keywords': [],
        'judgment_date': '',
        'parties': [],
        'amount': '',
    }
    
    # === 案由（header-first strategy: 法院文书前500字含案由，准确率远高于全文模糊匹配） ===
    cause_map = [
        (r'建设工程价款优先受偿权', '建设工程价款优先受偿权纠纷', '建设工程'),
        (r'建设工程施工合同', '建设工程施工合同纠纷', '建设工程'),
        (r'建设工程.*?纠纷', '建设工程合同纠纷', '建设工程'),
        (r'民间借贷', '民间借贷纠纷', '民间借贷'),
        (r'金融借款合同', '金融借款合同纠纷', '金融借款'),
        (r'买卖合同', '买卖合同纠纷', '合同'),
        (r'租赁合同', '租赁合同纠纷', '合同'),
        (r'承揽合同', '承揽合同纠纷', '合同'),
        (r'劳动争议|劳动.*?合同', '劳动争议', '劳动'),
        (r'离婚', '离婚纠纷', '婚姻家庭'),
        (r'继承', '继承纠纷', '婚姻家庭'),
        (r'侵害.*?商标|商标.*?侵权', '侵害商标权纠纷', '知识产权'),
        (r'著作权|专利', '知识产权纠纷', '知识产权'),
        (r'公司.*?纠纷|股权', '与公司有关的纠纷', '公司'),
        (r'引诱.*?卖淫|容留.*?卖淫|强迫.*?卖淫', '引诱、容留、介绍卖淫罪', '刑事'),
        (r'诈骗|盗窃|故意伤害|交通肇事|危险驾驶', '', '刑事'),
        (r'行政.*?诉讼|行政.*?处罚', '', '行政'),
        (r'破产.*?债权|破产.*?确认', '破产债权确认纠纷', '破产'),
        (r'破产', '破产', '破产'),
        (r'保证合同|担保', '保证合同纠纷', '担保'),
        (r'票据.*?纠纷', '票据纠纷', '票据'),
    ]

    # 优先在文书头部区域匹配（前 500 字符 + 标题），避免底部法律条款被误匹配
    header_text = text[:500] + "\n" + title
    for pat, case, cat in cause_map:
        if re.search(pat, header_text):
            if case and case not in meta['case_types']:
                meta['case_types'].append(case)
            if cat and cat not in meta['dispute_focus']:
                meta['dispute_focus'].append(cat)

    # 头部没找到才回退全文搜索
    if not meta['case_types']:
        for pat, case, cat in cause_map:
            if re.search(pat, text):
                if case and case not in meta['case_types']:
                    meta['case_types'].append(case)
                if cat and cat not in meta['dispute_focus']:
                    meta['dispute_focus'].append(cat)
    
    # === 内容类型和权威 ===
    if source_type == '司法解释':
        meta['content_type'] = '司法解释'
        meta['doc_authority'] = 5
    elif source_type == '公众号':
        if re.search(r'判决书|裁定书|裁判原文|判决如下|依照《', text):
            meta['content_type'] = '法院案例'
            meta['doc_authority'] = 4
        elif re.search(r'最高法|司法解释|指导案例|典型案例', text):
            meta['content_type'] = '案例分析'
            meta['doc_authority'] = 3
        elif re.search(r'法院.*?发布|公告|通知|通告', text):
            meta['content_type'] = '新闻通告'
            meta['doc_authority'] = 2
        else:
            meta['content_type'] = '普法文章'
            meta['doc_authority'] = 2
    elif source_type == '自办案件':
        meta['content_type'] = '裁判原文'
        meta['doc_authority'] = 4
    
    # === 法院 ===
    court_match = re.search(
        r'((?:最高|浙江省|江苏省|北京市|上海市|广东省|[\u4e00-\u9fff]{2,3}省|[\u4e00-\u9fff]+市|[\u4e00-\u9fff]+县|[\u4e00-\u9fff]+区)'
        r'(?:高级|中级|基层)?'
        r'人民法院|仲裁委员会|劳动人事争议仲裁院)',
        text
    )
    if court_match:
        cinfo = extract_court_info(court_match.group(1))
        meta.update(cinfo)
    
    # === 程序类型 ===
    if re.search(r'执行|申请执行|强制执行', text):
        meta['procedure_type'] = '执行'
    elif re.search(r'仲裁', text):
        meta['procedure_type'] = '仲裁'
    elif re.search(r'裁定.*?保全|财产保全|证据保全', text):
        meta['procedure_type'] = '保全'
    elif re.search(r'判决|裁定|调解', text):
        meta['procedure_type'] = '诉讼'
    
    # === 争议焦点关键词 ===
    focus_patterns = {
        '优先受偿权': r'优先受偿|优先权',
        '违约责任': r'违约|违约责任',
        '合同效力': r'合同无效|合同有效|合同效力|无效合同',
        '损害赔偿': r'损害赔偿|赔偿损失|侵权赔偿',
        '工程款': r'工程款|工程价款',
        '连带责任': r'连带责任|共同还款',
        '担保责任': r'担保责任|保证责任|保证人',
        '实际施工人': r'实际施工人',
        '破产债权': r'破产债权|债权申报',
        '举证责任': r'举证责任|举证不能',
        '诉讼时效': r'诉讼时效|除斥期间',
        '管辖权': r'管辖权|管辖异议',
    }
    for kw, pat in focus_patterns.items():
        if re.search(pat, text) and kw not in meta['keywords']:
            meta['keywords'].append(kw)
    
    # === 金额 ===
    amt = re.search(r'(\d[\d,]*\.?\d*)\s*[万元][元]?', text)
    if amt:
        meta['amount'] = amt.group(0)
    
    # === 日期 ===
    date_match = re.search(r'(\d{4})[-年](\d{1,2})[-月](\d{1,2})', text)
    if date_match:
        meta['judgment_date'] = f"{date_match.group(1)}-{date_match.group(2).zfill(2)}-{date_match.group(3).zfill(2)}"
    
    # === 当事人 ===
    party_matches = re.findall(r'(?:原告|被告|申请人|被申请人|上诉人|被上诉人)[：:]\s*([\u4e00-\u9fff]{2,4})', text)
    if party_matches:
        meta['parties'] = list(set(party_matches))[:5]
    
    return meta


def tag_document(db_path: str, doc_id: str, title: str, text: str,
                 source_type: str = "自办案件", kb_id: str = "",
                 ima_media_id: str = "", filepath: str = "") -> bool:
    """打标签、脱敏、合规拦截后写入数据库。

    返回 True 表示成功入库，False 表示被拒绝（合规拦截/重复等）。
    """
    # P0-1: 刑事案卷拦截（入库前第一道屏障）
    is_crim, crim_reasons = contains_criminal_material(filepath, text)
    if is_crim:
        print(f"  🚫 刑事案卷拦截: {'; '.join(crim_reasons)} — 已拒绝入库")
        return False

    # P0-2: PII 脱敏
    safe_text = anonymize_pii(text)

    # 提取元数据（基于脱敏后文本，避免 PII 进入元数据字段）
    meta = extract_metadata(safe_text, title, source_type)
    
    chash = content_hash(safe_text)
    
    conn = sqlite3.connect(db_path)
    try:
        # Check dedup
        existing = conn.execute(
            "SELECT doc_id FROM documents WHERE content_hash = ? AND doc_id != ?",
            (chash, doc_id)
        ).fetchone()
        if existing:
            print(f"  ⚠️ 重复内容，已存在: {existing[0]}")
            return False
        
        conn.execute("""
            INSERT OR REPLACE INTO documents
            (doc_id, title, source_type, content_type, doc_authority,
             kb_id, ima_media_id, case_types, dispute_focus, procedure_type,
             court, court_level, court_path, keywords, judgment_date,
             parties, amount, content_hash, text_preview, full_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id, title, source_type, meta['content_type'], meta['doc_authority'],
            kb_id, ima_media_id,
            json.dumps(meta['case_types'], ensure_ascii=False),
            json.dumps(meta['dispute_focus'], ensure_ascii=False),
            meta['procedure_type'],
            meta['court'], meta['court_level'], meta['court_path'],
            json.dumps(meta['keywords'], ensure_ascii=False),
            meta['judgment_date'],
            json.dumps(meta['parties'], ensure_ascii=False),
            meta['amount'],
            chash,
            safe_text[:500],
            safe_text[:20000]
        ))

        # 写入 FTS5 索引（含全文正文分词，保证摄入文档可被 FTS 通道召回）
        content_blob = tokenize(title + ' ' + ' '.join(meta['keywords']) +
                                ' ' + ' '.join(meta['case_types']) +
                                ' ' + ' '.join(meta['dispute_focus']) +
                                ' ' + safe_text[:5000])
        conn.execute("""INSERT OR REPLACE INTO docs_fts
            (doc_id, title, case_types, dispute_focus, keywords, court, content)
            VALUES (?,?,?,?,?,?,?)""",
            (doc_id, title,
             json.dumps(meta['case_types'], ensure_ascii=False),
             json.dumps(meta['dispute_focus'], ensure_ascii=False),
             json.dumps(meta['keywords'], ensure_ascii=False),
             meta.get('court', ''),
             content_blob))

        conn.commit()
        return True
    except sqlite3.IntegrityError as e:
        if 'UNIQUE' in str(e):
            print(f"  ⚠️ 已存在（hash冲突）")
        return False
    finally:
        conn.close()


def tag_guard_data(db_path: str, kb_id: str, guard_path: str):
    """从 Guard 格式 JSON 批量打标签（需自行提供 --guard 数据文件）"""
    if not os.path.exists(guard_path):
        print(f"❌ Guard 数据不存在: {guard_path}")
        return

    init_db(db_path)

    with open(guard_path) as f:
        guard = json.load(f)
    
    tagged = 0
    for r in guard['results']:
        fname = r['file'].split('/')[-1]
        title = fname.replace('.pdf', '')
        doc_id = title
        
        # 取文本：遍历所有页面，找最长的文本用于元数据提取
        text = ""
        for pg in r.get('pages', []):
            txt = pg.get('guard_text', pg.get('vlm_text', ''))
            if txt and len(txt) > len(text):
                text = txt
        
        if not text:
            print(f"  ⚠️ {title}: 无文本")
            continue
        
        meta = extract_metadata(text, title, '自办案件')
        tags = ', '.join(meta['keywords'][:4])
        ct = meta['case_types'][0] if meta['case_types'] else '其他'
        cl = meta['court_level'] or '?'
        
        print(f"  {title}: {ct} | {cl} | {tags}", end=' ', flush=True)
        
        if tag_document(db_path, doc_id, title, text, '自办案件', kb_id):
            print("✅")
            tagged += 1
        else:
            print("⏭️")
    
    print(f"\n标签: {tagged}/{len(guard['results'])}")


def tag_single(db_path: str, doc_id: str, title: str, text: str,
               source_type: str = "公众号", kb_id: str = "", filepath: str = "") -> bool:
    """单文档打标签（供自动化调用）"""
    init_db(db_path)
    return tag_document(db_path, doc_id, title, text, source_type, kb_id, filepath=filepath)


def scan_and_tag(db_path: str, directory: str, source_type: str = "公众号") -> int:
    """扫描目录中的 PDF/MD/TXT 文档，提取文本后打标入库。

    PDF 需要 pymupdf（pip install pymupdf）；MD/TXT 直接读取。
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        fitz = None

    exts = ('.pdf', '.md', '.txt')
    files = []
    for root, _, fns in os.walk(directory):
        for fn in sorted(fns):
            if fn.lower().endswith(exts):
                files.append(os.path.join(root, fn))

    if not files:
        print(f"❌ 目录中没有 PDF/MD/TXT 文件: {directory}")
        return 0

    init_db(db_path)
    tagged = 0
    for path in files:
        title = os.path.splitext(os.path.basename(path))[0]
        doc_id = title
        if path.lower().endswith('.pdf'):
            if fitz is None:
                print(f"  ⏭️ {title}: 跳过 PDF（未安装 pymupdf，pip install pymupdf）")
                continue
            try:
                doc = fitz.open(path)
                text = '\n\n'.join(p.get_text().strip() for p in doc if p.get_text().strip())
                doc.close()
            except Exception as e:
                print(f"  ⏭️ {title}: PDF 读取失败 {e}")
                continue
        else:
            try:
                with open(path, encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except Exception as e:
                print(f"  ⏭️ {title}: 读取失败 {e}")
                continue

        if len(text.strip()) < 50:
            print(f"  ⏭️ {title}: 文本太少")
            continue

        print(f"  {title}", end=' ', flush=True)
        if tag_document(db_path, doc_id, title, text, source_type, filepath=path):
            print("✅")
            tagged += 1
        else:
            print("⏭️")

    print(f"\n入库: {tagged}/{len(files)}")
    return tagged


def main():
    import argparse
    parser = argparse.ArgumentParser(description='元数据标签引擎 v2')
    parser.add_argument('--db', default=DEFAULT_DB, help='tags.db 路径')
    parser.add_argument('--kb-id', default='', help='IMA 知识库 ID（可选，仅推送公众号素材时使用）')
    parser.add_argument('--scan', metavar='DIR', help='扫描目录中的 PDF/MD/TXT 并打标入库')
    parser.add_argument('--source-type', default='公众号', help='来源类型（默认：公众号）')
    parser.add_argument('--from-guard', metavar='FILE', help='从 Guard 格式 JSON 批量打标签（需提供数据文件）')
    parser.add_argument('--init', action='store_true', help='仅初始化数据库')
    args = parser.parse_args()

    if args.init:
        init_db(args.db)
        print(f"✅ tags.db 初始化: {args.db}")
        return

    if args.scan:
        scan_and_tag(args.db, args.scan, args.source_type)
        return

    if args.from_guard:
        tag_guard_data(args.db, args.kb_id, args.from_guard)
        return

    print("用法: --scan 目录 (批量入库) | --init (初始化) | --from-guard 文件 (Guard 数据)")


if __name__ == '__main__':
    main()
