#!/usr/bin/env python3
"""
retrieve_first 检索门禁（Harness 层 · 硬约束）
harness_retrieve_first.py

三层编码：
  提示词层(弱·建议) → Skill 层(中·规则) → Harness 层(硬·门禁) ← 本脚本

法律类回答输出前强制执行检索来源校验。
FAIL = 回答违规（裸法条无出处 / 无任何来源标注），禁止输出，必须回退检索。

校验规则（按严重性排序）：
  P0 阻塞 / 裸法条引用无来源 → FAIL（禁止输出）
  P1 长回答无任何来源标注 → WARN（建议补充检索）
  P2 推断性表述无出处 → INFO（提示）

用法（供 legal-research Skill 调用）：
  echo "$answer" | python3 harness_retrieve_first.py --check-stdin --question "用户问题"
  退出码 0=PASS, 1=FAIL/WARN

位置：完整版包 scripts/harness_retrieve_first.py（随包分发，无需自行实现）
"""

import re
import sys
import argparse
import json

SCRIPT_DIR = __import__('os').path.dirname(__import__('os').path.abspath(__file__))

# ---- 合法来源标注（与 SKILL.md「来源优先级」和「新增来源标签」对齐）----
VALID_CITES = [
    r'\[本地KB·已核实\]',
    r'\[本地法律库·已核实\]',
    r'\[北大法宝·已核实\]',
    r'\[元典·已核实\]',
    r'\[ima知识库·已核实\]',
    r'\[元宝WSA·降级\]',
    r'\[WebSearch·已核实\]',
    r'\[指导案例·已核实\]',
    r'\[入库案例·已核实\]',
    r'\[公报/典型·已核实\]',
    r'\[法答网·已核实\]',
    r'\[裁判文书·已核实\]',
    r'\[CITE NEEDED\]',
    r'\[检索确认\]',
    r'\[法源\]',
]

# ---- 裸法条引用模式（兼容：单条 / 顿号并列 / 区间「至」/ 数字编号）----
LAW_NAME = (
    r'(民法典|刑法|公司法|劳动合同法|劳动法|行政诉讼法|民事诉讼法|刑事诉讼法'
    r'|合同法|物权法|侵权责任法|担保法|证券法|商标法|专利法|著作权法'
    r'|企业破产法|反垄断法|反不正当竞争法|消费者权益保护法'
    r'|建工解释|建设工程司法解释|民间借贷司法解释|担保制度解释'
    r'|九民纪要|公司法解释|劳动争议司法解释|合同编通则解释|建工合同司法解释'
    r'|渔业法|野生动物保护法|长江保护法)'  # 2026-09-26 补：渔业/生态类高频法律
)
ARTICLE = r'第\s*[一二三四五六七八九十百千零\d]+\s*条(?:之[一二三四五六七八九十]+)?'
BARE_LAW_REGEX = re.compile(
    LAW_NAME + r'\s*' + ARTICLE                        # 单条：民法典第807条
    + r'(?:\s*[、，]\s*' + ARTICLE + r')*'             # 顿号并列：「第807条、第808条」
    + r'(?:\s*(?:至|到|—|~|－|−)\s*' + ARTICLE + r')*' # 区间：「第703条至第734条」
)

# ---- 推断性表述（"根据""依据"但实际无出处）----
INFER_PATTERNS = [
    r'(根据|依据|按照|依照).{0,30}(规定|(?<!方)法|(?<!有)条|解释)',
]


def _has_cite_near(text: str, start: int, end: int, window: int = 80) -> bool:
    """检查 text[start:end] 位置前后 window 字符内是否有合法来源标注。

    2026-09-26 修：原实现为 _has_cite_near(text, target) 用 text.find(target)
    只检查目标串**首次出现处**，同一法条在文中多次出现时后续位置全部漏检。
    本版由调用方传入每个正则匹配的实际 start/end 位置。
    """
    ctx = text[max(0, start - window):end + window]
    return any(re.search(p, ctx) for p in VALID_CITES)


def check_answer(question: str, answer: str) -> dict:
    """
    校验回答的检索来源合规性。
    返回 {status: PASS|FAIL|WARN, reason: str, details: [str]}
    """
    issues = []
    text = answer or ""

    # P0 阻塞级：裸法条引用无来源
    for m in BARE_LAW_REGEX.finditer(text):
        bare = m.group(0)
        if not _has_cite_near(text, m.start(), m.end()):
            issues.append(f"BARE_LAW: 「{bare}」后无来源标注")

    # P1 警告级：长回答（>200字）无任何来源
    has_any_cite = any(re.search(p, text) for p in VALID_CITES)
    bare_len = len(re.sub(r'\s+', '', text))
    if not has_any_cite and bare_len > 200:
        issues.append("NO_CITE: 回答超过200字但无任何来源标注")

    # P2 提示级：推断性表述无出处
    for pat in INFER_PATTERNS:
        for m in re.finditer(pat, text):
            snippet = m.group(0)
            if not _has_cite_near(text, m.start(), m.end()):
                issues.append(f"INFER: 使用了「{snippet[:30]}...」但无来源标注")
                break  # 每类只报一次

    if not issues:
        return {"status": "PASS", "reason": "全部校验通过", "details": []}

    # 分级
    fails = [i for i in issues if i.startswith("BARE_LAW")]
    warns = [i for i in issues if i.startswith("NO_CITE")]
    infos = [i for i in issues if i.startswith("INFER")]

    if fails:
        return {"status": "FAIL", "reason": "; ".join(fails), "details": issues}
    if warns:
        return {"status": "WARN", "reason": "; ".join(warns), "details": issues}
    return {"status": "WARN", "reason": "; ".join(infos), "details": issues}


def main():
    parser = argparse.ArgumentParser(description='retrieve_first 检索门禁校验（Harness 层）')
    parser.add_argument('--check-stdin', action='store_true', help='从 stdin 读取回答')
    parser.add_argument('--check-file', help='从文件读取回答')
    parser.add_argument('--question', default='', help='用户问题（供 context，未强制）')
    parser.add_argument('--json', action='store_true', help='JSON 输出')
    args = parser.parse_args()

    if args.check_stdin:
        answer = sys.stdin.read()
    elif args.check_file:
        with open(args.check_file) as f:
            answer = f.read()
    else:
        print("❌ 需要 --check-stdin 或 --check-file，用法见 SKILL.md")
        sys.exit(2)

    result = check_answer(args.question, answer)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        icons = {"PASS": "✅", "WARN": "⚠️", "FAIL": "🚫"}
        print(f"{icons.get(result['status'],'?')} [{result['status']}] {result['reason']}")
        for d in result['details']:
            print(f"   - {d}")
        if result['status'] == 'FAIL':
            print("\n🔴 禁止输出。请先执行：")
            print("   python3 scripts/rag_ask.py \"$QUESTION\" --json")
            print("   基于 [本地KB·已核实] doc_id 出处重写回答后，重新过门禁。")

    sys.exit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
