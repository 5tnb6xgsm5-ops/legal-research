#!/usr/bin/env node
/**
 * validate-legal-citation.mjs
 *
 * legal-research 输出校验脚本（v2.19 强化版）
 *
 * 校验层级：
 * - P0 阻塞：法条引用后 80 字内无来源标注 → FAIL
 * - P0 阻塞：合并引用多个法条（如「《民法典》第577、580条」）→ FAIL
 * - P0 阻塞：笼统引用（「根据相关规定」「依据法律规定」）→ FAIL
 * - P1 警告：回答 >200 字但通篇无来源标注 → WARN
 * - P1 警告：引用案例未标注效力等级 → WARN
 * - P1 警告：地方文件未标注适用范围 → WARN
 * - P2 提示：法条未标注时效核验 → WARN
 *
 * 用法：
 *   node scripts/validate-legal-citation.mjs <research-report.md>
 *   node scripts/validate-legal-citation.mjs <research-report.md> --strict  # WARN 也视为失败
 */

import { readFileSync } from 'node:fs';

const file = process.argv[2];
const strict = process.argv.includes('--strict');

if (!file) {
  console.error('Usage: node scripts/validate-legal-citation.mjs <research-report.md> [--strict]');
  process.exit(2);
}

const content = readFileSync(file, 'utf-8');
const errors = [];   // P0 阻塞
const warnings = []; // P1/P2 警告

// ========== P0 · 法条引用必须带来源标注 ==========
const lawCitations = content.matchAll(/《[^》]+》第[一二三四五六七八九十百千\d]+条(第[一二三四五六七八九十\d]+款)?/g);

for (const match of lawCitations) {
  const citation = match[0];
  const position = match.index;
  const context = content.slice(position, position + citation.length + 80);
  const hasSource = /\[(本地KB|北大法宝|元典|WebSearch|IMA|企查查|CITE NEEDED)[^\]]*\]/.test(context);

  if (!hasSource) {
    const lineNumber = content.slice(0, position).split('\n').length;
    errors.push(
      `Line ${lineNumber}: 法条引用「${citation}」后 80 字内无来源标注。\n` +
      `  → 修复：在引用后补充 [来源] 标注，如「${citation} [北大法宝·已核实]」\n` +
      `  → 规则：SKILL.md 检索门禁 P0 阻塞项`
    );
  }
}

// ========== P0 · 禁止合并引用多个法条 ==========
// 匹配「《民法典》第577、580条」或「第577条至第580条」等合并引用
// 注意：实际文本是「第577、580条」（顿号后直接跟数字），不是「第577条、第580条」
const mergedCitations = content.matchAll(/《[^》]+》第[一二三四五六七八九十百千\d]+[、，]\d+条/g);

for (const match of mergedCitations) {
  const citation = match[0];
  const position = match.index;
  const context = content.slice(position, position + citation.length + 80);
  // 合并引用即使带来源标注也违规——必须拆分
  const lineNumber = content.slice(0, position).split('\n').length;
  errors.push(
    `Line ${lineNumber}: 合并引用多个法条「${citation}」。\n` +
    `  → 修复：拆分为独立引用，如「《民法典》第577条 [来源]；《民法典》第580条 [来源]」\n` +
    `  → 规则：references/citation-rules.md 操作禁令`
  );
}

// ========== P0 · 禁止笼统引用（无具体法条编号） ==========
const vaguePatterns = [
  /根据(相关|有关|法律)?规定/g,
  /依据(相关|有关|法律)?规定/g,
  /按照(相关|有关|法律)?规定/g,
  /依照(相关|有关|法律)?规定/g,
  /根据(法律|法规)的?规定/g,
];

for (const pattern of vaguePatterns) {
  const matches = content.matchAll(pattern);
  for (const match of matches) {
    const position = match.index;
    const context = content.slice(position, position + match[0].length + 60);
    // 如果后面 60 字内有具体法条编号，则不算笼统引用
    const hasSpecificCitation = /《[^》]+》第[一二三四五六七八九十百千\d]+条/.test(context);
    if (!hasSpecificCitation) {
      const lineNumber = content.slice(0, position).split('\n').length;
      errors.push(
        `Line ${lineNumber}: 笼统引用「${match[0]}」无具体法条编号。\n` +
        `  → 修复：替换为具体法条引用，如「根据《民法典》第577条 [来源]」\n` +
        `  → 规则：references/citation-rules.md 操作禁令`
      );
    }
  }
}

// ========== P1 · 案例引用必须标注效力等级 ==========
const caseCitations = content.matchAll(/(（\d{4}）[^\s，。；\]]+号|指导案例第?\d+号|公报案例(?!·)|参考案例(?!·)|典型案例(?!·))/g);

for (const match of caseCitations) {
  const citation = match[0];
  const position = match.index;
  const before = content.slice(Math.max(0, position - 1), position);
  if (before === '[') continue;

  const context = content.slice(position, position + citation.length + 100);
  const hasGrade = /\[(指导案例|公报案例|参考案例|典型案例|普通案例|入库案例|裁判文书)[^\]]*\]/.test(context);

  if (!hasGrade) {
    const lineNumber = content.slice(0, position).split('\n').length;
    warnings.push(
      `Line ${lineNumber}: 案例引用「${citation}」未标注效力等级。\n` +
      `  → 建议：补充 [效力等级]，如「${citation} [入库案例·已核实]」\n` +
      `  → 规则：references/citation-rules.md 案例效力分级`
    );
  }
}

// ========== P1 · 地方文件必须标注适用范围 ==========
const localDocPatterns = content.matchAll(/《(浙江省|江苏省|上海市|广东省|北京市|永康市|金华市)[^》]+》/g);

for (const match of localDocPatterns) {
  const citation = match[0];
  const position = match.index;
  const context = content.slice(position, position + citation.length + 100);
  const hasScope = /(仅适用于|适用范围|仅供参考|地方法院文件)/.test(context);

  if (!hasScope) {
    const lineNumber = content.slice(0, position).split('\n').length;
    warnings.push(
      `Line ${lineNumber}: 地方文件「${citation}」未标注适用范围。\n` +
      `  → 建议：补充适用范围标注，如「${citation} [地方法院文件·仅供参考，仅适用于浙江省]」\n` +
      `  → 规则：references/citation-rules.md 操作禁令`
    );
  }
}

// ========== P1 · 回答 >200 字但通篇无来源标注 ==========
const totalLength = content.length;
const sourceCount = (content.match(/\[(本地KB|北大法宝|元典|WebSearch|IMA|企查查|CITE NEEDED)[^\]]*\]/g) || []).length;

if (totalLength > 200 && sourceCount === 0) {
  warnings.push(
    `全文 ${totalLength} 字但通篇无来源标注。\n` +
    `  → 建议：至少为核心结论补充来源标注\n` +
    `  → 规则：SKILL.md 检索门禁 P1 警告项`
  );
}

// ========== P2 · 法条引用未标注时效核验 ==========
for (const match of lawCitations) {
  const citation = match[0];
  const position = match.index;
  const context = content.slice(position, position + citation.length + 120);
  const hasCurrency = /(✅现行有效|⚠️已修订|❌已废止|现行有效|已修订|已废止)/.test(context);

  if (!hasCurrency) {
    const lineNumber = content.slice(0, position).split('\n').length;
    warnings.push(
      `Line ${lineNumber}: 法条引用「${citation}」未标注时效核验。\n` +
      `  → 建议：补充时效标记，如「${citation}（✅现行有效）」\n` +
      `  → 规则：SKILL.md 检索原则性纲要·时效优先`
    );
  }
}

// ========== 输出结果 ==========
console.log(`\n📋 legal-research 校验报告：${file}\n`);

if (errors.length === 0 && warnings.length === 0) {
  console.log('✅ PASS · 所有硬规则通过\n');
  process.exit(0);
}

if (errors.length > 0) {
  console.log(`❌ FAIL · ${errors.length} 个 P0 阻塞项（必须修复后才能输出）\n`);
  errors.forEach((err, i) => console.log(`${i + 1}. ${err}\n`));
}

if (warnings.length > 0) {
  console.log(`⚠️  WARN · ${warnings.length} 个警告项（建议修复，不阻塞输出）\n`);
  warnings.forEach((warn, i) => console.log(`${i + 1}. ${warn}\n`));
}

if (errors.length > 0 || (strict && warnings.length > 0)) {
  console.log('🚫 校验失败，请修复上述问题后重新输出。\n');
  process.exit(1);
} else {
  console.log('✅ PASS（含警告）· 可输出，但建议修复警告项。\n');
  process.exit(0);
}
