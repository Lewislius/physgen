#!/usr/bin/env python3
"""Render the semantic/constraint audit CSV as a readable Markdown report.

The CSV remains the metadata source. Exact stage prompts are loaded from each
sample's trace.contexts.compiled.json because those are the strings actually
encoded for inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import textwrap
from typing import Any


STAGE_LABELS = {
    "setup": "准备/初始阶段",
    "onset": "起始/触发阶段",
    "evolution": "演化阶段",
    "completion": "完成阶段",
    "terminal": "终止/保持阶段",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="semantic audit CSV")
    parser.add_argument(
        "--output",
        type=Path,
        help="output Markdown path (default: CSV path with .md suffix)",
    )
    parser.add_argument(
        "--translations",
        type=Path,
        help=(
            "bilingual prompt-table Markdown containing reviewed stage translations "
            "(default: prompt_table_bilingual.md beside the CSV)"
        ),
    )
    return parser.parse_args()


def fenced(value: str, language: str = "text") -> list[str]:
    fence = "```"
    while fence in value:
        fence += "`"
    return [f"{fence}{language}", value, fence]


def quoted(value: str, width: int = 100) -> list[str]:
    """Wrap long prompt strings for source and rendered Markdown readability."""
    wrapped = textwrap.wrap(
        " ".join(value.split()),
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )
    return [f"> {line}" for line in wrapped] or [">"]


def find_artifact(sample_dir: Path, sample_id: str, filename: str) -> Path:
    matches = sorted((sample_dir / sample_id).glob(f"*/*/{filename}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {filename} for {sample_id}, found {len(matches)}"
        )
    return matches[0]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def constraint_ids(pair: dict[str, Any]) -> str:
    ordered: list[str] = []
    for group in pair.get("violation_constraint_ids", []):
        for value in group:
            text = str(value)
            if text not in ordered:
                ordered.append(text)
    return ", ".join(ordered) if ordered else "—"


def parse_stage_mapping(value: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for item in value.split(";"):
        if "→" not in item:
            continue
        stage_id, ids = item.split("→", 1)
        result[stage_id.strip()] = [part.strip() for part in ids.split(",") if part.strip()]
    return result


def one_line(value: str) -> str:
    return " ".join(value.split())


def split_numbered_prompts(value: str) -> list[str]:
    value = re.sub(r"^\[\d+\]\s*", "", value.strip())
    return [
        item.strip()
        for item in re.split(r"<br>\[\d+\]\s*", value)
        if item.strip()
    ]


def load_stage_translations(
    path: Path,
) -> dict[tuple[str, str], tuple[str, list[str], str, list[str]]]:
    """Load reviewed translations and their English alignment strings."""
    result: dict[tuple[str, str], tuple[str, list[str], str, list[str]]] = {}
    current_sample: str | None = None
    sample_pattern = re.compile(r"<summary><strong>(P\d{2})：")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        sample_match = sample_pattern.search(line)
        if sample_match:
            current_sample = sample_match.group(1)
            continue
        if not re.match(r"^\| (setup|onset|evolution|completion|terminal) /", line):
            continue
        if current_sample is None:
            raise RuntimeError(f"stage translation before sample heading at {path}:{line_number}")
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            raise RuntimeError(
                f"expected five prompt-table cells at {path}:{line_number}, got {len(cells)}"
            )
        stage_id = cells[0].split("/", 1)[0].strip()
        key = (current_sample, stage_id)
        if key in result:
            raise RuntimeError(f"duplicate stage translation for {current_sample}/{stage_id}")
        result[key] = (
            cells[2],
            split_numbered_prompts(cells[4]),
            cells[1],
            split_numbered_prompts(cells[3]),
        )
    return result


def render(csv_path: Path, translations_path: Path) -> str:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"no records found in {csv_path}")
    translations = load_stage_translations(translations_path)

    artifacts: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in rows:
        sample_id = row["P"]
        context_path = find_artifact(csv_path.parent, sample_id, "trace.contexts.compiled.json")
        plan_path = find_artifact(csv_path.parent, sample_id, "trace.plan.input.json")
        artifacts[sample_id] = (load_json(context_path), load_json(plan_path))

    lines = [
        "# 语义与约束使用报告（可读版）",
        "",
        "> 来源：`semantic_constraint_usage_bilingual.csv`。阶段正、负提示词读取自每个 P 的 "
        "`trace.contexts.compiled.json`，因此下文展示的是推理时实际编码的完整文本，而不是输入 plan "
        "中仅用于规划的原始描述。",
        "> 阶段中文译文读取自经过人工整理的 `prompt_table_bilingual.md`，只用于阅读；"
        "实际推理编码的仍是所列英文原文。",
        "",
        "## 阅读说明",
        "",
        "- `c1`、`c2` 等是当前 P 内部的约束 ID；不同 P 中相同编号的具体含义可能不同。",
        "- `positive` 是该阶段实际编码的正提示词。",
        "- `negative.N` 是实际编码的第 N 个负提示词；多个负提示词按照所列权重加权。",
        "- 各 P 使用折叠区块，避免所有长文本同时展开。",
        "",
        "## 总览",
        "",
        "| P | 时间路由 | 阶段文本实际选中约束 | Global couples | 仅校验约束 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {P} | {route} | {selected} | {couples} | {validation} |".format(
                P=row["P"],
                route=row["时间路由类型"] or "—",
                selected=row["阶段Prompt选中的约束"] or "—",
                couples=row["同时进入Global守恒文本的couples约束"] or "—",
                validation=row["仅校验、未进入生成张量的约束"] or "—",
            )
        )

    lines.extend(["", "## 每个 P 的完整内容", ""])
    for row in rows:
        sample_id = row["P"]
        context, plan = artifacts[sample_id]
        pairs = context.get("stage_pairs", [])
        encoded_lengths = context.get("actual_t5_token_lengths", {})
        selected_by_stage = parse_stage_mapping(row["阶段Prompt选中的约束"])
        lines.extend(
            [
                "<details>",
                f"<summary><strong>{sample_id}</strong> — {row['时间路由类型']}；"
                f"{row['阶段Prompt选中的约束']}</summary>",
                "",
                f"### {sample_id} 概要",
                "",
                f"- 时间路由类型：`{row['时间路由类型'] or '—'}`",
                f"- 阶段文本实际选中约束：`{row['阶段Prompt选中的约束'] or '—'}`",
                f"- 同时进入 Global 守恒文本的 couples 约束：`{row['同时进入Global守恒文本的couples约束'] or '—'}`",
                f"- 仅校验、未进入生成张量的约束：`{row['仅校验、未进入生成张量的约束'] or '—'}`",
                "- 实际编码文本键：",
                *[
                    f"  - `{key.strip()}`"
                    for key in row["实际编码文本键"].split(";")
                    if key.strip()
                ],
                "",
                "### Global Semantic（实际编码）",
                "",
                "#### 英文原文",
                "",
                *quoted(row["实际完整Global Semantic（英文）"]),
                "",
                "#### 中文解释",
                "",
                *quoted(row["实际完整Global Semantic（中文）"]),
                "",
                "### 各阶段实际使用的正、负提示词",
                "",
            ]
        )

        for pair in pairs:
            stage_id = str(pair["stage_id"])
            label = STAGE_LABELS.get(stage_id, stage_id)
            positive_key = f"stage.{stage_id}.positive"
            negative_texts = pair.get("violation_texts", [])
            negative_weights = pair.get("violation_weights", [])
            negative_kinds = pair.get("violation_kinds", [])
            negative_constraint_groups = pair.get("violation_constraint_ids", [])
            selected_ids = selected_by_stage.get(stage_id, [])
            translation_key = (sample_id, stage_id)
            if translation_key not in translations:
                raise RuntimeError(f"missing stage translation for {sample_id}/{stage_id}")
            positive_zh, negatives_zh, aligned_positive_en, aligned_negatives_en = translations[
                translation_key
            ]
            if one_line(aligned_positive_en) != one_line(str(pair["positive_text"])):
                raise RuntimeError(f"positive translation is misaligned for {sample_id}/{stage_id}")
            if [one_line(item) for item in aligned_negatives_en] != [
                one_line(str(item)) for item in negative_texts
            ]:
                raise RuntimeError(f"negative translations are misaligned for {sample_id}/{stage_id}")
            if len(negatives_zh) != len(negative_texts):
                raise RuntimeError(
                    f"negative translation count mismatch for {sample_id}/{stage_id}: "
                    f"{len(negatives_zh)} != {len(negative_texts)}"
                )
            lines.extend(
                [
                    f"#### {stage_id} / {label}",
                    "",
                    f"- 实际用于构造 Prompt 的约束：`{', '.join(selected_ids) or '—'}`",
                    f"- violation 中声明的候选约束：`{constraint_ids(pair)}`",
                    f"- minimal-pair 质量：`{pair.get('pair_quality', '—')}`",
                    f"- 正提示词键：`{positive_key}`；T5 token 数："
                    f"`{encoded_lengths.get(positive_key, '—')}`",
                    "",
                    "**正提示词（实际编码完整文本）**",
                    "",
                    *quoted(str(pair["positive_text"])),
                    "",
                    "**正提示词中文翻译（仅供阅读，不参与编码）**",
                    "",
                    *quoted(positive_zh),
                    "",
                ]
            )
            for index, negative_text in enumerate(negative_texts):
                negative_key = f"stage.{stage_id}.negative.{index}"
                weight = negative_weights[index] if index < len(negative_weights) else "—"
                kind = negative_kinds[index] if index < len(negative_kinds) else "—"
                ids = (
                    ", ".join(str(value) for value in negative_constraint_groups[index])
                    if index < len(negative_constraint_groups)
                    else "—"
                )
                if len(selected_ids) == len(negative_texts):
                    actually_used = selected_ids[index]
                elif len(selected_ids) == 1:
                    actually_used = selected_ids[0]
                else:
                    actually_used = ", ".join(selected_ids) or "—"
                lines.extend(
                    [
                        f"**负提示词 {index}（实际编码完整文本）**",
                        "",
                        f"- 键：`{negative_key}`；T5 token 数："
                        f"`{encoded_lengths.get(negative_key, '—')}`",
                        f"- 权重：`{weight}`；violation：`{kind}`",
                        f"- 实际用于构造该负提示词的约束：`{actually_used}`；"
                        f"violation 声明的候选约束：`{ids}`",
                        "",
                        *quoted(str(negative_text)),
                        "",
                        f"**负提示词 {index} 中文翻译（仅供阅读，不参与编码）**",
                        "",
                        *quoted(negatives_zh[index]),
                        "",
                    ]
                )

        lines.extend(["### 约束定义", ""])
        chinese_by_id: dict[str, str] = {}
        for item in row["全部约束中文解释"].splitlines():
            if ":" in item:
                key, value = item.split(":", 1)
                chinese_by_id[key.strip()] = value.strip()
        lines.extend(["| ID | 类型 | 中文解释 |", "|---|---|---|"])
        for constraint in plan.get("constraints", []):
            constraint_id = str(constraint.get("id", "—"))
            kind = str(constraint.get("type", "—"))
            explanation = chinese_by_id.get(constraint_id, "—").replace("|", "\\|")
            lines.append(f"| `{constraint_id}` | `{kind}` | {explanation} |")
        lines.extend(["", "<details>", "<summary>展开全部约束原始 JSON</summary>", ""])
        lines.extend(fenced(json.dumps(plan.get("constraints", []), ensure_ascii=False, indent=2), "json"))
        lines.extend(["", "</details>", ""])

        lines.extend(
            [
                "### 空间控制与未直接作为 Prompt 的字段",
                "",
                f"- 参与空间控制的 JSON：{row['参与空间控制的JSON'] or '—'}",
                f"- 空间控制编译结果：{one_line(row['空间控制编译结果']) or '—'}",
                f"- 没有原样作为 Prompt 使用的字段：{row['没有原样作为Prompt使用的字段'] or '—'}",
                "",
                "</details>",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    output = args.output or args.csv_path.with_suffix(".md")
    translations = args.translations or args.csv_path.with_name("prompt_table_bilingual.md")
    output.write_text(render(args.csv_path, translations), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
