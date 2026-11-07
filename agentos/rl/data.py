"""Multi-hop QA data for the search RL environment.

Raw HotpotQA / 2WikiMultihopQA / MuSiQue rows are converted to one record format,
split with fixed seeds, and given a per-question retrieval pool: the question's own
paragraphs plus distractor paragraphs drawn from other questions of the same split.
Document ids are opaque hashes so a policy cannot spot its own paragraphs by id.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import string
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

POOL_SIZE = 100


@dataclass
class Paragraph:
    doc_id: str
    title: str
    text: str


@dataclass
class QAItem:
    qid: str
    source: str
    question: str
    answers: list[str]
    qtype: str
    level: str
    support_titles: list[str]
    paragraphs: list[Paragraph]
    pool: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["own_doc_ids"] = [p.doc_id for p in self.paragraphs]
        del record["paragraphs"]
        return record


def doc_id(source: str, qid: str, index: int) -> str:
    digest = hashlib.sha1(f"{source}|{qid}|{index}".encode()).hexdigest()
    return f"d{digest[:10]}"


def normalize_answer(text: str) -> str:
    """SQuAD/HotpotQA answer normalisation."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _hotpot_like(row: dict[str, Any], source: str) -> QAItem:
    qid = str(row["id"])
    context = row["context"]
    paragraphs = [
        Paragraph(doc_id(source, qid, i), title, "".join(sentences).strip())
        for i, (title, sentences) in enumerate(zip(context["title"], context["sentences"]))
    ]
    return QAItem(
        qid=qid,
        source=source,
        question=row["question"],
        answers=[row["answer"]],
        qtype=row["type"],
        level=row.get("level") or "",
        support_titles=sorted(set(row["supporting_facts"]["title"])),
        paragraphs=paragraphs,
    )


def _musique(row: dict[str, Any]) -> QAItem:
    qid = str(row["id"])
    paragraphs = [
        Paragraph(doc_id("musique", qid, p["idx"]), p["title"], p["paragraph_text"].strip())
        for p in row["paragraphs"]
    ]
    hops = qid.split("__", 1)[0]
    return QAItem(
        qid=qid,
        source="musique",
        question=row["question"],
        answers=[row["answer"], *row.get("answer_aliases", [])],
        qtype=hops,
        level="",
        support_titles=sorted({p["title"] for p in row["paragraphs"] if p["is_supporting"]}),
        paragraphs=paragraphs,
    )


def load_raw(path: str | Path, source: str) -> list[QAItem]:
    rows = pq.read_table(path).to_pylist()
    if source == "musique":
        return [_musique(row) for row in rows if row.get("answerable", True)]
    return [_hotpot_like(row, source) for row in rows]


def stratified_sample(
    items: Sequence[QAItem], n: int, seed: int, key=lambda item: item.qtype
) -> list[QAItem]:
    """Sample ``n`` items keeping each stratum's share; deterministic for a seed."""
    groups: dict[str, list[QAItem]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    rng = random.Random(seed)
    for name in sorted(groups):
        rng.shuffle(groups[name])
    total = len(items)
    quotas = {name: n * len(group) // total for name, group in groups.items()}
    # Hand the rounding remainder to the largest strata.
    for name in sorted(groups, key=lambda g: (-len(groups[g]), g))[: n - sum(quotas.values())]:
        quotas[name] += 1
    chosen = [item for name in sorted(groups) for item in groups[name][: quotas[name]]]
    rng.shuffle(chosen)
    return chosen


def attach_pools(items: Sequence[QAItem], seed: int, pool_size: int = POOL_SIZE) -> None:
    """Pool = own paragraphs + distractors from other questions in the same split."""
    all_ids = [p.doc_id for item in items for p in item.paragraphs]
    for item in items:
        own = [p.doc_id for p in item.paragraphs]
        own_set = set(own)
        rng = random.Random(f"{seed}|{item.source}|{item.qid}")
        need = max(pool_size - len(own), 0)
        distractors: list[str] = []
        seen = set(own_set)
        while len(distractors) < need and len(seen) < len(all_ids):
            candidate = all_ids[rng.randrange(len(all_ids))]
            if candidate not in seen:
                seen.add(candidate)
                distractors.append(candidate)
        pool = own + distractors
        rng.shuffle(pool)
        item.pool = pool


def corpus_records(splits: dict[str, Sequence[QAItem]]) -> Iterable[dict[str, str]]:
    seen: set[str] = set()
    for items in splits.values():
        for item in items:
            for paragraph in item.paragraphs:
                if paragraph.doc_id not in seen:
                    seen.add(paragraph.doc_id)
                    yield {"doc_id": paragraph.doc_id, "title": paragraph.title, "text": paragraph.text}


def _token_set(text: str) -> frozenset[str]:
    return frozenset(normalize_answer(text).split())


def leakage_report(splits: dict[str, Sequence[QAItem]], near_dup: float = 0.8) -> dict[str, Any]:
    """Checks between train and every evaluation split."""
    train = splits["train"]
    train_questions = {normalize_answer(item.question) for item in train}
    train_sets = [_token_set(item.question) for item in train]
    train_titles = {p.title for item in train for p in item.paragraphs}
    train_answers = {normalize_answer(a) for item in train for a in item.answers}
    report: dict[str, Any] = {}
    for name, items in splits.items():
        if name == "train":
            continue
        exact = sum(normalize_answer(item.question) in train_questions for item in items)
        near = 0
        for item in items:
            tokens = _token_set(item.question)
            if any(len(tokens & other) / max(len(tokens | other), 1) >= near_dup for other in train_sets):
                near += 1
        support_seen = sum(any(title in train_titles for title in item.support_titles) for item in items)
        answer_seen = sum(any(normalize_answer(a) in train_answers for a in item.answers) for item in items)
        report[name] = {
            "n": len(items),
            "exact_question_in_train": exact,
            f"near_duplicate_question_jaccard>={near_dup}": near,
            "support_title_seen_in_train": support_seen,
            "answer_string_seen_in_train": answer_seen,
        }
    eval_ids = Counter(item.qid for name, items in splits.items() if name != "train" for item in items)
    report["qid_in_multiple_eval_splits"] = sum(count > 1 for count in eval_ids.values())
    return report


def split_summary(items: Sequence[QAItem]) -> dict[str, Any]:
    return {
        "n": len(items),
        "qtype": dict(Counter(item.qtype for item in items)),
        "level": dict(Counter(item.level or "-" for item in items)),
        "yes_no_answers": sum(normalize_answer(item.answers[0]) in {"yes", "no"} for item in items),
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
