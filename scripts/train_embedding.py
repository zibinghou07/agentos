"""Real sentence-transformers smoke/full training entrypoint; never fabricates a checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        import torch
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Install training dependencies first: pip install -e '.[rag]'") from exc
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    examples = [
        InputExample(texts=["什么是断点恢复", "Checkpoint 保存成功节点并在重启时跳过它们"]),
        InputExample(texts=["混合检索如何工作", "RRF 融合 BM25 和 Dense 排名"]),
        InputExample(texts=["参数错误如何恢复", "Repair 修改不符合 Schema 的参数"]),
    ]
    model = SentenceTransformer(args.model, device=device)
    loader = DataLoader(examples, shuffle=True, batch_size=2)
    loss = losses.MultipleNegativesRankingLoss(model)
    epochs = 1 if args.smoke else args.epochs
    output = Path("experiments/results/embedding/model_checkpoint")
    model.fit(train_objectives=[(loader, loss)], epochs=epochs, warmup_steps=0, output_path=str(output), show_progress_bar=True)
    log = {"model": args.model, "device": device, "epochs": epochs, "examples": len(examples), "smoke": args.smoke}
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(log)


if __name__ == "__main__":
    main()

