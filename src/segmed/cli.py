from __future__ import annotations

import argparse
import json

from segmed.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="segmed", description="Cardiac U-Net segmentation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a segmentation model")
    train_parser.add_argument("--config", required=True, help="Path to YAML config")
    train_parser.add_argument("--weights", help="Optional initial weights")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    eval_parser.add_argument("--config", required=True, help="Path to YAML config")
    eval_parser.add_argument("--weights", required=True, help="Weights to evaluate")
    eval_parser.add_argument("--split", default="Val", choices=["Train", "Val"], help="Dataset split")
    eval_parser.add_argument("--samples", type=int, default=8, help="Number of qualitative panels to save")

    pred_parser = subparsers.add_parser("predict", help="Predict one image")
    pred_parser.add_argument("--config", required=True, help="Path to YAML config")
    pred_parser.add_argument("--weights", required=True, help="Weights to use")
    pred_parser.add_argument("--image", required=True, help="Input image")
    pred_parser.add_argument("--output", help="Output panel path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "train":
        from segmed.pipeline import train

        path = train(config, args.weights)
        print(f"Saved final weights: {path}")
        return 0
    if args.command == "evaluate":
        from segmed.pipeline import evaluate

        summary = evaluate(config, args.weights, args.split, args.samples)
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "predict":
        from segmed.pipeline import predict_file

        path = predict_file(config, args.weights, args.image, args.output)
        print(f"Saved prediction panel: {path}")
        return 0
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
