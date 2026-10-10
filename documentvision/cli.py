"""
CLI Interface for DocumentVision Multimodal Benchmark.
"""

import os
import sys
import argparse
from documentvision.evaluator import DocumentVisionEvaluator
from documentvision.metrics.teds import TEDS
from documentvision.generators.document_synthesizer import DocumentSynthesizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="documentvision",
        description="DocumentVision: Multimodal Document Intelligence Benchmark (OCR vs VLM)",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run full multimodal document benchmark across all models")
    bench_parser.add_argument("--dataset", default="data/ground_truth.jsonl", help="Path to ground truth JSONL")
    bench_parser.add_argument("--output-dir", default="results", help="Directory for JSON and CSV exports")
    bench_parser.add_argument("--report-path", default="reports/documentvision_report.md", help="Path for Markdown report")

    # Evaluate TEDS
    teds_parser = subparsers.add_parser("evaluate-teds", help="Compute Tree Edit Distance-based Similarity between tables")
    teds_parser.add_argument("--pred", required=True, help="Predicted HTML table string or file path")
    teds_parser.add_argument("--gt", required=True, help="Ground truth HTML table string or file path")
    teds_parser.add_argument("--structure-only", action="store_true", help="Evaluate structure only (ignore text)")

    # Synthesize Docs
    synth_parser = subparsers.add_parser("synthesize-docs", help="Render synthetic high-resolution benchmark images")
    synth_parser.add_argument("--output-dir", default="data/images", help="Target directory for image PNGs")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand == "benchmark":
        print(f"[DocumentVision] Loading benchmark samples from: {args.dataset}")
        evaluator = DocumentVisionEvaluator.from_jsonl(args.dataset)
        print(f"[DocumentVision] Loaded {len(evaluator.samples)} multimodal document tasks.")

        summary = evaluator.run_full_benchmark()

        print("\n" + "=" * 95)
        print(f"{'Model Checkpoint':<26} | {'Composite':<10} | {'TEDS':<8} | {'KIE F1':<8} | {'ChartQA':<9} | {'Layout AP50':<11}")
        print("=" * 95)
        for model_id, _ in DocumentVisionEvaluator.MODELS:
            m = summary["models"][model_id]
            print(
                f"{model_id:<26} | {m['composite_score']*100:>8.1f}% | {m['teds']*100:>6.1f}% | "
                f"{m['kie_field_f1']*100:>6.1f}% | {m['chart_qa_acc']*100:>7.1f}% | {m['layout_ap50']*100:>9.1f}%"
            )
        print("=" * 95)

        # Export outputs
        os.makedirs(args.output_dir, exist_ok=True)
        summary_json = os.path.join(args.output_dir, "documentvision_summary.json")
        predictions_csv = os.path.join(args.output_dir, "documentvision_predictions.csv")
        evaluator.export_summary_json(summary, summary_json)
        evaluator.export_predictions_csv(predictions_csv)
        evaluator.generate_markdown_report(summary, args.report_path)

        print(f"\n[DocumentVision] Summary exported to: {summary_json}")
        print(f"[DocumentVision] Predictions exported to: {predictions_csv}")
        print(f"[DocumentVision] Research report generated at: {args.report_path}")
        print("\nBenchmark completed successfully!")

    elif args.subcommand == "evaluate-teds":
        pred_html = args.pred
        gt_html = args.gt

        if os.path.exists(pred_html):
            with open(pred_html, "r", encoding="utf-8") as f:
                pred_html = f.read()
        if os.path.exists(gt_html):
            with open(gt_html, "r", encoding="utf-8") as f:
                gt_html = f.read()

        teds = TEDS(structure_only=args.structure_only)
        score = teds.evaluate(pred_html, gt_html)
        metric_name = "TEDS-struct" if args.structure_only else "TEDS"
        print(f"[{metric_name} Score]: {score * 100:.2f}% (Similarity: {score:.4f})")

    elif args.subcommand == "synthesize-docs":
        print(f"[DocumentVision] Synthesizing document images in: {args.output_dir}")
        paths = DocumentSynthesizer.synthesize_all(args.output_dir)
        for name, p in paths.items():
            print(f"  Rendered {name}: {p}")
        print("[DocumentVision] All document images generated successfully!")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
