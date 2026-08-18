"""Genera el CSV de predicciones para la competencia."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifacts import write_json
from src.data import sha256_file
from src.inference import apply_output_template, predict_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "pipeline_test.csv"))
    parser.add_argument("--template", default=None, help="Template opcional Id,Prediction para validar filas y orden")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model", choices=["single", "ensemble"], default="single")
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    args = parser.parse_args()
    output, metrics = predict_file(args.input, args.artifacts, args.model)
    if args.template:
        output = apply_output_template(output, args.template)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, float_format="%.6f")
    metadata = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": str(Path(args.input).resolve()), "input_sha256": sha256_file(args.input),
        "output": str(output_path.resolve()), "rows": len(output), "model": args.model,
        "template": str(Path(args.template).resolve()) if args.template else None,
        "template_sha256": sha256_file(args.template) if args.template else None,
        "output_columns": output.columns.tolist(),
        "metrics_if_target_present": metrics,
    }
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    print(f"Predicciones guardadas: {output_path} ({len(output)} filas, modelo={args.model})")
    if metrics:
        print(f"RMSE: {metrics['rmse']:,.2f} | MAE: {metrics['mae']:,.2f} | R²: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
