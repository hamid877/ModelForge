import argparse
import logging
import sys
from pathlib import Path

# Ensure repository root is on sys.path when executed directly as a script
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backend.app.ml.bootstrap import (  # noqa: E402
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    bootstrap_model,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the bootstrap CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap ModelForge MLflow tracking, initial model, and champion alias."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to training data CSV (default: data/raw/churn.csv)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Path to save serialized model artifact "
            "(default: models/churn_model.joblib)"
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="MLflow registered model name (defaults to settings.model_name)",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default=None,
        help="Model alias to assign (defaults to settings.model_alias)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force candidate training and promotion evaluation even if champion exists"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run bootstrap workflow from command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = bootstrap_model(
            data_path=args.data_path,
            model_path=args.model_path,
            model_name=args.model_name,
            alias=args.alias,
            force=args.force,
        )
    except Exception as exc:
        print(f"Bootstrap error: {exc}", file=sys.stderr)
        return 1

    print(f"Bootstrap status: {result.status}")
    print(result.message)
    if result.metrics:
        print(f"Accuracy: {result.metrics.get('accuracy', 0.0):.4f}")
        print(f"F1 score: {result.metrics.get('f1_score', 0.0):.4f}")

    if result.status == "failed":
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
