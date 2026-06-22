"""Command line interface for the heritage perception analysis package."""

from __future__ import annotations

import argparse

from .image_pipeline import run_image_root
from .text_pipeline import run_text_csv


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Analyze cultural heritage reviews and images with LLM-based scoring.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    text_parser = subparsers.add_parser(
        "text",
        help="Analyze review text and score eight cultural value dimensions.",
    )
    text_parser.add_argument("--input", required=True, help="Input CSV path.")
    text_parser.add_argument("--output", required=True, help="Output CSV path.")
    text_parser.add_argument(
        "--text-column",
        default="comments",
        help="Column containing review text. Default: comments.",
    )
    text_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of reviews to process. Use 0 for all rows.",
    )
    text_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip rows already present in the output file.",
    )
    text_parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate review texts instead of processing unique reviews only.",
    )
    text_parser.add_argument(
        "--print-model-io",
        action="store_true",
        help="Print model input and output for each text pipeline step.",
    )

    image_parser = subparsers.add_parser(
        "images",
        help="Analyze image visual quality and score tourist attractiveness.",
    )
    image_parser.add_argument("--image-root", required=True, help="Root folder containing images.")
    image_parser.add_argument("--output", required=True, help="Output CSV path.")
    image_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search all nested folders instead of root and first-level folders only.",
    )
    image_parser.add_argument(
        "--limit-per-folder",
        type=int,
        default=1000,
        help="Maximum images per attraction folder. Use 0 for no limit.",
    )
    image_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip images already present in the output file.",
    )
    image_parser.add_argument(
        "--print-model-io",
        action="store_true",
        help="Print model input and output for each image pipeline step.",
    )
    image_parser.add_argument(
        "--include-image-data-uri",
        action="store_true",
        help="Print full base64 image data URIs when --print-model-io is enabled.",
    )

    return parser


def main() -> None:
    """Run the selected CLI command."""

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "text":
        run_text_csv(
            input_csv=args.input,
            output_csv=args.output,
            text_column=args.text_column,
            limit=args.limit,
            resume=not args.no_resume,
            drop_duplicates=not args.keep_duplicates,
            print_model_io=args.print_model_io,
        )
    elif args.command == "images":
        run_image_root(
            image_root=args.image_root,
            output_csv=args.output,
            recursive=args.recursive,
            limit_per_folder=args.limit_per_folder,
            resume=not args.no_resume,
            print_model_io=args.print_model_io,
            include_image_data_uri=args.include_image_data_uri,
        )
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
