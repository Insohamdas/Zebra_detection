"""CLI for testing live identification with a pre-recorded video file."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np

from zebraid.data.quality import QualityFilterConfig
from zebraid.data.stream import CCTVStreamConfig, VideoCaptureStreamSource
from zebraid.pipelines.live_identification import (
    IdentificationCandidate,
    LiveIdentificationPipeline,
)
from zebraid.pipelines.real_identify import create_real_identifier


def _mock_identify(frame: np.ndarray) -> IdentificationCandidate | None:
    """A deterministic mock matcher for offline pipeline testing.

    This does not perform zebra re-identification. It groups visually similar
    frames into pseudo IDs so you can verify stream orchestration behavior.
    """

    if frame.size == 0:
        return None

    # Works for both [0,1] and [0,255] ranges.
    mean_value = float(np.mean(frame))
    bucket = int(mean_value * 10) if mean_value <= 1.0 else int(mean_value // 25)
    bucket = max(0, min(bucket, 99))
    zebra_id = f"MOCK-{bucket:02d}"
    confidence = 0.9
    return IdentificationCandidate(zebra_id=zebra_id, confidence=confidence)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ZEBRAID live pipeline on a pre-recorded video file"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to a local video file (forest/camera recording)",
    )
    parser.add_argument(
        "--stream-id",
        default="video-test",
        help="Logical stream ID used in emitted frame IDs",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default="left",
        help="Expected zebra flank side",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Process every N-th frame",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum number of frames to process",
    )
    parser.add_argument(
        "--mode",
        choices=("quality-only", "mock-identify", "real-identify"),
        default="mock-identify",
        help="quality-only: quality filter + new IDs, mock-identify: pseudo stable IDs, real-identify: FAISS matching engine",
    )
    parser.add_argument(
        "--min-visual-quality",
        type=float,
        default=0.45,
        help="Minimum visual quality threshold",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.85,
        help="Match threshold for identification confidence",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON line per frame result",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    quality_config = QualityFilterConfig(
        min_visual_quality_score=args.min_visual_quality,
    )

    stream_config = CCTVStreamConfig(
        source=args.video,
        stream_id=args.stream_id,
        side=args.side,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )

    source = VideoCaptureStreamSource(stream_config)
    
    if args.mode == "mock-identify":
        identify_fn = _mock_identify
    elif args.mode == "real-identify":
        identify_fn = create_real_identifier(
            match_threshold=args.match_threshold,
            segment_input=False,
        )
    else:
        identify_fn = None
    
    pipeline = LiveIdentificationPipeline(
        source,
        identify_frame=identify_fn,
        quality_config=quality_config,
        match_threshold=args.match_threshold,
    )

    accepted = 0
    rejected = 0
    new_ids = 0

    for result in pipeline.run():
        if result.accepted:
            accepted += 1
        else:
            rejected += 1

        if result.is_new:
            new_ids += 1

        if args.json:
            payload = {
                "frame_id": result.frame.frame_id,
                "timestamp": result.frame.timestamp,
                "accepted": result.accepted,
                "zebra_id": result.zebra_id,
                "is_new": result.is_new,
                "confidence": result.confidence,
                "reasons": result.reasons,
            }
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(
                f"{result.frame.frame_id} accepted={result.accepted} "
                f"id={result.zebra_id} new={result.is_new} "
                f"confidence={result.confidence:.3f} reasons={','.join(result.reasons)}"
            )

    summary = {
        "processed": accepted + rejected,
        "accepted": accepted,
        "rejected": rejected,
        "new_ids": new_ids,
        "mode": args.mode,
    }

    print(json.dumps({"summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())