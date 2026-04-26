from zebraid.pipelines.video_test import build_parser


def test_video_test_parser_defaults() -> None:
    args = build_parser().parse_args(["--video", "forest.mp4"])

    assert args.video == "forest.mp4"
    assert args.stream_id == "video-test"
    assert args.side == "left"
    assert args.frame_stride == 5
    assert args.max_frames == 100
    assert args.mode == "mock-identify"
    assert args.min_visual_quality == 0.45
    assert args.match_threshold == 0.85
    assert args.json is False