from zebraid.data import download as download_module


def test_build_parser_defaults() -> None:
    args = download_module.build_parser().parse_args([])

    assert args.keyword == "zebra side view"
    assert args.max_num == 500
    assert args.output_dir == "data/raw/zebra_images"
    assert args.engine == "google"


def test_download_images_uses_crawler(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class FakeCrawler:
        def __init__(self, storage: dict[str, str]) -> None:
            calls["storage"] = storage

        def crawl(self, keyword: str, max_num: int) -> None:
            calls["keyword"] = keyword
            calls["max_num"] = max_num

    monkeypatch.setattr(download_module, "_load_crawler", lambda engine: FakeCrawler)

    output_path = download_module.download_images(
        "zebra side view",
        tmp_path / "zebra_images",
        max_num=3,
        engine="bing",
    )

    assert output_path == tmp_path / "zebra_images"
    assert output_path.exists()
    assert calls["storage"] == {"root_dir": str(output_path)}
    assert calls["keyword"] == "zebra side view"
    assert calls["max_num"] == 3