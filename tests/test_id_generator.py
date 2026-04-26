import re

import numpy as np

from zebraid.id_generator import (
    ITQBinarizer,
    generate_code,
    generate_dual_code,
    generate_readable_code,
    global_itq_code,
    local_patch_codes,
    pack_bits,
)


def test_global_itq_code_is_512_bits():
    descriptor = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)

    code = global_itq_code(descriptor)

    assert code.shape == (512,)
    assert set(np.unique(code)).issubset({0, 1})
    assert len(pack_bits(code)) == 64


def test_itq_binarizer_accepts_trained_projection():
    projection = np.eye(4, 2, dtype=np.float32)
    binarizer = ITQBinarizer(input_dim=4, output_bits=2, projection=projection)

    code = binarizer.transform_one(np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32))

    assert code.tolist() == [1, 0]


def test_local_patch_codes_have_paper_bit_lengths():
    zone_descriptors = {
        "shoulder": np.linspace(-1, 1, 96, dtype=np.float32),
        "torso": np.linspace(1, -1, 96, dtype=np.float32),
        "neck": np.ones(96, dtype=np.float32),
    }

    codes = local_patch_codes(zone_descriptors)

    assert codes.shoulder.shape == (128,)
    assert codes.torso.shape == (128,)
    assert codes.neck.shape == (64,)


def test_generate_readable_code_format_from_stripe_stats():
    stripe_stats = np.array(
        [
            12, 5.2, 1.1, 8.6, 0.2, 0.01,
            18, 6.1, 1.3, 7.4, 0.4, 0.02,
            15, 4.8, 1.0, 9.1, 0.1, 0.03,
        ],
        dtype=np.float32,
    )

    code = generate_readable_code(stripe_stats)

    assert re.match(r"^ZEB-\d{2}-\d{2}-\d{2}-\d{2}-[0-9A-F]{2}$", code)
    assert code.startswith("ZEB-45-05-08-")


def test_generate_code_and_dual_code_are_readable():
    code = generate_code(np.ones(1138, dtype=np.float32))
    dual = generate_dual_code(np.ones(1138, dtype=np.float32), np.zeros(96, dtype=np.float32))

    assert code.startswith("ZEB-")
    assert dual["global"].startswith("ZEB-")
    assert dual["local"].startswith("ZEB-")
