"""Biometric code generation helpers for ZEBRAID."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LocalPatchCodes:
    """Zone-level partial-body binary codes."""

    shoulder: np.ndarray
    torso: np.ndarray
    neck: np.ndarray


class ITQBinarizer:
    """Iterative Quantisation style binary projection.

    This implementation provides the code path for ITQ-style binarisation. If a
    trained PCA/ITQ rotation is available it can be supplied through
    ``projection`` and ``mean``. Otherwise a deterministic random orthonormal
    projection is used as an untrained fallback.
    """

    def __init__(
        self,
        input_dim: int,
        output_bits: int,
        projection: np.ndarray | None = None,
        mean: np.ndarray | None = None,
        seed: int = 17,
    ) -> None:
        self.input_dim = input_dim
        self.output_bits = output_bits
        self.mean = np.zeros(input_dim, dtype=np.float32) if mean is None else np.asarray(mean, dtype=np.float32)
        self.projection = (
            _orthonormal_projection(input_dim, output_bits, seed=seed)
            if projection is None
            else np.asarray(projection, dtype=np.float32)
        )
        if self.projection.shape != (input_dim, output_bits):
            raise ValueError("projection must have shape (input_dim, output_bits)")

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        """Return binary codes with shape ``(..., output_bits)``."""

        desc = np.asarray(descriptors, dtype=np.float32)
        if desc.shape[-1] != self.input_dim:
            raise ValueError(f"expected descriptor dimension {self.input_dim}")

        projected = (desc - self.mean) @ self.projection
        return (projected >= 0).astype(np.uint8)

    def transform_one(self, descriptor: np.ndarray) -> np.ndarray:
        """Return a one-dimensional binary code."""

        return self.transform(np.asarray(descriptor, dtype=np.float32).reshape(1, -1))[0]


def global_itq_code(descriptor: np.ndarray, binarizer: ITQBinarizer | None = None) -> np.ndarray:
    """Binarise a 1024D descriptor into a 512-bit global code."""

    descriptor = np.asarray(descriptor, dtype=np.float32).ravel()
    binarizer = binarizer or ITQBinarizer(input_dim=descriptor.shape[0], output_bits=512)
    return binarizer.transform_one(descriptor)


def local_patch_codes(
    zone_descriptors: dict[str, np.ndarray],
    *,
    shoulder_binarizer: ITQBinarizer | None = None,
    torso_binarizer: ITQBinarizer | None = None,
    neck_binarizer: ITQBinarizer | None = None,
) -> LocalPatchCodes:
    """Generate shoulder 128-bit, torso 128-bit, and neck 64-bit codes."""

    shoulder = np.asarray(zone_descriptors["shoulder"], dtype=np.float32).ravel()
    torso = np.asarray(zone_descriptors["torso"], dtype=np.float32).ravel()
    neck = np.asarray(zone_descriptors["neck"], dtype=np.float32).ravel()

    shoulder_binarizer = shoulder_binarizer or ITQBinarizer(shoulder.shape[0], 128, seed=101)
    torso_binarizer = torso_binarizer or ITQBinarizer(torso.shape[0], 128, seed=103)
    neck_binarizer = neck_binarizer or ITQBinarizer(neck.shape[0], 64, seed=107)

    return LocalPatchCodes(
        shoulder=shoulder_binarizer.transform_one(shoulder),
        torso=torso_binarizer.transform_one(torso),
        neck=neck_binarizer.transform_one(neck),
    )


def generate_readable_code(stripe_stats: np.ndarray | None = None, vector: np.ndarray | None = None) -> str:
    """Generate ``ZEB-[count]-[width]-[spacing]-[orient]-[checksum]``."""

    if stripe_stats is None:
        if vector is None:
            vector = np.zeros(18, dtype=np.float32)
        stripe_stats = _fallback_stats_from_vector(vector)

    stats = np.asarray(stripe_stats, dtype=np.float32).ravel()
    if stats.shape[0] < 18:
        raise ValueError("stripe_stats must contain 18 values")

    zones = stats.reshape(-1, 6)
    count = int(np.clip(round(float(np.sum(zones[:, 0]))), 0, 99))
    width = int(np.clip(round(float(np.mean(zones[:, 1]))), 0, 99))
    spacing = int(np.clip(round(float(np.mean(zones[:, 3]))), 0, 99))
    orient = int(np.clip(round(((float(np.mean(zones[:, 4])) + np.pi) / (2 * np.pi)) * 99), 0, 99))
    checksum = _checksum(f"{count:02d}-{width:02d}-{spacing:02d}-{orient:02d}")

    return f"ZEB-{count:02d}-{width:02d}-{spacing:02d}-{orient:02d}-{checksum}"


def generate_code(vector=None, stripe_stats: np.ndarray | None = None) -> str:
    """Generate a human-readable biometric ID."""

    return generate_readable_code(stripe_stats=stripe_stats, vector=vector)


def generate_dual_code(global_vec=None, local_vec=None, stripe_stats: np.ndarray | None = None) -> dict[str, str]:
    """Generate global/local biometric code strings for compatibility."""

    return {
        "global": generate_code(global_vec, stripe_stats=stripe_stats),
        "local": generate_code(local_vec, stripe_stats=stripe_stats),
    }


def pack_bits(bits: np.ndarray) -> bytes:
    """Pack a binary code into bytes."""

    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def _orthonormal_projection(input_dim: int, output_bits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((input_dim, output_bits)).astype(np.float32)
    q, _ = np.linalg.qr(matrix)
    if q.shape[1] < output_bits:
        # Wide projections cannot be fully orthonormal by QR on columns; normalize columns instead.
        norms = np.linalg.norm(matrix, axis=0, keepdims=True)
        return matrix / np.maximum(norms, 1e-8)
    return q[:, :output_bits].astype(np.float32)


def _fallback_stats_from_vector(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).ravel()
    digest = hashlib.blake2s(vec.tobytes(), digest_size=18).digest()
    values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    return values.reshape(3, 6).ravel()


def _checksum(payload: str) -> str:
    digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=1).hexdigest().upper()
    return digest[-2:]
