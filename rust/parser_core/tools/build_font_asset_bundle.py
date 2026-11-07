"""Pack parser font JSON sources into the checked-in compressed runtime bundle."""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

MAGIC = b"BDFNTBL2"
EXPECTED_ASSET_COUNT = 154


def main() -> None:
    asset_root = Path(__file__).resolve().parents[1] / "assets" / "font_runtime"
    source_paths = sorted(
        [*asset_root.glob("*.json"), *asset_root.joinpath("cmap").glob("*.json")]
    )
    if len(source_paths) != EXPECTED_ASSET_COUNT:
        raise SystemExit(
            f"expected {EXPECTED_ASSET_COUNT} JSON sources, found {len(source_paths)}"
        )

    payload = bytearray(MAGIC)
    payload.extend(struct.pack(">I", len(source_paths)))
    raw_json_bytes = 0
    compressed_json_bytes = 0
    zstd = shutil.which("zstd")
    if zstd is None:
        raise SystemExit("zstd CLI is required to regenerate the font asset bundle")

    for source_path in source_paths:
        name = source_path.relative_to(asset_root).as_posix().encode("utf-8")
        contents = source_path.read_bytes()
        if len(name) > 0xFFFF:
            raise SystemExit(f"asset path is too long: {source_path}")
        # Each source gets an independent frame so a lookup only inflates that table.
        completed = subprocess.run(  # noqa: S603
            [zstd, "-19", "--long=20", f"--stream-size={len(contents)}", "-q", "-c"],
            input=contents,
            check=True,
            stdout=subprocess.PIPE,
        )
        frame = completed.stdout
        payload.extend(struct.pack(">HQ", len(name), len(frame)))
        payload.extend(name)
        payload.extend(frame)
        raw_json_bytes += len(contents)
        compressed_json_bytes += len(frame)
    output_path = asset_root / "font_tables.zst"
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_bytes(payload)
    temporary_path.replace(output_path)
    print(
        f"assets={len(source_paths)} raw_json_bytes={raw_json_bytes} "
        f"container_bytes={len(payload)} compressed_json_bytes={compressed_json_bytes}"
    )


if __name__ == "__main__":
    main()
