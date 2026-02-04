"""Convert NetCDF files to Zarr stores (1:1).

Each .nc file in the input directory is converted to a .zarr store
with the same name.

Usage:
    python scripts/nc_to_zarr.py /data/hydro/mike_stage/
    python scripts/nc_to_zarr.py /data/hydro/mike_stage/ -o /data/zarr_stores/
    python scripts/nc_to_zarr.py /data/hydro/mike_stage/ --time-chunks 10
"""

import argparse
import sys
from pathlib import Path

import xarray as xr


def convert_file(nc_path: Path, output_path: Path, time_chunks: int = 1) -> Path:
    """Convert a single NetCDF file to a Zarr store.

    Parameters
    ----------
    nc_path : Path
        Input .nc file.
    output_path : Path
        Destination .zarr path.
    time_chunks : int
        Chunk size along the time dimension.

    Returns
    -------
    Path
        Path to the created Zarr store.
    """
    print(f"  Opening {nc_path.name}...")
    ds = xr.open_dataset(nc_path, engine="h5netcdf", chunks="auto")
    ds = ds.chunk({"time": time_chunks})

    print(f"  Writing {output_path.name}...")
    ds.to_zarr(output_path, mode="w")
    ds.close()

    return output_path


def nc_to_zarr(
    input_dir: Path,
    output_dir: Path | None = None,
    time_chunks: int = 1,
) -> list[Path]:
    """Convert all .nc files in a directory to .zarr stores.

    Parameters
    ----------
    input_dir : Path
        Directory containing .nc files.
    output_dir : Path, optional
        Output directory for .zarr stores. Defaults to
        ``{input_dir}_zarr`` (sibling directory).
    time_chunks : int
        Chunk size along the time dimension. Default 1.

    Returns
    -------
    list of Path
        Paths to the created Zarr stores.
    """
    input_dir = Path(input_dir)
    nc_files = sorted(input_dir.glob("*.nc"))
    if not nc_files:
        print(f"No .nc files found in {input_dir}")
        sys.exit(1)

    if output_dir is None:
        output_dir = input_dir.parent / (input_dir.name + "_zarr")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(nc_files)} NetCDF files in {input_dir}")
    for f in nc_files:
        print(f"  {f.name} ({f.stat().st_size / 1024**2:.0f} MB)")

    results = []
    for nc_file in nc_files:
        zarr_name = nc_file.stem + ".zarr"
        out = output_dir / zarr_name
        result = convert_file(nc_file, out, time_chunks)
        results.append(result)

    print(f"\n{'=' * 60}")
    print(f"Converted {len(results)} file(s) to {output_dir}")
    for p in results:
        print(f"  {p.name}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert NetCDF files to Zarr stores (1:1)."
    )
    parser.add_argument(
        "input_dir", type=Path, help="Directory containing .nc files"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for .zarr stores (default: {input_dir}_zarr)",
    )
    parser.add_argument(
        "--time-chunks",
        type=int,
        default=1,
        help="Chunk size along time dimension (default: 1)",
    )
    args = parser.parse_args()

    nc_to_zarr(args.input_dir, args.output_dir, args.time_chunks)


if __name__ == "__main__":
    main()
