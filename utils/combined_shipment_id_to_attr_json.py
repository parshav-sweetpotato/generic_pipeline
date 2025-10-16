"""
Combined Shipment ID to Attributes

Combine shipment IDs to attributes mapping from multiple per-product CSV files.
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Any


def find_csv_files(input_dir: Path, pattern: str) -> List[Path]:
    """Return list of CSV files matching pattern in the input directory (non-recursive)."""
    return sorted(input_dir.glob(pattern))


def extract_attr_columns(header: List[str]) -> List[str]:
    """Identify columns that start with 'attr_' (case-sensitive)."""
    return [c for c in header if c.startswith("attr_")]


def to_snake_case(key: str) -> str:
    """Convert arbitrary strings to lowercase snake_case suitable for JSON keys."""
    s = key.strip()
    # Insert underscore between camelCase and PascalCase boundaries
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    # Replace spaces, hyphens, slashes and other non-word chars with underscores
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    # Trim underscores and lowercase
    return s.strip("_").lower()


def load_attributes(files: List[Path], shipment_id_col: str = "shipment_id", product_col: str = "product") -> Dict[str, Dict[str, object]]:
    """
    Load and combine attributes from multiple CSV files.
    
    For each row:
      - Read shipment_id
      - Collect all attr_* columns, strip the 'attr_' prefix from keys
      - Skip attributes with empty or None values
    
    If the same shipment_id appears multiple times across files, attributes are merged.
    Later occurrences override earlier ones if the same attribute key repeats.
    
    Parameters:
    -----------
    files : List[Path]
        List of CSV file paths
    shipment_id_col : str
        Name of shipment ID column
    product_col : str
        Name of product column
    
    Returns:
    --------
    Dict[str, Dict[str, object]]
        shipment_id -> {"attrs": Dict[str, str], "product": Optional[str]}
    """
    combined: Dict[str, Dict[str, object]] = {}
    
    for path in files:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if shipment_id_col not in reader.fieldnames:
                raise ValueError(
                    f"File '{path.name}' is missing required column '{shipment_id_col}'."
                )
            
            attr_cols = extract_attr_columns(reader.fieldnames or [])
            for row in reader:
                shipment_id = (row.get(shipment_id_col) or "").strip()
                if not shipment_id:
                    continue
                
                entry = combined.setdefault(shipment_id, {"attrs": {}, "product": None})
                attrs: Dict[str, str] = entry["attrs"]  # type: ignore[assignment]
                
                if product_col in (reader.fieldnames or []):
                    prod_raw = row.get(product_col)
                    if prod_raw is not None:
                        prod = str(prod_raw).strip()
                        if prod:
                            entry["product"] = prod
                
                for col in attr_cols:
                    raw_val = row.get(col)
                    if raw_val is None:
                        continue
                    val = str(raw_val).strip()
                    if val == "" or val.lower() in {"nan", "none", "null"}:
                        continue
                    key = col[len("attr_"):]
                    snake_key = to_snake_case(key)
                    if not snake_key:
                        continue
                    attrs[snake_key] = val
    
    return combined


def combine_shipment_attributes(input_dir: str, pattern: str, output_path: str, shipment_id_col: str = "shipment_id") -> Dict[str, Any]:
    """
    Combine shipment ID to attribute mappings from multiple CSV files.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing input CSV files
    pattern : str
        Glob pattern for input CSV files
    output_path : str
        Output file path ('.json' writes JSON; anything else writes CSV)
    shipment_id_col : str
        Name of the shipment id column
    
    Returns:
    --------
    Dict[str, Any]
        Combined attributes dictionary
    """
    input_dir_path = Path(input_dir).expanduser().resolve()
    output_path_obj = Path(output_path)
    if not output_path_obj.is_absolute():
        output_path_obj = input_dir_path / output_path
    
    files = find_csv_files(input_dir_path, pattern)
    if not files:
        raise SystemExit(
            f"No files found in '{input_dir_path}' matching pattern '{pattern}'."
        )
    
    print(f"Found {len(files)} file(s): {[p.name for p in files]}")
    combined = load_attributes(files, shipment_id_col=shipment_id_col)
    
    # Write output based on extension
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path_obj.suffix.lower()
    
    if ext == ".json":
        # Single JSON object mapping shipment_id -> { attr: value }
        attrs_only = {sid: (entry.get("attrs") or {}) for sid, entry in combined.items()}
        with output_path_obj.open("w", encoding="utf-8") as out_f:
            json.dump(attrs_only, out_f, ensure_ascii=False, indent=2)
    else:
        # Three-column CSV: shipment_id, product, attrs_json
        with output_path_obj.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["shipment_id", "product", "attrs_json"])
            for shipment_id in sorted(combined.keys()):
                entry = combined[shipment_id]
                product = entry.get("product") or ""
                attrs_json = json.dumps(entry.get("attrs") or {}, ensure_ascii=False)
                writer.writerow([shipment_id, product, attrs_json])
    
    print(f"Wrote {len(combined)} shipment(s) with attributes to '{output_path_obj}'.")
    
    return combined


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Combine shipment_id to attribute mappings from CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing the input CSV files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for input CSV files (default: '*.csv')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="shipment_id_to_attr.json",
        help="Output file path ('.json' writes JSON; anything else writes CSV)."
    )
    parser.add_argument(
        "--shipment-id-col",
        type=str,
        default="shipment_id",
        help="Name of the shipment id column (default: shipment_id)"
    )
    args = parser.parse_args()
    
    combine_shipment_attributes(args.input_dir, args.pattern, args.output, args.shipment_id_col)

