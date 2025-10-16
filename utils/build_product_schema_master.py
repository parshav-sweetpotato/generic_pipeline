"""
Build Product Schema Master

Build a schema summary from per-product classification CSV files.
"""

import os
import json
import re
from typing import Dict, List, Optional

import pandas as pd


MISSING_STRINGS = {"", "none", "null", "nan", "na", "n/a", "undefined"}


def _clean_value(val: object) -> Optional[str]:
    """Clean a raw cell value, preserving original casing and spaces; return None if missing."""
    if pd.isna(val):
        return None
    s = str(val).strip().strip('"').strip("'")
    if not s:
        return None
    if s.strip().lower() in MISSING_STRINGS:
        return None
    return s


def _to_snake_case(name: str) -> str:
    """Convert PascalCase/camelCase/Title_Case to snake_case while respecting existing underscores."""
    # Normalize dashes and spaces to underscores
    s = re.sub(r"[\-\s]+", "_", name)
    # Handle transitions: Acronyms followed by words (e.g., 'HTTPServer' -> 'HTTP_Server')
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    # Lower-to-Upper transitions (e.g., 'originCountry' -> 'origin_Country')
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    return s.lower().strip("_")


def _attr_to_snake(col: str, drop_prefix: bool = True) -> str:
    """Normalize attribute column to snake_case and optionally drop 'attr_' prefix."""
    if col.startswith("attr_"):
        suffix = col[len("attr_"):]
        snake_suffix = _to_snake_case(suffix)
        return snake_suffix if drop_prefix else f"attr_{snake_suffix}"
    # Fallback for non-attr prefixed columns
    return _to_snake_case(col)


def product_name_from_filename(filename: str) -> str:
    """Extract product name from CSV filename."""
    name, _ = os.path.splitext(os.path.basename(filename))
    parts = name.split("_")
    # Attempt to strip leading 'hs' and code like '9011'
    if len(parts) >= 3 and parts[0].lower() == "hs" and parts[1].isdigit():
        product_parts = parts[2:]
    else:
        # Fallback: everything after potential 'hs_<code>_'
        product_parts = parts[2:] if len(parts) > 2 else parts
    # Join with spaces and tidy casing
    product = " ".join(product_parts).replace("  ", " ").strip()
    # Replace remaining underscores if any
    product = product.replace("_", " ")
    return product


def build_schema_for_csv(csv_path: str) -> Dict[str, List[str]]:
    """
    Build a mapping of attribute_name -> list of possible values (strings),
    considering only columns prefixed with 'attr_'.
    """
    df = pd.read_csv(csv_path)
    schema: Dict[str, List[str]] = {}
    attr_cols = [c for c in df.columns if c.startswith("attr_")]
    for col in attr_cols:
        # Convert column name to snake_case and drop attr_ prefix in the output
        snake_case_col = _attr_to_snake(col, drop_prefix=True)
        # Clean values and collect uniques, preserving original form
        raw_series = df[col].map(_clean_value).dropna()
        seen = set()
        values: List[str] = []
        for v in raw_series:
            key = v.strip().lower()
            if key not in seen:
                seen.add(key)
                values.append(v)
        schema[snake_case_col] = values
    return schema


def build_schema_master(input_dir: str, output_csv: str, file_pattern: str = "*.csv") -> pd.DataFrame:
    """
    Build product schema master from per-product CSV files.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing product CSV files
    output_csv : str
        Path to write product_schema_master.csv
    file_pattern : str
        Glob pattern for CSV files (default: "*.csv")
    
    Returns:
    --------
    pd.DataFrame
        Schema master DataFrame
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    rows = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        fpath = os.path.join(input_dir, fname)
        
        # Skip writing the output file itself if scanning the same directory
        try:
            if os.path.abspath(fpath) == os.path.abspath(output_csv):
                continue
        except Exception:
            pass
        
        try:
            product = product_name_from_filename(fname)
            schema = build_schema_for_csv(fpath)
            rows.append({
                "product": product,
                "file": fname,
                "schema_json": json.dumps(schema, ensure_ascii=False)
            })
        except Exception as e:
            rows.append({
                "product": product_name_from_filename(fname),
                "file": fname,
                "schema_json": json.dumps({"__error__": [str(e)]}, ensure_ascii=False)
            })
    
    if not rows:
        raise RuntimeError(f"No CSV files found in {input_dir}")
    
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote schema for {len(rows)} products to {output_csv}")
    
    return out_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build product schema master from product CSVs.")
    parser.add_argument("--input-dir", required=True, help="Directory containing product CSV files")
    parser.add_argument("--output-csv", required=True, help="Path to write product_schema_master.csv")
    parser.add_argument("--pattern", default="*.csv", help="File pattern (default: *.csv)")
    args = parser.parse_args()
    
    build_schema_master(args.input_dir, args.output_csv, args.pattern)

