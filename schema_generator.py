"""Schema generation utilities for building product configs from HS-4 codes.

This module samples shipment data, asks Gemini to propose product definitions,
then (after shipments are classified) derives attribute schemas and attribute
definitions in the config format expected by the generic pipeline.

Usage example:

    generator = SchemaGenerator()
    products_result = generator.generate_product_definition(
        hs_code="0904",
        shipment_csv="./output/shipment_master.csv",
        output_dir="./output/generated_schema"
    )

    attrs_result = generator.generate_attribute_configs_from_classifications(
        hs_code="0904",
        classified_csv="./output/shipment_master_classified.csv",
        product_definition_path=products_result.products_definition_path,
        output_dir="./output/generated_schema"
    )

The resulting dataclasses contain the absolute paths to the generated
`products_definition.json`, `product_attributes_schema.json`, and
`attribute_definitions.json` files.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple, Set

import dotenv
import google.generativeai as genai
import pandas as pd

from config_models import (
    AttributeDefinitions,
    AttributeSchema,
    AttributeSet,
    HSAttributeSchema,
    ProductDefinition,
    dump_attribute_definitions,
    dump_attribute_schema,
    dump_product_definition,
    load_product_definition,
)

logger = logging.getLogger(__name__)


DEFAULT_GENERATION_CONFIG = {
    "model_name": "gemini-2.0-flash",
    "temperature": 0.0,
    "max_total_samples": 120,
    "max_categories": 6,
    "max_attributes_per_category": 10,
    "max_values_per_attribute": 30,
    "max_retries": 3,
    "retry_delay": 3,
    "random_seed": 42,
    "use_response_schema": False,
}

MIN_ATTRIBUTES_PER_PRODUCT = 3
SUMMARY_VALUE_LIMIT = 8
SUMMARY_CANDIDATE_COLUMNS = [
    "shipment_origin",
    "shipment_destination",
    "port_of_lading_country",
    "port_of_unlading_country",
    "transport_method",
    "is_containerized",
    "industry_gics",
    "shipper_profile",
    "consignee_profile",
    "value_of_goods_usd",
    "weight_in_kg",
    "volume_teu",
]
DYNAMIC_COLUMN_KEYWORDS = [
    "grade",
    "variety",
    "process",
    "pack",
    "form",
    "moisture",
    "screen",
    "mesh",
    "quality",
    "color",
    "size",
    "type",
]
FALLBACK_ATTRIBUTE_COLUMNS: List[Tuple[str, str, str]] = [
    ("Origin_Country", "shipment_origin", "Country where the shipment originated."),
    ("Destination_Country", "shipment_destination", "Country where the shipment is delivered."),
    ("Port_Of_Lading_Country", "port_of_lading_country", "Country corresponding to the port of lading."),
    ("Port_Of_Unlading_Country", "port_of_unlading_country", "Country corresponding to the port of unlading."),
    ("Transport_Method", "transport_method", "Primary transport mode used for the shipment."),
    ("Containerization_Status", "is_containerized", "Whether the shipment travelled in containers (True/False)."),
    ("Industry_Gics", "industry_gics", "GICS industry classification associated with the shipment."),
    ("Value_Of_Goods_Usd", "value_of_goods_usd", "Declared value of the goods in USD."),
    ("Weight_In_Kg", "weight_in_kg", "Reported weight of the shipment in kilograms."),
    ("Volume_Teu", "volume_teu", "Shipment volume expressed in twenty-foot equivalent units (TEU)."),
]


@dataclass
class SchemaGenerationResult:
    """Holds absolute paths of generated schema artefacts."""

    products_definition_path: Optional[str] = None
    product_attributes_schema_path: Optional[str] = None
    attribute_definitions_path: Optional[str] = None
    product_definition: Optional[ProductDefinition] = None
    attribute_schema: Optional[AttributeSchema] = None
    attribute_definitions: Optional[AttributeDefinitions] = None


class SchemaGenerator:
    """Generate product schemas from shipment data using Gemini."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        generation_config: Optional[Dict[str, object]] = None,
    ) -> None:
        config = dict(DEFAULT_GENERATION_CONFIG)
        if generation_config:
            config.update(generation_config)
        self.config = config

        if not api_key and not os.getenv("GOOGLE_API_KEY"):
            dotenv_path = dotenv.find_dotenv(usecwd=True)
            if dotenv_path:
                dotenv.load_dotenv(dotenv_path)
            else:
                dotenv.load_dotenv()

        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Google API key must be provided via api_key or GOOGLE_API_KEY environment variable"
            )

        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(self.config["model_name"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_product_definition(
        self,
        hs_code: str,
        shipment_csv: str,
        output_dir: str,
        *,
        overwrite: bool = True,
    ) -> SchemaGenerationResult:
        """Infer product categories and persist a product definition config."""

        normalized_hs = self._normalize_hs_code(hs_code)
        logger.info("Inferring product definition for HS-4 %s", normalized_hs)

        goods_samples = self._collect_goods_samples(
            shipment_csv,
            normalized_hs,
            max_total=self.config["max_total_samples"],
        )
        if not goods_samples:
            raise RuntimeError(
                f"No goods descriptions found for HS code {normalized_hs} in {shipment_csv}"
            )

        paths = self._resolve_output_paths(output_dir)
        products_path = paths["products"]
        if not overwrite and os.path.exists(products_path):
            raise FileExistsError(
                f"Product definition already exists: {products_path}. Pass overwrite=True to replace it."
            )

        response = self._call_model_for_products(normalized_hs, goods_samples)

        categories_payload = response.get("product_categories", [])
        category_names: List[str] = []
        category_descriptions: Dict[str, str] = {}
        representative_goods: Dict[str, List[str]] = {}

        for entry in categories_payload:
            if not isinstance(entry, dict):
                continue
            raw_name = entry.get("name", "")
            name = str(raw_name).strip()
            if not name or name in category_names:
                continue
            category_names.append(name)
            description = str(entry.get("description", "")).strip()
            if description:
                category_descriptions[name] = description
            reps = entry.get("representative_goods") or []
            cleaned = [str(value).strip() for value in reps if str(value).strip()]
            if cleaned:
                representative_goods[name] = cleaned[:4]

        if not category_names:
            raise RuntimeError("Model returned no product categories")

        metadata = {
            "_generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "_source": "auto_inference",
        }
        if representative_goods:
            metadata["_representative_goods"] = representative_goods

        product_definition = ProductDefinition(
            hs_code=normalized_hs,
            product_categories=category_names,
            category_descriptions=category_descriptions,
            metadata=metadata,
        )

        dump_product_definition(product_definition, products_path)
        logger.info("✓ Product definition written to %s", products_path)

        return SchemaGenerationResult(
            products_definition_path=products_path,
            product_definition=product_definition,
        )

    def generate_attribute_configs_from_classifications(
        self,
        hs_code: str,
        classified_csv: str,
        output_dir: str,
        *,
        product_definition_path: Optional[str] = None,
        product_definition: Optional[ProductDefinition] = None,
        overwrite: bool = True,
    ) -> SchemaGenerationResult:
        """Derive attribute schema and definitions from classified shipments."""

        normalized_hs = self._normalize_hs_code(hs_code)
        paths = self._resolve_output_paths(output_dir)
        attributes_path = paths["attributes"]
        definitions_path = paths["definitions"]

        if not overwrite and (
            os.path.exists(attributes_path) or os.path.exists(definitions_path)
        ):
            raise FileExistsError(
                "Attribute configs already exist. Pass overwrite=True to replace them."
            )

        if product_definition is None:
            if not product_definition_path:
                raise ValueError(
                    "product_definition or product_definition_path must be provided"
                )
            product_definition = load_product_definition(product_definition_path)

        categories = list(product_definition.product_categories)
        if not categories:
            raise RuntimeError(
                "Product definition contains no categories to model attributes for"
            )

        df = pd.read_csv(classified_csv, low_memory=False, dtype={"hs_code": str})
        if "category" not in df.columns:
            raise KeyError(
                "Classified shipments CSV must contain a 'category' column produced by product classification"
            )
        df = df.copy()
        df["category"] = df["category"].astype(str).str.strip()
        if "goods_shipped" in df.columns:
            df["goods_shipped"] = df["goods_shipped"].astype(str).str.strip()

        products_map: Dict[str, AttributeSet] = {}
        definitions_map: Dict[str, str] = {}
        generation_notes: Dict[str, object] = {}

        for category in categories:
            subset = df[df["category"] == category]
            if subset.empty:
                logger.warning(
                    "No classified shipments found for category '%s'; skipping attribute inference.",
                    category,
                )
                continue

            goods_samples = self._collect_category_goods(
                subset,
                max_total=self.config["max_total_samples"],
            )
            summary_text = self._summarize_category_rows(subset)

            response = self._call_model_for_attributes(
                hs_code=normalized_hs,
                category=category,
                goods=goods_samples,
                summary_text=summary_text,
            )

            attr_map, def_map, telemetry = self._process_attribute_response(
                category=category,
                raw_attributes=response.get("attributes"),
                category_df=subset,
            )

            if not attr_map:
                logger.warning(
                    "No attributes inferred for category '%s' even after fallback; skipping.",
                    category,
                )
                continue

            products_map[category] = AttributeSet(attr_map)
            definitions_map.update(def_map)
            if telemetry:
                generation_notes[category] = telemetry

        if not products_map:
            raise RuntimeError(
                "No attribute schemas could be generated from classified shipments; ensure the classifier produced categories with sufficient data."
            )

        attribute_entry_metadata: Dict[str, object] = {"_product_categories": categories}
        if generation_notes:
            attribute_entry_metadata["_generation_notes"] = generation_notes

        attribute_schema = AttributeSchema(
            entries={
                normalized_hs: HSAttributeSchema(
                    version=datetime.utcnow().isoformat() + "Z",
                    products=products_map,
                    metadata=attribute_entry_metadata,
                )
            },
            metadata={
                "_generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "_source": "auto_inference",
                "_classified_rows": int(len(df)),
            },
        )

        attribute_definitions = AttributeDefinitions(
            definitions=definitions_map,
            metadata={"_source": "auto_inference"},
        )

        dump_attribute_schema(attribute_schema, attributes_path)
        dump_attribute_definitions(attribute_definitions, definitions_path)
        logger.info(
            "✓ Attribute schema written to %s; definitions to %s",
            attributes_path,
            definitions_path,
        )

        product_path = (
            os.path.abspath(product_definition_path)
            if product_definition_path
            else None
        )

        return SchemaGenerationResult(
            products_definition_path=product_path,
            product_attributes_schema_path=attributes_path,
            attribute_definitions_path=definitions_path,
            product_definition=product_definition,
            attribute_schema=attribute_schema,
            attribute_definitions=attribute_definitions,
        )

    def generate_from_csv(
        self,
        hs_code: str,
        shipment_csv: str,
        output_dir: str,
        *,
        overwrite: bool = True,
    ) -> SchemaGenerationResult:
        """Deprecated shim maintained for backward compatibility."""

        raise NotImplementedError(
            "Call generate_product_definition() and generate_attribute_configs_from_classifications() instead of generate_from_csv()."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_output_paths(self, output_dir: str) -> Dict[str, str]:
        """Resolve absolute paths for generated schema artefacts."""

        os.makedirs(output_dir, exist_ok=True)
        return {
            "products": os.path.abspath(os.path.join(output_dir, "products_definition.json")),
            "attributes": os.path.abspath(os.path.join(output_dir, "product_attributes_schema.json")),
            "definitions": os.path.abspath(os.path.join(output_dir, "attribute_definitions.json")),
        }

    def _call_model_for_products(
        self,
        hs_code: str,
        goods: Sequence[str],
    ) -> Dict[str, object]:
        """Ask Gemini to infer product categories for an HS-4 code."""

        if not goods:
            raise RuntimeError("Cannot infer product categories without any goods descriptions")

        sample_lines = "\n".join(
            f"{idx + 1}. {text}" for idx, text in enumerate(goods[: self.config["max_total_samples"]])
        )

        prompt = f"""
1) ROLE & GOAL
You are a Senior Commodity Schema Architect. Review the sampled shipment descriptions for HS-4 code {hs_code} and propose reusable product categories.

2) OUTPUT JSON
Return a JSON object with a single key "product_categories" (array). Each item MUST contain:
- "name": Concise Title Case label (e.g., "Bulk Whole Black Pepper").
- "description": One-sentence guidance describing what belongs in the category.
- "representative_goods": Array of 2-4 short substrings copied verbatim from the samples that support the category.

3) RULES
- Propose between 2 and {self.config['max_categories']} categories rooted in the evidence (processing stage, form, grading, packaging, destination usage, etc.).
- Avoid vague buckets such as "General" unless absolutely required; be specific and mutually exclusive.
- Do not invent categories unsupported by the text.

4) SAMPLED GOODS (DO NOT IGNORE)
{sample_lines}

5) FORMAT
- Output JSON only. No markdown, commentary, or trailing commas.
"""

        response_schema = {
            "type": "object",
            "properties": {
                "product_categories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "representative_goods": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 0,
                            },
                        },
                        "required": ["name", "description"],
                    },
                }
            },
            "required": ["product_categories"],
        }

        return self._invoke_model(prompt, response_schema=response_schema)

    def _call_model_for_attributes(
        self,
        hs_code: str,
        category: str,
        goods: Sequence[str],
        summary_text: str,
    ) -> Dict[str, object]:
        """Ask Gemini to infer attribute schema for a specific product category."""

        goods_section = "\n".join(
            f"{idx + 1}. {text}" for idx, text in enumerate(goods[: self.config["max_total_samples"]])
        ) or "1. No direct goods descriptions available"
        summary_section = summary_text.strip() or "No structured column summary available."

        prompt = f"""
1) ROLE & GOAL
You are a Senior Commodity Schema Architect. For HS-4 code {hs_code}, design a high-quality attribute schema for the product category "{category}" using the supplied shipment evidence.

2) OUTPUT JSON
Return an object with key "attributes" (array). Each attribute object MUST include:
- "name": Title_Case_With_Underscores attribute name suitable for downstream classifiers.
- "definition": One-sentence analyst-friendly description.
- "values": Array of representative, normalized values. Use insights from the summary when the raw goods text is sparse.

3) RULES
- Deliver at least {MIN_ATTRIBUTES_PER_PRODUCT} well-supported attributes when evidence exists. Prefer trade-relevant facets: origin, processing, grading, packaging, quality, physical form, organic status, moisture/mesh specs, etc.
- Reuse naming conventions consistently (e.g., Origin_Country, Processing_Method).
- Never output identifiers, invoice numbers, or purely logistical data as attributes.
- Normalize values to Title Case, merge obvious synonym variants, and keep to {self.config['max_values_per_attribute']} values per attribute.

4) EVIDENCE TO USE (DO NOT IGNORE)
- GOODS SAMPLES:\n{goods_section}
- STRUCTURED SUMMARY:\n{summary_section}

5) FORMAT
- JSON only; no markdown or commentary.
"""

        response_schema = {
            "type": "object",
            "properties": {
                "attributes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "definition": {"type": "string"},
                            "values": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["name", "definition", "values"],
                    },
                }
            },
            "required": ["attributes"],
        }

        return self._invoke_model(prompt, response_schema=response_schema)

    def _invoke_model(
        self,
        prompt: str,
        *,
        response_schema: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        supports_schema = bool(response_schema) and bool(self.config.get("use_response_schema", False))
        retries = 0
        last_exception: Optional[Exception] = None

        while retries <= self.config["max_retries"]:
            try:
                gen_kwargs = dict(
                    temperature=self.config["temperature"],
                    response_mime_type="application/json",
                )
                if supports_schema and response_schema:
                    gen_kwargs["response_schema"] = response_schema
                generation_config = genai.types.GenerationConfig(**gen_kwargs)
                response = self.model.generate_content(prompt, generation_config=generation_config)
                text = getattr(response, "text", "")
                if not text:
                    raise RuntimeError("Empty response from Gemini model")
                start = text.find("{")
                end = text.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    raise RuntimeError("Model response did not include JSON payload")
                return json.loads(text[start : end + 1])
            except TypeError as exc:
                last_exception = exc
                if supports_schema:
                    logger.info("Structured response unsupported; retrying without schema enforcement.")
                    supports_schema = False
                    retries += 1
                    time.sleep(self.config["retry_delay"])
                    continue
                break
            except Exception as exc:  # pragma: no cover - defensive parsing
                last_exception = exc
                if supports_schema and "Unknown field for Schema" in str(exc):
                    logger.info(
                        "Gemini rejected response schema; retrying with plain JSON parsing."
                    )
                    supports_schema = False
                    retries += 1
                    time.sleep(self.config["retry_delay"])
                    continue
                retries += 1
                logger.warning(
                    "Gemini invocation attempt %s/%s failed: %s",
                    retries,
                    self.config["max_retries"],
                    exc,
                )
                time.sleep(self.config["retry_delay"] * max(1, retries))

        if last_exception:
            logger.error("Gemini invocation failed after retries: %s", last_exception)
            raise RuntimeError("Gemini call failed") from last_exception
        raise RuntimeError("Gemini invocation failed without explicit exception")

    def _collect_goods_samples(
        self,
        shipment_csv: str,
        hs_code: str,
        *,
        max_total: int,
    ) -> List[str]:
        df = pd.read_csv(shipment_csv, low_memory=False, dtype={"hs_code": str})
        if "goods_shipped" not in df.columns:
            raise KeyError("Shipment CSV must contain 'goods_shipped' column")

        df = df.copy()
        df["hs_code"] = df["hs_code"].astype(str).str.strip()
        df["hs4"] = df["hs_code"].map(self._extract_hs4)
        df["goods_shipped"] = df["goods_shipped"].astype(str).str.strip()

        filtered = df[df["hs4"] == hs_code]
        if filtered.empty:
            available = sorted({code for code in df["hs4"] if code})
            logger.debug(
                "No rows matched HS-4 %s. Available HS-4 codes in file: %s",
                hs_code,
                available,
            )

        goods = filtered["goods_shipped"].dropna().tolist()
        uniques = list(dict.fromkeys(goods))
        if len(uniques) > max_total:
            rnd = random.Random(self.config.get("random_seed", 42))
            uniques = rnd.sample(uniques, max_total)
        return uniques

    def _collect_category_goods(
        self,
        category_df: pd.DataFrame,
        *,
        max_total: int,
    ) -> List[str]:
        if "goods_shipped" not in category_df.columns:
            return []

        goods = (
            category_df["goods_shipped"].dropna().astype(str).str.strip().tolist()
        )
        uniques = list(dict.fromkeys(goods))
        if len(uniques) > max_total:
            rnd = random.Random(self.config.get("random_seed", 42))
            uniques = rnd.sample(uniques, max_total)
        return uniques

    def _summarize_category_rows(self, category_df: pd.DataFrame) -> str:
        if category_df.empty:
            return ""

        lines: List[str] = []
        for column in SUMMARY_CANDIDATE_COLUMNS:
            if column not in category_df.columns:
                continue
            values = self._extract_top_values(
                category_df[column],
                limit=SUMMARY_VALUE_LIMIT,
            )
            if values:
                lines.append(f"{column}: {', '.join(values)}")

        observed_keywords: Set[str] = set()
        for column in category_df.columns:
            if column in SUMMARY_CANDIDATE_COLUMNS:
                continue
            lower = column.lower()
            matched = [kw for kw in DYNAMIC_COLUMN_KEYWORDS if kw in lower]
            if not matched:
                continue
            values = self._extract_top_values(
                category_df[column],
                limit=SUMMARY_VALUE_LIMIT,
            )
            if values:
                observed_keywords.update(matched)
                lines.append(f"{column}: {', '.join(values)}")

        if observed_keywords:
            lines.append(
                "Detected dynamic attribute hints: " + ", ".join(sorted(observed_keywords))
            )

        return "\n".join(lines)

    def _process_attribute_response(
        self,
        category: str,
        raw_attributes: Optional[List[Dict[str, object]]],
        category_df: pd.DataFrame,
    ) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, object]]:
        attr_map: Dict[str, List[str]] = {}
        definitions: Dict[str, str] = {}
        telemetry: Dict[str, object] = {}

        if isinstance(raw_attributes, list):
            for item in raw_attributes:
                if not isinstance(item, dict):
                    continue
                raw_name = item.get("name")
                if not raw_name:
                    continue
                attr_name = self._normalize_attribute_name(str(raw_name))
                if attr_name in attr_map:
                    continue

                definition = str(item.get("definition", "")).strip()
                values_raw = item.get("values", [])
                values_list = values_raw if isinstance(values_raw, list) else []
                normalized_values = self._normalize_attribute_values(values_list)

                attr_map[attr_name] = normalized_values[: self.config["max_values_per_attribute"]]
                if definition:
                    definitions[f"attr_{attr_name}"] = definition

        telemetry["model_attribute_count"] = sum(1 for values in attr_map.values() if values)

        fallback_attrs, fallback_defs, fallback_meta = self._fallback_attributes_from_dataframe(
            category_df=category_df,
            existing_attributes=attr_map,
        )

        for name, values in fallback_attrs.items():
            if not attr_map.get(name):
                attr_map[name] = values[: self.config["max_values_per_attribute"]]
        for key, definition in fallback_defs.items():
            definitions.setdefault(key, definition)

        for name, values in list(attr_map.items()):
            if not values:
                attr_map.pop(name)
                definitions.pop(f"attr_{name}", None)

        telemetry.update(fallback_meta)
        telemetry["final_attribute_count"] = len(attr_map)
        if attr_map and len(attr_map) < MIN_ATTRIBUTES_PER_PRODUCT:
            logger.warning(
                "Category '%s' produced only %s attributes after fallback (minimum expected %s)",
                category,
                len(attr_map),
                MIN_ATTRIBUTES_PER_PRODUCT,
            )

        if not attr_map:
            return {}, {}, {}

        return attr_map, definitions, telemetry

    def _fallback_attributes_from_dataframe(
        self,
        category_df: pd.DataFrame,
        existing_attributes: Dict[str, List[str]],
    ) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, object]]:
        fallback_attrs: Dict[str, List[str]] = {}
        fallback_defs: Dict[str, str] = {}
        details: Dict[str, object] = {"fallback_attributes": []}

        current_count = sum(1 for values in existing_attributes.values() if values)
        target_needed = max(0, MIN_ATTRIBUTES_PER_PRODUCT - current_count)

        for attr_name, column, definition in FALLBACK_ATTRIBUTE_COLUMNS:
            if target_needed <= 0:
                break
            if attr_name in existing_attributes and existing_attributes[attr_name]:
                continue
            if attr_name in fallback_attrs:
                continue
            if column not in category_df.columns:
                continue
            values = self._extract_top_values(
                category_df[column],
                limit=self.config["max_values_per_attribute"],
            )
            if not values:
                continue
            normalized = self._normalize_attribute_values(values)
            if not normalized:
                continue
            fallback_attrs[attr_name] = normalized
            fallback_defs[f"attr_{attr_name}"] = definition
            details["fallback_attributes"].append(attr_name)
            target_needed -= 1

        if target_needed > 0:
            for column in category_df.columns:
                lower = column.lower()
                if column in SUMMARY_CANDIDATE_COLUMNS:
                    continue
                if not any(keyword in lower for keyword in DYNAMIC_COLUMN_KEYWORDS):
                    continue
                attr_name = self._derive_attribute_name(column)
                if attr_name in existing_attributes and existing_attributes[attr_name]:
                    continue
                if attr_name in fallback_attrs:
                    continue
                values = self._extract_top_values(
                    category_df[column],
                    limit=self.config["max_values_per_attribute"],
                )
                if not values:
                    continue
                normalized = self._normalize_attribute_values(values)
                if not normalized:
                    continue
                fallback_attrs[attr_name] = normalized
                fallback_defs[f"attr_{attr_name}"] = (
                    f"Values observed for column '{column}' among shipments in this category."
                )
                details["fallback_attributes"].append(attr_name)
                target_needed -= 1
                if target_needed <= 0:
                    break

        return fallback_attrs, fallback_defs, details

    def _extract_top_values(
        self,
        values: Sequence[object] | pd.Series,
        *,
        limit: int,
    ) -> List[str]:
        if isinstance(values, pd.Series):
            iterable = values.dropna().tolist()
        else:
            iterable = [value for value in values if value is not None]

        seen: Set[str] = set()
        results: List[str] = []
        for value in iterable:
            text = str(value).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            results.append(text)
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _derive_attribute_name(column: str) -> str:
        sanitized = column.strip().replace("-", "_").replace(" ", "_")
        parts = [part for part in sanitized.split("_") if part]
        return "_".join(part[:1].upper() + part[1:] for part in parts)

    @staticmethod
    def _normalize_hs_code(value: str) -> str:
        hs4 = SchemaGenerator._extract_hs4(value)
        if not hs4:
            raise ValueError(f"HS code must contain at least 1 digit: {value!r}")
        if len(hs4) < 4:
            hs4 = hs4.zfill(4)
        return hs4

    @staticmethod
    def _extract_hs4(value: object) -> str:
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if not digits:
            return ""
        if len(digits) == 9:
            digits = digits.zfill(10)
        elif len(digits) < 4:
            digits = digits.zfill(4)
        return digits[:4]

    @staticmethod
    def _normalize_attribute_name(name: str) -> str:
        sanitized = name.strip().replace(" ", "_").replace("-", "_")
        parts = [part for part in sanitized.split("_") if part]
        return "_".join(part[:1].upper() + part[1:] for part in parts)

    @staticmethod
    def _normalize_attribute_values(values: Sequence[str]) -> List[str]:
        cleaned: List[str] = []
        seen = set()
        for value in values:
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if not stripped:
                continue
            title_case = " ".join(word.capitalize() for word in stripped.split())
            if title_case.lower() in seen:
                continue
            seen.add(title_case.lower())
            cleaned.append(title_case)
        return cleaned

__all__ = ["SchemaGenerator", "SchemaGenerationResult"]


