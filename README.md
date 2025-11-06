# Generic Product Classification Pipeline

This pipeline ingests shipment data, classifies each record into product groups, and optionally extracts attribute values per product. You can run it in two modes depending on whether you already maintain the configuration files yourself.

## 1. Manual Workflow (config files supplied)
Use this when you already have:
- `products_definition.json` – list of product categories and descriptions
- `product_attributes_schema.json` – allowed attributes/values per product
- `attribute_definitions.json` – analyst-friendly attribute descriptions

```bash
python pipeline_runner.py \
  --data-source ./shipments.csv \
  --products-definition ./config/products_definition.json \
  --product-attributes-schema ./config/product_attributes_schema.json \
  --attribute-definitions ./config/attribute_definitions.json \
  --output-dir ./output/manual_run
```

The runner will load shipments, classify products, apply the provided attribute schema, and emit:
- Per-product CSVs in `<output-dir>/per_product_classifications/`
- Aggregated JSON/CSV summaries
- Optional helper artefacts (`product_schema_master.csv`, `shipment_id_to_attr.json`)

## 2. Automatic Workflow (configs inferred from data)
Use this when you want the pipeline to design products and attribute schemas on the fly from a single HS-4 shipment dataset.

```bash
python pipeline_runner.py \
  --data-source ./shipments.csv \
  --products-definition ./tmp_schema/products_definition.json \
  --product-attributes-schema ./tmp_schema/product_attributes_schema.json \
  --attribute-definitions ./tmp_schema/attribute_definitions.json \
  --output-dir ./output/auto_run \
  --auto-generate-schema \
  --hs-code 9011 \
  --schema-output-dir ./tmp_schema
```

This mode executes:
1. **Product inference** – creates `products_definition.json`
2. **Product classification** – labels each shipment using the inferred products
3. **Attribute inference** – builds attribute schema/definitions from the classified rows
4. **Attribute classification** – same outputs as the manual workflow

## Tips
- The shipment CSV must contain `hs_code`, `goods_shipped`, and `shipment_id` columns.
- For automatic mode, ensure `--hs-code` matches the dominant HS-4 in your data.
- Environment variables (e.g., `GOOGLE_API_KEY`) can be loaded from `.env` automatically.

