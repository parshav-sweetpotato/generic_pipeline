"""
Generic Product Classification Pipeline

A reusable, product-agnostic pipeline for classifying shipment data
through product categorization and attribute extraction stages.

Main components:
- data_loader: Template for loading shipment data
- products_classifier: LLM-based product categorization
- product_attribute_classifier: Multi-attribute extraction and classification
- pipeline_runner: Orchestration script for end-to-end workflow
"""

__version__ = "1.0.0"

