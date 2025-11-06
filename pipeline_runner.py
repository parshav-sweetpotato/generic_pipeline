"""
Generic Pipeline Runner

Orchestrates the complete classification workflow:
1. Data loading
2. Product classification
3. Attribute classification
4. Output generation
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dotenv
from data_loader import load_data, validate_shipment_data
from products_classifier import classify_products
from product_attribute_classifier import classify_attributes
from schema_generator import SchemaGenerator, SchemaGenerationResult
from utils.build_product_schema_master import build_schema_master
from utils.combined_shipment_id_to_attr_json import combine_shipment_attributes


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Load environment variables from nearest .env (workspace root)
DOTENV_PATH = dotenv.find_dotenv(usecwd=True)
if DOTENV_PATH:
    dotenv.load_dotenv(DOTENV_PATH)
else:
    dotenv.load_dotenv()


def setup_output_directories(base_output_dir: str) -> Dict[str, str]:
    """Create output directory structure."""
    paths = {
        'base': base_output_dir,
        'per_product': os.path.join(base_output_dir, 'per_product_classifications'),
        'checkpoints': os.path.join(base_output_dir, 'checkpoints')
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths


def run_pipeline(
    data_source: str,
    products_definition: str,
    product_attributes_schema: str,
    attribute_definitions: str,
    output_dir: str,
    skip_data_loading: bool = False,
    skip_product_classification: bool = False,
    skip_attribute_classification: bool = False,
    run_helpers: bool = True,
    resume: bool = True,
    config_overrides: Optional[Dict[str, Any]] = None,
    *,
    auto_generate_schema: bool = False,
    hs_code: Optional[str] = None,
    schema_output_dir: Optional[str] = None,
    schema_generation_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the complete classification pipeline.
    
    Parameters:
    -----------
    data_source : str
        Input data source path
    products_definition : str
        Path to products definition JSON
    product_attributes_schema : str
        Path to product attributes schema JSON
    attribute_definitions : str
        Path to attribute definitions JSON
    output_dir : str
        Base output directory
    skip_data_loading : bool
        Skip data loading step
    skip_product_classification : bool
        Skip product classification step
    skip_attribute_classification : bool
        Skip attribute classification step
    run_helpers : bool
        Run helper utilities at end
    resume : bool
        Resume from checkpoints if available
    config_overrides : Dict[str, Any], optional
        Configuration overrides for attribute classification
    
    Returns:
    --------
    Dict[str, Any]
        Pipeline execution summary
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("GENERIC PRODUCT CLASSIFICATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info(f"Output directory: {output_dir}")
    
    # Setup directories
    paths = setup_output_directories(output_dir)
    
    # Define file paths
    shipment_master_csv = os.path.join(output_dir, "shipment_master.csv")
    shipment_master_classified_csv = os.path.join(output_dir, "shipment_master_classified.csv")
    classifications_json = os.path.join(output_dir, "classifications.json")
    classifications_flat_csv = os.path.join(output_dir, "classifications_flat.csv")
    product_schema_master_csv = os.path.join(output_dir, "product_schema_master.csv")
    shipment_id_to_attr_json = os.path.join(output_dir, "shipment_id_to_attr.json")
    
    results = {
        'status': 'success',
        'steps_completed': [],
        'errors': [],
        'timing': {}
    }
    
    schema_generator: Optional[SchemaGenerator] = None
    product_schema_result: Optional[SchemaGenerationResult] = None
    attribute_schema_result: Optional[SchemaGenerationResult] = None
    schema_dir: Optional[str] = None

    if auto_generate_schema and skip_product_classification:
        logger.error("Auto schema generation requires the product classification stage to run.")
        results['errors'].append('schema_generation: requires product classification step')
        results['status'] = 'failed'
        return results

    # Step 1: Data Loading
    if not skip_data_loading:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA LOADING")
        logger.info("=" * 80)
        step_start = time.time()
        
        try:
            logger.info(f"Loading data from: {data_source}")
            df = load_data(data_source, shipment_master_csv)
            validate_shipment_data(df)
            logger.info(f"✓ Data loaded successfully: {len(df)} records")
            logger.info(f"✓ Output: {shipment_master_csv}")
            results['steps_completed'].append('data_loading')
            results['data_loading'] = {'records': len(df)}
        except NotImplementedError as e:
            logger.error("Data loader is not implemented. Please implement the load_data() function.")
            logger.error(str(e))
            results['errors'].append('data_loading: Not implemented')
            return results
        except Exception as e:
            logger.error(f"✗ Data loading failed: {str(e)}")
            results['errors'].append(f'data_loading: {str(e)}')
            results['status'] = 'failed'
            return results
        
        results['timing']['data_loading'] = time.time() - step_start
    else:
        logger.info("Skipping data loading (using existing shipment_master.csv)")
        shipment_master_csv = os.path.join(output_dir, "shipment_master.csv")
        if not os.path.exists(shipment_master_csv):
            logger.error(f"Expected file not found: {shipment_master_csv}")
            results['errors'].append('data_loading: File not found')
            results['status'] = 'failed'
            return results

    # Optional: Schema generation
    if auto_generate_schema:
        logger.info("\n" + "=" * 80)
        logger.info("SCHEMA GENERATION - PRODUCT INFERENCE")
        logger.info("=" * 80)

        if skip_data_loading and not os.path.exists(shipment_master_csv):
            logger.error(
                "Auto schema generation requires a shipment master CSV. Provide one or run without --skip-data-loading."
            )
            results['errors'].append('schema_generation: missing shipment master CSV')
            results['status'] = 'failed'
            return results

        if not hs_code:
            logger.error("HS code is required when auto-generating schema")
            results['errors'].append('schema_generation: missing hs_code')
            results['status'] = 'failed'
            return results

        schema_dir = schema_output_dir or os.path.join(output_dir, "generated_schema")

        try:
            schema_generator = SchemaGenerator(generation_config=schema_generation_overrides)
            product_schema_result = schema_generator.generate_product_definition(
                hs_code=hs_code,
                shipment_csv=shipment_master_csv,
                output_dir=schema_dir,
            )
            products_definition = product_schema_result.products_definition_path or products_definition
            results['steps_completed'].append('product_inference')
            results['product_inference'] = {
                'hs_code': hs_code,
                'output_dir': schema_dir,
                'categories': product_schema_result.product_definition.product_categories
                if product_schema_result.product_definition
                else None,
            }
        except Exception as e:
            logger.error(f"✗ Product inference failed: {str(e)}")
            results['errors'].append(f'product_inference: {str(e)}')
            results['status'] = 'failed'
            return results

    # Step 2: Product Classification
    if not skip_product_classification:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: PRODUCT CLASSIFICATION")
        logger.info("=" * 80)
        step_start = time.time()
        
        try:
            checkpoint_file = os.path.join(paths['checkpoints'], 'product_classification.pkl')
            logger.info(f"Classifying products...")
            logger.info(f"Products definition: {products_definition}")
            
            classify_kwargs = dict(config_overrides or {})
            df_classified = classify_products(
                input_csv=shipment_master_csv,
                products_definition_path=products_definition,
                output_csv=shipment_master_classified_csv,
                checkpoint_file=checkpoint_file,
                resume=resume,
                **classify_kwargs
            )
            
            logger.info(f"✓ Product classification completed: {len(df_classified)} records")
            logger.info(f"✓ Output: {shipment_master_classified_csv}")
            results['steps_completed'].append('product_classification')
            results['product_classification'] = {
                'records': len(df_classified),
                'categories': df_classified['category'].nunique()
            }

            if auto_generate_schema:
                logger.info("\n" + "=" * 80)
                logger.info("SCHEMA GENERATION - ATTRIBUTE INFERENCE")
                logger.info("=" * 80)

                if schema_generator is None:
                    schema_generator = SchemaGenerator(generation_config=schema_generation_overrides)
                if schema_dir is None:
                    schema_dir = os.path.join(output_dir, "generated_schema")

                try:
                    attribute_schema_result = schema_generator.generate_attribute_configs_from_classifications(
                        hs_code=hs_code,
                        classified_csv=shipment_master_classified_csv,
                        output_dir=schema_dir,
                        product_definition_path=products_definition,
                        product_definition=(
                            product_schema_result.product_definition
                            if product_schema_result and product_schema_result.product_definition
                            else None
                        ),
                        overwrite=True,
                    )
                    if attribute_schema_result.product_attributes_schema_path:
                        product_attributes_schema = attribute_schema_result.product_attributes_schema_path
                    if attribute_schema_result.attribute_definitions_path:
                        attribute_definitions = attribute_schema_result.attribute_definitions_path
                    if attribute_schema_result.attribute_schema:
                        product_entry = attribute_schema_result.attribute_schema.entries.get(hs_code)
                    else:
                        product_entry = None
                    results['steps_completed'].append('attribute_schema_generation')
                    results['attribute_schema_generation'] = {
                        'hs_code': hs_code,
                        'output_dir': schema_dir,
                        'products_with_attributes': (
                            list(product_entry.products.keys()) if product_entry else None
                        ),
                    }
                except Exception as e:
                    logger.error(f"✗ Attribute inference failed: {str(e)}")
                    results['errors'].append(f'attribute_schema_generation: {str(e)}')
                    results['status'] = 'failed'
                    return results

        except Exception as e:
            logger.error(f"✗ Product classification failed: {str(e)}")
            results['errors'].append(f'product_classification: {str(e)}')
            results['status'] = 'failed'
            return results
        
        results['timing']['product_classification'] = time.time() - step_start
    else:
        logger.info("Skipping product classification (using existing classified file)")
        if not os.path.exists(shipment_master_classified_csv):
            logger.error(f"Expected file not found: {shipment_master_classified_csv}")
            results['errors'].append('product_classification: File not found')
            results['status'] = 'failed'
            return results
    
    # Step 3: Attribute Classification
    if not skip_attribute_classification:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: ATTRIBUTE CLASSIFICATION")
        logger.info("=" * 80)
        step_start = time.time()
        
        try:
            checkpoint_file = os.path.join(paths['checkpoints'], 'attribute_classification.json')
            logger.info(f"Classifying attributes...")
            logger.info(f"Attribute schema: {product_attributes_schema}")
            logger.info(f"Attribute definitions: {attribute_definitions}")
            
            attr_results = classify_attributes(
                input_csv=shipment_master_classified_csv,
                product_attributes_schema_path=product_attributes_schema,
                attribute_definitions_path=attribute_definitions,
                output_json=classifications_json,
                output_csv=classifications_flat_csv,
                output_folder=paths['per_product'],
                checkpoint_file=checkpoint_file,
                config=config_overrides
            )
            
            logger.info(f"✓ Attribute classification completed")
            logger.info(f"✓ Outputs:")
            logger.info(f"  - JSON: {classifications_json}")
            logger.info(f"  - Flat CSV: {classifications_flat_csv}")
            logger.info(f"  - Per-product CSVs: {paths['per_product']}/")
            
            results['steps_completed'].append('attribute_classification')
            results['attribute_classification'] = {
                'products_processed': attr_results['metadata']['products_processed'],
                'token_usage': attr_results['metadata']['token_usage']
            }
        except Exception as e:
            logger.error(f"✗ Attribute classification failed: {str(e)}")
            results['errors'].append(f'attribute_classification: {str(e)}')
            results['status'] = 'failed'
            return results
        
        results['timing']['attribute_classification'] = time.time() - step_start
    else:
        logger.info("Skipping attribute classification")
    
    # Step 4: Helper Utilities (Optional)
    if run_helpers:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: HELPER UTILITIES")
        logger.info("=" * 80)
        step_start = time.time()
        
        # Build product schema master
        try:
            logger.info("Building product schema master...")
            schema_df = build_schema_master(
                input_dir=paths['per_product'],
                output_csv=product_schema_master_csv
            )
            logger.info(f"✓ Product schema master created: {product_schema_master_csv}")
            results['steps_completed'].append('schema_master')
        except Exception as e:
            logger.warning(f"Schema master generation failed: {str(e)}")
            results['errors'].append(f'schema_master: {str(e)}')
        
        # Combine shipment attributes
        try:
            logger.info("Combining shipment attributes...")
            combined = combine_shipment_attributes(
                input_dir=paths['per_product'],
                pattern="*.csv",
                output_path=shipment_id_to_attr_json
            )
            logger.info(f"✓ Shipment attributes combined: {shipment_id_to_attr_json}")
            results['steps_completed'].append('combine_attributes')
        except Exception as e:
            logger.warning(f"Combining shipment attributes failed: {str(e)}")
            results['errors'].append(f'combine_attributes: {str(e)}')
        
        results['timing']['helpers'] = time.time() - step_start
    
    # Summary
    total_time = time.time() - start_time
    results['timing']['total'] = total_time
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Status: {results['status'].upper()}")
    logger.info(f"Steps completed: {', '.join(results['steps_completed'])}")
    if results['errors']:
        logger.info(f"Errors: {len(results['errors'])}")
        for error in results['errors']:
            logger.info(f"  - {error}")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info(f"End time: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generic Product Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python pipeline_runner.py \\
    --data-source ./raw_data.csv \\
    --products-definition ./config/products_definition.json \\
    --product-attributes-schema ./config/product_attributes_schema.json \\
    --attribute-definitions ./config/attribute_definitions.json \\
    --output-dir ./output
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data-source',
        required=True,
        help='Input data source path'
    )
    parser.add_argument(
        '--products-definition',
        required=True,
        help='Path to products definition JSON'
    )
    parser.add_argument(
        '--product-attributes-schema',
        required=True,
        help='Path to product attributes schema JSON'
    )
    parser.add_argument(
        '--attribute-definitions',
        required=True,
        help='Path to attribute definitions JSON'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Base output directory'
    )
    
    # Optional arguments
    parser.add_argument(
        '--skip-data-loading',
        action='store_true',
        help='Skip data loading step (use existing shipment_master.csv)'
    )
    parser.add_argument(
        '--skip-product-classification',
        action='store_true',
        help='Skip product classification step'
    )
    parser.add_argument(
        '--skip-attribute-classification',
        action='store_true',
        help='Skip attribute classification step'
    )
    parser.add_argument(
        '--no-helpers',
        action='store_true',
        help='Skip helper utilities at end'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from checkpoints'
    )
    
    # Configuration overrides
    parser.add_argument(
        '--model',
        default='gemini-2.0-flash',
        help='LLM model name (default: gemini-2.0-flash)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for classification (default: 10)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Number of parallel workers (default: 5)'
    )
    parser.add_argument(
        '--token-budget',
        type=int,
        default=20_000_000,
        help='Maximum token budget (default: 20000000)'
    )
    parser.add_argument(
        '--items-per-call',
        type=int,
        default=10,
        help='Items per LLM call for attribute classification (default: 10)'
    )
    parser.add_argument(
        '--auto-generate-schema',
        action='store_true',
        help='Infer product schema from the input data using Gemini before classification'
    )
    parser.add_argument(
        '--hs-code',
        type=str,
        help='HS-4 code to target when auto-generating schema'
    )
    parser.add_argument(
        '--schema-output-dir',
        type=str,
        help='Directory where generated schema files should be written (default: <output-dir>/generated_schema)'
    )
    parser.add_argument(
        '--schema-model',
        type=str,
        default=None,
        help='Override the Gemini model used for schema generation (default: gemini-2.0-flash)'
    )
    parser.add_argument(
        '--schema-temperature',
        type=float,
        default=None,
        help='Override temperature for schema generation (default: 0.0)'
    )
    parser.add_argument(
        '--schema-max-samples',
        type=int,
        default=None,
        help='Maximum total goods descriptions to sample for schema generation (default: 120)'
    )
    
    args = parser.parse_args()
    
    # Build configuration overrides
    config_overrides = {
        'model_name': args.model,
        'batch_size': args.batch_size,
        'max_workers': args.max_workers,
        'max_token_budget': args.token_budget,
        'items_per_call': args.items_per_call,
    }
    
    schema_overrides = {}
    if args.schema_model:
        schema_overrides['model_name'] = args.schema_model
    if args.schema_temperature is not None:
        schema_overrides['temperature'] = args.schema_temperature
    if args.schema_max_samples is not None:
        schema_overrides['max_total_samples'] = args.schema_max_samples

    if args.auto_generate_schema and not args.hs_code:
        parser.error('--hs-code is required when --auto-generate-schema is set')

    # Run pipeline
    try:
        results = run_pipeline(
            data_source=args.data_source,
            products_definition=args.products_definition,
            product_attributes_schema=args.product_attributes_schema,
            attribute_definitions=args.attribute_definitions,
            output_dir=args.output_dir,
            skip_data_loading=args.skip_data_loading,
            skip_product_classification=args.skip_product_classification,
            skip_attribute_classification=args.skip_attribute_classification,
            run_helpers=not args.no_helpers,
            resume=not args.no_resume,
            config_overrides=config_overrides,
            auto_generate_schema=args.auto_generate_schema,
            hs_code=args.hs_code,
            schema_output_dir=args.schema_output_dir,
            schema_generation_overrides=schema_overrides or None,
        )
        
        if results['status'] == 'failed':
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

