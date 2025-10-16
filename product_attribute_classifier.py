"""
Product Attribute Classifier Module

Extract and classify product attributes from goods descriptions using
deterministic matching, LLM classification, and heuristic enrichment.
"""

import os
import json
import time
import re
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
import google.generativeai as genai


logger = logging.getLogger(__name__)


# Negation detection constants
NEGATION_WORDS = [
    'not', 'no', 'without', 'w/o', 'minus', 'excluding', 'free of', 'free-from', 'non'
]
NEGATION_PREFIXES = ['un', 'non']


class DeterministicMatcher:
    """
    Robust deterministic multi-attribute matcher with negation awareness.
    
    Features:
    - Whole-token & multiword value matching (case-insensitive)
    - Longest-value precedence
    - Negation detection (rejects matches in negated phrases)
    - Boundary-aware matching
    """
    
    def __init__(self, attr_values: Dict[str, List[str]], config: Dict[str, Any]):
        self.attr_values = attr_values
        self.config = config
        self.min_len = config.get('deterministic_min_token_chars', 3)
        self.patterns: Dict[str, List[Tuple[str, re.Pattern, int]]] = {}
        self._compile()
        self.stats = {
            'attempted': 0,
            'matched': 0,
            'negation_blocked': 0,
            'attr_attempted': 0,
            'attr_matched': 0,
        }
    
    def _compile(self):
        """Compile regex patterns for all attribute values."""
        for attr, values in self.attr_values.items():
            compiled: List[Tuple[str, re.Pattern, int]] = []
            seen_norm = set()
            for v in values:
                if not isinstance(v, str):
                    continue
                vv = v.strip()
                if len(vv) < self.min_len:
                    continue
                norm = ' '.join(vv.lower().split())
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)
                
                # Handle density-like patterns (e.g., "500 G/L")
                dens_like = False
                if re.search(r"\b\d{3,4}\s*g/?l\b", norm.replace(' ', '')) or 'g/l' in norm or 'gr/l' in norm or 'grams/liter' in norm:
                    dens_like = True
                
                if dens_like:
                    number_part = re.findall(r"\d+", norm)
                    number_regex = r"(?:" + r"|".join(sorted({re.escape(n) for n in number_part}, key=len, reverse=True)) + r")" if number_part else r"\d+"
                    dens_pattern = number_regex + r"\s*(?:g|gr|grams)\s*/?\s*(?:l|liter)"
                    pattern = re.compile(rf"(?<![A-Za-z0-9]){dens_pattern}(?![A-Za-z0-9])", re.IGNORECASE)
                    compiled.append((v, pattern, len(v)))
                else:
                    escaped = re.escape(norm)
                    pattern = re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
                    compiled.append((v, pattern, len(v)))
            
            compiled.sort(key=lambda x: (-x[2], x[0]))
            self.patterns[attr] = compiled
    
    def _has_negation(self, text: str, start: int, end: int, token_lower: str) -> bool:
        """Check if match is in a negated context."""
        window_chars = self.config.get('deterministic_negation_window_chars', 28)
        prefix_slice = text[max(0, start-4):start].lower()
        
        for pref in NEGATION_PREFIXES:
            if prefix_slice.endswith(pref) and token_lower.startswith(prefix_slice[-len(pref):]):
                return True
        
        look_back = text[max(0, start-window_chars):start].lower()
        look_back = re.sub(r'[^a-z0-9]+', ' ', look_back)
        for w in NEGATION_WORDS:
            if look_back.endswith(' ' + w) or look_back == w or look_back.endswith(w.replace(' ', '')):
                return True
        
        return False
    
    def classify_goods(self, goods_list: List[str]) -> List[Dict[str, Any]]:
        """Classify a list of goods descriptions."""
        results: List[Dict[str, Any]] = []
        for g in goods_list:
            self.stats['attempted'] += 1
            lower_txt = ' ' + ' '.join(g.lower().split()) + ' '
            attr_assign: Dict[str, str] = {}
            
            for attr, pats in self.patterns.items():
                self.stats['attr_attempted'] += 1
                chosen = None
                chosen_pos = 10**9
                
                for canonical, rgx, _length in pats:
                    m = rgx.search(lower_txt)
                    if not m:
                        continue
                    
                    if self.config.get('enable_negation_guard') and self._has_negation(lower_txt, m.start(), m.end(), canonical.lower()):
                        self.stats['negation_blocked'] += 1
                        continue
                    
                    pos = m.start()
                    if chosen is None or pos < chosen_pos:
                        chosen = canonical
                        chosen_pos = pos
                    break
                
                if chosen:
                    attr_assign[attr] = chosen
                    self.stats['attr_matched'] += 1
            
            if attr_assign:
                self.stats['matched'] += 1
            results.append({'goods_shipped': g, 'attributes': attr_assign})
        
        return results
    
    def coverage(self) -> Dict[str, Any]:
        """Return coverage statistics."""
        total_attrs = self.stats['attr_attempted']
        return {
            'rows_with_any_match': self.stats['matched'],
            'rows_total': self.stats['attempted'],
            'attribute_matches': self.stats['attr_matched'],
            'attribute_attempted': total_attrs,
            'attribute_match_rate': round(self.stats['attr_matched']/max(1, total_attrs), 6),
            'negation_blocked': self.stats['negation_blocked']
        }


class TokenTracker:
    """Track token usage against a budget."""
    
    def __init__(self, budget: int):
        self.budget = int(budget)
        self.used = 0
    
    def add(self, tokens: int):
        self.used += int(tokens)
        return self.used <= self.budget
    
    def remaining(self) -> int:
        return max(0, self.budget - self.used)
    
    def usage(self) -> int:
        return self.used


def load_product_attribute_schema(path: str) -> Dict[str, Any]:
    """
    Load attribute/value schema from JSON.
    
    Expected structure:
    {
      "hs_code": {
          "version": "...",
          "products": {
             "ProductName": {
                "attributes": { "AttrType": ["Value1", "Value2", ...] }
             }
          }
      }
    }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Attribute schema file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    normalized: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for hs4, hs_block in raw.items():
        if not isinstance(hs_block, dict):
            continue
        products = hs_block.get('products', {}) or {}
        for product_name, pdata in products.items():
            if not isinstance(pdata, dict):
                continue
            attrs = pdata.get('attributes', {}) or {}
            cleaned_attrs: Dict[str, List[str]] = {}
            for attr_type, values in attrs.items():
                if not isinstance(values, list):
                    continue
                cleaned = [v.strip() for v in values if isinstance(v, str) and v.strip()]
                cleaned_attrs[attr_type] = cleaned
            if cleaned_attrs:
                normalized.setdefault(hs4, {})[product_name] = cleaned_attrs
    
    return normalized


def load_attribute_definitions(path: str) -> Dict[str, str]:
    """Load attribute definitions for prompting."""
    if not os.path.exists(path):
        logger.warning(f"Attribute definitions file not found: {path}")
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            definitions = json.load(f)
        attr_definitions = {k: v for k, v in definitions.items() if k.startswith('attr_')}
        logger.info(f"Loaded {len(attr_definitions)} attribute definitions")
        return attr_definitions
    except Exception as e:
        logger.warning(f"Failed to load attribute definitions: {e}")
        return {}


def load_flat_products(csv_path: str) -> pd.DataFrame:
    """Load and prepare classified shipment data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path, low_memory=False, dtype={'hs_code': str})
    required = ['hs_code', 'goods_shipped', 'category']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Input CSV missing columns: {missing}")
    
    df = df.copy()
    df['hs_code'] = df['hs_code'].astype(str).str.strip().str[:4].str.zfill(4)
    df['goods_shipped'] = df['goods_shipped'].astype(str).str.strip()
    df['category'] = df['category'].astype(str).str.strip()
    
    if 'shipment_id' in df.columns:
        df['shipment_id'] = df['shipment_id'].astype(str).str.strip()
    else:
        df['shipment_id'] = [str(i) for i in range(1, len(df) + 1)]
    
    df = df.dropna(subset=['goods_shipped'])
    return df[['hs_code', 'category', 'goods_shipped', 'shipment_id']]


def build_multiitem_response_schema(attr_values: Dict[str, List[str]], allow_out_of_schema: bool = True):
    """Build JSON schema for multi-item batch classification."""
    attr_props = {}
    for attr, vals in attr_values.items():
        if allow_out_of_schema:
            attr_props[attr] = {'type': 'string'}
        else:
            enum_vals = sorted({v for v in vals if v})
            if 'None' not in enum_vals:
                enum_vals.append('None')
            attr_props[attr] = {'type': 'string', 'enum': enum_vals}
    
    return {
        'type': 'object',
        'properties': {
            'items': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'item_preview': {'type': 'string'},
                        'attributes': {
                            'type': 'object',
                            'properties': attr_props,
                            'required': []
                        }
                    },
                    'required': ['item_preview', 'attributes']
                }
            }
        },
        'required': ['items']
    }


def build_batch_prompt(hs4: str, product: str, attr_values: Dict[str, List[str]], goods_list: List[str], batch_num: int, total_batches: int, attr_definitions: Dict[str, str], config: Dict[str, Any]) -> str:
    """Build classification prompt for a batch of goods."""
    max_preview = config.get('max_hints_preview', 25)
    
    lines: List[str] = []
    lines.append("TASK: For each goods description pick exactly one allowed value per attribute, or 'None' if nothing clearly fits.")
    lines.append(f"HS4 {hs4} | Product {product} | Batch {batch_num}/{total_batches}")
    
    if attr_definitions:
        lines.append("\nATTRIBUTE DEFINITIONS:")
        for attr in attr_values.keys():
            attr_key = f"attr_{attr}" if not attr.startswith('attr_') else attr
            definition = attr_definitions.get(attr_key, "")
            if definition:
                lines.append(f"- {attr}: {definition}")
            else:
                lines.append(f"- {attr}: [Definition not available]")
    
    lines.append("\nALLOWED VALUE PREVIEW:")
    for attr, vals in attr_values.items():
        pv = ", ".join(vals[:max_preview]) + (" ..." if len(vals) > max_preview else "")
        lines.append(f"- {attr}: {pv}")
    
    lines.append("""

You are a **Senior Commodity Specialist** with deep experience in international trade, commodity classification, and named-entity extraction from shipment descriptions. Your sole task: **extract attribute values from a single text field `Goods shipped` and map them to the provided taxonomy**.

---

# **Step-by-step Workflow**:
1. **Carefully read** the entire `Goods shipped` text. It may contain multiple pieces of information, but you must focus only on the attributes listed.
2. **For each attribute**, identify if any of the allowed values are clearly mentioned in the text. You must pick exactly one value per attribute.
3. **Prioritize exact matches** of the allowed values. If multiple allowed values could match, prefer (in order):
   a. exact token match (whole-word)
   b. longest / most specific textual match
   c. contextual semantic match — choose the most industry-standard option
4. **Extract obvious technical patterns**: For technical specifications like density (e.g., "500 G/L", "550 GR/L"), percentages, measurements, or other clearly identifiable values, ALWAYS extract them as custom values even if not in the allowed list. These are valuable data points that should never be missed.
5. **Use "None" only when completely missing**: If the attribute is not present or cannot be inferred or matched (not even as a custom), then and only then use "None".
---

## Hard rules (follow exactly)

1. **Normalization:** Ignore case, punctuation, and hyphen/space differences. Treat obvious spelling errors, plural/singular forms, and verb endings as matches when intent is clear. Be language-agnostic.
2. **Prefer allowed values:** You **must** prioritize matching values from the allowed list for each attribute. **If no allowed value is a fit**, you may come up with additional value for the attribute similar to the already provided allowed values but always remember product is NOT an attribute.
3. **Extract obvious technical patterns:** For technical specifications like density (e.g., "500 G/L", "550 GR/L"), percentages, measurements, or other clearly identifiable values, ALWAYS extract them as custom values even if not in the allowed list. These are valuable data points that should never be missed.
4. **Output format:** For each attribute, return either:
   * `"value"`: a string from the allowed list
   * or
   * `"custom"`: a string **not** in the allowed list, when no provided value is appropriate or when extracting obvious technical patterns.
   Example:
   ```json
   {
     "Form": { "value": "Powder" },
     "Color": { "custom": "Light reddish-brown" },
     "Density": { "custom": "500 G/L" }
   }
   ```
5. **One unique value per attribute:** Return a **single, clear, unique** value per attribute (no arrays, no multiple choices).
6. **Use `"None"` only when completely missing:** If the attribute is not present or cannot be inferred or matched (not even as a `"custom"`), then and only then use:
   ```json
   "AttributeName": "None"
   ```
7. **Inference only when obvious:** Apply domain-specific inference **only** when the conclusion is unambiguous (e.g., "not ground nor crushed" -> `Form = "Whole"`). Avoid speculative or creative guesses.
8. **Units & minor variants:** Ignore minor unit/abbreviation differences (g/l, gl, gr/l, kg, etc.) when they don't impact the attribute value. Do not convert or normalize numeric quantities unless the taxonomy explicitly requires a numeric value.
9. **Strict output format:** Output must be **ONLY** the JSON block described above. No extra text, no comments, no trailing commas.

---

""")
    
    lines.append("\nGOODS:")
    for i, g in enumerate(goods_list, start=1):
        lines.append(f"{i}. {g}")
    lines.append("\nIMPORTANT: Return results in the SAME ORDER as the input list (1, 2, 3...).")
    lines.append("For each item, include 'item_preview' with the first 30 characters of the goods description as a verification guardrail.")
    lines.append("Do NOT echo the full goods_shipped text - only return item_preview (first ~30 chars) and attributes.")
    lines.append("\nRETURN JSON ONLY.")
    
    return "\n".join(lines)


def heuristic_fill(attrs: Dict[str, str], goods_text: str, attr_values: Dict[str, List[str]], config: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    """Apply heuristic extraction for missing attributes."""
    if not config.get('enable_heuristic_fill', False):
        return attrs, []
    
    filled: List[str] = []
    norm_text = goods_text.lower()
    norm_text_simple = re.sub(r'[#,;/\\()]', ' ', norm_text)
    
    for attr, allowed_list in attr_values.items():
        if attr not in attrs or attrs[attr] != 'None':
            continue
        
        candidates = []
        for val in allowed_list:
            lv = val.lower()
            if lv and lv in norm_text_simple:
                start_idx = norm_text_simple.find(lv)
                if start_idx >= 0:
                    end_idx = start_idx + len(lv)
                    left_ok = start_idx == 0 or norm_text_simple[start_idx-1].isspace()
                    right_ok = end_idx == len(norm_text_simple) or norm_text_simple[end_idx:end_idx+1].isspace()
                    if left_ok and right_ok:
                        candidates.append(val)
        
        if candidates:
            chosen = sorted(candidates, key=lambda x: (-len(x), x))[0]
            attrs[attr] = chosen
            filled.append(attr)
    
    return attrs, filled


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from response text."""
    if not text:
        return {}
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end >= 0:
            return json.loads(text[start:end+1])
    except Exception:
        return {}
    return {}


def sanitize_filename(name: str) -> str:
    """Sanitize string for use in filenames."""
    safe = ''.join(ch if ch.isalnum() or ch in (' ', '_', '-') else '_' for ch in name)
    safe = '_'.join(safe.strip().split())
    return safe[:120] or 'product'


def classify_batch(hs4: str, product: str, attr_values: Dict[str, List[str]], goods_list: List[str], model, tracker: TokenTracker, batch_num: int, total_batches: int, attr_definitions: Dict[str, str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Classify a batch of goods descriptions."""
    if tracker.remaining() <= 0:
        return []
    
    # Deterministic first pass
    det_stats = {}
    if config.get('deterministic_first_pass'):
        dm = DeterministicMatcher(attr_values, config)
        deterministic_results = dm.classify_goods(goods_list)
        det_stats = dm.coverage()
    else:
        deterministic_results = [{'goods_shipped': g, 'attributes': {}} for g in goods_list]
        det_stats = {
            'rows_with_any_match': 0,
            'rows_total': len(goods_list),
            'attribute_matches': 0,
            'attribute_attempted': len(goods_list) * len(attr_values),
            'attribute_match_rate': 0.0,
            'negation_blocked': 0
        }
    
    full_attr_count = len(attr_values)
    all_resolved = config.get('deterministic_first_pass') and all(len(r['attributes']) == full_attr_count for r in deterministic_results)
    
    if config.get('dry_run_deterministic_only'):
        all_resolved = True
    
    if all_resolved:
        out = []
        for rec in deterministic_results:
            comp = {a: rec['attributes'].get(a, 'None') for a in attr_values.keys()}
            out.append({
                'goods_shipped': rec['goods_shipped'],
                'attributes': comp,
                '_validation_meta': {
                    'invalid': 0,
                    'total': len(attr_values),
                    'deterministic_only': True,
                    'deterministic_stats': det_stats
                }
            })
        return out
    
    # LLM classification
    response_schema = build_multiitem_response_schema(attr_values, config.get('allow_out_of_schema_values', True))
    prompt = build_batch_prompt(hs4, product, attr_values, goods_list, batch_num, total_batches, attr_definitions, config)
    
    retries = 0
    supports_schema = config.get('use_structured_output', True)
    
    while retries <= config.get('max_retries', 3):
        try:
            gen_kwargs = dict(
                temperature=config.get('temperature', 0.0),
                response_mime_type='application/json'
            )
            if supports_schema:
                gen_kwargs['response_schema'] = response_schema
            
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(**gen_kwargs))
            text = getattr(response, 'text', '')
            usage = getattr(response, 'usage_metadata', None)
            
            used_tokens = 0
            if usage:
                try:
                    used_tokens = int(getattr(usage, 'prompt_token_count', 0)) + int(getattr(usage, 'candidates_token_count', 0))
                except Exception:
                    used_tokens = 0
            if used_tokens == 0:
                used_tokens = (len(prompt) + len(text)) // 4
            
            tracker.add(used_tokens)
            
            parsed = parse_json_response(text)
            out: List[Dict[str, Any]] = []
            invalid_counter = 0
            total_assignments = 0
            out_of_schema_counter = 0
            
            if isinstance(parsed, dict) and isinstance(parsed.get('items'), list):
                # Use positional matching - response order must match input order
                for i, item in enumerate(parsed['items']):
                    if not isinstance(item, dict):
                        continue
                    
                    # Get the corresponding goods description by position
                    if i >= len(goods_list):
                        logger.warning(f"Response has more items than input ({i+1} > {len(goods_list)})")
                        break
                    
                    gs = goods_list[i]  # Use input goods by position
                    
                    # Verify preview as guardrail (optional - log mismatch but continue)
                    item_preview = item.get('item_preview', '')
                    if item_preview:
                        expected_preview = gs[:30]
                        if not expected_preview.startswith(item_preview[:15]):
                            logger.debug(f"Preview mismatch at position {i}: expected '{expected_preview[:20]}...' got '{item_preview[:20]}...'")
                    
                    attrs_in = item.get('attributes', {})
                    
                    norm_attrs: Dict[str, str] = {}
                    raw_attrs: Dict[str, Any] = {}
                    custom_flags: List[str] = []
                    
                    for attr in attr_values.keys():
                        total_assignments += 1
                        raw_v = None
                        if isinstance(attrs_in, dict):
                            raw_v = attrs_in.get(attr, 'None')
                        val = raw_v
                        was_custom_nested = False
                        
                        if isinstance(val, dict):
                            if 'value' in val and isinstance(val['value'], str):
                                val = val['value']
                            elif 'custom' in val and isinstance(val['custom'], str):
                                val = val['custom']
                                was_custom_nested = True
                                if val not in attr_values[attr] and val != 'None':
                                    out_of_schema_counter += 1
                                    custom_flags.append(attr)
                            else:
                                val = 'None'
                        
                        if not isinstance(val, str):
                            val = 'None' if val is None else str(val)
                        
                        if not was_custom_nested and val not in attr_values[attr] and val != 'None':
                            if config.get('allow_out_of_schema_values'):
                                out_of_schema_counter += 1
                                custom_flags.append(attr)
                            else:
                                invalid_counter += 1
                                if config.get('log_invalid_values'):
                                    logger.debug(f"Invalid value coerced -> hs4={hs4} product={product} attr={attr} raw='{val}'")
                                val = 'None'
                        
                        norm_attrs[attr] = val
                        if config.get('record_raw_values'):
                            raw_attrs[attr] = raw_v
                    
                    # Merge deterministic assignments
                    if config.get('deterministic_first_pass'):
                        det_lookup = next((d for d in deterministic_results if d['goods_shipped'] == gs), None)
                        if det_lookup:
                            for a, v_det in det_lookup['attributes'].items():
                                if v_det and v_det in attr_values.get(a, []):
                                    if norm_attrs.get(a) != v_det:
                                        norm_attrs[a] = v_det
                                        if a in custom_flags:
                                            custom_flags.remove(a)
                    
                    record: Dict[str, Any] = {'goods_shipped': gs, 'attributes': norm_attrs}
                    if custom_flags:
                        record['custom_attributes'] = custom_flags
                    if config.get('record_raw_values'):
                        record['raw_attributes'] = raw_attrs
                    out.append(record)
            
            if not out:
                for gs in goods_list:
                    for _attr in attr_values.keys():
                        total_assignments += 1
                    out.append({'goods_shipped': gs, 'attributes': {a: 'None' for a in attr_values.keys()}})
            
            if out:
                out[0]['_validation_meta'] = {
                    'invalid': invalid_counter,
                    'total': total_assignments,
                    'deterministic_stats': det_stats,
                    'out_of_schema': out_of_schema_counter if config.get('allow_out_of_schema_values') else 0
                }
            
            # Heuristic enrichment
            heuristic_total_fills = 0
            for rec in out:
                rec_attrs, filled = heuristic_fill(rec['attributes'], rec['goods_shipped'], attr_values, config)
                if filled:
                    heuristic_total_fills += len(filled)
                    rec['attributes'] = rec_attrs
                    if config.get('record_raw_values'):
                        rec.setdefault('heuristic_filled', filled)
            
            if out:
                out[0]['_validation_meta']['heuristic_fills'] = heuristic_total_fills
            
            return out
        
        except TypeError:
            if supports_schema:
                logger.info('Structured schema unsupported; retrying without schema.')
                supports_schema = False
                retries += 1
                time.sleep(config.get('retry_delay', 4))
                continue
            retries += 1
            time.sleep(config.get('retry_delay', 4) * retries)
        except Exception as e:
            msg = str(e)
            if supports_schema and ('response_schema' in msg or 'Unknown field' in msg):
                logger.info('API rejected response_schema; retrying without schema.')
                supports_schema = False
                retries += 1
                time.sleep(config.get('retry_delay', 4))
                continue
            retries += 1
            logger.warning(f"Retry {retries} failed for batch: {e}")
            time.sleep(config.get('retry_delay', 4) * retries)
    
    return []


def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """Load classification checkpoint."""
    if not os.path.exists(checkpoint_file):
        return {
            'products': {},
            'token_usage': 0,
            'timestamp': datetime.now().isoformat(),
            'version': 2
        }
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'products' not in data:
            data['products'] = {}
        return data
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return {
            'products': {},
            'token_usage': 0,
            'timestamp': datetime.now().isoformat(),
            'version': 2
        }


def save_checkpoint(checkpoint_file: str, state: Dict[str, Any]):
    """Save classification checkpoint."""
    try:
        state['timestamp'] = datetime.now().isoformat()
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logger.info('Checkpoint saved.')
    except Exception as e:
        logger.error(f'Failed to save checkpoint: {e}')


def classify_attributes(
    input_csv: str,
    product_attributes_schema_path: str,
    attribute_definitions_path: str,
    output_json: str,
    output_csv: str,
    output_folder: str,
    checkpoint_file: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classify product attributes from goods descriptions.
    
    Parameters:
    -----------
    input_csv : str
        Path to classified shipment master CSV
    product_attributes_schema_path : str
        Path to product attributes schema JSON
    attribute_definitions_path : str
        Path to attribute definitions JSON
    output_json : str
        Path for aggregated JSON output
    output_csv : str
        Path for combined flat CSV output
    output_folder : str
        Directory for per-product CSV files
    checkpoint_file : str, optional
        Path to checkpoint file
    config : Dict[str, Any], optional
        Configuration dictionary
    api_key : str, optional
        Google API key
    
    Returns:
    --------
    Dict[str, Any]
        Results dictionary with metadata and classifications
    """
    logger.info('Starting attribute classification...')
    
    # Default config
    default_config = {
        'hs_filter': None,
        'sample_goods_per_product': None,
        'items_per_call': 10,
        'max_token_budget': 20_000_000,
        'model_name': 'gemini-2.5-flash-lite',
        'retry_delay': 4,
        'max_retries': 3,
        'temperature': 0.0,
        'use_structured_output': True,
        'save_overall_csv': True,
        'validation_mode': 'coerce',
        'log_invalid_values': True,
        'record_raw_values': True,
        'enable_heuristic_fill': True,
        'max_hints_preview': 25,
        'deterministic_first_pass': True,
        'deterministic_min_token_chars': 3,
        'enable_negation_guard': True,
        'deterministic_negation_window_chars': 28,
        'dry_run_deterministic_only': False,
        'debug_deterministic_examples': 0,
        'allow_out_of_schema_values': True,
        'add_custom_flag_columns': False,
    }
    
    if config:
        default_config.update(config)
    config = default_config
    
    # Configure API
    if api_key:
        genai.configure(api_key=api_key)
    elif os.getenv('GOOGLE_API_KEY'):
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    else:
        raise ValueError("Google API key required via api_key parameter or GOOGLE_API_KEY environment variable")
    
    model = genai.GenerativeModel(config['model_name'])
    
    # Load schemas
    schema_map = load_product_attribute_schema(product_attributes_schema_path)
    hs_available = set(schema_map.keys())
    attr_definitions = load_attribute_definitions(attribute_definitions_path)
    
    # Load data
    df = load_flat_products(input_csv)
    if config['hs_filter']:
        allowed = {str(h)[:4] for h in config['hs_filter']}
        df = df[df['hs_code'].isin(allowed)]
    df = df[df['hs_code'].isin(hs_available)]
    
    # Group goods by hs4+product
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (hs4, product), sub in df.groupby(['hs_code', 'category']):
        sub2 = sub[['goods_shipped', 'shipment_id']].dropna(subset=['goods_shipped'])
        goods_to_shipments: Dict[str, List[str]] = {}
        for r in sub2.to_dict(orient='records'):
            g = r['goods_shipped']
            sid = r['shipment_id']
            if not isinstance(g, str) or not g.strip():
                continue
            if sid is None or (isinstance(sid, float) and pd.isna(sid)) or not isinstance(sid, str) or not str(sid).strip():
                sid = ''
            goods_to_shipments.setdefault(g, []).append(str(sid))
        
        unique_goods = list(goods_to_shipments.keys())
        if config['sample_goods_per_product'] and isinstance(config['sample_goods_per_product'], int):
            unique_goods = unique_goods[:config['sample_goods_per_product']]
        
        if unique_goods:
            grouped.setdefault(hs4, {})[product] = {
                'goods': sorted(unique_goods),
                'shipments': goods_to_shipments
            }
    
    checkpoint_file = checkpoint_file or 'attribute_classification_checkpoint.json'
    state = load_checkpoint(checkpoint_file)
    tracker = TokenTracker(config['max_token_budget'])
    tracker.used = int(state.get('token_usage', 0))
    
    items_per_call = max(1, int(config['items_per_call']))
    os.makedirs(output_folder, exist_ok=True)
    
    hs_list = sorted(grouped.keys())
    for hs4 in hs_list:
        for product, pdata in tqdm(grouped[hs4].items(), desc=f'HS {hs4}', leave=False):
            if tracker.remaining() <= 0:
                logger.warning('Token budget exhausted; stopping.')
                break
            
            attr_values = schema_map.get(hs4, {}).get(product)
            if not attr_values:
                continue
            
            goods_list = pdata['goods']
            shipments_map = pdata['shipments']
            key = f"{hs4}::{product}"
            prod_state = state['products'].setdefault(key, {'completed': 0, 'classifications': []})
            
            # Merge shipment ids for already-classified goods
            existing_index = {c['goods_shipped']: c for c in prod_state['classifications']}
            for g, sids in shipments_map.items():
                if g in existing_index:
                    existing = existing_index[g].setdefault('shipment_ids', [])
                    for sid in sids:
                        if sid not in existing:
                            existing.append(sid)
            
            done_goods = {c['goods_shipped'] for c in prod_state['classifications']}
            remaining_goods = [g for g in goods_list if g not in done_goods]
            total_batches = (len(remaining_goods) + items_per_call - 1) // items_per_call if remaining_goods else 0
            
            for b_start in range(0, len(remaining_goods), items_per_call):
                if tracker.remaining() <= 0:
                    logger.warning('Token budget exhausted mid-product.')
                    break
                
                batch_goods = remaining_goods[b_start:b_start+items_per_call]
                batch_num = (b_start // items_per_call) + 1
                results = classify_batch(hs4, product, attr_values, batch_goods, model, tracker, batch_num, total_batches, attr_definitions, config)
                
                if results and '_validation_meta' in results[0]:
                    meta = results[0].pop('_validation_meta')
                    state.setdefault('validation_stats', {'invalid': 0, 'total': 0})
                    state['validation_stats']['invalid'] += meta.get('invalid', 0)
                    state['validation_stats']['total'] += meta.get('total', 0)
                    if config.get('allow_out_of_schema_values'):
                        state.setdefault('out_of_schema_stats', {'count': 0})
                        state['out_of_schema_stats']['count'] += meta.get('out_of_schema', 0)
                    if 'heuristic_fills' in meta:
                        state.setdefault('heuristic_stats', {'fills': 0})
                        state['heuristic_stats']['fills'] += meta.get('heuristic_fills', 0)
                
                for entry in results:
                    gs = entry['goods_shipped']
                    if gs not in done_goods:
                        entry['shipment_ids'] = shipments_map.get(gs, [])
                        prod_state['classifications'].append(entry)
                        done_goods.add(gs)
                
                prod_state['completed'] = len(prod_state['classifications'])
                state['token_usage'] = tracker.usage()
                save_checkpoint(checkpoint_file, state)
            
            # Save per-product CSV
            try:
                fname = f"hs_{hs4}_{sanitize_filename(product)}.csv"
                rows = []
                for entry in prod_state['classifications']:
                    shipment_ids = entry.get('shipment_ids') or ['']
                    for sid in shipment_ids:
                        base = {
                            'hs_code': hs4,
                            'product': product,
                            'shipment_id': sid,
                            'goods_shipped': entry['goods_shipped']
                        }
                        for attr, val in entry['attributes'].items():
                            base[f'attr_{attr}'] = val
                            if config.get('add_custom_flag_columns'):
                                is_custom = 0
                                if entry.get('custom_attributes') and attr in entry['custom_attributes']:
                                    is_custom = 1
                                base[f'attr_{attr}_is_custom'] = is_custom
                        rows.append(base)
                if rows:
                    pd.DataFrame(rows).to_csv(os.path.join(output_folder, fname), index=False, encoding='utf-8')
                    logger.info(f"Saved per-product CSV {fname} ({len(rows)} rows)")
            except Exception as e:
                logger.warning(f"Failed to save product CSV {product}: {e}")
        
        if tracker.remaining() <= 0:
            break
    
    # Aggregated JSON output
    aggregated = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'products_processed': len(state['products']),
            'hs_codes_processed': sorted({k.split('::')[0] for k in state['products'].keys()}),
            'token_usage': tracker.usage(),
            'token_remaining': tracker.remaining(),
            'items_per_call': items_per_call
        },
        'products': state['products']
    }
    
    if 'validation_stats' in state:
        aggregated['metadata']['validation_invalid'] = state['validation_stats'].get('invalid', 0)
        aggregated['metadata']['validation_total_assignments'] = state['validation_stats'].get('total', 0)
        if state['validation_stats'].get('total', 0) > 0:
            aggregated['metadata']['validation_invalid_ratio'] = round(state['validation_stats']['invalid'] / max(1, state['validation_stats']['total']), 6)
    if 'heuristic_stats' in state:
        aggregated['metadata']['heuristic_fills'] = state['heuristic_stats'].get('fills', 0)
    if config.get('allow_out_of_schema_values') and 'out_of_schema_stats' in state:
        aggregated['metadata']['out_of_schema_values'] = state['out_of_schema_stats'].get('count', 0)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved aggregated JSON -> {output_json}")
    
    # Overall flat CSV
    if config.get('save_overall_csv'):
        try:
            flat_rows: List[Dict[str, Any]] = []
            for key, pdata in state['products'].items():
                hs4, product = key.split('::', 1)
                for entry in pdata.get('classifications', []):
                    shipment_ids = entry.get('shipment_ids') or ['']
                    for sid in shipment_ids:
                        base = {'hs_code': hs4, 'product': product, 'shipment_id': sid, 'goods_shipped': entry['goods_shipped']}
                        for attr, val in entry['attributes'].items():
                            base[f'attr_{attr}'] = val
                        flat_rows.append(base)
            if flat_rows:
                pd.DataFrame(flat_rows).to_csv(output_csv, index=False, encoding='utf-8')
                logger.info(f"Saved combined CSV -> {output_csv} (rows={len(flat_rows)})")
        except Exception as e:
            logger.warning(f"Failed to save overall CSV: {e}")
    
    # Final checkpoint save
    state['token_usage'] = tracker.usage()
    save_checkpoint(checkpoint_file, state)
    
    logger.info('[SUMMARY]')
    logger.info(f"  -> Products processed: {len(state['products'])}")
    logger.info(f"  -> Token usage: {tracker.usage():,} (remaining {tracker.remaining():,})")
    if 'validation_stats' in state:
        inval = state['validation_stats']['invalid']
        tot = state['validation_stats']['total']
        ratio = (inval / tot) if tot else 0.0
        logger.info(f"  -> Validation: invalid {inval} / {tot} assignments ({ratio:.4%})")
    if 'heuristic_stats' in state:
        logger.info(f"  -> Heuristic fills: {state['heuristic_stats'].get('fills', 0)}")
    if config.get('allow_out_of_schema_values') and 'out_of_schema_stats' in state:
        logger.info(f"  -> Out-of-schema values: {state['out_of_schema_stats'].get('count', 0)}")
    
    return aggregated

