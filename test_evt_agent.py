"""
STAR: Semantic-enhanced Two-stage Agent for model peRformance prediction

Full pipeline:
1. Read requirements from the test set
2. Check and load/generate benchmark and model feature databases
3. Check and generate CPMF prediction results
4. Check and generate reference model cache
5. Semantic adjustment module: refine predictions based on textual features, reference matrix, and statistical expectations
"""
import pandas as pd
import numpy as np
import json
import re
import subprocess
import sys
from pathlib import Path
from openai import OpenAI
from typing import Dict, Optional, Tuple
from scipy import stats

# Import agents
from agents.benchmark_agent import BenchmarkAgent
from agents.model_agent import ModelAgent

# ============================================================================
# Path configuration
# ============================================================================
DATA_DIR = Path(__file__).parent / "data" / "opencompass_cache"
BASELINE_DIR = Path(__file__).parent / "crosspred"

RATE = "0.4"
suffix = "gpt-4o"
# Data files
TRAIN_DATA_FILE = DATA_DIR / f"train_data_wide_{RATE}.csv"
TEST_DATA_FILE = DATA_DIR / f"test_data_wide_{RATE}.csv"

# Feature databases
MODELS_FEATURES_FILE = DATA_DIR / "models_features_db.json"
BENCHMARK_FEATURES_FILE = DATA_DIR / "benchmark_features_db.json"
MODELS_KNOWLEDGE_FILE = DATA_DIR / "models_knowledge_db.json"

# Component knowledge bases
VISION_MODEL_KB_FILE = DATA_DIR / "vision_model_knowledge_db.json"
LANGUAGE_MODEL_KB_FILE = DATA_DIR / "language_model_knowledge_db.json"

# CPMF prediction results
PMF_FILE = BASELINE_DIR / f"cpmf_vlm_predictions_{RATE}.csv"

# Final results (filename is dynamically generated based on model)
RESULTS_FILE_TEMPLATE = f"star_results_{RATE}_{suffix}.csv"
RESULTS_DETAIL_FILE_TEMPLATE = f"star_results_{RATE}_{suffix}.json"

# ============================================================================
# Global variables
# ============================================================================
_models_features_db = None
_benchmark_features_db = None
_models_knowledge_db = None
_vision_model_kb = None
_language_model_kb = None
_model_agent = None
_benchmark_agent = None

# ============================================================================
# Organization profile database (region, size, credibility)
# ============================================================================
ORGANIZATION_PROFILES = {
    # US large companies
    "OpenAI": {"region": "US", "size": "large", "credibility": 0.95, "chinese_focus": False, "specialties": ["reasoning", "general"]},
    "Google": {"region": "US", "size": "large", "credibility": 0.95, "chinese_focus": False, "specialties": ["multimodal", "efficiency"]},
    "Google DeepMind": {"region": "US", "size": "large", "credibility": 0.95, "chinese_focus": False, "specialties": ["reasoning", "research"]},
    "Meta": {"region": "US", "size": "large", "credibility": 0.90, "chinese_focus": False, "specialties": ["open-source", "efficiency"]},
    "Meta AI": {"region": "US", "size": "large", "credibility": 0.90, "chinese_focus": False, "specialties": ["open-source", "multimodal"]},
    "Microsoft": {"region": "US", "size": "large", "credibility": 0.90, "chinese_focus": False, "specialties": ["efficiency", "enterprise"]},
    "NVIDIA": {"region": "US", "size": "large", "credibility": 0.90, "chinese_focus": False, "specialties": ["efficiency", "hardware-optimized"]},
    "Anthropic": {"region": "US", "size": "medium", "credibility": 0.90, "chinese_focus": False, "specialties": ["safety", "reasoning"]},
    
    # Chinese large companies
    "Alibaba": {"region": "CN", "size": "large", "credibility": 0.88, "chinese_focus": True, "specialties": ["multilingual", "chinese", "efficiency"]},
    "Alibaba Cloud": {"region": "CN", "size": "large", "credibility": 0.88, "chinese_focus": True, "specialties": ["multilingual", "chinese", "long-context"]},
    "Alibaba Group": {"region": "CN", "size": "large", "credibility": 0.88, "chinese_focus": True, "specialties": ["multilingual", "chinese"]},
    "Tencent": {"region": "CN", "size": "large", "credibility": 0.85, "chinese_focus": True, "specialties": ["chinese", "multimodal"]},
    "Baidu": {"region": "CN", "size": "large", "credibility": 0.85, "chinese_focus": True, "specialties": ["chinese", "search"]},
    "ByteDance": {"region": "CN", "size": "large", "credibility": 0.85, "chinese_focus": True, "specialties": ["chinese", "video"]},
    "Huawei": {"region": "CN", "size": "large", "credibility": 0.85, "chinese_focus": True, "specialties": ["chinese", "efficiency"]},
    "Ant Group": {"region": "CN", "size": "large", "credibility": 0.85, "chinese_focus": True, "specialties": ["chinese", "finance"]},
    
    # Chinese research institutions
    "Shanghai AI Laboratory": {"region": "CN", "size": "medium", "credibility": 0.88, "chinese_focus": True, "specialties": ["research", "open-source", "chinese"]},
    "Beijing Academy of Artificial Intelligence": {"region": "CN", "size": "medium", "credibility": 0.85, "chinese_focus": True, "specialties": ["research", "vision"]},
    "BAAI": {"region": "CN", "size": "medium", "credibility": 0.85, "chinese_focus": True, "specialties": ["research", "vision"]},
    "Tsinghua University": {"region": "CN", "size": "medium", "credibility": 0.88, "chinese_focus": True, "specialties": ["research", "chinese"]},
    "OpenGVLab": {"region": "CN", "size": "medium", "credibility": 0.85, "chinese_focus": True, "specialties": ["vision", "multimodal"]},
    "SenseTime": {"region": "CN", "size": "medium", "credibility": 0.82, "chinese_focus": True, "specialties": ["vision", "chinese"]},
    "Megvii": {"region": "CN", "size": "medium", "credibility": 0.80, "chinese_focus": True, "specialties": ["vision"]},
    "InternLM": {"region": "CN", "size": "medium", "credibility": 0.85, "chinese_focus": True, "specialties": ["chinese", "long-context"]},
    "THUDM": {"region": "CN", "size": "medium", "credibility": 0.85, "chinese_focus": True, "specialties": ["chinese", "dialogue"]},
    "Zhipu AI": {"region": "CN", "size": "medium", "credibility": 0.85, "chinese_focus": True, "specialties": ["chinese", "dialogue"]},
    "DeepSeek-AI": {"region": "CN", "size": "medium", "credibility": 0.85, "chinese_focus": True, "specialties": ["reasoning", "efficiency", "chinese"]},
    "Moonshot AI": {"region": "CN", "size": "medium", "credibility": 0.82, "chinese_focus": True, "specialties": ["chinese", "long-context"]},
    "01.AI": {"region": "CN", "size": "medium", "credibility": 0.85, "chinese_focus": True, "specialties": ["chinese", "multilingual"]},
    "01-ai": {"region": "CN", "size": "medium", "credibility": 0.85, "chinese_focus": True, "specialties": ["chinese", "multilingual"]},
    "Xiaomi": {"region": "CN", "size": "large", "credibility": 0.80, "chinese_focus": True, "specialties": ["chinese", "mobile"]},
    "vivo AI Lab": {"region": "CN", "size": "medium", "credibility": 0.78, "chinese_focus": True, "specialties": ["chinese", "mobile"]},
    
    # European companies
    "Mistral AI": {"region": "EU", "size": "medium", "credibility": 0.85, "chinese_focus": False, "specialties": ["efficiency", "open-source", "multilingual"]},
    "Hugging Face": {"region": "EU", "size": "medium", "credibility": 0.85, "chinese_focus": False, "specialties": ["open-source", "community"]},
    "HuggingFaceTB": {"region": "EU", "size": "medium", "credibility": 0.82, "chinese_focus": False, "specialties": ["efficiency", "small-models"]},
    "AllenAI": {"region": "US", "size": "medium", "credibility": 0.85, "chinese_focus": False, "specialties": ["research", "open-source"]},
    
    # Japanese companies
    "Sony": {"region": "JP", "size": "large", "credibility": 0.85, "chinese_focus": False, "specialties": ["consumer-electronics", "imaging"]},
    
    # Others
    "LMSYS": {"region": "US", "size": "small", "credibility": 0.80, "chinese_focus": False, "specialties": ["research", "benchmarking"]},
    "MosaicML": {"region": "US", "size": "small", "credibility": 0.78, "chinese_focus": False, "specialties": ["efficiency", "training"]},
    "Adept AI": {"region": "US", "size": "small", "credibility": 0.75, "chinese_focus": False, "specialties": ["agents"]},
    "Rhymes AI": {"region": "US", "size": "small", "credibility": 0.70, "chinese_focus": False, "specialties": ["multimodal"]},
    "H2O.ai": {"region": "US", "size": "small", "credibility": 0.75, "chinese_focus": False, "specialties": ["efficiency", "enterprise"]},
    "Recursion": {"region": "US", "size": "small", "credibility": 0.70, "chinese_focus": False, "specialties": ["biotech"]},
    "IBM": {"region": "US", "size": "large", "credibility": 0.85, "chinese_focus": False, "specialties": ["enterprise", "multilingual"]},
    "IBM Granite Team": {"region": "US", "size": "large", "credibility": 0.85, "chinese_focus": False, "specialties": ["enterprise"]},
    "Apple": {"region": "US", "size": "large", "credibility": 0.90, "chinese_focus": False, "specialties": ["efficiency", "mobile", "vision"]},
    "Technology Innovation Institute": {"region": "AE", "size": "medium", "credibility": 0.78, "chinese_focus": False, "specialties": ["multilingual", "open-source"]},
    "Nous Research": {"region": "US", "size": "small", "credibility": 0.75, "chinese_focus": False, "specialties": ["open-source", "fine-tuning"]},
    "Princeton NLP": {"region": "US", "size": "small", "credibility": 0.80, "chinese_focus": False, "specialties": ["research", "efficiency"]},
    "LAION": {"region": "EU", "size": "small", "credibility": 0.75, "chinese_focus": False, "specialties": ["datasets", "vision"]},
}

def get_organization_profile(org_name: str) -> dict:
    """Get organization profile with fuzzy matching support"""
    if not org_name:
        return {"region": "Unknown", "size": "unknown", "credibility": 0.5, "chinese_focus": False, "specialties": []}
    
    # Exact match
    if org_name in ORGANIZATION_PROFILES:
        return ORGANIZATION_PROFILES[org_name]
    
    # Fuzzy match
    org_lower = org_name.lower()
    for name, profile in ORGANIZATION_PROFILES.items():
        if name.lower() in org_lower or org_lower in name.lower():
            return profile
    
    # Guess based on name
    if any(kw in org_lower for kw in ["china", "chinese", "beijing", "shanghai", "shenzhen", "tsinghua", "peking"]):
        return {"region": "CN", "size": "small", "credibility": 0.70, "chinese_focus": True, "specialties": []}
    
    return {"region": "Unknown", "size": "small", "credibility": 0.60, "chinese_focus": False, "specialties": []}


# ============================================================================
# Step 1: Feature database check and generation
# ============================================================================

def load_models_features_db() -> dict:
    """Load model feature database"""
    global _models_features_db
    if _models_features_db is None:
        if MODELS_FEATURES_FILE.exists():
            print(f"[1.1] Loading model feature database: {MODELS_FEATURES_FILE}")
            with open(MODELS_FEATURES_FILE, 'r', encoding='utf-8') as f:
                _models_features_db = json.load(f)
            model_count = sum(len(family.get('models', {})) for family in _models_features_db.get('model_families', {}).values())
            print(f"      Loaded features for {model_count} models")
        else:
            print(f"[1.1] Model feature database not found, will generate when needed")
            _models_features_db = {"model_families": {}}
    return _models_features_db


def load_benchmark_features_db() -> dict:
    """Load benchmark feature database"""
    global _benchmark_features_db
    if _benchmark_features_db is None:
        if BENCHMARK_FEATURES_FILE.exists():
            print(f"[1.2] Loading benchmark feature database: {BENCHMARK_FEATURES_FILE}")
            with open(BENCHMARK_FEATURES_FILE, 'r', encoding='utf-8') as f:
                _benchmark_features_db = json.load(f)
            bench_count = len(_benchmark_features_db.get('benchmarks', {}))
            print(f"      Loaded features for {bench_count} benchmarks")
        else:
            print(f"[1.2] Benchmark feature database not found, will generate when needed")
            _benchmark_features_db = {"benchmarks": {}}
    return _benchmark_features_db


def get_or_generate_model_features(model_name: str, llm_config: dict) -> dict:
    """Get or generate model features"""
    global _model_agent
    
    db = load_models_features_db()
    
    # Search across all families
    for family_name, family_data in db.get("model_families", {}).items():
        models = family_data.get("models", {})
        if model_name in models:
            print(f"      ✓ Read model features from database: {model_name}")
            return models[model_name]
    
    # Not found, use ModelAgent to generate
    print(f"      ✗ {model_name} not found in database, calling ModelAgent to generate...")
    
    if _model_agent is None:
        _model_agent = ModelAgent(llm_config=llm_config)
    
    features = _model_agent.get_features(model_name, fetch_community=True)
    
    # Save to database
    feature_dict = features.to_dict()
    
    # Update database (simplified; ideally should update the corresponding family)
    if "model_families" not in db:
        db["model_families"] = {}
    
    # Create or update family
    family_name = features.model_family or "Unknown"
    if family_name not in db["model_families"]:
        db["model_families"][family_name] = {
            "organization": features.organization,
            "models": {}
        }
    
    db["model_families"][family_name]["models"][model_name] = feature_dict
    
    # Save database
    with open(MODELS_FEATURES_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    
    print(f"      ✓ Model features generated and saved to database")
    return feature_dict


def get_or_generate_benchmark_features(benchmark_name: str, llm_config: dict) -> dict:
    """Get or generate benchmark features"""
    global _benchmark_agent
    
    db = load_benchmark_features_db()
    benchmarks = db.get("benchmarks", {})
    
    # Exact match
    if benchmark_name in benchmarks:
        #print(f"      ✓ Read benchmark features from database: {benchmark_name}")
        return benchmarks[benchmark_name]
    
    # Fuzzy match
    for name, info in benchmarks.items():
        if benchmark_name.lower() in name.lower() or name.lower() in benchmark_name.lower():
            print(f"      ✓ Fuzzy matched benchmark: {benchmark_name} -> {name}")
            return info
    
    # Not found, use BenchmarkAgent to generate
    print(f"      ✗ {benchmark_name} not found in database, calling BenchmarkAgent to generate...")
    
    if _benchmark_agent is None:
        _benchmark_agent = BenchmarkAgent(llm_config=llm_config)
    
    features = _benchmark_agent.get_features(benchmark_name)
    
    # Save to database
    feature_dict = features.to_dict()
    db["benchmarks"][benchmark_name] = feature_dict
    
    with open(BENCHMARK_FEATURES_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    
    print(f"      ✓ Benchmark features generated and saved to database")
    return feature_dict


# ============================================================================
# Step 2: CPMF prediction result check and generation
# ============================================================================

def check_or_generate_cpmf_predictions() -> bool:
    """Check or generate CPMF prediction results"""
    if PMF_FILE.exists():
        print(f"[2] ✓ CPMF predictions already exist: {PMF_FILE}")
        return True
    
    print(f"[2] ✗ CPMF predictions not found, generating...")
    print(f"    Calling CrossPred/run_cpmf_vlm.py...")
    
    # Call CPMF script
    cpmf_script = BASELINE_DIR / "run_cpmf_vlm.py"
    
    if not cpmf_script.exists():
        print(f"    Error: CPMF script not found: {cpmf_script}")
        return False
    
    try:
        # Run CPMF
        cmd = [
            sys.executable,
            str(cpmf_script),
            "--train_data", str(TRAIN_DATA_FILE),
            "--test_data", str(TEST_DATA_FILE),
            "--models_db", str(MODELS_FEATURES_FILE),
            "--benchmarks_db", str(BENCHMARK_FEATURES_FILE),
            "--output_path", str(PMF_FILE),
            "--dim", "10",
            "--draws", "500",
            "--tune", "500",
        ]
        
        print(f"    Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"    ✓ CPMF predictions generated successfully")
            return True
        else:
            print(f"    ✗ CPMF generation failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"    ✗ CPMF generation error: {e}")
        return False


def load_pmf_predictions() -> dict:
    """
    Load CPMF prediction results (format: [prediction±uncertainty]/[true_value])
    
    Returns:
        dict: {(model, benchmark): {'prediction': float, 'uncertainty': float, 'true_score': float}}
    """
    if not PMF_FILE.exists():
        print(f"Error: CPMF prediction file not found: {PMF_FILE}")
        return {}
    
    df = pd.read_csv(PMF_FILE)
    
    # Parse [prediction±uncertainty]/[true_value] format
    pmf_preds = {}
    for _, row in df.iterrows():
        model = row['Model']
        for col in df.columns[1:]:
            val = row[col]
            if pd.notna(val) and isinstance(val, str) and '/' in val:
                # Format: [67.59±5.73]/[66.80]
                match = re.match(r'\[([0-9.]+)±([0-9.]+)\]/\[([0-9.]+)\]', val)
                if match:
                    pred = float(match.group(1))
                    uncertainty = float(match.group(2))
                    true_score = float(match.group(3))
                    pmf_preds[(model, col)] = {
                        'prediction': pred,
                        'uncertainty': uncertainty,
                        'true_score': true_score
                    }
    
    print(f"[2] Loaded CPMF predictions: {len(pmf_preds)} samples (with uncertainty)")
    return pmf_preds


# ============================================================================
# Step 3: Evidence extraction module (from models_knowledge_db.json)
# ============================================================================

def load_models_knowledge_db() -> dict:
    """Load model knowledge database"""
    global _models_knowledge_db
    if _models_knowledge_db is None:
        if MODELS_KNOWLEDGE_FILE.exists():
            print(f"[3.1] Loading model knowledge database: {MODELS_KNOWLEDGE_FILE}")
            with open(MODELS_KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
                _models_knowledge_db = json.load(f)
            org_count = len(_models_knowledge_db.get('organizations', {}))
            family_count = sum(
                len(org.get('families', {})) 
                for org in _models_knowledge_db.get('organizations', {}).values()
            )
            print(f"      Loaded {org_count} organizations, {family_count} model families")
        else:
            print(f"[3.1] ⚠ Model knowledge database not found: {MODELS_KNOWLEDGE_FILE}")
            _models_knowledge_db = {"organizations": {}}
    return _models_knowledge_db


def load_component_knowledge_db(component_type: str = "vision") -> dict:
    """Load component knowledge base (vision/language)"""
    global _vision_model_kb, _language_model_kb
    
    if component_type == "vision":
        if _vision_model_kb is None and VISION_MODEL_KB_FILE.exists():
            with open(VISION_MODEL_KB_FILE, 'r', encoding='utf-8') as f:
                _vision_model_kb = json.load(f)
        return _vision_model_kb or {"components": {}}
    else:
        if _language_model_kb is None and LANGUAGE_MODEL_KB_FILE.exists():
            with open(LANGUAGE_MODEL_KB_FILE, 'r', encoding='utf-8') as f:
                _language_model_kb = json.load(f)
        return _language_model_kb or {"components": {}}


def get_component_info(component_name: str, component_type: str = "vision") -> Optional[dict]:
    """Get detailed info for a component (vision model or language model)"""
    kb = load_component_knowledge_db(component_type)
    components = kb.get("components", {})
    
    # Exact match
    if component_name in components:
        return components[component_name]
    
    # Fuzzy match
    name_lower = component_name.lower()
    for name, info in components.items():
        if name.lower() == name_lower or name_lower in name.lower():
            return info
    return None


def find_model_family(model_name: str) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
    """
    Find the family a model belongs to
    
    Returns:
        (org_name, family_name, family_data) or (None, None, None)
    """
    db = load_models_knowledge_db()
    
    for org_name, org_data in db.get('organizations', {}).items():
        for family_name, family_data in org_data.get('families', {}).items():
            if model_name in family_data.get('models', {}):
                return org_name, family_name, family_data
    
    return None, None, None


def get_model_info(model_name: str) -> Optional[dict]:
    """Get information for a single model"""
    org_name, family_name, family_data = find_model_family(model_name)
    if family_data:
        return family_data.get('models', {}).get(model_name)
        return None
    

def extract_family_evidence(model_name: str, train_df: pd.DataFrame, benchmark_cols: list) -> dict:
    """
    Extract evidence for the target model's family:
    1. Family technical summary and community feedback
    2. Historical performance of same-family models on the training set
    """
    org_name, family_name, family_data = find_model_family(model_name)
    
    evidence = {
        "org_name": org_name,
        "family_name": family_name,
        "technical_summary": None,
        "sentiment_score": None,
        "positive_aspects": [],
        "negative_aspects": [],
        "community_summary": None,
        "family_models": [],
        "family_performance": {}
    }
    
    if not family_data:
        return evidence
    
    # 1. Extract family-level information
    evidence["technical_summary"] = family_data.get("technical_summary")
    evidence["sentiment_score"] = family_data.get("sentiment_score")
    evidence["positive_aspects"] = family_data.get("positive_aspects", [])
    evidence["negative_aspects"] = family_data.get("negative_aspects", [])
    evidence["community_summary"] = family_data.get("community_summary")
    
    # 2. Extract same-family model list and their info
    family_models = family_data.get("models", {})
    for name, info in family_models.items():
        evidence["family_models"].append({
            "name": name,
            "parameters": info.get("parameters"),
            "language_model": info.get("language_model"),
            "vision_model": info.get("vision_model"),
            "release_date": info.get("release_date"),
            "open_source": info.get("open_source"),
            "technical_summary": info.get("technical_summary"),
        })
    
    # 3. Extract historical performance of same-family models on training set
    for model_info in evidence["family_models"]:
        name = model_info["name"]
        if name in train_df['Model'].values and name != model_name:
            row = train_df[train_df['Model'] == name].iloc[0]
            scores = {}
            for bench in benchmark_cols:
                if bench in row and pd.notna(row[bench]):
                    scores[bench] = float(row[bench])
            if scores:
                evidence["family_performance"][name] = {
                    "parameters": model_info["parameters"],
                    "scores": scores
                }
    
    return evidence


def _parse_param_size(param_str: str) -> float:
    """Parse parameter count string to numeric value (unit: B)"""
    if not param_str:
        return 0
    param_str = str(param_str).upper().replace(" ", "")
    try:
        if "B" in param_str:
            return float(param_str.replace("B", ""))
        elif "M" in param_str:
            return float(param_str.replace("M", "")) / 1000
        else:
            return float(param_str)
    except:
        return 0



def extract_all_evidence(
    model_name: str, 
    benchmark_name: str,
    train_df: pd.DataFrame, 
    benchmark_cols: list,
    llm_config: dict
    ) -> dict:
    """
    Extract all evidence needed for prediction
    """
    # 1. Family evidence
    family_evidence = extract_family_evidence(model_name, train_df, benchmark_cols)
    
    # 2. Similar architecture evidence (commented out: CPMF already predicts based on historical model patterns)
    # similar_evidence = extract_similar_architecture_evidence(model_name, train_df, benchmark_cols)
    similar_evidence = {"target_vision_model": None, "target_language_model": None, "similar_models": [], "similar_performance": {}}
    
    # 3. Target model's own information
    model_info = get_model_info(model_name)
    
    return {
        "model_name": model_name,
        "benchmark_name": benchmark_name,
        "model_info": model_info,
        "family_evidence": family_evidence,
        "similar_evidence": similar_evidence,
    }

def extract_rank_similar_models_evidence(
    client:OpenAI,
    evidence: dict,
    benchmark_name: str,
    cpmf_pred: float,
    train_df: pd.DataFrame,
    llm_config: dict,
    llm_model: str = "gpt-4o",
    rank_range: int = 4  # rank range (above and below)
    ) -> Optional[dict]:
    """
    Extract models with similar rankings on the target benchmark for multi-dimensional comparison
    
    Comparison dimensions: organization, LLM, Vision Encoder, training paradigm, parameters, release date
    """
    model_name = evidence["model_name"]
    model_info = evidence.get("model_info") or {}
    family_evidence = evidence.get("family_evidence", {})
    
    # Target model info
    target_org = family_evidence.get("org_name", "Unknown")
    target_vision = model_info.get("vision_model", "Unknown")
    target_llm = model_info.get("language_model", "Unknown")
    target_params = model_info.get("parameters", "Unknown")
    target_release_date = model_info.get("release_date", "Unknown")
    target_org_profile = get_organization_profile(target_org)
    
    # Get all scores for this benchmark in the training set
    if benchmark_name not in train_df.columns:
        #print(f"    [rank_similar] Skipping: benchmark {benchmark_name} not in training set")
        return None
    
    bench_data = train_df[['Model', benchmark_name]].dropna()
    if len(bench_data) < 3:
        #print(f"    [rank_similar] Skipping: benchmark {benchmark_name} has too few samples ({len(bench_data)})")
        return None
    
    #print(f"    [rank_similar] Found {len(bench_data)} models with scores on {benchmark_name}")
    
    # Sort by score
    bench_data = bench_data.sort_values(benchmark_name, ascending=False).reset_index(drop=True)
    bench_data['rank'] = bench_data.index + 1
    
    # Find the rank position corresponding to the CPMF prediction
    bench_data['diff_from_pred'] = abs(bench_data[benchmark_name] - cpmf_pred)
    pred_rank_idx = bench_data['diff_from_pred'].idxmin()
    pred_rank = bench_data.loc[pred_rank_idx, 'rank']
    
    # Get models with similar rankings (within rank_range above and below)
    rank_min = max(1, pred_rank - rank_range)
    rank_max = min(len(bench_data), pred_rank + rank_range)
    similar_rank_models = bench_data[(bench_data['rank'] >= rank_min) & (bench_data['rank'] <= rank_max)]
    
    # Get detailed info for these models
    comparison_models = []
    db = load_models_knowledge_db()
    
    for _, row in similar_rank_models.iterrows():
        ref_model_name = row['Model']
        ref_score = row[benchmark_name]
        ref_rank = row['rank']
        
        # Get detailed info from knowledge base
        ref_org, ref_family, ref_family_data = find_model_family(ref_model_name)
        ref_info = {}
        if ref_family_data:
            ref_info = ref_family_data.get('models', {}).get(ref_model_name, {})
        
        ref_vision = ref_info.get("vision_model", "Unknown")
        ref_llm = ref_info.get("language_model", "Unknown")
        ref_params = ref_info.get("parameters", "Unknown")
        ref_release_date = ref_info.get("release_date", "Unknown")
        ref_org_profile = get_organization_profile(ref_org or "Unknown")
        
        # Get component detailed info
        ref_vision_info = get_component_info(ref_vision, "vision") if ref_vision and ref_vision != "Unknown" else None
        ref_llm_info = get_component_info(ref_llm, "language") if ref_llm and ref_llm != "Unknown" else None
        
        comparison_models.append({
            "name": ref_model_name,
            "score": ref_score,
            "rank": ref_rank,
            "org": ref_org or "Unknown",
            "org_region": ref_org_profile.get("region", "Unknown"),
            "org_size": ref_org_profile.get("size", "unknown"),
            "org_chinese_focus": ref_org_profile.get("chinese_focus", False),
            "vision_model": ref_vision,
            "vision_org": ref_vision_info.get("organization", "Unknown") if ref_vision_info else "Unknown",
            "language_model": ref_llm,
            "llm_org": ref_llm_info.get("organization", "Unknown") if ref_llm_info else "Unknown",
            "parameters": ref_params,
            "release_date": ref_release_date,
            "technical_summary": ref_info.get("technical_summary", "")[:200] if ref_info.get("technical_summary") else ""
        })
    
    if not comparison_models:
        #print(f"    [rank_similar] Skipping: unable to get reference model details")
        return None
    
    #print(f"    [rank_similar] Found {len(comparison_models)} similarly-ranked reference models")
    
    # Normalize scores to 0-100 (to avoid LLM judgment bias from different score ranges like MME)
    cpmf_pred_norm = normalize_to_100(cpmf_pred, benchmark_name)
    
    # Format reference model info as readable text (using normalized scores)
    ref_models_text = []
    for m in comparison_models:
        score_norm = normalize_to_100(m['score'], benchmark_name)
        ref_models_text.append(
            f"- {m['name']} (Rank {m['rank']}, Score {score_norm:.1f}, Released: {m['release_date']})\n"
            f"  Organization: {m['org']} \n"
            f"  Parameters: {m['parameters']}, Vision: {m['vision_model']}, LLM: {m['language_model']}\n"
            f"  Technical summary: {m['technical_summary']}"
        )

    #print("ref_models_text:", ref_models_text)

    # ------------------------------------------------------------------
    # Step 2: Cross-Model Comparison Prompt
    # Operationalizes EVT's violation-detection sub-process:
    # the LLM contrasts the target with capability-similar models to
    # detect whether the CPMF prior is plausibly over- or under-estimated.
    # ------------------------------------------------------------------
    prompt = f"""You are checking whether the CPMF prior for a newly released model is consistent with observations from capability-similar models.

    Target: {model_name} on {benchmark_name}
    - Organization: {target_org} (Size: {target_org_profile.get('size', 'unknown')})
    - Parameters: {target_params}
    - Release Date: {target_release_date}
    - Vision Model: {target_vision}
    - Language Model: {target_llm}

    CPMF prior prediction: {cpmf_pred_norm:.1f} (expected rank ~{pred_rank} of {len(bench_data)}; all scores normalized to 0-100)

    Capability-similar models (top-k models with cosine similarity > tau in observed performance vectors) and their scores on {benchmark_name}:
    {chr(10).join(ref_models_text)}

    Task: Compare the CPMF prediction against this peer group. Consider:
    - What is the observed score range and median among peers?
    - Does the CPMF prior fall within, above, or below this peer range?
    - Could any peer-target capability gap (e.g., domain specialization, parameter count, training data scale) explain a deviation from peers?

    Return a JSON object with the following fields:
    ```json
    {{
        "peer_score_range": [<min>, <median>, <max>],
        "deviation_flag": "overestimate | underestimate | consistent",
        "deviation_magnitude": "estimated points difference from peer median",
        "capability_gap_analysis": "textual analysis of why the target may differ from peers",
        "reasoning": "brief justification"
    }}
    ```

    Output JSON only."""

    #print("prompt:", prompt)

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.3,
        )
        #print("Reference model comparison successful!")
        # Debug: print full response structure
        # print(f"    [DEBUG] choices length: {len(response.choices)}")
        # if response.choices:
        #     msg = response.choices[0].message
        #     print(f"    [DEBUG] message type: {type(msg)}, content type: {type(msg.content)}")
        #     print(f"    [DEBUG] message content: {msg}")
        result_text = response.choices[0].message.content
        #print("result_text:", result_text)
        if not result_text:
            return None
        
        if "```json" in result_text:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', result_text)
            json_str = match.group(1) if match else None
        else:
            match = re.search(r'\{[\s\S]*\}', result_text)
            json_str = match.group(0) if match else None
        
        result = json.loads(json_str) if json_str else None
        if result:
            # Add reference model info
            result['reference_models'] = comparison_models
            result['cpmf_pred_rank'] = pred_rank
            result['total_models'] = len(bench_data)
        return result
        
    except Exception as e:
        print(f"    Similarly-ranked model evidence extraction failed: {e}")
        return None


def extract_family_evolution_evidence(
    client:OpenAI,
    evidence: dict,
    benchmark_info: dict,
    llm_config: dict,
    llm_model: str = "gpt-4o",
    ) -> Optional[dict]:
    """
    Step 1: Extract architecture evolution evidence from same organization and series
    
    Analyze architectural changes within the same organization and model series, assess potential performance improvements on benchmarks
    
    Returns:
        {
            "architecture_changes": [{"from": "old_arch", "to": "new_arch", "component": "vision/llm"}],
            "performance_trend": "improving/stable/declining",
            "trend_evidence": ["evidence1", "evidence2"],
            "expected_improvement": "expected improvement description",
            "confidence": "high/medium/low"
        }
    """
    model_name = evidence["model_name"]
    benchmark_name = evidence["benchmark_name"]
    model_info = evidence.get("model_info") or {}
    family_evidence = evidence.get("family_evidence", {})
    
    # Get target model's components
    target_vision = model_info.get("vision_model")
    target_llm = model_info.get("language_model")
    
    if not target_vision and not target_llm:
        return None
    
    # Build same-series model architecture comparison
    family_models = family_evidence.get("family_models", [])
    family_perf = family_evidence.get("family_performance", {})
    
    if not family_models:
        #print(f"    [family_evolution] Skipping: family_models is empty")
        return None
    
    # If only the target model itself exists, evolution analysis is not possible
    #if len(family_models) == 1 and family_models[0].get("name") == model_name:
        #print(f"    [family_evolution] Skipping: only the target model in the series")
        # But still try to analyze organization background
    
    # Get component detailed info
    vision_info = get_component_info(target_vision, "vision") if target_vision else None
    llm_info = get_component_info(target_llm, "language") if target_llm else None
    
    # Get organization info
    org_name = family_evidence.get("org_name", "")
    org_profile = get_organization_profile(org_name)
    
    # Build architecture comparison text (scores normalized to 0-100)
    arch_comparison = []
    for m in family_models:
        m_name = m.get("name", "")
        m_vision = m.get("vision_model", "N/A")
        m_llm = m.get("language_model", "N/A")
        m_params = m.get("parameters", "N/A")
        m_date = m.get("release_date", "N/A")
        
        perf_str = ""
        if m_name in family_perf and benchmark_name in family_perf[m_name].get("scores", {}):
            raw_score = family_perf[m_name]['scores'][benchmark_name]
            norm_score = normalize_to_100(raw_score, benchmark_name)
            perf_str = f", {benchmark_name}={norm_score:.1f}"
        
        arch_comparison.append(f"  - {m_name}: vision={m_vision}, llm={m_llm}, params={m_params}, date={m_date}{perf_str}")
    
    # Build component knowledge
    component_knowledge = []
    if vision_info:
        component_knowledge.append(f"[Target Vision Model: {target_vision}]")
        component_knowledge.append(f"  Organization: {vision_info.get('organization', 'N/A')}")
        component_knowledge.append(f"  Architecture: {vision_info.get('architecture', 'N/A')}")
        component_knowledge.append(f"  Features: {', '.join(vision_info.get('key_features', [])[:3])}")
        component_knowledge.append(f"  Training Data: {vision_info.get('training_data', 'N/A')[:200]}")
    
    if llm_info:
        component_knowledge.append(f"[Target Language Model: {target_llm}]")
        component_knowledge.append(f"  Organization: {llm_info.get('organization', 'N/A')}")
        component_knowledge.append(f"  Parameters: {llm_info.get('parameters', 'N/A')}")
        component_knowledge.append(f"  Architecture: {llm_info.get('architecture', 'N/A')}")
        component_knowledge.append(f"  Features: {', '.join(llm_info.get('key_features', [])[:3])}")
    
    # Build benchmark info
    bench_desc = f"Benchmark: {benchmark_name}"
    if benchmark_info.get('category'):
        bench_desc += f" ({benchmark_info['category']})"
    if benchmark_info.get('key_capabilities'):
        bench_desc += f"\nKey capabilities: {', '.join(benchmark_info['key_capabilities'][:5])}"
    
    # Build organization info description
    org_desc = f"""## Organization Profile
    - Organization: {org_name}
    - Region: {org_profile.get('region', 'Unknown')} ({'Chinese company, may optimize for Chinese' if org_profile.get('chinese_focus') else 'Non-Chinese company, may not specialize in Chinese'})
    - Size: {org_profile.get('size', 'unknown')} ({'Large company, higher credibility' if org_profile.get('size') == 'large' else 'Small/medium company'})
    - Credibility: {org_profile.get('credibility', 0.5):.2f}
    - Specialties: {', '.join(org_profile.get('specialties', []))}"""
        
    # ------------------------------------------------------------------
    # Step 1: Intra-Family Analysis Prompt
    # Operationalizes EVT's expectation-grounding sub-process:
    # the LLM positions the target model within its family lineage,
    # examining architectural evolution and identifying reference
    # scores from observed family members.
    # ------------------------------------------------------------------
    prompt = f"""You are analyzing the expected performance of a newly released model on a benchmark, given its family lineage.

    Target: {model_name} on {benchmark_name}
    CPMF prior prediction: (provided downstream; this step only positions the target within its family)
    Statistical uncertainty: (provided downstream)

    Model technical summary (filtered to remove benchmark scores):
    {chr(10).join(component_knowledge)}

    {org_desc}

    Target benchmark:
    {bench_desc}

    Family members and their observed scores on {benchmark_name}:
    {chr(10).join(arch_comparison)}

    Task: Analyze the target model's expected position within its family. Consider:
    - How does the target's architecture / training paradigm differ from its predecessors?
    - Does the family show a consistent scaling or improvement trend on this benchmark category?
    - Are there architectural changes (e.g., new vision encoder, new training data, new alignment procedure) that would specifically affect performance on {benchmark_name}?

    Return a JSON object with the following fields:
    ```json
    {{
        "reference_scores": [
            {{"model": "...", "score": 0.0, "relevance": "high/medium/low"}}
        ],
        "architectural_deltas": "textual description of how target differs from predecessors",
        "expected_direction": "up | down | stable",
        "magnitude_hint": "small (<2 points) | moderate (2--5) | large (>5)",
        "reasoning": "brief justification grounded in retrieved evidence"
    }}
    ```

    Output JSON only."""

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3,
        )
        result_text = response.choices[0].message.content
        #result_text = response.choices[0].message.content.strip() if response.choices else None
        #print("family_result_text:", result_text)
        if not result_text:
            return None
        
        # Parse JSON
        if "```json" in result_text:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', result_text)
            json_str = match.group(1) if match else None
        else:
            match = re.search(r'\{[\s\S]*\}', result_text)
            json_str = match.group(0) if match else None
        
        return json.loads(json_str) if json_str else None
        
    except Exception as e:
        print(f"    Architecture evolution evidence extraction failed: {e}")
        return None


def generate_evidence_summary(
    client:OpenAI,
    evidence: dict, 
    llm_config: dict,
    cpmf_pred: float = None,
    train_df: pd.DataFrame = None,
    llm_model: str = "gpt-4o",
    ) -> Tuple[str, Optional[dict]]:
    """
    Generate evidence summary (based on models_knowledge_db.json)
    
    Contains:
    1. Target model information
    2. Historical performance of same-family models
    3. Multi-dimensional comparison with similarly-ranked models
    4. Technical summary and community feedback
    5. LLM-extracted benchmark-related evidence
    
    Returns:
        (evidence_summary_text, llm_extracted_evidence)
    """
    lines = []
    model_name = evidence["model_name"]
    benchmark_name = evidence["benchmark_name"]
    model_info = evidence.get("model_info") or {}
    family_evidence = evidence.get("family_evidence", {})
    similar_evidence = evidence.get("similar_evidence", {})
    
    lines.append("=" * 60)
    
    # 1. Target model basic info
    lines.append(f"[Target Model] {model_name}")
    lines.append(f"  Organization: {family_evidence.get('org_name', 'N/A')}")
    lines.append(f"  Series: {family_evidence.get('family_name', 'N/A')}")
    if model_info.get('parameters'):
        lines.append(f"  Parameters: {model_info.get('parameters')}")
    if model_info.get('language_model'):
        lines.append(f"  Language Model: {model_info.get('language_model')}")
    if model_info.get('vision_model'):
        lines.append(f"  Vision Model: {model_info.get('vision_model')}")
    if model_info.get('release_date'):
        lines.append(f"  Release Date: {model_info.get('release_date')}")
    
    # 2. Get Benchmark info (fetched early for LLM extraction)
    bench_info = get_or_generate_benchmark_features(benchmark_name, llm_config)
    
    # 3. Two-step LLM evidence extraction
    # Step 1: Same-organization same-series architecture evolution evidence
    evolution_evidence = extract_family_evolution_evidence(client,evidence, bench_info, llm_config,llm_model=llm_model)
    
    # Step 2: Multi-dimensional comparison with similarly-ranked models
    rank_similar_evidence = None
    if cpmf_pred is not None and train_df is not None:
        print(f"    [rank_similar] Starting extraction, cpmf_pred={cpmf_pred:.1f}, train_df has {len(train_df)} rows")
        rank_similar_evidence = extract_rank_similar_models_evidence(
            client,evidence, benchmark_name, cpmf_pred, train_df, llm_config,llm_model=llm_model
        )
        #print(rank_similar_evidence)
    else:
        print(f"    [rank_similar] Skipping: cpmf_pred={cpmf_pred}, train_df={'available' if train_df is not None else 'unavailable'}")
    
    # Merge into llm_evidence
    llm_evidence = {
        "family_evolution": evolution_evidence,
        "rank_similar": rank_similar_evidence
    }
    
    # 3.1 Display architecture evolution evidence
    if evolution_evidence:
        lines.append(f"\n[Step 1: Same-Series Architecture Evolution Analysis]")
        
        # Architecture changes
        if evolution_evidence.get('architecture_changes'):
            lines.append(f"  Architecture Changes:")
            for change in evolution_evidence['architecture_changes'][:3]:
                lines.append(f"    • {change.get('component_type', 'N/A')}: {change.get('from', 'N/A')} → {change.get('to', 'N/A')} ({change.get('improvement_type', '')})")
        
        # Performance trend
        trend = evolution_evidence.get('performance_trend', 'unknown')
        trend_label = {"improving": "📈 Improving", "stable": "➡️ Stable", "declining": "📉 Declining"}.get(trend, trend)
        lines.append(f"  Performance Trend: {trend_label}")
        
        # Trend evidence
        if evolution_evidence.get('trend_evidence'):
            for ev in evolution_evidence['trend_evidence'][:2]:
                lines.append(f"    ✓ {ev}")
        
        # Expected improvement
        if evolution_evidence.get('expected_improvement'):
            lines.append(f"  Expected: {evolution_evidence['expected_improvement']}")
        
        # Reasoning
        if evolution_evidence.get('reasoning'):
            lines.append(f"  Reasoning: {evolution_evidence['reasoning']}")
    
    # 3.2 Display similarly-ranked model comparison analysis
    if rank_similar_evidence:
        lines.append(f"\n[Step 2: Similarly-Ranked Model Multi-Dimensional Comparison]")
        
        # Predicted rank info
        pred_rank = rank_similar_evidence.get('cpmf_pred_rank', 'N/A')
        total_models = rank_similar_evidence.get('total_models', 'N/A')
        lines.append(f"  CPMF Predicted Rank: ~{pred_rank}/{total_models}")
        
        # Reference model list
        ref_models = rank_similar_evidence.get('reference_models', [])
        if ref_models:
            lines.append(f"  Similarly-Ranked Reference Models ({len(ref_models)}):")
            for m in ref_models[:5]:
                lines.append(f"    Rank {m['rank']}: {m['name']} = {m['score']:.1f} ({m['org']}, {m['parameters']})")
        
        # Comparison summary
        comparison = rank_similar_evidence.get('comparison_summary', {})
        if comparison:
            lines.append(f"  Comparison Analysis:")
            if comparison.get('org_comparison'):
                lines.append(f"    Organization: {comparison['org_comparison'][:80]}")
            if comparison.get('vision_comparison'):
                lines.append(f"    Vision: {comparison['vision_comparison'][:80]}")
            if comparison.get('llm_comparison'):
                lines.append(f"    LLM: {comparison['llm_comparison'][:80]}")
            if comparison.get('param_comparison'):
                lines.append(f"    Parameters: {comparison['param_comparison'][:80]}")
        
        # Training paradigm differences
        if rank_similar_evidence.get('inferred_training_diff'):
            lines.append(f"  Training Paradigm Inference: {rank_similar_evidence['inferred_training_diff'][:120]}")
        
        # Rank adjustment suggestion
        rank_adj = rank_similar_evidence.get('rank_adjustment', {})
        if rank_adj:
            dir_label = {"higher": "↑ Higher", "lower": "↓ Lower", "same": "→ Same"}.get(rank_adj.get('direction', ''), rank_adj.get('direction', ''))
            lines.append(f"  Rank Adjustment: {dir_label} ({rank_adj.get('magnitude', '')})")
            if rank_adj.get('reason'):
                lines.append(f"    Reason: {rank_adj['reason'][:100]}")
        
        # Score adjustment suggestion
        score_adj = rank_similar_evidence.get('score_adjustment', {})
        if score_adj:
            dir_label = {"up": "↑ Up", "down": "↓ Down", "neutral": "→ Neutral"}.get(score_adj.get('direction', ''), score_adj.get('direction', ''))
            lines.append(f"  Score Adjustment: {dir_label} {score_adj.get('magnitude', '')}")
            if score_adj.get('reason'):
                lines.append(f"    Reason: {score_adj['reason'][:100]}")
        
        # Reasoning
        if rank_similar_evidence.get('reasoning'):
            lines.append(f"  Comprehensive Analysis: {rank_similar_evidence['reasoning'][:200]}")
    
    # 4. Model technical summary (supplementary)
    model_tech = model_info.get('technical_summary')
    if model_tech and "Not provided" not in model_tech:
        lines.append(f"\n  [Model Technical Summary]")
        summary = model_tech
        lines.append(f"    {summary}")
    
    # 5. Series technical summary (supplementary)
    family_tech = family_evidence.get('technical_summary')
    if family_tech and "Not provided" not in family_tech:
        lines.append(f"\n  [Series Technical Summary]")
        summary = family_tech
        lines.append(f"    {summary}")
    
    # 6. Community feedback
    sentiment = family_evidence.get('sentiment_score')
    if sentiment:
        sentiment_label = "Positive" if sentiment >= 0.7 else ("Neutral" if sentiment >= 0.4 else "Negative")
        lines.append(f"\n  [Community Feedback]")
        lines.append(f"    Sentiment Score: {sentiment:.2f} ({sentiment_label})")
        if family_evidence.get('positive_aspects'):
            lines.append(f"    Strengths: {'; '.join(family_evidence.get('positive_aspects', [])[:3])}")
        if family_evidence.get('negative_aspects'):
            lines.append(f"    Weaknesses: {'; '.join(family_evidence.get('negative_aspects', [])[:3])}")
        if family_evidence.get('community_summary'):
            summary = family_evidence['community_summary'][:200] + "..." if len(family_evidence['community_summary']) > 200 else family_evidence['community_summary']
            lines.append(f"    Community Summary: {summary}")
    
    # 7. Same-family model historical performance
    family_perf = family_evidence.get('family_performance', {})
    if family_perf:
        lines.append(f"\n[Same-Family Model Historical Performance] (target benchmark: {benchmark_name})")
        for name, data in family_perf.items():
            if benchmark_name in data.get('scores', {}):
                score = data['scores'][benchmark_name]
                param = data.get('parameters', 'N/A')
                lines.append(f"  {name}: {score:.1f} (params: {param})")
    
    # 8. Similar architecture model historical performance (showing full architecture)
    similar_perf = similar_evidence.get('similar_performance', {})
    if similar_perf:
        lines.append(f"\n[Similar Architecture Model Historical Performance] (target benchmark: {benchmark_name})")
        target_vision = similar_evidence.get('target_vision_model')
        target_lang = similar_evidence.get('target_language_model')
        if target_vision or target_lang:
            lines.append(f"  Target model architecture: vision={target_vision}, llm={target_lang}")
        
        for name, data in list(similar_perf.items())[:5]:
            if benchmark_name in data.get('scores', {}):
                score = data['scores'][benchmark_name]
                param = data.get('parameters', 'N/A')
                vision = data.get('vision_model', 'N/A')
                lang = data.get('language_model', 'N/A')
                lines.append(f"  {name}: {score:.1f} (params: {param}, vision={vision}, llm={lang})")
    
    # 9. Benchmark info
    if bench_info.get('category'):
        lines.append(f"\n[Target Benchmark] {benchmark_name}")
        lines.append(f"  Category: {bench_info.get('category', 'N/A')}")
        if bench_info.get('task'):
            lines.append(f"  Task: {bench_info.get('task', '')[:100]}")
        if bench_info.get('difficulty'):
            lines.append(f"  Difficulty: {bench_info.get('difficulty', '')[:80]}")
    
    lines.append("=" * 60)
    return "\n".join(lines), llm_evidence


def parse_param_size(param_str: str) -> float:
    """Parse parameter count string to numeric value (unit: B)"""
    if not param_str:
        return 0
    param_str = str(param_str).upper().replace(" ", "")
    try:
        if "B" in param_str:
            return float(param_str.replace("B", ""))
        elif "M" in param_str:
            return float(param_str.replace("M", "")) / 1000
        else:
            return float(param_str)
    except:
        return 0

# ============================================================================
# Step 4: Semantic adjustment module
# ============================================================================


def semantic_adjustment(
    client: OpenAI,
    model_name: str,
    benchmark_name: str,
    score_matrix_text: str,
    llm_evidence: dict,
    cpmf_pred: float,
    cpmf_uncertainty: float = 0.0,
    llm_model: str = "gpt-4o"
    ) -> Optional[dict]:
    """
    Semantic adjustment module
    
    Adjust CPMF predictions based on evidence (reference model scores + LLM-extracted evidence)
    """
    #print("score_matrix_text: ", score_matrix_text)
    
    # ------------------------------------------------------------------
    # Step 3: Credibility-Aware Aggregation Prompt
    # Operationalizes EVT's credibility-weighted update: the LLM
    # synthesizes A_1 (intra-family analysis), A_2 (cross-model
    # comparison), and sigma_{mn} into the adjustment Delta_{mn},
    # credibility weight w_{mn}, and explanation E_{mn}.
    # The prompt explicitly encodes the source authority hierarchy
    # that drives EVT's credibility weighting.
    # Final prediction is computed downstream as:
    #     R_tilde_{mn} = R_hat_{mn} + w * Delta
    # ------------------------------------------------------------------
    # Extract evidence channels (A1, A2) from the upstream LLM evidence dict
    a1_intra_family = (llm_evidence or {}).get('family_evolution', llm_evidence)
    a2_cross_model = (llm_evidence or {}).get('rank_similar', None)

    prompt1 = f"""You are producing the final adjustment to a CPMF benchmark score prediction, integrating two evidence channels under EVT principles.

    Target: {model_name} on {benchmark_name}
    CPMF prior prediction (R_hat): {cpmf_pred:.2f}
    Statistical uncertainty (sigma_mn): {cpmf_uncertainty:.2f}
    Note: All scores are normalized to 0-100 range.

    Reference scores (same-series / family observed scores on {benchmark_name}):
    {score_matrix_text if score_matrix_text else 'No reference data'}

    Evidence channel 1 (intra-family analysis, A_1):
    {a1_intra_family}

    Evidence channel 2 (cross-model comparison, A_2):
    {a2_cross_model if a2_cross_model is not None else 'Not available'}

    Task: Produce two outputs that capture distinct judgments:
    - Delta (adjustment): what correction should be applied to the CPMF prior. May be positive, negative, or zero.
    - w (credibility weight, in [0, 1]): how reliable this correction is, controlling how much of Delta to apply.

    Source authority hierarchy (use this to weight evidence):
      1. Tier 1 (highest): Official technical reports, peer-reviewed papers
      2. Tier 2: HuggingFace model cards, official documentation
      3. Tier 3: Community posts, blog posts, third-party reviews

    Set w HIGHER when:
    - Multiple sources converge on consistent conclusions, especially across tiers.
    - Direct family reference scores are available (e.g., a sibling model with documented behavior on the same benchmark).
    - sigma_mn is high, indicating the CPMF prior is uncertain and leaves room for evidence-based correction.

    Set w LOWER when:
    - Evidence comes only from Tier 3 sources without corroboration.
    - Channels 1 and 2 disagree on the direction or magnitude of correction.
    - sigma_mn is low, indicating CPMF is already confident and external evidence should not override it.

    Return a JSON object with the following fields:
    ```json
    {{
        "delta": <signed numeric value, the correction Delta>,
        "credibility_weight": <float in [0, 1], the weight w>,
        "evidence_summary": [
            {{"source": "...", "tier": 1, "claim": "..."}}
        ],
        "explanation": "natural-language rationale traceable to specific evidence items",
        "adjustment": <same signed value as delta; kept for downstream compatibility>,
        "final_prediction": <{cpmf_pred:.2f} + credibility_weight * delta>,
        "confidence": "high/medium/low",
        "analysis": "brief overall analysis (mirrors explanation; kept for downstream compatibility)"
    }}
    ```

    The final prediction will be computed as R_tilde_mn = R_hat_mn + w * Delta.
    Output JSON only."""

    # prompt2 retained for backward compatibility (not currently invoked).
    # It mirrors prompt1 under the same EVT credibility-weighted update formulation.
    prompt2 = prompt1

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt1}],
            max_tokens=1000,
            temperature=0.3,
        )
        
        result_text = response.choices[0].message.content
        if not result_text:
            print(f"  Semantic adjustment: LLM returned empty")
            return None
        
        result_text = result_text.strip()
        
        # Parse JSON
        json_str = None
        if "```json" in result_text:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', result_text)
            json_str = match.group(1) if match else None
        elif "```" in result_text:
            match = re.search(r'```\s*([\s\S]*?)\s*```', result_text)
            json_str = match.group(1) if match else None
        
        if not json_str:
            match = re.search(r'\{[\s\S]*\}', result_text)
            json_str = match.group(0) if match else None
        
        if not json_str:
            print(f"  Semantic adjustment: unable to parse JSON")
            return None
        
        data = json.loads(json_str)

        # Extract Delta and credibility weight w under EVT principles.
        # Final prediction: R_tilde_mn = R_hat_mn + w * Delta
        # Fallbacks preserve backward compatibility with the older field name.
        delta = data.get("delta", data.get("adjustment", 0))
        try:
            delta = float(delta)
        except (TypeError, ValueError):
            delta = 0.0

        w = data.get("credibility_weight", 1.0)
        try:
            w = float(w)
        except (TypeError, ValueError):
            w = 1.0
        # Clamp w to [0, 1] as specified in the prompt.
        w = max(0.0, min(1.0, w))

        adjustment = w * delta
        final_pred = cpmf_pred + adjustment

        return {
            "predicted_score": final_pred,
            "cpmf_prediction": cpmf_pred,
            "adjustment": adjustment,
            "delta": delta,
            "credibility_weight": w,
            "confidence": str(data.get("confidence", "medium")),
            "analysis": str(data.get("explanation", data.get("analysis", ""))),
            "evidence_summary": data.get("evidence_summary", []),
        }
        
    except Exception as e:
        print(f"  Semantic adjustment failed: {e}")
        return None


# ============================================================================
# Step 5: Evaluation metrics (standard five metrics)
# ============================================================================

# Benchmark score range mapping
BENCHMARK_RANGES = {
    "MME": (0, 2800),
    "LLaVABench": (0, 120),
}


def normalize_to_100(score, benchmark_name):
    """Normalize score to 0-100"""
    min_s, max_s = BENCHMARK_RANGES.get(benchmark_name, (0, 100))
    return np.clip((score - min_s) / (max_s - min_s) * 100, 0, 100)


def denormalize_from_100(score_norm, benchmark_name):
    """Denormalize score back to original range"""
    min_s, max_s = BENCHMARK_RANGES.get(benchmark_name, (0, 100))
    return score_norm / 100 * (max_s - min_s) + min_s


def needs_normalization(benchmark_name):
    """Check if benchmark needs normalization (score range is not 0-100)"""
    return benchmark_name in BENCHMARK_RANGES


def calculate_metrics(results_df: pd.DataFrame) -> dict:
    """
    Calculate five standard metrics: MAE, RMSE (0-100), SRCC, KRCC, MAE@3
    
    Reference: data/calculate_metrics.py
    """
    if len(results_df) == 0:
        print("No valid samples!")
        return None
    
    # Normalize to 0-100
    results_df['pred_norm'] = results_df.apply(
        lambda r: normalize_to_100(r['predicted_score'], r['benchmark_name']), axis=1
    )
    results_df['true_norm'] = results_df.apply(
        lambda r: normalize_to_100(r['true_score'], r['benchmark_name']), axis=1
    )
    
    # 1. MAE (0-100)
    mae = np.mean(np.abs(results_df['pred_norm'] - results_df['true_norm']))
    
    # 2. RMSE (0-100)
    rmse = np.sqrt(np.mean((results_df['pred_norm'] - results_df['true_norm']) ** 2))
    
    # 3. SRCC (Spearman Rank Correlation Coefficient)
    srcc, _ = stats.spearmanr(results_df['predicted_score'], results_df['true_score'])
    
    # 4. KRCC (Kendall Rank Correlation Coefficient)
    krcc, _ = stats.kendalltau(results_df['predicted_score'], results_df['true_score'])
    
    # 5. MAE@3 (proportion of rank errors ≤ 3)
    all_rank_errors = []
    for bench in results_df['benchmark_name'].unique():
        bench_data = results_df[results_df['benchmark_name'] == bench]
        if len(bench_data) < 2:
            continue
        pred_ranks = (-bench_data['predicted_score'].values).argsort().argsort() + 1
        true_ranks = (-bench_data['true_score'].values).argsort().argsort() + 1
        all_rank_errors.extend(np.abs(pred_ranks - true_ranks).tolist())
    
    mae3 = (np.array(all_rank_errors) <= 3).mean() if all_rank_errors else 0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'srcc': srcc,
        'krcc': krcc,
        'mae3': mae3,
        'n_samples': len(results_df),
        'n_benchmarks': results_df['benchmark_name'].nunique(),
        'n_models': results_df['model_name'].nunique()
    }


# ============================================================================
# Result saving and loading (with checkpoint resume support)
# ============================================================================

def load_existing_results(results_detail_file: Path) -> Tuple[dict, list]:
    """
    Load previously saved detailed results
    
    Returns:
        (set of predicted samples {(model, benchmark)}, existing results list)
    """
    existing_keys = set()
    existing_results = []
    
    if results_detail_file.exists():
        try:
            with open(results_detail_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_results = data.get('results', [])
                for r in existing_results:
                    key = (r.get('model_name'), r.get('benchmark_name'))
                    existing_keys.add(key)
                print(f"[Checkpoint] Loaded {len(existing_keys)} previously predicted samples")
        except Exception as e:
            print(f"[Checkpoint] Loading failed: {e}, starting from scratch")
    
    return existing_keys, existing_results


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results_with_evidence(results: list, results_detail_file: Path, metrics: dict = None):
    """
    Save detailed results (with evidence) to JSON file
    
    Args:
        results: prediction results list
        results_detail_file: save path
        metrics: evaluation metrics
    """
    data = {
        "metadata": {
            "version": "2.0",
            "saved_at": pd.Timestamp.now().isoformat(),
            "total_samples": len(results),
        },
        "metrics": metrics,
        "results": results
    }
    
    with open(results_detail_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    print(f"[Save] Detailed results saved to: {results_detail_file}")


# ============================================================================
# Main pipeline
# ============================================================================

def run_evt_agent(llm_config: dict, max_samples: Optional[int] = None, batch_id: int = 0, total_batches: int = 1):
    """
    Run STAR Agent full pipeline
    
    Args:
        llm_config: LLM configuration
        max_samples: maximum test samples (None means all)
    """
    print("=" * 70)
    print("STAR Agent Pipeline")
    print("=" * 70)
    
    # ========================================================================
    # Step 1: Check and load feature databases
    # ========================================================================
    print("\n[Step 1] Check and load feature databases")
    print("-" * 70)
    load_models_features_db()
    load_benchmark_features_db()
    
    # ========================================================================
    # Step 2: Check and generate CPMF prediction results
    # ========================================================================
    print("\n[Step 2] Check and generate CPMF prediction results")
    print("-" * 70)
    if not check_or_generate_cpmf_predictions():
        print("Error: CPMF prediction generation failed")
        return None, None
    
    pmf_preds = load_pmf_predictions()
    if not pmf_preds:
        print("Error: Unable to load CPMF prediction results")
        return None, None
    
    # ========================================================================
    # Step 3: Load model knowledge database (evidence extraction)
    # ========================================================================
    print("\n[Step 3] Load model knowledge database (evidence extraction)")
    print("-" * 70)
    knowledge_db = load_models_knowledge_db()
    if not knowledge_db.get('organizations'):
        print("Warning: Model knowledge database is empty, evidence extraction will be limited")
    
    # ========================================================================
    # Step 4: Load training data
    # ========================================================================
    print("\n[Step 4] Load training data")
    print("-" * 70)
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    # Extract benchmark columns from training data
    meta_cols = ['Model', 'Parameters', 'Organization', 'OpenSource', 'Time']
    benchmark_cols = [c for c in train_df.columns if c not in meta_cols]
    print(f"Training data: {len(train_df)} models, {len(benchmark_cols)} benchmarks")
    
    # ========================================================================
    # Step 5: Iterate over test samples and predict (with checkpoint resume)
    # ========================================================================
    print("\n[Step 5] Semantic adjustment module - prediction refinement")
    print("-" * 70)
    
    # Initialize LLM client
    client = OpenAI(
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url")
    )
    llm_model = llm_config.get("model", "gpt-4o")
    
    # Generate file suffix based on model name (take company name before / or full model name)
    model_suffix = llm_model.split("/")[0] if "/" in llm_model else llm_model.replace("-", "_")
    
    # When running in batches, filename includes batch info
    if total_batches > 1:
        batch_suffix = f"_batch{batch_id}of{total_batches}"
        results_file = DATA_DIR / RESULTS_FILE_TEMPLATE.replace(".csv", f"{batch_suffix}.csv")
        results_detail_file = DATA_DIR / RESULTS_DETAIL_FILE_TEMPLATE.replace(".json", f"{batch_suffix}.json")
    else:
        results_file = DATA_DIR / RESULTS_FILE_TEMPLATE
        results_detail_file = DATA_DIR / RESULTS_DETAIL_FILE_TEMPLATE
    print(f"[File Paths] Results will be saved to: {results_file.name}, {results_detail_file.name}")
    
    # Load previously saved results (checkpoint resume)
    existing_keys, existing_results = load_existing_results(results_detail_file)
    
    # Build sample list from CPMF prediction results
    all_samples = list(pmf_preds.items())
    if max_samples:
        all_samples = all_samples[:max_samples]
    
    # Batch processing: split samples by batch_id and total_batches
    total_samples = len(all_samples)
    batch_size = (total_samples + total_batches - 1) // total_batches  # ceiling division
    start_idx = batch_id * batch_size
    end_idx = min(start_idx + batch_size, total_samples)
    samples = all_samples[start_idx:end_idx]
    
    print(f"[Batch Info] Total samples: {total_samples}, Current batch: {batch_id}/{total_batches}, Range: [{start_idx}, {end_idx}), Batch size: {len(samples)}")
    
    # Restore from existing results
    results = existing_results.copy()
    cpmf_only_results = []
    direction_stats = {"up": 0, "down": 0, "keep": 0}
    skipped = 0
    resumed = 0
    
    for idx, ((model_name, benchmark_name), cpmf_data) in enumerate(samples):
        true_score = cpmf_data['true_score']
        cpmf_pred = cpmf_data['prediction']
        cpmf_uncertainty = cpmf_data['uncertainty']
        
        # Check if already predicted (checkpoint resume)
        sample_key = (model_name, benchmark_name)
        if sample_key in existing_keys:
            print(f"[{idx+1}/{len(samples)}] {model_name} on {benchmark_name} - ✓ Already predicted, skipping")
            resumed += 1
            # Record pure CPMF result (for comparison)
            cpmf_only_results.append({
                'model_name': model_name,
                'benchmark_name': benchmark_name,
                'true_score': true_score,
                'predicted_score': cpmf_pred,
            })
            continue
        
        print(f"[{idx+1}/{len(samples)}] {model_name} on {benchmark_name}")
        print(f"  CPMF: {cpmf_pred:.1f} ± {cpmf_uncertainty:.1f}, True: {true_score:.1f}")
        
        # Record pure CPMF result
        cpmf_only_results.append({
            'model_name': model_name,
            'benchmark_name': benchmark_name,
            'true_score': true_score,
            'predicted_score': cpmf_pred,
        })
        
        # Extract evidence
        evidence = extract_all_evidence(model_name, benchmark_name, train_df, benchmark_cols, llm_config)
        
        # Generate evidence summary (with LLM-extracted relevant evidence)
        evidence_summary, llm_evidence = generate_evidence_summary(
            client,evidence, llm_config, cpmf_pred=cpmf_pred, train_df=train_df,llm_model=llm_model
        )
        #print("llm_evidence:", llm_evidence)
        
        # Build reference matrix text (same-series models are most important) - normalize scores to 0-100
        score_matrix_lines = []
        
        # Target model info
        model_info = evidence.get("model_info") or {}
        target_params = model_info.get('parameters', 'N/A')
        family_name = evidence.get("family_evidence", {}).get("family_name", "N/A")
        score_matrix_lines.append(f"Target model: {model_name} ({target_params}), Series: {family_name}")
        
        # Normalize CPMF prediction
        cpmf_pred_norm = normalize_to_100(cpmf_pred, benchmark_name)
        cpmf_uncertainty_norm = cpmf_uncertainty / (BENCHMARK_RANGES.get(benchmark_name, (0, 100))[1] - BENCHMARK_RANGES.get(benchmark_name, (0, 100))[0]) * 100
        
        # Same-series model scores (most important evidence) - normalized to 0-100
        family_perf = evidence.get("family_evidence", {}).get("family_performance", {})
        if family_perf:
            score_matrix_lines.append(f"\n★★★ Same-Series Model Historical Scores ({len(family_perf)} models, normalized to 0-100):")
            for name, data in family_perf.items():
                scores = data.get('scores', {})
                if benchmark_name in scores:
                    raw_score = scores[benchmark_name]
                    norm_score = normalize_to_100(raw_score, benchmark_name)
                    score_matrix_lines.append(f"  {name} ({data.get('parameters', 'N/A')}): {norm_score:.1f}")
        else:
            score_matrix_lines.append("\n★★★ Same-Series Model Historical Scores: No data")
        
        # Commented out: similar architecture model scores (auxiliary reference) - CPMF already predicts based on historical model patterns
        # similar_perf = evidence.get("similar_evidence", {}).get("similar_performance", {})
        # if similar_perf:
        #     ...(reference model display logic)
        
        score_matrix_text = "\n".join(score_matrix_lines) if score_matrix_lines else None
        
        # Call semantic adjustment module (using normalized scores)
        prediction = semantic_adjustment(
            client, model_name, benchmark_name, 
            score_matrix_text, llm_evidence, 
            cpmf_pred_norm, cpmf_uncertainty=cpmf_uncertainty_norm,
            llm_model=llm_model
        )
        
        # If prediction succeeded, denormalize the predicted value back to original range
        if prediction and needs_normalization(benchmark_name):
            prediction['predicted_score'] = denormalize_from_100(prediction['predicted_score'], benchmark_name)
            prediction['adjustment'] = prediction['predicted_score'] - cpmf_pred  # recalculate adjustment in original range
        
        if prediction:
            # Track adjustment direction
            adj = prediction['adjustment']
            if adj > 0.5:
                direction = 'up'
            elif adj < -0.5:
                direction = 'down'
            else:
                direction = 'keep'
            direction_stats[direction] = direction_stats.get(direction, 0) + 1
            
            results.append({
                'model_name': model_name,
                'benchmark_name': benchmark_name,
                'true_score': true_score,
                'cpmf_prediction': cpmf_pred,
                'predicted_score': prediction['predicted_score'],
                'adjustment': adj,
                'direction': direction,
                'confidence': prediction['confidence'],
                'analysis': prediction['analysis'],
                # Save evidence
                'evidence': {
                    'family_evolution': llm_evidence.get('family_evolution'),
                    'rank_similar': llm_evidence.get('rank_similar'),
                }
            })
            
            print(f"  Adjusted prediction: {prediction['predicted_score']:.1f} (delta={adj:+.1f})")
            print(f"    Analysis: {prediction['analysis'][:60]}...")
            
            # Incremental save (save after each prediction to prevent loss on interruption)
            save_results_with_evidence(results, results_detail_file)
        else:
            # On failure, use original CPMF prediction
            results.append({
                'model_name': model_name,
                'benchmark_name': benchmark_name,
                'true_score': true_score,
                'cpmf_prediction': cpmf_pred,
                'predicted_score': cpmf_pred,
                'adjustment': 0,
                'direction': 'fallback',
                'confidence': 'fallback',
                'analysis': 'Semantic adjustment failed',
                # Save evidence (even if prediction fails)
                'evidence': {
                    'family_evolution': llm_evidence.get('family_evolution') if llm_evidence else None,
                    'rank_similar': llm_evidence.get('rank_similar') if llm_evidence else None,
                    'score_matrix_text': score_matrix_text,
                    'evidence_summary': evidence_summary,
                }
            })
            print(f"  Semantic adjustment failed, using CPMF: {cpmf_pred:.1f}")
            
            # Incremental save
            save_results_with_evidence(results, results_detail_file)
    
    # ========================================================================
    # Step 6: Calculate evaluation metrics
    # ========================================================================
    print("\n[Step 6] Calculate evaluation metrics")
    print("-" * 70)
    
    if not results:
        print("Error: No valid prediction results")
        return None, None
    
    results_df = pd.DataFrame(results)
    cpmf_df = pd.DataFrame(cpmf_only_results)
    
    # Calculate STAR Agent metrics
    evt_metrics = calculate_metrics(results_df)
    # Calculate pure CPMF metrics
    cpmf_metrics = calculate_metrics(cpmf_df)
    
    # ========================================================================
    # Output results
    # ========================================================================
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Total samples: {len(samples)}")
    print(f"Valid predictions: {len(results)}")
    print(f"Resumed: {resumed}")
    print(f"Skipped: {skipped}")
    
    # Adjustment direction statistics
    print("\n[Adjustment Direction Statistics]")
    total_adj = sum(direction_stats.values())
    for direction, count in direction_stats.items():
        pct = 100 * count / total_adj if total_adj > 0 else 0
        print(f"  {direction}: {count} ({pct:.1f}%)")
    
    print("\n[CPMF Baseline]")
    print(f"  Samples: {cpmf_metrics['n_samples']} (Models: {cpmf_metrics['n_models']}, Benchmarks: {cpmf_metrics['n_benchmarks']})")
    print(f"\n  Score-Loss Metrics (0-100):")
    print(f"    MAE (↓):   {cpmf_metrics['mae']:.4f}")
    print(f"    RMSE (↓):  {cpmf_metrics['rmse']:.4f}")
    print(f"\n  Rank Metrics:")
    print(f"    SRCC (↑):  {cpmf_metrics['srcc']:.4f}")
    print(f"    KRCC (↑):  {cpmf_metrics['krcc']:.4f}")
    print(f"    MAE@3 (↑): {cpmf_metrics['mae3']:.4f}")
    
    print("\n[STAR Agent (CPMF + Semantic Adjustment)]")
    print(f"  Samples: {evt_metrics['n_samples']} (Models: {evt_metrics['n_models']}, Benchmarks: {evt_metrics['n_benchmarks']})")
    print(f"\n  Score-Loss Metrics (0-100):")
    print(f"    MAE (↓):   {evt_metrics['mae']:.4f}")
    print(f"    RMSE (↓):  {evt_metrics['rmse']:.4f}")
    print(f"\n  Rank Metrics:")
    print(f"    SRCC (↑):  {evt_metrics['srcc']:.4f}")
    print(f"    KRCC (↑):  {evt_metrics['krcc']:.4f}")
    print(f"    MAE@3 (↑): {evt_metrics['mae3']:.4f}")
    
    # Calculate improvement
    print("\n[Improvement over CPMF]")
    mae_imp = (cpmf_metrics['mae'] - evt_metrics['mae']) / cpmf_metrics['mae'] * 100
    rmse_imp = (cpmf_metrics['rmse'] - evt_metrics['rmse']) / cpmf_metrics['rmse'] * 100
    srcc_imp = (evt_metrics['srcc'] - cpmf_metrics['srcc']) / abs(cpmf_metrics['srcc']) * 100 if cpmf_metrics['srcc'] != 0 else 0
    krcc_imp = (evt_metrics['krcc'] - cpmf_metrics['krcc']) / abs(cpmf_metrics['krcc']) * 100 if cpmf_metrics['krcc'] != 0 else 0
    mae3_imp = (evt_metrics['mae3'] - cpmf_metrics['mae3']) / cpmf_metrics['mae3'] * 100 if cpmf_metrics['mae3'] != 0 else 0
    
    print(f"  MAE (↓):   {mae_imp:+.2f}%")
    print(f"  RMSE (↓):  {rmse_imp:+.2f}%")
    print(f"  SRCC (↑):  {srcc_imp:+.2f}%")
    print(f"  KRCC (↑):  {krcc_imp:+.2f}%")
    print(f"  MAE@3 (↑): {mae3_imp:+.2f}%")
    
    # Save results
    results_df.to_csv(results_file, index=False, encoding='utf-8')
    print(f"\nResults saved to: {results_file}")
    
    # Save detailed results (with evidence and metrics)
    save_results_with_evidence(results, results_detail_file, metrics={
        'evt_agent': evt_metrics,
        'cpmf_baseline': cpmf_metrics,
    })
    
    return results_df, evt_metrics


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="STAR Agent test pipeline (supports batch parallelism)")
    parser.add_argument("--batch_id", type=int, default=0, help="Current batch ID (starting from 0)")
    parser.add_argument("--total_batches", type=int, default=1, help="Total number of batches")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples (for testing)")
    args = parser.parse_args()
    
    # LLM configuration (read from environment variables)
    llm_config = {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "model": os.environ.get("LLM_MODEL", "gpt-4o"),
        "base_url": os.environ.get("OPENAI_BASE_URL"),
    }
    
    if not llm_config["api_key"]:
        print("Error: Please set the OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Run STAR Agent
    results_df, metrics = run_evt_agent(
        llm_config=llm_config,
        max_samples=args.max_samples,
        batch_id=args.batch_id,
        total_batches=args.total_batches,
    )
    
    # Print batch completion hint
    if args.total_batches > 1:
        print(f"\n[Info] Batch {args.batch_id} completed. After all batches finish, please manually merge the result files.")
