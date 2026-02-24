#!/usr/bin/env python3
"""
STAR Database Generation Tool

Generates knowledge databases required by the STAR framework from OpenCompass VLM Leaderboard data:
  - models_features_db.json          Model features database (used by CPMF statistical stage)
  - benchmark_features_db.json       Benchmark features database (used by CPMF + semantic adjustment stage)
  - models_knowledge_db.json         Model knowledge database (evidence extraction, includes LLM summaries and sentiment analysis)
  - vision_model_knowledge_db.json   Vision model component knowledge database
  - language_model_knowledge_db.json Language model component knowledge database

Usage:
  python data/gen_databases.py build-features    # Steps 1-4: Download data + match links + extract families -> models_features_db.json (no LLM required)
  python data/gen_databases.py build-benchmarks  # Step B: Use BenchmarkAgent -> benchmark_features_db.json (requires LLM)
  python data/gen_databases.py build-knowledge   # Step 5: Use ModelAgent -> models_knowledge_db.json (requires LLM + network)
  python data/gen_databases.py build-components  # Step 6: Generate component knowledge databases (requires LLM)
  python data/gen_databases.py all               # Run all steps

Dependencies:
  - Steps 1-4 do not require LLM, only network access to HuggingFace API
  - Steps 5-6 require LLM API (configured via environment variables)
  - It is recommended to run datadown.py first to download base data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import os
import csv
import ssl
import time
import argparse
import urllib.request
import urllib.parse
from datetime import datetime
from collections import defaultdict
from typing import Dict, Optional, List, Set, Tuple

# ============================================================================
# Path Configuration
# ============================================================================
DATA_DIR = Path(__file__).parent / "opencompass_cache"

# Intermediate files
MODEL_LINKS_DB = DATA_DIR / "model_links_db.json"
MODEL_LINKS_V2_DB = DATA_DIR / "model_links_db_v2.json"

# Final outputs
MODELS_FEATURES_FILE = DATA_DIR / "models_features_db.json"
BENCHMARK_FEATURES_FILE = DATA_DIR / "benchmark_features_db.json"
MODELS_KNOWLEDGE_FILE = DATA_DIR / "models_knowledge_db.json"
VISION_MODEL_KB_FILE = DATA_DIR / "vision_model_knowledge_db.json"
LANGUAGE_MODEL_KB_FILE = DATA_DIR / "language_model_knowledge_db.json"
CSV_PATH = DATA_DIR / "opencompass_vlm_full.csv"
RAW_JSON_PATH = DATA_DIR / "opencompass_vlm_raw.json"

# SSL configuration (for HuggingFace API requests)
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# ============================================================================
# Model Family Matching Rules: (family_name, regex_pattern, organization)
# ============================================================================
MODEL_FAMILY_PATTERNS = [
    # Alibaba Qwen series
    ("Qwen-VL", r"^Qwen-VL(?!2).*", "Alibaba"),
    ("Qwen2-VL", r"^Qwen2-VL-\d+B.*", "Alibaba"),
    ("Qwen2.5-VL", r"^Qwen2\.5-VL-\d+B.*", "Alibaba"),
    ("Ovis1.5", r"^Ovis1\.5-.*", "Alibaba"),
    ("Ovis1.6", r"^Ovis1\.6-.*", "Alibaba"),
    ("Ovis2", r"^Ovis2-\d+B.*", "Alibaba"),
    # OpenGVLab InternVL series
    ("InternVL-Chat-V1.5", r"^InternVL-Chat-V1\.5.*|^Mini-InternVL-Chat.*V1\.5.*", "Shanghai AI Laboratory"),
    ("InternVL2", r"^InternVL2-\d+B(?!-MPO).*", "Shanghai AI Laboratory"),
    ("InternVL2-MPO", r"^InternVL2-\d+B-MPO.*", "Shanghai AI Laboratory"),
    ("InternVL2.5", r"^InternVL2\.5-\d+B(?!-MPO).*", "Shanghai AI Laboratory"),
    ("InternVL2.5-MPO", r"^InternVL2\.5-\d+B-MPO.*", "Shanghai AI Laboratory"),
    ("InternVL3", r"^InternVL3-\d+B.*", "Shanghai AI Laboratory"),
    # InternLM-XComposer series
    ("InternLM-XComposer", r"^InternLM-XComposer(?!2).*", "Shanghai AI Laboratory"),
    ("InternLM-XComposer2", r"^InternLM-XComposer2(?!\.5).*", "Shanghai AI Laboratory"),
    ("InternLM-XComposer2.5", r"^InternLM-XComposer2\.5.*", "Shanghai AI Laboratory"),
    # LLaVA series
    ("LLaVA-v1", r"^LLaVA-v1-.*", "University of Wisconsin"),
    ("LLaVA-v1.5", r"^LLaVA-v1\.5-.*", "University of Wisconsin"),
    ("LLaVA-Next", r"^LLaVA-Next-(?!Interleave|OneVision).*", "University of Wisconsin"),
    ("LLaVA-Next-Interleave", r"^LLaVA-Next-Interleave.*", "ByteDance"),
    ("LLaVA-OneVision", r"^LLaVA-OneVision-.*", "ByteDance"),
    ("LLaVA-InternLM", r"^LLaVA-InternLM.*", "Shanghai AI Laboratory"),
    ("LLaVA-LLaMA-3", r"^LLaVA-LLaMA-3.*|^LLaVA-Next-Llama3.*", "Shanghai AI Laboratory"),
    # OpenBMB MiniCPM series
    ("MiniCPM-V", r"^MiniCPM-V(?!-2).*", "OpenBMB"),
    ("MiniCPM-V-2", r"^MiniCPM-V-2(?!\.6).*", "OpenBMB"),
    ("MiniCPM-V-2.6", r"^MiniCPM-V-2\.6.*|^MiniCPM-o-2\.6.*", "OpenBMB"),
    ("MiniCPM-Llama3-V2.5", r"^MiniCPM-Llama3-V2\.5.*", "OpenBMB"),
    # OpenAI GPT series
    ("GPT-4v", r"^GPT-4v.*", "OpenAI"),
    ("GPT-4o", r"^GPT-4o(?!-mini).*", "OpenAI"),
    ("GPT-4o-mini", r"^GPT-4o-mini.*", "OpenAI"),
    ("GPT-4.1", r"^GPT-4\.1(?!-mini|-nano).*", "OpenAI"),
    ("GPT-4.1-mini", r"^GPT-4\.1-mini.*", "OpenAI"),
    ("GPT-4.1-nano", r"^GPT-4\.1-nano.*", "OpenAI"),
    ("GPT-4.5", r"^GPT-4\.5.*", "OpenAI"),
    ("GPT-5", r"^GPT-5(?!-mini|-nano).*", "OpenAI"),
    ("GPT-5-mini", r"^GPT-5-mini.*", "OpenAI"),
    ("GPT-5-nano", r"^GPT-5-nano.*", "OpenAI"),
    ("ChatGPT-4o", r"^ChatGPT-4o.*", "OpenAI"),
    # Anthropic Claude series
    ("Claude3", r"^Claude3(?!\.5|\.7)-.*", "Anthropic"),
    ("Claude3.5-Sonnet", r"^Claude3\.5-Sonnet.*", "Anthropic"),
    ("Claude3.7-Sonnet", r"^Claude3\.7-Sonnet.*", "Anthropic"),
    # Google Gemini series
    ("Gemini-1.0", r"^Gemini-1\.0-.*", "Google"),
    ("Gemini-1.5", r"^Gemini-1\.5-.*", "Google"),
    ("Gemini-2.0", r"^Gemini-2\.0-.*", "Google"),
    ("Gemini-2.5", r"^Gemini-2\.5-.*", "Google"),
    ("Gemma3", r"^Gemma3-\d+B.*", "Google"),
    ("PaliGemma", r"^PaliGemma-.*", "Google"),
    # DeepSeek series
    ("DeepSeek-VL", r"^DeepSeek-VL-(?!2).*", "DeepSeek"),
    ("DeepSeek-VL2", r"^DeepSeek-VL2.*", "DeepSeek"),
    ("Janus", r"^Janus-.*", "DeepSeek"),
    # Meta series
    ("Llama-3.2-Vision", r"^Llama-3\.2-\d+B-Vision.*", "Meta"),
    ("Chameleon", r"^Chameleon-\d+B.*", "Meta"),
    # AllenAI Molmo series
    ("Molmo", r"^Molmo-\d+B.*|^MolmoE-.*", "AllenAI"),
    # NVIDIA series
    ("Eagle-X5", r"^Eagle-X5-\d+B.*", "Nvidia"),
    ("VILA1.5", r"^VILA1\.5-\d+B.*|^Llama-3-VILA1\.5.*", "NVIDIA"),
    ("NVLM", r"^NVLM-.*", "Nvidia"),
    # ByteDance series
    ("SAIL-VL", r"^SAIL-VL-\d+B.*", "ByteDance"),
    ("SAIL-VL-1.5", r"^SAIL-VL-1\.5-\d+B.*|^SAIL-VL-1d5.*", "ByteDance"),
    ("SAIL-VL-1.6", r"^SAIL-VL-1\.6-\d+B.*|^SAIL-VL-1d6.*", "ByteDance"),
    ("Valley", r"^Valley-Eagle.*|^valley2.*", "ByteDance"),
    # Zhipu AI series
    ("GLM-4v", r"^GLM-4v(?!-Plus).*", "Zhipu AI"),
    ("GLM-4v-Plus", r"^GLM-4v-Plus.*", "Zhipu AI"),
    ("CogVLM", r"^CogVLM-\d+B.*", "Zhipu AI"),
    ("CogVLM2", r"^CogVLM2-.*", "Zhipu AI"),
    # 01-AI Yi series
    ("Yi-VL", r"^Yi-VL-\d+B.*", "01-AI"),
    # HuggingFace series
    ("IDEFICS", r"^IDEFICS-\d+B.*", "Hugging Face"),
    ("IDEFICS2", r"^IDEFICS2-.*", "Hugging Face"),
    ("Idefics3", r"^Idefics3-.*", "HuggingFace"),
    ("SmolVLM", r"^SmolVLM(?!2)-.*", "HuggingFace"),
    ("SmolVLM2", r"^SmolVLM2.*", "HuggingFace"),
    # TIGER Lab
    ("Mantis", r"^Mantis-\d+B-.*", "TIGER Lab"),
    # NYU Cambrian
    ("Cambrian", r"^Cambrian-\d+B.*", "NYU"),
    # Microsoft series
    ("Phi-3-Vision", r"^Phi-3-Vision.*", "Microsoft"),
    ("Phi-3.5-Vision", r"^Phi-3\.5-Vision.*", "Microsoft"),
    ("Phi-4", r"^Phi-4-.*", "Microsoft"),
    ("Kosmos", r"^Kosmos.*", "Microsoft"),
    # Moondream series
    ("Moondream", r"^Moondream\d+.*", "Moondream"),
    # Mistral AI
    ("Pixtral", r"^Pixtral-.*", "Mistral AI"),
    # BAAI series
    ("Bunny", r"^Bunny-.*", "BAAI"),
    ("Emu", r"^Emu\d+_chat.*", "BAAI"),
    ("Aquila-VL", r"^Aquila-VL-.*", "BAAI"),
    # ShareGPT4V
    ("ShareGPT4V", r"^ShareGPT4V-\d+B.*", "Shanghai AI Laboratory"),
    # H2O.ai
    ("H2OVL", r"^H2OVL-.*", "H2O.ai"),
    # WeChat/WePOINTS
    ("POINTS", r"^POINTS-.*", "WeChat"),
    ("POINTS1.5", r"^POINTS1\.5-.*", "WeChat"),
    # Slime
    ("Slime", r"^Slime-\d+B.*", "Institute of Automation"),
    # Moonshot
    ("Kimi-VL", r"^Kimi-VL-.*", "moonshot.ai"),
    # Salesforce
    ("XGen-MM", r"^XGen-MM-.*", "Salesforce"),
    # IBM Granite
    ("granite-vision", r"^granite-vision-.*", "IBM"),
    # StepFun
    ("Step-1.5V", r"^Step-1\.5V.*", "StepFun"),
    ("Step-1o", r"^Step-1o.*", "StepFun"),
    # Ant Group BailingMM
    ("BailingMM", r"^BailingMM.*|^bailingMM.*", "Ant Group"),
    # Vintern
    ("Vintern", r"^Vintern-.*", "Fifth Civil Defender"),
    # DataCanvas MMAlaya
    ("MMAlaya", r"^MMAlaya\d*.*", "DataCanvas"),
    # VITA series
    ("VITA", r"^VITA(?!-1\.5).*", "NJU"),
    ("VITA-1.5", r"^VITA-1\.5.*", "NJU"),
    ("Long-VITA", r"^Long-VITA.*", "Tencent"),
]

# ============================================================================
# Family Introduction URL Mapping
# ============================================================================
FAMILY_INTRO_URLS = {
    "InternVL-Chat-V1.5": "https://internvl.github.io/blog/2024-04-30-InternVL-1.5/",
    "InternVL2": "https://internvl.github.io/blog/2024-07-02-InternVL-2.0/",
    "InternVL2-MPO": "https://internvl.github.io/blog/2024-07-02-InternVL-2.0/",
    "InternVL2.5": None,
    "InternVL2.5-MPO": "https://internvl.github.io/blog/2024-12-20-InternVL-2.5-MPO/",
    "InternVL3": "https://internvl.github.io/blog/2025-04-11-InternVL-3.0/",
    "Qwen-VL": "https://qwen.ai/blog?id=qwen-vl",
    "Qwen2-VL": "https://qwen.ai/blog?id=qwen2-vl",
    "Qwen2.5-VL": "https://qwen.ai/blog?id=qwen2.5-vl",
    "LLaVA-v1": "https://llava-vl.github.io/",
    "LLaVA-v1.5": "https://llava-vl.github.io/",
    "LLaVA-Next": "https://llava-vl.github.io/blog/2024-01-30-llava-next/",
    "LLaVA-Next-Interleave": "https://llava-vl.github.io/blog/2024-01-30-llava-next/",
    "LLaVA-OneVision": "https://llava-vl.github.io/blog/2024-08-05-llava-onevision/",
    "LLaVA-InternLM": "https://github.com/InternLM/xtuner",
    "LLaVA-LLaMA-3": "https://github.com/InternLM/xtuner",
    "GPT-4v": "https://openai.com/research/gpt-4v-system-card",
    "GPT-4o": "https://openai.com/index/hello-gpt-4o/",
    "GPT-4o-mini": "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",
    "ChatGPT-4o": "https://openai.com/index/hello-gpt-4o/",
    "GPT-4.1": "https://openai.com/index/gpt-4-1/",
    "GPT-4.1-mini": "https://openai.com/index/gpt-4-1/",
    "GPT-4.1-nano": "https://openai.com/index/gpt-4-1/",
    "GPT-4.5": "https://openai.com/index/introducing-gpt-4-5/",
    "GPT-5": "https://openai.com/index/introducing-gpt-5/",
    "GPT-5-mini": "https://openai.com/index/introducing-gpt-5/",
    "GPT-5-nano": "https://openai.com/index/introducing-gpt-5/",
    "Claude3": "https://www.anthropic.com/news/claude-3-family",
    "Claude3.5-Sonnet": "https://www.anthropic.com/news/claude-3-5-sonnet",
    "Claude3.7-Sonnet": "https://www.anthropic.com/news/claude-3-7-sonnet",
    "Gemini-1.0": "https://blog.google/technology/ai/google-gemini-ai/",
    "Gemini-1.5": "https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/",
    "Gemini-2.0": "https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/",
    "Gemini-2.5": "https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/",
    "Gemma3": "https://blog.google/technology/developers/gemma-3/",
    "PaliGemma": "https://ai.google.dev/gemma/docs/paligemma",
    "DeepSeek-VL": "https://github.com/deepseek-ai/DeepSeek-VL",
    "DeepSeek-VL2": "https://github.com/deepseek-ai/DeepSeek-VL2",
    "Janus": "https://github.com/deepseek-ai/Janus",
    "MiniCPM-V": "https://github.com/OpenBMB/MiniCPM-V",
    "MiniCPM-V-2": "https://github.com/OpenBMB/MiniCPM-V",
    "MiniCPM-V-2.6": "https://github.com/OpenBMB/MiniCPM-V",
    "MiniCPM-Llama3-V2.5": "https://github.com/OpenBMB/MiniCPM-V",
    "Phi-3-Vision": "https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/",
    "Phi-3.5-Vision": "https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/",
    "Phi-4": "https://azure.microsoft.com/en-us/blog/phi-4-a-small-language-model-specialist-in-complex-reasoning/",
    "CogVLM": "https://github.com/THUDM/CogVLM",
    "CogVLM2": "https://github.com/THUDM/CogVLM2",
    "GLM-4v": "https://open.bigmodel.cn/dev/api#glm-4v",
    "GLM-4v-Plus": "https://open.bigmodel.cn/dev/api#glm-4v",
    "Yi-VL": "https://01.ai/blog/yi-vl-release",
    "Cambrian": "https://cambrian-mllm.github.io/",
    "IDEFICS": "https://huggingface.co/blog/idefics",
    "IDEFICS2": "https://huggingface.co/blog/idefics2",
    "Idefics3": "https://huggingface.co/blog/idefics3",
    "InternLM-XComposer": "https://github.com/InternLM/InternLM-XComposer",
    "InternLM-XComposer2": "https://github.com/InternLM/InternLM-XComposer",
    "InternLM-XComposer2.5": "https://github.com/InternLM/InternLM-XComposer",
    "Chameleon": "https://ai.meta.com/blog/meta-fair-research-new-releases/",
    "Llama-3.2-Vision": "https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/",
    "Eagle-X5": "https://github.com/NVlabs/Eagle",
    "NVLM": "https://research.nvidia.com/labs/adlr/NVLM-1/",
    "Emu": "https://github.com/baaivision/Emu",
    "H2OVL": "https://h2o.ai/blog/",
    "Molmo": "https://molmo.allenai.org/blog",
    "Ovis1.5": "https://github.com/AIDC-AI/Ovis",
    "Ovis1.6": "https://github.com/AIDC-AI/Ovis",
    "Ovis2": "https://github.com/AIDC-AI/Ovis",
    "Pixtral": "https://mistral.ai/news/pixtral-12b/",
    "Kimi-VL": "https://kimi.moonshot.cn/",
    "Step-1.5V": "https://www.stepfun.com/",
    "Step-1o": "https://www.stepfun.com/",
    "VILA1.5": "https://github.com/NVlabs/VILA",
    "SmolVLM": "https://huggingface.co/blog/smolvlm",
    "SmolVLM2": "https://huggingface.co/blog/smolvlm2",
    "SAIL-VL": "https://github.com/BytedanceDouyinContent/SAIL-VL",
    "SAIL-VL-1.5": "https://github.com/BytedanceDouyinContent/SAIL-VL",
    "SAIL-VL-1.6": "https://github.com/BytedanceDouyinContent/SAIL-VL",
    "Valley": "https://github.com/bytedance-research/Valley",
    "ShareGPT4V": "https://github.com/ShareGPT4Omni/ShareGPT4V",
    "VITA": "https://github.com/VITA-MLLM/VITA",
    "VITA-1.5": "https://github.com/VITA-MLLM/VITA",
    "Long-VITA": "https://github.com/VITA-MLLM/Long-VITA",
    "Mantis": "https://github.com/TIGER-AI-Lab/Mantis",
    "Slime": "https://github.com/yifanzhang114/SliME",
    "POINTS": "https://github.com/WePOINTS/WePOINTS",
    "POINTS1.5": "https://github.com/WePOINTS/WePOINTS",
    "Moondream": "https://moondream.ai/",
    "XGen-MM": "https://github.com/salesforce/LAVIS/tree/xgen-mm",
    "Aquila-VL": "https://github.com/FlagAI-Open/Aquila-VL",
    "Bunny": "https://github.com/BAAI-DCAI/Bunny",
    "MMAlaya": "https://github.com/DataCanvas/MMAlaya",
    "BailingMM": None,
    "Vintern": None,
}


# ============================================================================
# Utility Functions
# ============================================================================

def http_get(url: str, timeout: int = 15) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as resp:
            return resp.read().decode('utf-8', errors='ignore')
    except Exception:
        return None


def normalize_name(name: str) -> str:
    name = re.sub(r'\s*\([^)]*\)', '', name)
    return re.sub(r'[-_.\s]+', '', name.lower())


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Optional[dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def sort_models_by_size(models: List[str]) -> List[str]:
    """Sort model name list by parameter size"""
    def extract_size(name):
        match = re.search(r'(\d+(?:\.\d+)?)\s*B', name, re.IGNORECASE)
        return float(match.group(1)) if match else 0
    return sorted(models, key=lambda x: (extract_size(x), x))


def load_csv_models() -> Set[str]:
    """Load the set of model names from CSV leaderboard data"""
    if not CSV_PATH.exists():
        print(f"  ⚠ CSV file not found: {CSV_PATH}, please run datadown.py first")
        return set()
    models = set()
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Model"):
                models.add(row["Model"])
    return models


# ============================================================================
# Phase 1: Build models_features_db.json (no LLM required)
# ============================================================================

# ---- Step 1: Download OpenVLM leaderboard, extract model metadata ----

def download_model_links() -> dict:
    """Extract all model links and metadata from OpenVLM.json"""
    url = "http://opencompass.openxlab.space/assets/OpenVLM.json"
    print("[Step 1] Downloading OpenVLM data...")
    content = http_get(url, timeout=60)
    if not content:
        raise RuntimeError("Failed to download OpenVLM.json")
    data = json.loads(content)

    results = data.get("results", {})
    print(f"  Total {len(results)} models")

    models = {}
    for model_name, model_data in results.items():
        meta = model_data.get("META", {})
        method_info = meta.get("Method", [])
        official_url = method_info[1] if len(method_info) > 1 else None

        entry = {
            "parameters": meta.get("Parameters", ""),
            "language_model": meta.get("Language Model", ""),
            "vision_model": meta.get("Vision Model", ""),
            "organization": meta.get("Org", ""),
            "release_date": meta.get("Time", ""),
            "open_source": meta.get("OpenSource", ""),
            "official_url": official_url,
            "huggingface": None,
            "arxiv": None,
        }
        # Infer huggingface/github from official_url
        if official_url:
            if "huggingface.co" in official_url:
                parts = official_url.split("huggingface.co/")
                if len(parts) > 1:
                    entry["huggingface"] = parts[1].split("?")[0].rstrip("/")
            elif "github.com" in official_url:
                entry["github"] = official_url
        models[model_name] = entry

    db = {
        "metadata": {
            "description": "Model links database for VLM evaluation",
            "version": "1.0",
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "total_models": len(models),
        },
        "models": models,
    }
    save_json(MODEL_LINKS_DB, db)
    print(f"  ✓ Saved to {MODEL_LINKS_DB}")
    return db


# ---- Step 2: Auto-match HuggingFace links + arXiv ----

def search_huggingface(name: str, limit: int = 15) -> List[Dict]:
    """Search models on HuggingFace API"""
    base = name.split("(")[0].strip()
    terms = list(dict.fromkeys([
        base, base.replace(" ", "-"), base.replace("-", " "),
        re.sub(r'[-_]?v?\d+\.?\d*[-_]?', '', base, flags=re.IGNORECASE),
    ]))
    results, seen = [], set()
    for term in [t.strip() for t in terms[:3] if t.strip()]:
        try:
            url = f"https://huggingface.co/api/models?search={urllib.parse.quote(term)}&limit={limit}"
            content = http_get(url, timeout=10)
            if content:
                for r in json.loads(content):
                    rid = r.get("id", "")
                    if rid and rid not in seen:
                        seen.add(rid)
                        results.append(r)
        except Exception:
            continue
    return results


def find_best_hf_match(model_name: str, results: List[Dict]) -> Optional[str]:
    """Find the best matching HuggingFace repo ID from search results"""
    if not results:
        return None
    name_n = normalize_name(model_name.split("(")[0].strip())

    # Exact match
    for r in results:
        repo = r.get("id", "")
        repo_name = repo.split("/")[-1] if "/" in repo else repo
        if normalize_name(repo_name) == name_n:
            return repo

    # Fuzzy match
    candidates = []
    for r in results:
        repo = r.get("id", "")
        repo_name = repo.split("/")[-1] if "/" in repo else repo
        rn = normalize_name(repo_name)
        score = 0
        if name_n in rn:
            score = 100 - (len(rn) - len(name_n))
        elif rn in name_n:
            score = 100 - (len(name_n) - len(rn))
        nc = re.sub(r'\d+b?$', '', name_n)
        rc = re.sub(r'\d+b?$', '', rn)
        if nc and rc:
            if nc == rc:
                score = max(score, 90)
            elif nc in rc or rc in nc:
                score = max(score, 70)
        if score > 50:
            candidates.append((repo, score, r.get("downloads", 0)))
    if candidates:
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return candidates[0][0]
    return None


def extract_arxiv_from_hf(hf_id: str) -> Optional[str]:
    """Extract arXiv ID from HuggingFace model page"""
    if not hf_id or hf_id.startswith("spaces/"):
        return None
    patterns = [
        r'arxiv\.org/abs/(\d{4}\.\d{4,5})',
        r'arxiv\.org/pdf/(\d{4}\.\d{4,5})',
        r'arXiv:(\d{4}\.\d{4,5})',
        r'\[(\d{4}\.\d{4,5})\]',
    ]
    # Try README
    content = http_get(f"https://huggingface.co/{hf_id}/raw/main/README.md")
    if content:
        for p in patterns:
            m = re.findall(p, content, re.IGNORECASE)
            if m:
                return m[0]
    # Try API
    content = http_get(f"https://huggingface.co/api/models/{hf_id}")
    if content:
        try:
            data = json.loads(content)
            card = data.get("cardData", {})
            for key in ["paper_url", "arxiv", "paper"]:
                if key in card and "arxiv" in str(card[key]).lower():
                    m = re.search(r'(\d{4}\.\d{4,5})', str(card[key]))
                    if m:
                        return m.group(1)
            for tag in data.get("tags", []):
                if "arxiv" in str(tag).lower():
                    m = re.search(r'(\d{4}\.\d{4,5})', str(tag))
                    if m:
                        return m.group(1)
        except Exception:
            pass
    return None


def enrich_model_links(db: dict) -> dict:
    """Auto-match HuggingFace links and arXiv for open-source models"""
    models = db["models"]
    to_process = [
        (n, m) for n, m in models.items()
        if m.get("open_source") == "Yes" and (not m.get("huggingface") or not m.get("arxiv"))
    ]
    print(f"[Step 2] Matching HuggingFace/arXiv links... ({len(to_process)} to process)")

    for i, (name, info) in enumerate(to_process, 1):
        # Match HuggingFace
        if not info.get("huggingface"):
            results = search_huggingface(name)
            best = find_best_hf_match(name, results) if results else None
            if best:
                info["huggingface"] = best
            time.sleep(0.5)

        # Extract arXiv from HuggingFace
        if info.get("huggingface") and not info.get("arxiv"):
            arxiv_id = extract_arxiv_from_hf(info["huggingface"])
            if arxiv_id:
                info["arxiv"] = arxiv_id
            time.sleep(0.3)

        if i % 20 == 0:
            save_json(MODEL_LINKS_DB, db)
            print(f"  Progress: {i}/{len(to_process)}")

    save_json(MODEL_LINKS_DB, db)
    hf_count = sum(1 for m in models.values() if m.get("huggingface"))
    arxiv_count = sum(1 for m in models.values() if m.get("arxiv"))
    print(f"  ✓ HuggingFace: {hf_count}, arXiv: {arxiv_count}")
    return db


# ---- Step 3: Extract model family relationships ----

def extract_families(db: dict) -> dict:
    """Extract model families based on regex rules"""
    models = db.get("models", {})
    families = defaultdict(lambda: {"organization": "", "models": []})
    assigned = set()

    for family_name, pattern, org in MODEL_FAMILY_PATTERNS:
        regex = re.compile(pattern)
        for model_name in models:
            if model_name not in assigned and regex.match(model_name):
                families[family_name]["organization"] = org
                families[family_name]["models"].append(model_name)
                assigned.add(model_name)

    result = {}
    for fn in sorted(families):
        result[fn] = {
            "organization": families[fn]["organization"],
            "models": sort_models_by_size(families[fn]["models"]),
        }
    db["model_families"] = result
    save_json(MODEL_LINKS_DB, db)
    print(f"[Step 3] Extracted model families: {len(result)} families, "
          f"{len(assigned)}/{len(models)} models assigned")
    return db


# ---- Step 4: Reorganize structure -> models_features_db.json + model_links_db_v2.json ----

def reorganize_to_v2(db: dict) -> dict:
    """Reorganize to organizations -> families -> models structure (model_links_db_v2.json)"""
    old_models = db.get("models", {})
    old_families = db.get("model_families", {})
    organizations = defaultdict(lambda: {"families": {}})

    for model_name, info in old_models.items():
        org = info.get("organization", "Unknown")
        # Find the model's family
        family_name = None
        for fn, fd in old_families.items():
            if model_name in fd.get("models", []):
                family_name = fn
                break
        if not family_name:
            family_name = model_name.split("-")[0] if "-" in model_name else model_name

        if family_name not in organizations[org]["families"]:
            organizations[org]["families"][family_name] = {
                "intro_url": FAMILY_INTRO_URLS.get(family_name),
                "models": {},
            }

        is_open = info.get("open_source") == "Yes"
        hf_id = info.get("huggingface")
        entry = {k: v for k, v in {
            "parameters": info.get("parameters") or None,
            "language_model": info.get("language_model") or None,
            "vision_model": info.get("vision_model") or None,
            "release_date": info.get("release_date") or None,
            "open_source": is_open or None,
            "main_url": f"https://huggingface.co/{hf_id}" if is_open and hf_id else None,
            "huggingface": hf_id if is_open else None,
            "arxiv": info.get("arxiv"),
        }.items() if v is not None}

        organizations[org]["families"][family_name]["models"][model_name] = entry

    v2 = {
        "metadata": {
            "description": "Model links database organized by organization and family",
            "version": "2.0",
            "structure": "organizations -> families -> models",
        },
        "organizations": dict(organizations),
    }
    save_json(MODEL_LINKS_V2_DB, v2)
    print(f"[Step 4a] Generated v2 structure: {len(organizations)} organizations")
    return v2


def flatten_v2_to_features(v2: dict) -> dict:
    """
    Flatten v2 structure to models_features_db.json:
    {model_families: {family: {organization, models: {model: features}}}}
    """
    features_db = {"model_families": {}}
    for org_name, org_data in v2.get("organizations", {}).items():
        for family_name, family_data in org_data.get("families", {}).items():
            if family_name not in features_db["model_families"]:
                features_db["model_families"][family_name] = {
                    "organization": org_name,
                    "models": {},
                }
            features_db["model_families"][family_name]["models"].update(
                family_data.get("models", {})
            )
    save_json(MODELS_FEATURES_FILE, features_db)
    family_count = len(features_db["model_families"])
    model_count = sum(len(f["models"]) for f in features_db["model_families"].values())
    print(f"[Step 4b] Generated models_features_db.json: {family_count} families, {model_count} models")
    return features_db


def cmd_build_features():
    """Phase 1: Build models_features_db.json (Steps 1-4)"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download or load cache
    db = load_json(MODEL_LINKS_DB)
    if db and db.get("models"):
        print(f"[Step 1] Using existing model_links_db.json ({len(db['models'])} models)")
    else:
        db = download_model_links()

    # Step 2: Match HuggingFace/arXiv
    db = enrich_model_links(db)

    # Step 3: Extract families
    db = extract_families(db)

    # Step 4: Reorganize + generate features DB
    v2 = reorganize_to_v2(db)
    flatten_v2_to_features(v2)

    print("\n✓ Phase 1 complete: models_features_db.json generated")


# ============================================================================
# Phase 1.5: Build benchmark_features_db.json (requires LLM)
# ============================================================================

def get_llm_config() -> dict:
    """Get LLM configuration from environment variables"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable")
    return {
        "api_key": api_key,
        "model": os.environ.get("LLM_MODEL", "gpt-4o"),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/"),
    }


def extract_benchmark_subcategories() -> Dict[str, List[str]]:
    """Extract subcategory fields for each benchmark from OpenVLM.json"""
    # Prefer using local cached raw JSON
    data = load_json(RAW_JSON_PATH)
    if not data:
        content = http_get("http://opencompass.openxlab.space/assets/OpenVLM.json", timeout=60)
        if not content:
            return {}
        data = json.loads(content)

    subcats = defaultdict(set)
    for model_data in data.get("results", {}).values():
        for key, value in model_data.items():
            if key != "META" and isinstance(value, dict):
                subcats[key].update(
                    s for s in value.keys()
                    if not s.startswith("Overall") and s not in ("Dir Name", "Final Score")
                )
    return {k: sorted(v) for k, v in sorted(subcats.items())}


def get_benchmark_names_from_csv() -> List[str]:
    """Extract all benchmark column names from CSV"""
    if not CSV_PATH.exists():
        return []
    import pandas as pd
    df = pd.read_csv(CSV_PATH, nrows=0)
    meta_cols = {"Model", "Parameters", "Organization", "OpenSource", "Time"}
    return [c for c in df.columns if c not in meta_cols]


def cmd_build_benchmarks():
    """Phase 1.5: Build benchmark_features_db.json"""
    from agents.benchmark_agent import BenchmarkAgent

    llm_config = get_llm_config()
    agent = BenchmarkAgent(llm_config=llm_config)

    # Get benchmark list
    benchmarks = get_benchmark_names_from_csv()
    if not benchmarks:
        print("❌ Benchmark list not found, please run datadown.py first")
        return

    # Extract subcategory info
    subcats_map = extract_benchmark_subcategories()

    # Load existing data (supports resume)
    result = load_json(BENCHMARK_FEATURES_FILE) or {"benchmarks": {}}

    print(f"[Step B] Generating benchmark features database... ({len(benchmarks)} benchmarks)")
    processed = 0

    for i, bench_name in enumerate(benchmarks, 1):
        # Skip existing entries
        if bench_name in result["benchmarks"]:
            continue

        print(f"  [{i}/{len(benchmarks)}] {bench_name}")
        try:
            features = agent.get_features(bench_name)
            feat_dict = features.to_dict()

            # Supplement subcategory info (extracted from OpenVLM data)
            if bench_name in subcats_map and not feat_dict.get("subcategories"):
                feat_dict["subcategories"] = subcats_map[bench_name]

            result["benchmarks"][bench_name] = feat_dict
            processed += 1
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # Save immediately
        save_json(BENCHMARK_FEATURES_FILE, result)

    print(f"\n✓ benchmark_features_db.json generated (processed {processed}, total {len(result['benchmarks'])})")


# ============================================================================
# Phase 2: Build models_knowledge_db.json (requires LLM)
# ============================================================================


def cmd_build_knowledge():
    """Phase 2: Build models_knowledge_db.json (Step 5)"""
    from agents.model_agent import ModelAgent

    llm_config = get_llm_config()
    agent = ModelAgent(
        llm_config=llm_config,
        google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
        google_cse_id=os.environ.get("GOOGLE_CSE_ID", ""),
        jina_api_key=os.environ.get("JINA_API_KEY", ""),
    )

    # Load v2 data
    v2 = load_json(MODEL_LINKS_V2_DB)
    if not v2:
        print("❌ model_links_db_v2.json not found, please run build-features first")
        return

    # Load model names from CSV (only process models on the leaderboard)
    csv_models = load_csv_models()
    if not csv_models:
        print("⚠ No CSV data, will process all models")

    # Load existing results (supports resume)
    result = load_json(MODELS_KNOWLEDGE_FILE) or {
        "metadata": {
            "description": "Model knowledge database with technical summaries and community feedback",
            "version": "2.0",
            "structure": "organizations -> families -> models",
            "generated_at": datetime.now().isoformat(),
        },
        "organizations": {},
    }

    organizations = v2.get("organizations", {})
    total_families = sum(len(o.get("families", {})) for o in organizations.values())
    idx = 0
    processed = 0

    print(f"[Step 5] Generating model knowledge database... ({total_families} families)")

    for org_name, org_data in organizations.items():
        if org_name not in result["organizations"]:
            result["organizations"][org_name] = {"families": {}}

        for family_name, family_data in org_data.get("families", {}).items():
            idx += 1
            models = family_data.get("models", {})

            # Only process models that exist in CSV
            if csv_models:
                relevant = [m for m in models if m in csv_models]
                if not relevant:
                    continue

            # Skip already processed families
            existing = result["organizations"].get(org_name, {}).get("families", {}).get(family_name, {})
            if existing.get("technical_summary"):
                continue

            print(f"  [{idx}/{total_families}] {org_name} / {family_name}")

            family_result = {
                "intro_url": family_data.get("intro_url"),
                "models": {},
            }

            # Get family-level features (technical summary + community feedback)
            try:
                ff = agent.get_family_features(family_name)
                family_result["technical_summary"] = ff.technical_summary
                family_result["sentiment_score"] = ff.sentiment_score
                family_result["hype_level"] = ff.hype_level
                family_result["positive_aspects"] = ff.positive_aspects
                family_result["negative_aspects"] = ff.negative_aspects
                family_result["community_summary"] = ff.community_summary
                family_result["web_mentions_count"] = ff.web_mentions_count
            except Exception as e:
                print(f"    ✗ Family features extraction failed: {e}")

            # Get model-level features (open-source models only)
            for model_name, model_data in models.items():
                if csv_models and model_name not in csv_models:
                    continue
                model_result = dict(model_data)
                if model_data.get("open_source") and model_data.get("main_url"):
                    try:
                        mf = agent.get_model_features(model_name)
                        if mf.technical_summary:
                            model_result["technical_summary"] = mf.technical_summary
                    except Exception as e:
                        print(f"    ✗ Model {model_name} features extraction failed: {e}")
                family_result["models"][model_name] = model_result

            result["organizations"][org_name]["families"][family_name] = family_result
            processed += 1

            # Save immediately (supports resume)
            result["metadata"]["last_updated"] = datetime.now().isoformat()
            save_json(MODELS_KNOWLEDGE_FILE, result)

    print(f"\n✓ Phase 2 complete: models_knowledge_db.json generated (processed {processed} families)")


# ============================================================================
# Phase 3: Build component knowledge databases (requires LLM)
# ============================================================================

def extract_components(v2: dict) -> Tuple[Dict, Dict]:
    """Extract all language_model / vision_model components from v2 database"""
    lm_map, vm_map = {}, {}
    for org_data in v2.get("organizations", {}).values():
        for family_data in org_data.get("families", {}).values():
            for model_name, model_data in family_data.get("models", {}).items():
                lm = (model_data.get("language_model") or "").split("<br>")[0].strip()
                vm = (model_data.get("vision_model") or "").split("<br>")[0].strip()
                if lm:
                    lm_map.setdefault(lm, []).append(model_name)
                if vm:
                    vm_map.setdefault(vm, []).append(model_name)
    return lm_map, vm_map


def generate_component_summary(client, name: str, comp_type: str, readme: Optional[str]) -> Optional[dict]:
    """Use LLM to generate component technical summary from README"""
    if readme and len(readme) > 8000:
        readme = readme[:8000] + "\n...[truncated]..."

    if readme and len(readme) > 200:
        prompt = f"""Based on the following HuggingFace README, extract technical information for {comp_type} "{name}".
## README Content
{readme}
## Please return JSON
```json
{{"name":"{name}","type":"{comp_type}","organization":"developer organization","parameters":"parameter count","architecture":"architecture","key_features":["feature1","feature2","feature3"],"summary":"one sentence summary"}}
```
Output only JSON."""
    else:
        prompt = f"""Based on your knowledge, provide technical information for {comp_type} "{name}", return JSON:
```json
{{"name":"{name}","type":"{comp_type}","organization":"developer organization","parameters":"parameter count","architecture":"architecture","key_features":["feature1","feature2","feature3"],"summary":"one sentence summary"}}
```
Output only JSON."""

    try:
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800, temperature=0.3,
        )
        text = resp.choices[0].message.content or ""
        # Parse JSON
        m = re.search(r'```json\s*([\s\S]*?)\s*```', text) or re.search(r'\{[\s\S]*\}', text)
        if m:
            json_str = m.group(1) if '```' in m.group(0) else m.group(0)
            return json.loads(json_str)
    except Exception as e:
        print(f"    ✗ LLM generation failed: {e}")
    return None


def cmd_build_components():
    """Phase 3: Build component knowledge databases (Step 6)"""
    from openai import OpenAI

    llm_config = get_llm_config()
    client = OpenAI(api_key=llm_config["api_key"], base_url=llm_config["base_url"])

    v2 = load_json(MODEL_LINKS_V2_DB)
    if not v2:
        print("❌ model_links_db_v2.json not found, please run build-features first")
        return

    lm_map, vm_map = extract_components(v2)
    print(f"[Step 6] Generating component knowledge databases: Language Models={len(lm_map)}, Vision Models={len(vm_map)}")

    for comp_type, comp_map, output_path in [
        ("Language Model", lm_map, LANGUAGE_MODEL_KB_FILE),
        ("Vision Model", vm_map, VISION_MODEL_KB_FILE),
    ]:
        result = load_json(output_path) or {
            "metadata": {"description": f"{comp_type} knowledge database", "version": "1.0"},
            "components": {},
        }

        total = len(comp_map)
        for idx, (name, used_by) in enumerate(sorted(comp_map.items()), 1):
            existing = result["components"].get(name, {})
            if existing.get("summary") and len(existing.get("summary", "")) > 10:
                continue

            print(f"  [{idx}/{total}] {name}")

            # Try searching HuggingFace and fetching README
            hf_results = search_huggingface(name)
            hf_id = find_best_hf_match(name, hf_results) if hf_results else None
            readme = None
            if hf_id:
                readme = http_get(f"https://huggingface.co/{hf_id}/raw/main/README.md")

            knowledge = generate_component_summary(client, name, comp_type, readme)
            if knowledge:
                knowledge["huggingface"] = hf_id
                knowledge["used_by_count"] = len(used_by)
                knowledge["used_by_examples"] = used_by[:5]
                result["components"][name] = knowledge
            else:
                result["components"][name] = {
                    "name": name, "type": comp_type,
                    "huggingface": hf_id,
                    "used_by_count": len(used_by),
                    "used_by_examples": used_by[:5],
                    "summary": "N/A",
                }

            result["metadata"]["last_updated"] = datetime.now().isoformat()
            save_json(output_path, result)

        print(f"  ✓ {comp_type} knowledge database complete: {len(result['components'])} components")

    print("\n✓ Phase 3 complete: Component knowledge databases generated")


# ============================================================================
# Main
# ============================================================================

def main():
    commands = ["build-features", "build-benchmarks", "build-knowledge", "build-components", "all"]
    parser = argparse.ArgumentParser(
        description="STAR Database Generation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  build-features    Steps 1-4: Download data + match links + extract families -> models_features_db.json (no LLM required)
  build-benchmarks  Step B: Use BenchmarkAgent -> benchmark_features_db.json (requires LLM)
  build-knowledge   Step 5: Use ModelAgent -> models_knowledge_db.json (requires LLM + network)
  build-components  Step 6: Generate component knowledge databases -> vision/language_model_knowledge_db.json (requires LLM)
  all               Run all steps

Environment variables (required for build-benchmarks / build-knowledge / build-components):
  OPENAI_API_KEY    LLM API Key
  OPENAI_BASE_URL   LLM API Base URL (optional)
  LLM_MODEL         LLM model name (default: gpt-4o)
  GOOGLE_API_KEY    Google Search API Key (optional, used by ModelAgent / BenchmarkAgent)
  GOOGLE_CSE_ID     Google Custom Search Engine ID (optional)
  JINA_API_KEY      Jina AI API Key (optional)
""")
    parser.add_argument("command", choices=commands, help="Command to execute")
    args = parser.parse_args()

    print("=" * 60)
    print("STAR Database Generation Tool")
    print("=" * 60)

    if args.command in ("build-features", "all"):
        cmd_build_features()

    if args.command in ("build-benchmarks", "all"):
        cmd_build_benchmarks()

    if args.command in ("build-knowledge", "all"):
        cmd_build_knowledge()

    if args.command in ("build-components", "all"):
        cmd_build_components()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
