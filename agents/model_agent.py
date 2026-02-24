"""
Model Agent - Model family and individual model information collection

Two-level structure:
1. Family level (FamilyFeatures): Technical summary + community feedback (web search)
2. Model level (ModelFeatures): Technical info + technical summary (no community feedback)
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import requests
import json
import re
import os

# Import async fetcher module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_sources.async_fetcher import AsyncFetcher


# ============================================================================
# Data Class Definitions
# ============================================================================

@dataclass
class FamilyFeatures:
    """Model family features (technical summary + community feedback)"""
    family_name: str
    organization: Optional[str] = None
    intro_url: Optional[str] = None
    
    # Models in this family
    models: List[str] = field(default_factory=list)
    
    # Technical summary (fetched from intro_url)
    technical_summary: Optional[str] = None
    
    # Community feedback (web search)
    sentiment_score: float = 0.5
    hype_level: str = "medium"
    confidence: float = 0.5
    positive_aspects: List[str] = field(default_factory=list)
    negative_aspects: List[str] = field(default_factory=list)
    reported_issues: List[str] = field(default_factory=list)
    community_summary: Optional[str] = None
    web_mentions_count: int = 0
    
    last_update: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() 
                if v is not None and v != [] and v != False}


@dataclass
class ModelFeatures:
    """Individual model features (technical info only, no community feedback)"""
    model_name: str
    
    # Technical info
    parameters: Optional[str] = None
    language_model: Optional[str] = None
    vision_model: Optional[str] = None
    organization: Optional[str] = None
    release_date: Optional[str] = None
    open_source: bool = False
    
    # Link info
    huggingface_id: Optional[str] = None
    arxiv_id: Optional[str] = None
    main_url: Optional[str] = None  # Open-source models: HF Model Card
    
    # Family info
    model_family: Optional[str] = None
    family_models: List[str] = field(default_factory=list)
    
    # Technical summary
    technical_summary: Optional[str] = None
    
    last_update: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() 
                if v is not None and v != [] and v != False}
    
    def get_brief(self) -> str:
        """Get brief information"""
        parts = []
        if self.parameters:
            parts.append(f"Parameters: {self.parameters}")
        if self.language_model:
            parts.append(f"Language Model: {self.language_model}")
        if self.vision_model:
            parts.append(f"Vision Model: {self.vision_model}")
        return " | ".join(parts) if parts else "No technical info"


# ============================================================================
# ModelAgent Main Class
# ============================================================================

class ModelAgent:
    """
    Model Information Collection Agent
    
    Features:
    1. Read technical info from model_links_db.json
    2. Search HuggingFace Discussions and Google for community feedback
    3. Use LLM to analyze community content
    """
    
    # Database path (v2 structure: organizations -> families -> models)
    MODEL_LINKS_DB = Path(__file__).parent.parent / "data" / "opencompass_cache" / "model_links_db_v2.json"
    
    def __init__(
        self,
        llm_config: Optional[Dict] = None,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        jina_api_key: Optional[str] = None,
        use_jina: bool = True
    ):
        """
        Initialize ModelAgent
        
        Args:
            llm_config: LLM configuration
            google_api_key: Google Custom Search API Key
            google_cse_id: Google Custom Search Engine ID
            jina_api_key: Jina API Key
            use_jina: Whether to use Jina AI
        """
        self.llm_config = llm_config or {}
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        self.google_cse_id = google_cse_id or os.environ.get("GOOGLE_CSE_ID")
        self.jina_api_key = jina_api_key or os.environ.get("JINA_API_KEY")
        
        self._openai_client = None
        self._model_db = None  # Lazy loading
        
        # Async fetcher
        self._fetcher = AsyncFetcher(
            max_concurrent=20,
            timeout=30,
            use_jina=use_jina and bool(jina_api_key or os.environ.get("JINA_API_KEY")),
            jina_api_key=jina_api_key or os.environ.get("JINA_API_KEY")
        )
    
    # ========================================================================
    # Public Methods
    # ========================================================================
    
    def get_family_features(self, family_name: str) -> FamilyFeatures:
        """
        Get model family features (including community feedback)
        
        Args:
            family_name: Family name, e.g., "InternVL2", "GPT-4o"
        
        Returns:
            FamilyFeatures instance
        """
        features = FamilyFeatures(family_name=family_name)
        
        print(f"Starting family info collection: {family_name}")
        
        # 1. Read family info from database
        print(f"[1/3] Loading family info...")
        self._load_family_info(features, family_name)
        
        # 2. Generate technical summary from intro_url
        if features.intro_url:
            print(f"[2/3] Reading technical intro and generating summary...")
            self._generate_family_summary(features)
        else:
            print(f"[2/3] No intro_url, skipping technical summary")
        
        # 3. Web search for community feedback
        print(f"[3/3] Searching for community feedback...")
        self._fetch_family_feedback(features)
        
        features.last_update = datetime.now().isoformat()
        print(f"Family info collection complete")
        return features
    
    def get_model_features(self, model_name: str) -> ModelFeatures:
        """
        Get individual model features (technical info only, no community feedback)
        
        Args:
            model_name: Model name
        
        Returns:
            ModelFeatures instance
        """
        features = ModelFeatures(model_name=model_name)
        
        print(f"Starting model info collection: {model_name}")
        
        # 1. Read basic info from database
        print(f"[1/2] Loading basic info...")
        self._load_technical_info(features, model_name)
        
        # 2. Generate technical summary (open-source models only, from HF Model Card)
        if features.open_source and features.main_url:
            print(f"[2/2] Generating technical summary...")
            self._generate_model_summary(features)
        else:
            print(f"[2/2] Closed-source model, technical summary inherited from family")
        
        features.last_update = datetime.now().isoformat()
        print(f"Model info collection complete")
        return features
    
    def get_family_text(self, family_name: str) -> str:
        """Get text description of family features"""
        f = self.get_family_features(family_name)
        
        lines = [f"{'='*60}", f"Family: {f.family_name}", f"{'='*60}"]
        
        if f.organization:
            lines.append(f"Organization: {f.organization}")
        if f.intro_url:
            lines.append(f"Intro URL: {f.intro_url}")
        if f.models:
            lines.append(f"Model count: {len(f.models)}")
            lines.append(f"Model list: {', '.join(f.models[:5])}{'...' if len(f.models) > 5 else ''}")
        
        if f.technical_summary:
            lines.append(f"\n[Technical Summary]\n{f.technical_summary}")
        
        if f.web_mentions_count > 0:
            lines.append(f"\n[Community Feedback]")
            lines.append(f"  Sentiment score: {f.sentiment_score:.2f}")
            lines.append(f"  Hype level: {f.hype_level}")
            lines.append(f"  Web mentions: {f.web_mentions_count}")
            
            if f.positive_aspects:
                lines.append(f"\n  [Positive Aspects]")
                for aspect in f.positive_aspects[:5]:
                    lines.append(f"    - {aspect}")
            
            if f.negative_aspects:
                lines.append(f"\n  [Negative Aspects]")
                for aspect in f.negative_aspects[:5]:
                    lines.append(f"    - {aspect}")
            
            if f.community_summary:
                lines.append(f"\n  [Discussion Summary]\n  {f.community_summary}")
        
        return "\n".join(lines)
    
    def get_model_text(self, model_name: str) -> str:
        """Get text description of model features"""
        f = self.get_model_features(model_name)
        
        lines = [f"{'='*60}", f"Model: {f.model_name}", f"{'='*60}"]
        
        # Technical info
        lines.append("\n[Technical Info]")
        if f.parameters:
            lines.append(f"  Parameters: {f.parameters}")
        if f.language_model:
            lines.append(f"  Language Model: {f.language_model}")
        if f.vision_model:
            lines.append(f"  Vision Model: {f.vision_model}")
        if f.organization:
            lines.append(f"  Organization: {f.organization}")
        if f.release_date:
            lines.append(f"  Release Date: {f.release_date}")
        lines.append(f"  Open Source: {'Yes' if f.open_source else 'No'}")
        
        # Links
        if f.main_url or f.huggingface_id or f.arxiv_id:
            lines.append("\n[Links]")
            if f.main_url:
                lines.append(f"  Main Reference: {f.main_url}")
            if f.huggingface_id:
                lines.append(f"  HuggingFace: https://huggingface.co/{f.huggingface_id}")
            if f.arxiv_id:
                lines.append(f"  arXiv: https://arxiv.org/abs/{f.arxiv_id}")
        
        # Family
        if f.model_family:
            lines.append(f"\n[Family] {f.model_family}")
        
        if f.technical_summary:
            lines.append(f"\n[Technical Summary]\n{f.technical_summary}")
        
        return "\n".join(lines)
    
    # ========================================================================
    # Technical Info (from database + online sources)
    # ========================================================================
    
    def _load_model_db(self) -> Dict:
        """Load model database (singleton)"""
        if self._model_db is None:
            try:
                with open(self.MODEL_LINKS_DB, "r", encoding="utf-8") as f:
                    self._model_db = json.load(f)
            except Exception as e:
                print(f"Failed to load model database: {e}")
                self._model_db = {"organizations": {}}
        return self._model_db
    
    def _load_technical_info(self, features: ModelFeatures, model_name: str):
        """Read basic technical info from database (v2 structure)"""
        db = self._load_model_db()
        organizations = db.get("organizations", {})
        
        # Traverse organizations -> families -> models to find the model
        found = False
        for org_name, org_data in organizations.items():
            for family_name, family_data in org_data.get("families", {}).items():
                models = family_data.get("models", {})
                
                # Exact match
                model_info = models.get(model_name)
                
                # Fuzzy match
                if not model_info:
                    model_info, matched_name = self._fuzzy_match_model(model_name, models)
                    if matched_name:
                        model_name = matched_name
                
                if model_info:
                    # Populate model info
                    features.parameters = model_info.get("parameters")
                    features.language_model = model_info.get("language_model")
                    features.vision_model = model_info.get("vision_model")
                    features.organization = org_name
                    features.release_date = model_info.get("release_date")
                    features.open_source = model_info.get("open_source", False)
                    features.huggingface_id = model_info.get("huggingface")
                    features.arxiv_id = model_info.get("arxiv")
                    
                    # main_url: open-source models use their own, closed-source inherit from family intro_url
                    if model_info.get("main_url"):
                        features.main_url = model_info["main_url"]
                    elif family_data.get("intro_url"):
                        features.main_url = family_data["intro_url"]
                    
                    # Family info
                    features.model_family = family_name
                    features.family_models = list(models.keys())
                    
                    print(f"  ✓ Loaded from database: {org_name} / {family_name}")
                    found = True
                    break
            if found:
                break
        
        if not found:
            print(f"  ✗ Model not found in database: {model_name}")
    
    def _fuzzy_match_model(self, model_name: str, models: Dict) -> tuple:
        """Fuzzy match model name, returns (model_info, matched_name)"""
        def normalize(name: str) -> str:
            return name.lower().replace(" ", "").replace("_", "").replace("-", "").replace(".", "")
        
        target = normalize(model_name)
        
        for name, info in models.items():
            if normalize(name) == target:
                return (info, name)
        
        for name, info in models.items():
            normalized = normalize(name)
            if target in normalized or normalized in target:
                return (info, name)
        
        return (None, None)
    
    def _load_family_info(self, features: FamilyFeatures, family_name: str):
        """Read family info from database"""
        db = self._load_model_db()
        
        for org_name, org_data in db.get("organizations", {}).items():
            families = org_data.get("families", {})
            if family_name in families:
                family_data = families[family_name]
                features.organization = org_name
                features.intro_url = family_data.get("intro_url")
                features.models = list(family_data.get("models", {}).keys())
                print(f"  ✓ Loaded from database: {org_name} / {family_name} ({len(features.models)} models)")
                return
        
        print(f"  ✗ Family not found in database: {family_name}")
    
    def _generate_family_summary(self, features: FamilyFeatures):
        """Generate family technical summary from intro_url"""
        if not features.intro_url:
            return
        
        # 1. Try direct fetch first
        content = self._fetch_url_content(features.intro_url)
        
        if content:
            print(f"       Fetched {len(content)} characters")
            _, summary = self._generate_structured_summary(
                features.family_name, content, check_sufficiency=False
            )
            features.technical_summary = summary
            print(f"  ✓ Family technical summary generated")
        else:
            # 2. Fetch failed, use LLM search mode
            print(f"  ✗ Cannot fetch directly, switching to LLM search mode...")
            summary = self._get_summary_via_llm_search(features.family_name, features.intro_url)
            if summary:
                features.technical_summary = summary
                print(f"  ✓ LLM search mode succeeded")
            else:
                print(f"  ✗ LLM search mode also failed")
    
    def _get_summary_via_llm_search(self, name: str, url: str) -> Optional[str]:
        """Use LLM + web_search to get content summary (when direct fetch fails)"""
        client = self._get_llm_client()
        if not client:
            return None
        
        prompt = f"""Please research the following AI model/series and provide technical information.

Model Name: {name}
Reference URL: {url}

Please search the web for detailed technical information about this model, then generate a summary in the following format:

1. Introduction
(Brief description including positioning, key features, parameter scale, etc.)

2. Key Enhancements
(Main improvements and highlights, list 3-5 key points)

3. Model Architecture Updates
(Architecture updates, if any)

Please output the summary directly, no additional explanation needed. If no relevant information is found, please state "No relevant technical information found"."""

        try:
            model = self.llm_config.get("model", "gpt-4o")
            response = client.responses.create(
                model=model,
                input=prompt,
                tools=[{"type": "web_search_preview"}]
            )
            
            # Extract text from response.output
            # output[0] is web_search_call, output[1] is message
            result = None
            for item in response.output:
                if hasattr(item, 'content'):  # ResponseOutputMessage
                    for content in item.content:
                        if hasattr(content, 'text'):  # ResponseOutputText
                            result = content.text.strip()
                            break
                    if result:
                        break
            
            if result and "No relevant" not in result and len(result) > 100:
                return result
            return None
            
        except Exception as e:
            print(f"       LLM search failed: {e}")
            return None
    
    def _fetch_family_feedback(self, features: FamilyFeatures):
        """Web search for family community feedback (no HuggingFace discussions)"""
        web_data = self._get_web_discussions(features.family_name)
        if not web_data:
            print(f"  ✗ No web discussions")
            return
        
        features.web_mentions_count = web_data.get("count", 0)
        all_discussions = web_data.get("mentions", [])
        print(f"       Found {features.web_mentions_count} discussions")
        
        if all_discussions:
            analysis = self._analyze_discussions(features.family_name, all_discussions)
            if analysis:
                features.positive_aspects = analysis.get("positive", [])
                features.negative_aspects = analysis.get("negative", [])
                features.reported_issues = analysis.get("issues", [])
                features.sentiment_score = analysis.get("sentiment", 0.5)
                features.community_summary = analysis.get("summary", "")
        
        features.confidence = min(features.web_mentions_count / 30, 1.0)
        features.hype_level = self._calculate_hype_level(features.web_mentions_count)
    
    def _generate_model_summary(self, features: ModelFeatures):
        """Generate technical summary for open-source models (from HF Model Card)"""
        if not features.main_url:
            return
        
        # 1. Try direct fetch first
        content = self._fetch_url_content(features.main_url)
        
        if content:
            print(f"       Fetched {len(content)} characters")
            _, summary = self._generate_structured_summary(
                features.model_name, content, check_sufficiency=False
            )
            features.technical_summary = summary
            print(f"  ✓ Model technical summary generated")
        else:
            # 2. Fetch failed, use LLM search mode
            print(f"  ✗ Cannot fetch directly, switching to LLM search mode...")
            summary = self._get_summary_via_llm_search(features.model_name, features.main_url)
            if summary:
                features.technical_summary = summary
                print(f"  ✓ LLM search mode succeeded")
            else:
                print(f"  ✗ LLM search mode also failed")
    
    # ========================================================================
    # Technical Summary Generation (legacy method, kept for compatibility)
    # ========================================================================
    
    def _generate_technical_summary(self, features: ModelFeatures):
        """
        Generate structured technical summary
        
        Workflow:
        1. First read main_url (primary reference link)
        2. LLM determines if information is sufficient
        3. If not, read additional links (HuggingFace/arXiv/GitHub, etc.)
        """
        client = self._get_llm_client()
        if not client:
            print("  ✗ LLM client unavailable, skipping technical summary")
            return
        
        collected_content = ""
        
        # 1. Read main_url first
        if features.main_url:
            print(f"  [2.1] Reading main reference link: {features.main_url[:50]}...")
            main_content = self._fetch_url_content(features.main_url)
            if main_content:
                collected_content = main_content
                print(f"       Fetched {len(main_content)} characters")
                
                # LLM determines if information is sufficient
                is_sufficient, summary = self._generate_structured_summary(
                    features.model_name, collected_content, check_sufficiency=True
                )
                
                if is_sufficient:
                    features.technical_summary = summary
                    print(f"  ✓ main_url info sufficient, summary generated")
                    return
                else:
                    print(f"  [2.2] main_url info insufficient, fetching additional links...")
        else:
            print(f"  [2.1] No main_url, trying other sources")
        
        # 2. Fetch supplementary links
        extra_content = self._fetch_supplementary_content(features)
        if extra_content:
            collected_content += f"\n\n--- Supplementary Sources ---\n{extra_content}"
            print(f"       Additional {len(extra_content)} characters fetched")
        
        # 3. Generate final summary
        if collected_content:
            _, summary = self._generate_structured_summary(
                features.model_name, collected_content, check_sufficiency=False
            )
            features.technical_summary = summary
            print(f"  ✓ Technical summary generated")
        else:
            print(f"  ✗ Unable to fetch technical info")
    
    def _fetch_url_content(self, url: str) -> Optional[str]:
        """Fetch content from a single URL"""
        contents = self._fetcher.fetch_urls([url])
        content = contents.get(url, "")
        if content and not content.startswith("Error:") and len(content) > 200:
            return content[:15000]
        return None
    
    def _fetch_hf_model_card(self, repo_id: str) -> Optional[str]:
        """Fetch Model Card (README.md) from HuggingFace"""
        try:
            url = f"https://huggingface.co/{repo_id}/raw/main/README.md"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.text[:15000]
        except Exception as e:
            print(f"       Failed to fetch HuggingFace Model Card: {e}")
        return None
    
    def _fetch_supplementary_content(self, features: ModelFeatures) -> Optional[str]:
        """Fetch supplementary content (HuggingFace/arXiv/GitHub, etc.)"""
        urls_to_fetch = []
        
        # HuggingFace Model Card (if main_url is not HF)
        if features.huggingface_id and features.main_url and "huggingface.co" not in features.main_url:
            urls_to_fetch.append(f"https://huggingface.co/{features.huggingface_id}/raw/main/README.md")
        
        # arXiv
        if features.arxiv_id:
            urls_to_fetch.append(f"https://arxiv.org/abs/{features.arxiv_id}")
        
        # GitHub
        if features.github_url:
            github_raw = self._convert_github_to_raw(features.github_url)
            if github_raw:
                urls_to_fetch.append(github_raw)
        
        if not urls_to_fetch:
            return None
        
        print(f"       Trying supplementary: {[u[:40]+'...' for u in urls_to_fetch[:2]]}")
        contents = self._fetcher.fetch_urls(urls_to_fetch[:2])
        
        combined = ""
        for url, content in contents.items():
            if content and not content.startswith("Error:") and len(content) > 200:
                combined += f"\n[Source: {url}]\n{content[:8000]}\n"
        
        return combined[:15000] if combined else None
    
    def _fetch_technical_report(self, features: ModelFeatures) -> Optional[str]:
        """Fetch technical report/paper content"""
        urls_to_fetch = []
        
        # Priority: main_url > arXiv > GitHub > official site
        if features.main_url:
            urls_to_fetch.append(features.main_url)
        
        if features.arxiv_id:
            urls_to_fetch.append(f"https://arxiv.org/abs/{features.arxiv_id}")
        if features.github_url:
            github_raw = self._convert_github_to_raw(features.github_url)
            if github_raw:
                urls_to_fetch.append(github_raw)
        if features.official_url and features.official_url not in urls_to_fetch:
            urls_to_fetch.append(features.official_url)
        
        if not urls_to_fetch:
            return None
        
        print(f"       Trying to read: {urls_to_fetch[:3]}")
        
        # Use Jina or direct fetch
        contents = self._fetcher.fetch_urls(urls_to_fetch[:3])
        
        combined = ""
        for url, content in contents.items():
            if content and not content.startswith("Error:") and len(content) > 200:
                combined += f"\n[Source: {url}]\n{content[:8000]}\n"
        
        return combined[:20000] if combined else None
    
    def _convert_github_to_raw(self, github_url: str) -> Optional[str]:
        """Convert GitHub URL to README raw URL"""
        if "github.com" not in github_url:
            return None
        
        # Extract owner/repo
        match = re.search(r'github\.com/([^/]+)/([^/]+)', github_url)
        if match:
            owner, repo = match.groups()
            repo = repo.replace('.git', '')
            return f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md"
        return None
    
    def _generate_structured_summary(
        self, 
        model_name: str, 
        content: str, 
        check_sufficiency: bool = False
    ) -> tuple:
        """
        LLM generates structured technical summary
        
        Args:
            model_name: Model name
            content: Collected text content
            check_sufficiency: Whether to check if info is sufficient
        
        Returns:
            (is_sufficient, summary_text)
        """
        client = self._get_llm_client()
        if not client:
            return (False, "")
        
        # Reference format example
        example = """
Example format (Qwen3-VL-2B-Thinking-FP8):

1. Introduction
This repository contains an FP8 quantized version of the Qwen3-VL-2B-Thinking model. The quantization method is fine-grained fp8 quantization with block size of 128, and its performance metrics are nearly identical to those of the original BF16 model. Qwen3-VL delivers comprehensive upgrades across the board: superior text understanding & generation, deeper visual perception & reasoning, extended context length, enhanced spatial and video dynamics comprehension, and stronger agent interaction capabilities.

2. Key Enhancements
- Visual Agent: Operates PC/mobile GUIs—recognizes elements, understands functions, invokes tools, completes tasks.
- Visual Coding Boost: Generates Draw.io/HTML/CSS/JS from images/videos.
- Advanced Spatial Perception: Judges object positions, viewpoints, and occlusions; provides stronger 2D grounding and enables 3D grounding.
- Long Context & Video Understanding: Native 256K context, expandable to 1M; handles hours-long video with full recall.
- Enhanced Multimodal Reasoning: Excels in STEM/Math—causal analysis and logical, evidence-based answers.
- Upgraded Visual Recognition: Broader, higher-quality pretraining to "recognize everything"—celebrities, products, landmarks, etc.
- Expanded OCR: Supports 32 languages; robust in low light, blur, and tilt.

3. Model Architecture Updates
- Interleaved-MRoPE: Full-frequency allocation over time, width, and height via robust positional embeddings.
- DeepStack: Fuses multi-level ViT features to capture fine-grained details.
- Text–Timestamp Alignment: Precise, timestamp-grounded event localization for stronger video temporal modeling.
"""
        
        sufficiency_instruction = ""
        if check_sufficiency:
            sufficiency_instruction = """
IMPORTANT: First evaluate if the provided content contains enough information to write a meaningful summary.
- If YES: Set "sufficient": true and provide the summary.
- If NO (missing key details about architecture, capabilities, or enhancements): Set "sufficient": false and provide a partial summary with what's available.
"""
        
        prompt = f"""Analyze the following technical content about "{model_name}" and generate a structured summary.

{sufficiency_instruction}

Content:
{content[:12000]}

---

Generate a summary with EXACTLY these 3 sections:

1. Introduction
- Brief description of what the model is
- Key positioning and main capabilities
- Quantization/variant info if applicable

2. Key Enhancements
- List the main improvements and features
- Focus on capabilities and use cases
- Use bullet points

3. Model Architecture Updates
- Technical architecture changes
- Novel components or techniques
- Training methodology if mentioned

{example}

---

Return JSON format:
{{
    "sufficient": true/false,  // Only if checking sufficiency
    "summary": "1. Introduction\\n...\\n\\n2. Key Enhancements\\n...\\n\\n3. Model Architecture Updates\\n..."
}}

Constraints:
- ONLY include information explicitly stated in the content
- Do NOT fabricate or guess missing details
- Write "Not provided in source material" for missing sections
- Keep concise but comprehensive (300-600 words total)

Return JSON only."""
        
        try:
            model = self.llm_config.get("model", "gpt-4o")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0,
            )
            result = response.choices[0].message.content.strip()
            
            # Parse JSON
            if "```" in result:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result)
                result = match.group(1) if match else result
            
            data = json.loads(result)
            is_sufficient = data.get("sufficient", True)
            summary = data.get("summary", "")
            
            return (is_sufficient, summary)
            
        except Exception as e:
            print(f"       LLM summary generation failed: {e}")
            return (False, "")
    
    # ========================================================================
    # Community Feedback (real-time search)
    # ========================================================================
    
    def _fetch_community_feedback(self, features: ModelFeatures, days_after_release: int = 30):
        """Fetch community feedback"""
        model_name = features.model_name
        all_discussions = []
        
        # Calculate time range
        if features.release_date:
            try:
                # Support multiple date formats
                for fmt in ["%Y/%m/%d", "%Y-%m-%d", "%Y/%m/%d"]:
                    try:
                        start_date = datetime.strptime(features.release_date, fmt)
                        break
                    except:
                        continue
                else:
                    start_date = datetime.now() - timedelta(days=days_after_release)
            except:
                start_date = datetime.now() - timedelta(days=days_after_release)
        else:
            start_date = datetime.now() - timedelta(days=days_after_release)
        
        features.time_decay_factor = self._calculate_time_decay(start_date)
        
        # 1. Get HuggingFace discussions
        print(f"  [3.1] Fetching HuggingFace discussions...")
        hf_data = self._get_hf_discussions(model_name, features.huggingface_id)
        if hf_data:
            features.hf_discussions_count = hf_data.get("count", 0)
            all_discussions.extend(hf_data.get("discussions", []))
            print(f"       Found {features.hf_discussions_count} discussions")
        
        # 2. Get web discussions
        print(f"  [3.2] Searching web discussions...")
        web_data = self._get_web_discussions(model_name)
        if web_data:
            features.web_mentions_count = web_data.get("count", 0)
            all_discussions.extend(web_data.get("mentions", []))
            print(f"       Found {features.web_mentions_count} discussions")
        
        # 3. LLM analysis
        if all_discussions:
            print(f"  [3.3] LLM analyzing discussion content...")
            analysis = self._analyze_discussions(model_name, all_discussions)
            if analysis:
                features.positive_aspects = analysis.get("positive", [])
                features.negative_aspects = analysis.get("negative", [])
                features.reported_issues = analysis.get("issues", [])
                features.sentiment_score = analysis.get("sentiment", 0.5)
                features.community_summary = analysis.get("summary", "")
        
        # 4. Calculate confidence and hype level
        total_posts = features.hf_discussions_count + features.web_mentions_count
        features.confidence = min(total_posts / 50, 1.0)
        features.hype_level = self._calculate_hype_level(total_posts)
    
    def _get_hf_discussions(self, model_name: str, hf_repo_id: Optional[str] = None) -> Optional[Dict]:
        """Get discussions from HuggingFace"""
        # Prefer repo ID from database
        repo_id = hf_repo_id or self._find_hf_repo(model_name)
        
        if not repo_id:
            return None
        
        discussions = []
        try:
            url = f"https://huggingface.co/api/models/{repo_id}/discussions"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("discussions", [])
                
                for item in items[:20]:
                    discussions.append({
                        "title": item.get("title", ""),
                        "text": item.get("content", "")[:500] if item.get("content") else "",
                        "status": item.get("status", ""),
                        "source": "huggingface"
                    })
                
                return {"count": len(discussions), "discussions": discussions}
        except Exception as e:
            print(f"       Failed to fetch HuggingFace discussions: {e}")
        
        return None
    
    def _find_hf_repo(self, model_name: str) -> Optional[str]:
        """Find HuggingFace repo ID"""
        HF_API = "https://huggingface.co/api/models"
        
        if "/" in model_name:
            try:
                resp = requests.get(f"{HF_API}/{model_name}", timeout=10)
                if resp.status_code == 200:
                    return model_name
            except:
                pass
            return None
        
        try:
            resp = requests.get(HF_API, params={"search": model_name, "limit": 10}, timeout=10)
            if resp.status_code == 200:
                results = resp.json()
                name_lower = model_name.lower().replace(" ", "-").replace("_", "-")
                
                for result in results:
                    repo_id = result.get("id", "")
                    repo_name = repo_id.split("/")[-1].lower() if "/" in repo_id else repo_id.lower()
                    
                    if repo_name == name_lower or repo_name.startswith(name_lower):
                        return repo_id
        except:
            pass
        
        return None
    
    def _get_web_discussions(self, model_name: str) -> Optional[Dict]:
        """Get web discussions via Google Search"""
        if not self.google_api_key or not self.google_cse_id:
            return None
        
        query = f'"{model_name}" review OR benchmark OR discussion OR experience'
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "num": 10,
            }
            
            resp = requests.get(url, params=params, timeout=10)
            
            if resp.status_code != 200:
                return None
            
            items = resp.json().get("items", [])
            if not items:
                return None
            
            # Extract URLs and fetch content
            urls = [item.get("link") for item in items if item.get("link")]
            contents = self._fetcher.fetch_urls(urls)
            
            # Combine results
            mentions = []
            fetch_success = 0
            
            for item in items:
                url = item.get("link", "")
                snippet = item.get("snippet", "")
                fetched = contents.get(url, "")
                
                is_success = fetched and not fetched.startswith("Error:") and len(fetched) > 100
                
                mentions.append({
                    "title": item.get("title", ""),
                    "text": (fetched if is_success else snippet)[:2000],
                    "url": url,
                    "source": "web",
                    "is_full_content": is_success,
                })
                
                if is_success:
                    fetch_success += 1
            
            print(f"       Fetched: {fetch_success}/{len(urls)} successful")
            return {"count": len(mentions), "mentions": mentions}
            
        except Exception as e:
            print(f"       Google search failed: {e}")
            return None
    
    # ========================================================================
    # LLM Analysis
    # ========================================================================
    
    def _get_llm_client(self):
        """Get OpenAI client"""
        if self._openai_client is None:
            try:
                from openai import OpenAI
                api_key = self.llm_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
                base_url = self.llm_config.get("base_url") or os.environ.get("OPENAI_BASE_URL")
                if api_key:
                    self._openai_client = OpenAI(
                        api_key=api_key,
                        base_url=base_url
                    ) if base_url else OpenAI(api_key=api_key)
            except ImportError:
                print("openai library not installed")
        return self._openai_client
    
    def _analyze_discussions(self, model_name: str, discussions: List[Dict]) -> Optional[Dict]:
        """Use LLM to analyze discussion content"""
        client = self._get_llm_client()
        if not client or not discussions:
            return None
        
        # Build discussion text
        discussion_text = ""
        for d in discussions[:30]:
            title = d.get("title", "")
            text = d.get("text", "")
            source = d.get("source", "")
            discussion_text += f"\n[{source}] {title}\n{text}\n"
        
        if not discussion_text.strip():
            return None
        
        prompt = f"""Please analyze the following community discussions about the "{model_name}" model, and extract key information related specifically to **model performance**.

[Important] Only focus on:
- Model performance on various tasks/benchmarks
- Performance comparison with other models
- Advantages and limitations of the model's capabilities
- Real-world user feedback on effectiveness

[Ignore] Any information unrelated to performance, including:
- Hardware deployment issues
- Installation or configuration problems
- Technical implementation details

Discussion content:
{discussion_text[:10000]}

Please return in JSON format:
{{
    "sentiment": sentiment score between 0.0 and 1.0,
    "positive": ["positive aspect 1", "positive aspect 2", ...] up to 5 items,
    "negative": ["negative aspect 1", "negative aspect 2", ...] up to 5 items,
    "issues": ["issue 1", "issue 2", ...] up to 5 items,
    "summary": "A summary of the discussion"
}}

Return JSON only."""

        try:
            model = self.llm_config.get("model", "gpt-4o")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0,
            )
            result = response.choices[0].message.content.strip()
            
            # Parse JSON
            if "```" in result:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result)
                result = match.group(1) if match else result
            
            analysis = json.loads(result)
            print(f"       LLM analysis complete, sentiment score: {analysis.get('sentiment', 0.5)}")
            return analysis
            
        except Exception as e:
            print(f"       LLM analysis failed: {e}")
            return None
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _calculate_hype_level(self, total_posts: int) -> str:
        """Calculate hype level"""
        if total_posts < 5:
            return "low"
        elif total_posts < 20:
            return "medium"
        else:
            return "high"
    
    def _calculate_time_decay(self, release_date: datetime) -> float:
        """Calculate time decay factor"""
        days_since_release = (datetime.now() - release_date).days
        
        if days_since_release <= 30:
            return 1.0
        elif days_since_release <= 90:
            return 0.75
        elif days_since_release <= 180:
            return 0.5
        else:
            return 0.25
    
    # ========================================================================
    # Batch Query Methods
    # ========================================================================
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Quick model basic info lookup (database only, no search)"""
        db = self._load_model_db()
        
        for org_name, org_data in db.get("organizations", {}).items():
            for family_name, family_data in org_data.get("families", {}).items():
                models = family_data.get("models", {})
                if model_name in models:
                    info = models[model_name].copy()
                    info["organization"] = org_name
                    info["family"] = family_name
                    info["intro_url"] = family_data.get("intro_url")
                    return info
                # Fuzzy match
                matched, name = self._fuzzy_match_model(model_name, models)
                if matched:
                    info = matched.copy()
                    info["organization"] = org_name
                    info["family"] = family_name
                    info["intro_url"] = family_data.get("intro_url")
                    return info
        return None
    
    def get_model_family(self, model_name: str) -> Optional[Dict]:
        """Get model's family"""
        db = self._load_model_db()
        
        for org_name, org_data in db.get("organizations", {}).items():
            for family_name, family_data in org_data.get("families", {}).items():
                models = family_data.get("models", {})
                if model_name in models:
                    return {
                        "family_name": family_name,
                        "organization": org_name,
                        "intro_url": family_data.get("intro_url"),
                        "models": list(models.keys())
                    }
        return None
    
    def list_models_by_org(self, organization: str) -> List[str]:
        """List models by organization"""
        db = self._load_model_db()
        org_lower = organization.lower()
        result = []
        
        for org_name, org_data in db.get("organizations", {}).items():
            if org_lower in org_name.lower():
                for family_data in org_data.get("families", {}).values():
                    result.extend(family_data.get("models", {}).keys())
        return result
    
    def list_open_source_models(self) -> List[str]:
        """List all open-source models"""
        db = self._load_model_db()
        result = []
        
        for org_data in db.get("organizations", {}).values():
            for family_data in org_data.get("families", {}).values():
                for name, info in family_data.get("models", {}).items():
                    if info.get("open_source"):
                        result.append(name)
        return result
    
    def list_families(self, organization: str = None) -> List[Dict]:
        """List all families (optionally filter by organization)"""
        db = self._load_model_db()
        result = []
        
        for org_name, org_data in db.get("organizations", {}).items():
            if organization and organization.lower() not in org_name.lower():
                continue
            for family_name, family_data in org_data.get("families", {}).items():
                result.append({
                    "family_name": family_name,
                    "organization": org_name,
                    "intro_url": family_data.get("intro_url"),
                    "model_count": len(family_data.get("models", {}))
                })
        return result


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    import os
    
    agent = ModelAgent(
        llm_config={
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "model": os.environ.get("LLM_MODEL", "gpt-4o"),
            "base_url": os.environ.get("OPENAI_BASE_URL"),
        },
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        google_cse_id=os.environ.get("GOOGLE_CSE_ID"),
        jina_api_key=os.environ.get("JINA_API_KEY"),
    )
    
    # Test family level (includes community feedback)
    # print(agent.get_family_text("InternVL2"))
    
    # Test model level (technical info only)
    # print(agent.get_model_text("GPT-4o-mini"))
