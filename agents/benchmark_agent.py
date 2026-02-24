"""
Benchmark Agent - Collect benchmark information
Output: BenchmarkFeatures

Data source priority:
1. HuggingFace Datasets API - Get dataset config, README, etc.
2. Extract arXiv ID from README, get paper info
3. If not found on HuggingFace, search via Google
4. LLM extraction and summarization
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import requests
import re
import os
import json

# Import async fetcher
from data_sources.async_fetcher import AsyncFetcher


@dataclass
class BenchmarkFeatures:
    """
    Benchmark Features Data Class
    Stores benchmark information to assist model score prediction
    """
    benchmark_name: str
    
    # === Core Info (5 key fields) ===
    task: Optional[str] = None                    # Task description (1-2 sentences)
    category: Optional[str] = None                # Ability category (vision/reasoning/knowledge/math...)
    evaluation_metric: Optional[str] = None       # Evaluation metric (accuracy/F1/GPT-score...)
    num_samples: Optional[int] = None             # Number of samples
    difficulty: Optional[str] = None              # Difficulty characteristics (1-2 sentences)
    
    # === Subcategories ===
    subcategories: List[str] = field(default_factory=list)  # List of subcategories
    subcategory_descriptions: Dict[str, str] = field(default_factory=dict)  # {subcategory_name: description}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filter empty values"""
        result = {"benchmark_name": self.benchmark_name}
        if self.task:
            result["task"] = self.task
        if self.category:
            result["category"] = self.category
        if self.evaluation_metric:
            result["evaluation_metric"] = self.evaluation_metric
        if self.num_samples:
            result["num_samples"] = self.num_samples
        if self.difficulty:
            result["difficulty"] = self.difficulty
        if self.subcategories:
            result["subcategories"] = self.subcategories
        if self.subcategory_descriptions:
            result["subcategory_descriptions"] = self.subcategory_descriptions
        return result
    
    @staticmethod
    def get_schema() -> Dict[str, str]:
        """Get schema description for LLM prompting"""
        return {
            "task": "Task description (1-2 sentences), explaining input/output format and specific requirements",
            "category": "Ability category, e.g., vision, reasoning, knowledge, math, multimodal, etc.",
            "evaluation_metric": "Evaluation metric and calculation method, such as accuracy, F1, GPT-4 score, etc.",
            "num_samples": "Number of samples (integer)",
            "difficulty": "Task difficulty (1-2 sentences), describing the main challenges of the task",
        }
    
    @staticmethod
    def get_subcategory_schema() -> str:
        """Get schema for subcategory description"""
        return "Provide a 1-sentence description for each subcategory, explaining the specific ability or content scope it tests"
    
    def to_text(self) -> str:
        """Convert to text description (for LLM use)"""
        lines = [f"Benchmark: {self.benchmark_name}"]
        
        if self.category:
            lines.append(f"- Category: {self.category}")
        if self.task:
            lines.append(f"- Task: {self.task}")
        if self.evaluation_metric:
            lines.append(f"- Evaluation Metric: {self.evaluation_metric}")
        if self.num_samples:
            lines.append(f"- Samples: {self.num_samples}")
        if self.difficulty:
            lines.append(f"- Difficulty: {self.difficulty}")
        
        # Subcategory descriptions
        if self.subcategory_descriptions:
            lines.append(f"\nSubcategories ({len(self.subcategory_descriptions)}):")
            for subcat, desc in self.subcategory_descriptions.items():
                lines.append(f"  - {subcat}: {desc}")
        
        return "\n".join(lines)


class BenchmarkAgent:
    """
    Benchmark Information Collection Agent
    
    Workflow (similar to TechnicalAgent):
    1. Precisely match benchmark dataset from HuggingFace Datasets
    2. Extract paper links from dataset card (README)
    3. Summarize both HuggingFace info and paper
    4. Only search the internet if benchmark not found on HuggingFace
    """
    
    # HuggingFace Datasets API
    HF_DATASETS_API = "https://huggingface.co/api/datasets"
    ARXIV_API = "http://export.arxiv.org/api/query"
    
    def __init__(
        self, 
        llm_config: Optional[Dict] = None,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        jina_api_key: Optional[str] = None,
        use_jina: bool = True
    ):
        """
        Initialize BenchmarkAgent
        
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
        
        # Async fetcher with cache
        self._fetcher = AsyncFetcher(
            max_concurrent=10,
            timeout=30,
            use_jina=use_jina and bool(self.jina_api_key or os.environ.get("JINA_API_KEY")),
            jina_api_key=self.jina_api_key or os.environ.get("JINA_API_KEY")
        )
        
        # Cache for fetched benchmark info
        self._cache: Dict[str, BenchmarkFeatures] = {}
        
        # Load benchmark links database (for subcategories)
        self._links_db = self._load_links_db()
    
    def _load_links_db(self) -> Dict:
        """Load benchmark_links_db.json for subcategory info"""
        try:
            from pathlib import Path
            db_path = Path(__file__).parent.parent / "data" / "opencompass_cache" / "benchmark_links_db.json"
            if db_path.exists():
                with open(db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Failed to load benchmark_links_db.json: {e}")
        return {"benchmarks": {}, "variants": {}}
    
    def _get_subcategories(self, benchmark_name: str) -> List[str]:
        """Get subcategories for a benchmark from links_db"""
        # Check in benchmarks
        if benchmark_name in self._links_db.get("benchmarks", {}):
            return self._links_db["benchmarks"][benchmark_name].get("subcategories", [])
        # Check in variants
        if benchmark_name in self._links_db.get("variants", {}):
            return self._links_db["variants"][benchmark_name].get("subcategories", [])
        return []
    
    # ========================================================================
    # Public Methods
    # ========================================================================
        
    def get_features(self, benchmark_name: str) -> BenchmarkFeatures:
        """
        Collect benchmark features (main entry point)
        
        Workflow:
        1. Precisely match benchmark dataset from HuggingFace Datasets
        2. Extract paper links from dataset card (README)
        3. Summarize both HuggingFace info and paper
        4. Only search the internet if benchmark not found on HuggingFace
        
        Each step progressively enriches the features.
        
        Args:
            benchmark_name: Benchmark name, e.g., "MMLU", "HumanEval"
            
        Returns:
            BenchmarkFeatures: Benchmark features
        """
        # Check cache
        if benchmark_name in self._cache:
            print(f"Using cached info for {benchmark_name}")
            return self._cache[benchmark_name]
        
        features = BenchmarkFeatures(benchmark_name=benchmark_name)
        readme = None
        paper_content = None
        all_content = []  # Collect all content for subcategory description generation
        
        print(f"Starting benchmark info collection: {benchmark_name}")
        
        # =====================================================================
        # 1. Get info from HuggingFace Datasets (priority)
        # =====================================================================
        print(f"------------------1. Fetching from HuggingFace Datasets------------------")
        readme = self._get_huggingface_readme(benchmark_name)
        
        if readme:
            all_content.append(f"[HuggingFace README]\n{readme[:8000]}")
            print(f"[LLM] Extracting info from HuggingFace README...")
            extracted = self._extract_info_with_llm(benchmark_name, readme, "huggingface", features)
            if extracted:
                self._apply_extracted_info(features, extracted)
                print(f"[Step 1] Features enriched from HuggingFace")
        
        # =====================================================================
        # 2. Extract arXiv ID from README to get paper info
        # =====================================================================
        print(f"------------------2. Extracting paper links from README------------------")
        arxiv_id = self._find_arxiv_id_from_readme(readme) if readme else None
        
        if arxiv_id:
            print(f"Extracted arXiv ID from README: {arxiv_id}")
            paper_info = self._get_arxiv_paper_by_id(arxiv_id)
            if paper_info:
                # Use paper abstract to enrich features
                paper_content = f"Title: {paper_info.get('title', '')}\nAbstract: {paper_info.get('abstract', '')}"
                all_content.append(f"[arXiv Paper]\n{paper_content}")
                
                print(f"[LLM] Enriching info from arXiv paper...")
                extracted = self._extract_info_with_llm(benchmark_name, paper_content, "arxiv", features)
                if extracted:
                    self._apply_extracted_info(features, extracted)
                    print(f"[Step 2] Features enriched from arXiv")
        else:
            print(f"No arXiv link found in README")
        
        # =====================================================================
        # 3. If not found on HuggingFace, search Google
        # =====================================================================
        web_content = None
        if not readme:
            print(f"------------------3. Not found on HuggingFace, searching Google------------------")
            web_info = self._search_google(benchmark_name)
            if web_info:
                web_content = "\n\n".join([
                    f"[{item.get('title', '')}]: {item.get('content', '')[:2000]}" 
                    for item in web_info[:5]
                ])
                all_content.append(f"[Web Search]\n{web_content[:8000]}")
                print(f"Retrieved {len(web_info)} web pages")
                print(f"[LLM] Extracting info from Google search results...")
                extracted = self._extract_info_with_llm(benchmark_name, web_content, "google", features)
                if extracted:
                    self._apply_extracted_info(features, extracted)
                    print(f"[Step 3] Features enriched from Google")
        else:
            print(f"------------------3. Found on HuggingFace, skipping Google search------------------")
        
        # =====================================================================
        # 4. Load subcategories and generate descriptions (based on collected content)
        # =====================================================================
        print(f"------------------4. Loading subcategories and generating descriptions------------------")
        subcategories = self._get_subcategories(benchmark_name)
        if subcategories:
            features.subcategories = subcategories
            print(f"Found {len(subcategories)} subcategories: {subcategories[:5]}{'...' if len(subcategories) > 5 else ''}")
            
            # Generate descriptions based on collected content (README + paper + web)
            if all_content:
                combined_content = "\n\n".join(all_content)
                descriptions = self._generate_subcategory_descriptions(
                    benchmark_name, 
                    subcategories, 
                    combined_content  # Pass collected content
                )
                if descriptions:
                    features.subcategory_descriptions = descriptions
                    print(f"Generated descriptions for {len(descriptions)} subcategories")
            else:
                print(f"No content collected, skipping subcategory description generation")
        else:
            print(f"No subcategories found for {benchmark_name}")
        
        # Cache result
        self._cache[benchmark_name] = features
        
        print(f"--------------------Benchmark info collection completed--------------------")
        return features
    
    def _apply_extracted_info(self, features: BenchmarkFeatures, extracted: Dict):
        """
        Apply extracted info to features object.
        Only updates fields that are currently empty/None (won't overwrite existing values).
        """
        # Helper function: only update if current value is None/empty
        def update_if_empty(current, new_value):
            if current is None or current == "":
                return new_value
            return current
        
        # Core fields (5 key fields after simplification)
        features.task = update_if_empty(features.task, extracted.get("task"))
        features.category = update_if_empty(features.category, extracted.get("category"))
        features.evaluation_metric = update_if_empty(
            features.evaluation_metric, 
            extracted.get("evaluation_metric") or extracted.get("metric")
        )
        features.difficulty = update_if_empty(
            features.difficulty, 
            extracted.get("difficulty") or extracted.get("difficulty_characteristics")
        )
        
        # Numeric fields
        if features.num_samples is None and extracted.get("num_samples"):
            features.num_samples = extracted.get("num_samples")
    
    def _generate_subcategory_descriptions(
        self, 
        benchmark_name: str, 
        subcategories: List[str],
        source_content: str
    ) -> Dict[str, str]:
        """
        Use LLM to generate descriptions for each subcategory based on collected content
        
        Args:
            benchmark_name: Benchmark name
            subcategories: List of subcategory names
            source_content: Collected content from HuggingFace/arXiv/Web (used as reference)
        
        Returns:
            Dict mapping subcategory name to description
        """
        client = self._get_llm_client()
        if not client:
            print("LLM client not configured, skipping subcategory descriptions")
            return {}
        
        if not subcategories:
            return {}
        
        # Build prompt
        subcats_str = "\n".join([f"- {s}" for s in subcategories])
        
        prompt = f"""Based on the reference materials below, please provide a one-sentence description for each subcategory of benchmark "{benchmark_name}".

Reference materials:
{source_content[:15000]}

Subcategory list:
{subcats_str}

Instructions:
1. Write one sentence (15-30 words) describing the specific ability or content tested by each subcategory.
2. Descriptions should be based on information found in the reference materials, do not make wild guesses.
3. If certain subcategories are not mentioned in the materials, make a reasonable inference based on the subcategory name.
4. Return the result in JSON format, with key as the subcategory name and value as its description.

Example output format:
{{
  "Color": "Tests the model's ability to recognize and understand color information in images.",
  "Count": "Tests the model's ability to count the number of objects in an image.",
  "OCR": "Tests the model's ability to recognize and understand textual content in images."
}}

Return only the JSON object, and nothing else."""

        try:
            model = self.llm_config.get("model", "gpt-4o")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3,
            )
            result = response.choices[0].message.content.strip()
            
            # Parse JSON
            if "```" in result:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result)
                result = match.group(1) if match else result
            
            descriptions = json.loads(result)
            
            # Validate: only keep descriptions for known subcategories
            valid_descriptions = {
                k: v for k, v in descriptions.items() 
                if k in subcategories and isinstance(v, str)
            }
            
            return valid_descriptions
            
        except Exception as e:
            print(f"Failed to generate subcategory descriptions: {e}")
            return {}
    
    def get_features_text(self, benchmark_name: str) -> str:
        """Get text description of benchmark features"""
        features = self.get_features(benchmark_name)
        return features.to_text()
    
    # ========================================================================
    # HuggingFace Datasets Methods
    # ========================================================================
    
    def _get_huggingface_readme(self, benchmark_name: str) -> Optional[str]:
        """
        Get README from HuggingFace Datasets API
        
        Returns:
            (info_dict, readme_text) tuple
        """
        dataset_id = self._find_hf_dataset(benchmark_name)
        if not dataset_id:
            return None, None
        
        try:
            resp = requests.get(f"{self.HF_DATASETS_API}/{dataset_id}", timeout=10)
            
            if resp.status_code != 200:
                print(f"HuggingFace API returned {resp.status_code}")
                return None, None
            
            
            # Get README (dataset card)
            readme = self._get_dataset_readme(dataset_id)
            if readme:
                print(f"Retrieved dataset card README, length: {len(readme)} chars")
            else:
                print(f"No dataset card README found")
            
            return readme
            
        except Exception as e:
            print(f"HuggingFace API error: {e}")
            return None, None
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize name for matching (ignore case, underscores, hyphens)
        
        Example: "TextVQA_VAL" -> "textvqaval"
                 "textvqa-val" -> "textvqaval"
        """
        return name.lower().replace(" ", "").replace("_", "").replace("-", "").replace(".", "")
    
    def _find_hf_dataset(self, benchmark_name: str) -> Optional[str]:
        """
        Find HuggingFace dataset by benchmark name (precise matching)
        
        Matching strategy:
        1. Search benchmark name directly
        2. Precise match dataset name (ignore case, underscores, hyphens)
        3. Return the highest download count match
        """
        try:
            # Search datasets (try both original and lowercase)
            search_terms = [benchmark_name, benchmark_name.lower()]
            
            all_results = []
            for term in search_terms:
                resp = requests.get(
                    self.HF_DATASETS_API, 
                    params={"search": term, "limit": 20},
                    timeout=10
                )
                
                if resp.status_code == 200:
                    results = resp.json()
                    all_results.extend(results)
            
            if not all_results:
                print(f"No related datasets found on HuggingFace for {benchmark_name}")
                return None
            
            # Deduplicate by dataset_id
            seen_ids = set()
            unique_results = []
            for result in all_results:
                dataset_id = result.get("id", "")
                if dataset_id and dataset_id not in seen_ids:
                    seen_ids.add(dataset_id)
                    unique_results.append(result)
            
            # Normalize benchmark name (ignore case, underscores, hyphens)
            name_normalized = self._normalize_name(benchmark_name)
            
            # Exact match: dataset name equals or starts/ends with benchmark name
            exact_matches = []
            for result in unique_results:
                dataset_id = result.get("id", "")
                dataset_name = dataset_id.split("/")[-1] if "/" in dataset_id else dataset_id
                dataset_name_normalized = self._normalize_name(dataset_name)
                
                # Exact match (normalized strings are equal)
                if dataset_name_normalized == name_normalized:
                    print(f"Exact match HuggingFace dataset: {dataset_id}")
                    return dataset_id
                
                # Partial match (normalized starts/ends with benchmark name)
                if dataset_name_normalized.startswith(name_normalized) or dataset_name_normalized.endswith(name_normalized):
                    exact_matches.append((dataset_id, result.get("downloads", 0)))
            
            # Return highest download partial match
            if exact_matches:
                exact_matches.sort(key=lambda x: x[1], reverse=True)
                best_match = exact_matches[0][0]
                print(f"Found HuggingFace dataset (partial match): {best_match}")
                return best_match
            
            # If no exact match, check if first result is related
            first_result = unique_results[0]
            dataset_id = first_result.get("id", "")
            dataset_name = dataset_id.split("/")[-1] if "/" in dataset_id else dataset_id
            dataset_name_normalized = self._normalize_name(dataset_name)
            
            # Check if contains benchmark name
            if name_normalized in dataset_name_normalized:
                print(f"Found HuggingFace dataset (contains match): {dataset_id}")
                return dataset_id
            
            print(f"No exact match found on HuggingFace: {benchmark_name}")
            return None
            
        except Exception as e:
            print(f"HuggingFace search error: {e}")
            return None
    
    def _get_dataset_readme(self, dataset_id: str) -> Optional[str]:
        """Get dataset README.md (dataset card), strip YAML front matter"""
        try:
            url = f"https://huggingface.co/datasets/{dataset_id}/raw/main/README.md"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                content = resp.text
                
                # Strip YAML front matter (between --- markers)
                if content.startswith('---'):
                    # Find the closing ---
                    end_marker = content.find('---', 3)
                    if end_marker != -1:
                        # Skip YAML and return only Markdown content
                        content = content[end_marker + 3:].strip()
                
                return content
        except:
            pass
        return None
    
    def _find_arxiv_id_from_readme(self, readme: str) -> Optional[str]:
        """
        Extract arXiv ID from README
        
        Supported formats:
        - arxiv:2009.03300
        - https://arxiv.org/abs/2009.03300
        - arXiv:2009.03300v1
        """
        if not readme:
            return None
        
        # Match various arXiv ID formats
        patterns = [
            r'arxiv[:/](?:abs/)?(\d{4}\.\d{4,5}(?:v\d+)?)',  # arxiv:2009.03300 or arxiv.org/abs/2009.03300
            r'https?://arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)',  # full URL
            r'\[(\d{4}\.\d{4,5})\]',  # [2009.03300]
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, readme, re.IGNORECASE)
            if matches:
                # Return first match (remove version number)
                arxiv_id = matches[0].split('v')[0]
                return arxiv_id
        
        return None
    
    def _get_arxiv_paper_by_id(self, arxiv_id: str) -> Optional[Dict]:
        """
        Get paper info by arXiv ID
        
        Args:
            arxiv_id: arXiv paper ID, e.g., "2009.03300"
            
        Returns:
            Paper info dictionary
        """
        try:
            resp = requests.get(
                self.ARXIV_API, 
                params={"id_list": arxiv_id},
                timeout=10
            )
            
            if resp.status_code != 200:
                return None
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            entry = root.find('atom:entry', ns)
            if entry is None:
                return None
            
            title = entry.find('atom:title', ns)
            abstract = entry.find('atom:summary', ns)
            
            return {
                "title": title.text.strip().replace('\n', ' ') if title is not None else "",
                "abstract": abstract.text.strip().replace('\n', ' ') if abstract is not None else "",
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "arxiv_id": arxiv_id
            }
            
        except Exception as e:
            print(f"Failed to get arXiv paper: {e}")
            return None
    
    # ========================================================================
    # Google Search
    # ========================================================================
    
    def _search_google(self, benchmark_name: str) -> Optional[List[Dict]]:
        """
        Search benchmark's official page and related info via Google
        """
        if not self.google_api_key or not self.google_cse_id:
            print("Google Search API not configured, skipping")
            return None
        
        query = f'"{benchmark_name}" benchmark dataset evaluation'
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "num": 5,
            }
            
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                print(f"Google search failed: {resp.status_code}")
                return None
            
            items = resp.json().get("items", [])
            if not items:
                return None
            
            # Fetch web page content
            urls = [item.get("link") for item in items if item.get("link")]
            contents = self._fetcher.fetch_urls(urls)
            
            results = []
            for item in items:
                url = item.get("link", "")
                content = contents.get(url, item.get("snippet", ""))
                
                results.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "snippet": item.get("snippet", ""),
                    "content": content[:3000] if content else item.get("snippet", "")
                })
            
            return results
            
        except Exception as e:
            print(f"Google search failed: {e}")
            return None
    
    # ========================================================================
    # LLM Info Extraction
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
    
    def _extract_info_with_llm(
        self, 
        benchmark_name: str, 
        content: str, 
        source: str,
        current_features: Optional[BenchmarkFeatures] = None
    ) -> Optional[Dict]:
        """
        Use LLM to extract benchmark info from paper/web content.
        If current_features is provided, LLM will focus on filling missing fields.
        
        Args:
            benchmark_name: Name of the benchmark
            content: Content to extract info from
            source: Source type (huggingface, arxiv, google)
            current_features: Current features to enrich (optional)
        """
        client = self._get_llm_client()
        if not client:
            print("LLM client not configured")
            return None
        
        if not content:
            print(f"No content to extract from {source}")
            return None
        
        schema = BenchmarkFeatures.get_schema()
        schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
        
        # Build current state description
        current_state_text = "None"
        if current_features:
            current_dict = current_features.to_dict()
            current_dict.pop("benchmark_name", None)
            current_dict.pop("subcategories", None)
            current_dict.pop("subcategory_descriptions", None)
            filled_fields = {k: v for k, v in current_dict.items() if v is not None and v != ""}
            if filled_fields:
                current_state_text = json.dumps(filled_fields, indent=2, ensure_ascii=False)
        
        prompt = f"""Please extract the key feature information for the benchmark "{benchmark_name}" from the following content.

Known information:
{current_state_text}

Fields to extract:
{schema_text}

Source ({source}) content:
{content[:12000]}

Please return the result in JSON format, following these rules:
- Return only JSON, do not include any other text.
- Use null for fields that cannot be determined.
- Focus on supplementing the fields that are still unknown.

[IMPORTANT] Each field must have a concrete description:
- task: Do not just write "QA"; describe the specific task, e.g., "Given an image and a question, the model is required to count the number of objects in the image and answer."
- difficulty: Describe the task's challenges, e.g., "The task requires precise recognition of small objects and handling occlusions."
- evaluation_metric: Describe the evaluation method, e.g., "Accuracy is used for evaluation, and the answer must exactly match the annotation."

Ensure each field has a specific description of 1-2 sentences."""
        try:
            model = self.llm_config.get("model", "gpt-4o")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0,
            )
            result = response.choices[0].message.content.strip()
            
            # Parse JSON
            if "```" in result:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result)
                result = match.group(1) if match else result
            
            extracted = json.loads(result)
            print(f"LLM extraction completed, category: {extracted.get('category')}")
            return extracted
            
        except Exception as e:
            print(f"LLM info extraction failed: {e}")
            return None
    
    def clear_cache(self):
        """Clear cache"""
        self._cache.clear()
        self._fetcher.clear_cache()
