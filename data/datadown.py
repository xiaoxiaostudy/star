"""
Download OpenCompass OpenVLM Leaderboard data and organize into a table.
"""

import requests
import json
import pandas as pd
from pathlib import Path

# Data URL
OPENCOMPASS_VLM_URL = "http://opencompass.openxlab.space/assets/OpenVLM.json"

def download_opencompass_data():
    """Download OpenCompass VLM data"""
    print(f"Downloading data: {OPENCOMPASS_VLM_URL}")
    response = requests.get(OPENCOMPASS_VLM_URL, timeout=60)
    response.raise_for_status()
    data = response.json()
    print(f"Download complete, data updated at: {data.get('time', 'unknown')}")
    return data

def parse_to_dataframe(data: dict) -> pd.DataFrame:
    """
    Parse OpenCompass JSON data into a DataFrame.
    
    Returns:
        DataFrame where each row is a model, each column is a benchmark Overall score.
    """
    results = data.get("results", {})
    
    rows = []
    all_benchmarks = set()
    
    for model_name, model_data in results.items():
        row = {"Model": model_name}
        
        # Extract META info
        meta = model_data.get("META", {})
        row["Parameters"] = meta.get("Parameters", "")
        row["Organization"] = meta.get("Org", "")
        row["OpenSource"] = meta.get("OpenSource", "")
        row["Time"] = meta.get("Time", "")
        
        # Extract Overall scores for each benchmark
        for key, value in model_data.items():
            if key == "META":
                continue
            if isinstance(value, dict):
                # Get Overall score (if available)
                if "Overall" in value:
                    benchmark_name = key
                    score = value["Overall"]
                    # Handle "N/A" cases
                    if score != "N/A" and score is not None:
                        try:
                            row[benchmark_name] = float(score)
                            all_benchmarks.add(benchmark_name)
                        except (ValueError, TypeError):
                            pass
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by model name
    df = df.sort_values("Model").reset_index(drop=True)
    
    return df, list(all_benchmarks)

def main():
    # Download data
    data = download_opencompass_data()
    
    # Parse into DataFrame
    df, benchmarks = parse_to_dataframe(data)
    
    print(f"\n{'='*60}")
    print(f"Data Statistics:")
    print(f"{'='*60}")
    print(f"Number of models: {len(df)}")
    print(f"Number of benchmarks: {len(benchmarks)}")
    print(f"\nBenchmark list:")
    for b in sorted(benchmarks):
        non_null = df[b].notna().sum() if b in df.columns else 0
        print(f"  - {b}: {non_null} models with scores")
    
    # Save full data
    output_dir = Path(__file__).parent / "opencompass_cache"
    output_dir.mkdir(exist_ok=True)
    
    # Save full CSV
    csv_path = output_dir / "opencompass_vlm_full.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull data saved to: {csv_path}")
    
    # Keep only benchmark score columns (for matrix factorization)
    meta_cols = ["Model", "Parameters", "Organization", "OpenSource", "Time"]
    benchmark_cols = [c for c in df.columns if c not in meta_cols]
    
    # Create pure score matrix
    score_df = df[["Model"] + benchmark_cols].copy()
    score_csv_path = output_dir / "opencompass_vlm_scores.csv"
    score_df.to_csv(score_csv_path, index=False)
    print(f"Score matrix saved to: {score_csv_path}")
    
    # Save raw JSON
    json_path = output_dir / "opencompass_vlm_raw.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Raw JSON saved to: {json_path}")
    
    # Preview first 20 rows
    print(f"\n{'='*60}")
    print(f"Data Preview (first 20 models):")
    print(f"{'='*60}")
    
    # Select main benchmarks for display
    main_benchmarks = [
        "MMBench_TEST_EN_V11", "MMBench_TEST_CN_V11", 
        "MMMU_VAL", "MathVista", "MMVet", "AI2D", "OCRBench", "MMStar"
    ]
    display_cols = ["Model", "Organization", "Parameters"] + [b for b in main_benchmarks if b in df.columns]
    
    preview_df = df[display_cols].head(20)
    
    # Set pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    print(preview_df.to_string(index=False))
    
    # Display score distribution statistics
    print(f"\n{'='*60}")
    print(f"Main Benchmark Score Statistics:")
    print(f"{'='*60}")
    for b in main_benchmarks:
        if b in df.columns:
            stats = df[b].describe()
            print(f"\n{b}:")
            print(f"  Valid data: {int(stats['count'])} models")
            print(f"  Mean: {stats['mean']:.1f}")
            print(f"  Max: {stats['max']:.1f}")
            print(f"  Min: {stats['min']:.1f}")
            print(f"  Std: {stats['std']:.1f}")

    return df

if __name__ == "__main__":
    df = main()
