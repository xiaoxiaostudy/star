"""
CPMF Training Script for VLM Datasets (Constrained Probabilistic Matrix Factorization)
"""
import argparse
import json
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats

# Add crosspred directory to sys.path so method/ and utils/ can be imported
sys.path.insert(0, str(Path(__file__).parent))

from method.baseline_pmf import UniformRandomBaseline, GlobalMeanBaseline, MeanOfMeansBaseline
from method.pmf import PMF
from method.pmf_with_profile import CPMF
from utils.metric import rmse

# Project root & data directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "opencompass_cache"


def parse_parameters(param_str):
    """Parse parameter count string, return numeric value (unit: B)"""
    if param_str is None or pd.isna(param_str) or param_str == '':
        return 0.0
    param_str = str(param_str).strip()
    match = re.search(r'([\d.]+)\s*[Bb]?', param_str)
    return float(match.group(1)) if match else 0.0


class FeatureExtractor:
    """Extract model and benchmark features from feature databases"""

    def __init__(self, models_db_path, benchmarks_db_path):
        with open(models_db_path, 'r') as f:
            self.models_db = json.load(f)
        with open(benchmarks_db_path, 'r') as f:
            self.benchmarks_db = json.load(f)

        self.model_features = {}
        self._build_model_features()
        self.benchmark_features = {}
        self._build_benchmark_features()
        self._build_category_maps()

    def _build_model_features(self):
        for family_name, family_data in self.models_db.get('model_families', {}).items():
            for model_name, model_data in family_data.get('models', {}).items():
                self.model_features[model_name] = {
                    'family': family_name,
                    'organization': family_data.get('organization', 'Unknown'),
                    'parameters': parse_parameters(model_data.get('parameters', '0')),
                    'vision_model': family_data.get('base_vision_model', 'Unknown'),
                    'language_model': family_data.get('base_language_model', 'Unknown'),
                    'open_source': 1.0 if model_data.get('open_source', False) else 0.0,
                    'sentiment_score': model_data.get('sentiment_score', 0.5),
                }

    def _build_benchmark_features(self):
        for bench_name, bench_data in self.benchmarks_db.get('benchmarks', {}).items():
            num_samples = bench_data.get('num_samples', 0)
            if isinstance(num_samples, str):
                num_samples = int(re.sub(r'[^\d]', '', num_samples) or 0)
            subcategories = bench_data.get('subcategories', [])
            self.benchmark_features[bench_name] = {
                'category': bench_data.get('category', 'unknown'),
                'num_samples': num_samples,
                'num_subcategories': len(subcategories) if subcategories else 0,
            }

    def _build_category_maps(self):
        organizations = set(f.get('organization', 'Unknown') for f in self.model_features.values())
        self.org_to_idx = {org: i for i, org in enumerate(sorted(organizations))}
        vision_models = set(f.get('vision_model', 'Unknown') for f in self.model_features.values())
        self.vision_to_idx = {v: i for i, v in enumerate(sorted(vision_models))}
        language_models = set(f.get('language_model', 'Unknown') for f in self.model_features.values())
        self.lang_to_idx = {l: i for i, l in enumerate(sorted(language_models))}
        categories = set(f.get('category', 'unknown') for f in self.benchmark_features.values())
        self.category_to_idx = {c: i for i, c in enumerate(sorted(categories))}

    def get_model_profile(self, model_name, csv_row=None):
        """Get model feature vector: [parameters, organization one-hot]"""
        features = []
        if model_name in self.model_features:
            mf = self.model_features[model_name]
            features.append(mf['parameters'] / 100.0)
            org_onehot = [0.0] * len(self.org_to_idx)
            org_onehot[self.org_to_idx.get(mf['organization'], 0)] = 1.0
            features.extend(org_onehot)
        else:
            if csv_row is not None:
                features.append(parse_parameters(csv_row.get('Parameters', '0')) / 100.0)
            else:
                features.append(0.0)
            features.extend([0.0] * len(self.org_to_idx))
        return np.array(features, dtype=np.float32)

    def get_benchmark_profile(self, bench_name):
        """Get benchmark feature vector"""
        features = []
        if bench_name in self.benchmark_features:
            bf = self.benchmark_features[bench_name]
            features.append(bf['num_samples'] / 50000.0)
            features.append(bf['num_subcategories'] / 50.0)
            cat_onehot = [0.0] * len(self.category_to_idx)
            cat_onehot[self.category_to_idx.get(bf['category'], 0)] = 1.0
            features.extend(cat_onehot)
        else:
            features.extend([0.0, 0.0])
            features.extend([0.0] * len(self.category_to_idx))
        return np.array(features, dtype=np.float32)


class VLMMatrixManager:
    """Matrix manager adapted for VLM datasets"""

    def __init__(self, train_path, test_path, feature_extractor=None):
        self.train_path = train_path
        self.test_path = test_path
        self.feature_extractor = feature_extractor

    def load_data(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        meta_cols = ['Model', 'Parameters', 'Organization', 'OpenSource', 'Time']
        benchmark_cols = [col for col in train_df.columns if col not in meta_cols]

        print(f"Training models: {len(train_df)}, Test models: {len(test_df)}, Benchmarks: {len(benchmark_cols)}")

        train_matrix = train_df[benchmark_cols].values.astype(float)
        test_matrix = test_df[benchmark_cols].values.astype(float)

        train = train_matrix.copy()
        test = np.ones_like(train_matrix) * np.nan

        train_model_to_idx = {model: idx for idx, model in enumerate(train_df['Model'])}
        for idx, model in enumerate(test_df['Model']):
            if model in train_model_to_idx:
                test[train_model_to_idx[model], :] = test_matrix[idx, :]

        self.train_df = train_df
        self.test_df = test_df
        self.benchmark_cols = benchmark_cols
        self.model_names = train_df['Model'].tolist()

        train_nonnull = (~np.isnan(train)).sum()
        test_nonnull = (~np.isnan(test)).sum()
        print(f"Training samples: {train_nonnull}, Test samples: {test_nonnull}")

        return train, test, benchmark_cols, self.model_names

    def get_model_profiles(self):
        if self.feature_extractor is None:
            return None
        profiles = []
        for i, model_name in enumerate(self.model_names):
            csv_row = self.train_df.iloc[i].to_dict()
            profiles.append(self.feature_extractor.get_model_profile(model_name, csv_row))
        profiles = np.array(profiles)
        print(f"Model feature dimensions: {profiles.shape}")
        return profiles

    def get_benchmark_profiles(self):
        if self.feature_extractor is None:
            return None
        profiles = []
        for bench_name in self.benchmark_cols:
            profiles.append(self.feature_extractor.get_benchmark_profile(bench_name))
        profiles = np.array(profiles)
        print(f"Benchmark feature dimensions: {profiles.shape}")
        return profiles

    def normalize_data(self, train, test):
        mu = np.nanmean(train, axis=0, keepdims=True)
        sigma = np.nanstd(train, axis=0, keepdims=True)
        num_samples = (~np.isnan(train)).sum(axis=0, keepdims=True)
        mu[num_samples < 3] = np.nan
        sigma[num_samples < 3] = np.nan
        global_mean = np.nanmean(train)
        global_std = np.nanstd(train)
        mu[np.isnan(mu)] = global_mean
        sigma[np.isnan(sigma)] = global_std
        sigma[sigma == 0] = 1.0
        train_norm = (train - mu) / sigma
        test_norm = (test - mu) / sigma
        return train_norm, test_norm, mu, sigma


def normalize_score_to_100(score, benchmark_name):
    """Normalize score to 0-100"""
    BENCHMARK_RANGES = {"MME": (0, 2800), "LLaVABench": (0, 120)}
    min_s, max_s = BENCHMARK_RANGES.get(benchmark_name, (0, 100))
    return np.clip((score - min_s) / (max_s - min_s) * 100, 0, 100)


def calculate_all_metrics(pred, test, mu, sigma, train_original, benchmark_cols=None):
    """Calculate evaluation metrics: MAE, RMSE (0-100), SRCC, KRCC, MAE@3"""
    pred_original = pred * sigma + mu
    test_original = test * sigma + mu

    nan_mask = np.isnan(test)
    overall_pred_orig = pred_original[~nan_mask]
    overall_test_orig = test_original[~nan_mask]

    if len(overall_pred_orig) == 0:
        return {'mae': 0, 'rmse': 0, 'srcc': 0, 'krcc': 0, 'mae3': 0}, pred_original, test_original

    # Normalize to 0-100
    n_models, n_benchmarks = test.shape
    pred_norm_list, test_norm_list = [], []
    for j in range(n_benchmarks):
        bench_name = benchmark_cols[j] if benchmark_cols else f"bench_{j}"
        for i in range(n_models):
            if not np.isnan(test[i, j]):
                pred_norm_list.append(normalize_score_to_100(pred_original[i, j], bench_name))
                test_norm_list.append(normalize_score_to_100(test_original[i, j], bench_name))

    pred_norm = np.array(pred_norm_list)
    test_norm = np.array(test_norm_list)
    mae = np.mean(np.abs(pred_norm - test_norm))
    rmse_val = np.sqrt(np.mean((pred_norm - test_norm) ** 2))

    srcc, _ = stats.spearmanr(overall_pred_orig, overall_test_orig)
    krcc, _ = stats.kendalltau(overall_pred_orig, overall_test_orig)

    # MAE@3
    all_rank_errors = []
    for j in range(n_benchmarks):
        mask = ~np.isnan(test[:, j])
        if mask.sum() < 2:
            continue
        pred_scores = pred_original[mask, j]
        true_scores = test_original[mask, j]
        pred_ranks = (-pred_scores).argsort().argsort() + 1
        true_ranks = (-true_scores).argsort().argsort() + 1
        all_rank_errors.extend(np.abs(pred_ranks - true_ranks).tolist())
    mae3 = (np.array(all_rank_errors) <= 3).mean() if all_rank_errors else 0.0

    metrics = {'mae': mae, 'rmse': rmse_val, 'srcc': srcc, 'krcc': krcc, 'mae3': mae3}
    return metrics, pred_original, test_original


def compute_predictions_with_uncertainty(model, mu, sigma):
    """Compute prediction mean and standard deviation (uncertainty) from MCMC samples"""
    all_samples = []
    is_cpmf = "U_comb" in model.trace.posterior
    for cnt in model.trace.posterior.draw.values:
        if is_cpmf:
            U = model.trace.posterior["U_comb"].sel(chain=0, draw=cnt)
            V = model.trace.posterior["V_comb"].sel(chain=0, draw=cnt)
        else:
            U = model.trace.posterior["U"].sel(chain=0, draw=cnt)
            V = model.trace.posterior["V"].sel(chain=0, draw=cnt)
        sample_R_orig = model.predict(U, V) * sigma + mu
        all_samples.append(sample_R_orig)
    all_samples = np.array(all_samples)
    return np.mean(all_samples, axis=0), np.std(all_samples, axis=0)


def get_latent_matrices(model):
    """Extract posterior mean of latent factor matrices from trained CPMF/PMF model"""
    is_cpmf = "U_comb" in model.trace.posterior
    key_u, key_v = ("U_comb", "V_comb") if is_cpmf else ("U", "V")
    U = model.trace.posterior[key_u].mean(dim=["chain", "draw"]).values
    V = model.trace.posterior[key_v].mean(dim=["chain", "draw"]).values
    return U, V


class ColdStartPredictor:
    """Feature-based cold start predictor"""

    def __init__(self, model_profiles, U_matrix, k_neighbors=5):
        self.model_profiles = model_profiles
        self.U_matrix = U_matrix
        self.k_neighbors = k_neighbors

    def predict_scores(self, new_profile, V_matrix):
        # Cosine similarity
        new_norm = new_profile / (np.linalg.norm(new_profile) + 1e-8)
        train_norms = self.model_profiles / (np.linalg.norm(self.model_profiles, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(train_norms, new_norm)

        top_k_idx = np.argsort(similarities)[-self.k_neighbors:]
        weights = np.maximum(similarities[top_k_idx], 0) + 1e-8
        weights = weights / weights.sum()

        u_new = np.average(self.U_matrix[top_k_idx], weights=weights, axis=0)
        predictions = np.dot(u_new, V_matrix.T)

        similar_preds = np.dot(self.U_matrix[top_k_idx], V_matrix.T)
        uncertainty = np.std(similar_preds, axis=0)

        return predictions, uncertainty, {'similar_model_indices': top_k_idx, 'weights': weights}


def save_predictions_table(pred_original, test_original, model_names, benchmark_cols, output_path, std_original=None):
    """Save prediction comparison table (test set models only)"""
    test_model_mask = ~np.isnan(test_original).all(axis=1)
    test_model_indices = np.where(test_model_mask)[0]
    test_model_names = [model_names[i] for i in test_model_indices]

    wide_data = {'Model': test_model_names}
    for j, bench in enumerate(benchmark_cols):
        col_values = []
        for i in test_model_indices:
            pred_val = pred_original[i, j]
            true_val = test_original[i, j]
            if np.isnan(true_val):
                col_values.append("-")
            elif std_original is not None:
                col_values.append(f"[{pred_val:.2f}±{std_original[i, j]:.2f}]/[{true_val:.2f}]")
            else:
                col_values.append(f"[{pred_val:.2f}]/[{true_val:.2f}]")
        wide_data[bench] = col_values

    pd.DataFrame(wide_data).to_csv(output_path, index=False)
    print(f"Prediction comparison table saved to: {output_path}")


def main(args):
    np.random.seed(args.random_seed)

    print("=" * 60)
    print("CrossPred (CPMF) for VLM Benchmark")
    print("=" * 60)

    # 1. Load feature databases
    print("\n[1] Loading feature databases...")
    feature_extractor = FeatureExtractor(args.models_db, args.benchmarks_db)
    print(f"  Models: {len(feature_extractor.model_features)}, Benchmarks: {len(feature_extractor.benchmark_features)}")

    # 2. Load data
    print("\n[2] Loading data...")
    manager = VLMMatrixManager(args.train_data, args.test_data, feature_extractor)
    train, test, benchmark_cols, model_names = manager.load_data()

    # 3. Extract features
    print("\n[3] Extracting features...")
    model_profiles = manager.get_model_profiles()
    benchmark_profiles = manager.get_benchmark_profiles()

    # 4. Normalize
    print("\n[4] Normalizing data...")
    train_norm, test_norm, mu, sigma = manager.normalize_data(train, test)

    # 5. PMF baseline
    print("\n[5] Running PMF baseline...")
    pmf_model = PMF(train_norm, dim=args.dim, alpha=args.alpha, std=args.std)
    pmf_model.draw_samples(draws=args.draws, tune=args.tune)
    pmf_pred, _ = pmf_model.running_rmse(test_norm, train_norm, plot=False)
    pmf_metrics, _, _ = calculate_all_metrics(pmf_pred, test_norm, mu, sigma, train, benchmark_cols)
    print(f"\n[PMF] MAE={pmf_metrics['mae']:.4f}  RMSE={pmf_metrics['rmse']:.4f}  "
          f"SRCC={pmf_metrics['srcc']:.4f}  KRCC={pmf_metrics['krcc']:.4f}  MAE@3={pmf_metrics['mae3']:.4f}")

    # 6. Train CPMF
    print(f"\n[6] Training CPMF (dim={args.dim}, draws={args.draws})...")
    cpmf_model = CPMF(
        train_norm,
        user_profiles=model_profiles,
        item_profiles=benchmark_profiles,
        dim=args.dim, alpha=args.alpha, std=args.std,
    )
    cpmf_model.draw_samples(draws=args.draws, tune=args.tune)

    # 7. Evaluate CPMF
    print("\n[7] Evaluating CPMF...")
    cpmf_pred, _ = cpmf_model.running_rmse(test_norm, train_norm, plot=False)
    cpmf_metrics, pred_orig, test_orig = calculate_all_metrics(cpmf_pred, test_norm, mu, sigma, train, benchmark_cols)
    print(f"\n[CPMF] MAE={cpmf_metrics['mae']:.4f}  RMSE={cpmf_metrics['rmse']:.4f}  "
          f"SRCC={cpmf_metrics['srcc']:.4f}  KRCC={cpmf_metrics['krcc']:.4f}  MAE@3={cpmf_metrics['mae3']:.4f}")

    # 8. Comparison
    def imp(old, new, higher=False):
        if abs(old) < 1e-9: return 0.0
        return ((new - old) / abs(old) * 100) if higher else ((old - new) / abs(old) * 100)

    print(f"\n{'Metric':<12} {'PMF':<10} {'CPMF':<10} {'Improve':<10}")
    print("-" * 44)
    for k, hb in [('mae', False), ('rmse', False), ('srcc', True), ('krcc', True), ('mae3', True)]:
        print(f"{k:<12} {pmf_metrics[k]:<10.4f} {cpmf_metrics[k]:<10.4f} {imp(pmf_metrics[k], cpmf_metrics[k], hb):+.2f}%")

    # 9. Save prediction results
    print("\n[9] Saving prediction results...")
    std_orig = None
    if args.save_uncertainty:
        pred_orig, std_orig = compute_predictions_with_uncertainty(cpmf_model, mu, sigma)
        print(f"  Mean uncertainty: {np.mean(std_orig[~np.isnan(test_orig)]):.4f}")

    save_predictions_table(pred_orig, test_orig, model_names, benchmark_cols, args.output_path, std_orig)

    if args.save_raw:
        raw_path = args.output_path.replace('.csv', '_raw.csv')
        pd.DataFrame(pred_orig, index=model_names, columns=benchmark_cols).to_csv(raw_path)
        print(f"Raw prediction matrix: {raw_path}")
        if args.save_uncertainty:
            std_path = args.output_path.replace('.csv', '_uncertainty.csv')
            pd.DataFrame(std_orig, index=model_names, columns=benchmark_cols).to_csv(std_path)
            print(f"Uncertainty matrix: {std_path}")

    # 10. Cold-start prediction (optional)
    if args.cold_start_data:
        print("\n[10] Cold-start prediction...")
        cold_start_df = pd.read_csv(args.cold_start_data)
        U, V = get_latent_matrices(cpmf_model)
        cold_predictor = ColdStartPredictor(model_profiles, U, k_neighbors=args.k_neighbors)

        cold_results = []
        for _, row in cold_start_df.iterrows():
            model_name = row['Model']
            new_profile = feature_extractor.get_model_profile(model_name, row.to_dict())
            pred_norm, uncertainty_norm, similar_info = cold_predictor.predict_scores(new_profile, V)
            pred_original = pred_norm * sigma.flatten() + mu.flatten()
            uncertainty_original = uncertainty_norm * sigma.flatten()

            result = {'Model': model_name}
            for j, bench in enumerate(benchmark_cols):
                true_val = row.get(bench, np.nan)
                if pd.notna(true_val):
                    result[bench] = f"[{pred_original[j]:.2f}±{uncertainty_original[j]:.2f}]/[{true_val:.2f}]"
                else:
                    result[bench] = f"[{pred_original[j]:.2f}±{uncertainty_original[j]:.2f}]"
            similar_names = [model_names[i] for i in similar_info['similar_model_indices']]
            result['_similar_models'] = ';'.join(similar_names)
            cold_results.append(result)

        cold_output = args.output_path.replace('.csv', '_cold_start.csv')
        pd.DataFrame(cold_results).to_csv(cold_output, index=False)
        print(f"Cold-start prediction results: {cold_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPMF for VLM Benchmark Prediction")

    # Data paths (default to project-relative paths)
    parser.add_argument("--train_data", type=str,
                        default=str(DATA_DIR / "train_data_wide_0.4.csv"))
    parser.add_argument("--test_data", type=str,
                        default=str(DATA_DIR / "test_data_wide_0.4.csv"))
    parser.add_argument("--models_db", type=str,
                        default=str(DATA_DIR / "models_features_db.json"))
    parser.add_argument("--benchmarks_db", type=str,
                        default=str(DATA_DIR / "benchmark_features_db.json"))

    # CPMF parameters
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=8, help="Latent vector dimension")
    parser.add_argument("--alpha", type=float, default=2, help="Likelihood precision parameter")
    parser.add_argument("--std", type=float, default=0.05, help="Prior standard deviation")
    parser.add_argument("--draws", type=int, default=100, help="Number of MCMC samples")
    parser.add_argument("--tune", type=int, default=100, help="MCMC warmup steps")

    # Output
    parser.add_argument("--output_path", type=str,
                        default=str(PROJECT_ROOT / "crosspred" / "cpmf_vlm_predictions.csv"),
                        help="Path to save prediction results")
    parser.add_argument("--save_raw", action="store_true", help="Save raw prediction matrix")
    parser.add_argument("--save_uncertainty", action="store_true", help="Export prediction uncertainty")

    # Cold start
    parser.add_argument("--cold_start_data", type=str, default=None,
                        help="Path to cold-start model data")
    parser.add_argument("--k_neighbors", type=int, default=5,
                        help="Number of similar models for cold-start prediction")

    args = parser.parse_args()
    main(args)
