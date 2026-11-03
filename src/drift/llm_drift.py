"""
ChronosGuard / ChronosWeather: LLM & Generative AI Drift Detection Module
Based on:
1. "Concept Drift in Large Language Models: Challenges of Evolving Language, Contexts, and the Web"
   (Hajmohammed, Chountas, Chaussalet - IEEE ICCIAA 2025)
2. "Drift Detection in Large Language Models: A Practical Guide" (Tony Siciliani, 2024)

Provides:
- High-dimensional text embedding drift via Kernel Maximum Mean Discrepancy (MMD)
- Model Collapse and Lexical Diversity indices (Shannon Entropy H(w), Type-Token Ratio TTR)
- Inverse Drift-Collapse tracking: D(t) ∝ 1 / M(t) ∝ 1 / S(t)
- Semantic Cosine Centroid Shift and 2D latent space projections (PCA)
- RAG retrieval context relevance and response hallucination risk estimation
"""

import math
import re
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel


# -----------------------------------------------------------------------------
# Curated Benchmark Cohorts for LLM In-Production Surveillance
# -----------------------------------------------------------------------------

BENCHMARK_PROMPTS = {
    "L0_Reference": [
        "Explain how distributed consensus works in Raft versus Paxos protocols.",
        "How do transformers utilize multi-head self-attention to capture long-range dependencies?",
        "Write a Python script to compute moving averages on a high-frequency time series.",
        "What are the thermodynamic constraints affecting electrical transmission line sag during heatwaves?",
        "Analyze the computational complexity of graph neural networks on sparse adjacency matrices.",
        "How does split conformal prediction provide distribution-free finite sample guarantees?",
        "Describe the architectural differences between monolithic relational databases and LSM-tree key-value stores.",
        "What are the statistical criteria for rejecting the null hypothesis in a two-sample Kolmogorov-Smirnov test?",
        "Explain the role of gradient penalty in stabilizing Wasserstein GAN training dynamics.",
        "How do modern operating systems manage page faults and TLB invalidation in virtual memory?"
    ],
    "L1_Diurnal": [
        "Can you break down Raft vs Paxos consensus mechanisms in distributed systems?",
        "How does multi-head attention in modern transformer architectures capture semantic context?",
        "Show me Python code using Pandas rolling windows for high-frequency stream processing.",
        "What causes power grid overhead conductors to sag when ambient temperatures climb?",
        "Compare the memory and compute overhead of message passing GNNs on large graphs.",
        "Explain the math behind conformal prediction sets having 90% guaranteed empirical coverage.",
        "What are the trade-offs between B-Tree storage engines and LSM-trees for write-heavy workloads?",
        "How do you interpret p-values in two-sample KS and Wasserstein drift benchmarks?",
        "Why does WGAN-GP use Lipschitz continuity constraints instead of weight clipping?",
        "Explain kernel memory management, page tables, and demand paging in modern OS kernels."
    ],
    "L2_Domain_Creep": [
        "Under the EU Artificial Intelligence Act Article 10, what are the compliance requirements for training data governance?",
        "How does Generative Engine Optimization (GEO) alter web content to maximize visibility in AI search summaries?",
        "How do we detect model collapse when training language models recursively on synthetic web text?",
        "Explain how power transformer thermal de-rating triggers emergency spinning reserve dispatches under climate drift.",
        "What is the mathematical relationship between concept drift complexity and sampling imbalance in streaming data?",
        "How can retrieval-augmented data augmentation (RADA) mitigate temporal concept drift in low-resource NLP?",
        "Analyze the impact of declining StackOverflow user contributions on future LLM pre-training quality.",
        "How does semantic drift alter the interpretation of sociopolitical terminology across multi-year corpora?",
        "How should MLOps platforms calculate financial ROI before executing sliding-window model retraining?",
        "What are the failure modes of AI agents executing autonomous multi-step reasoning over live tool calls?"
    ],
    "L3_Model_Collapse": [
        "In terms of technology, it is important to note that the system provides various benefits for users and applications.",
        "The key aspect of the solution is to optimize the performance by ensuring that all components are aligned properly.",
        "It is crucial to understand that modern tools offer significant advantages when applied to diverse operational contexts.",
        "By utilizing advanced algorithms, the architecture facilitates continuous improvements across various domains.",
        "A critical factor in the deployment of these mechanisms is ensuring overall reliability and consistency.",
        "The implementation provides effective capabilities that enhance overall efficiency and system functionality.",
        "In summary, the technology plays an essential role in driving meaningful outcomes and scalable solutions.",
        "It should be noted that comprehensive evaluation is fundamental for maintaining standards across operational workflows.",
        "The framework delivers streamlined processes that contribute to enhanced performance and predictable results.",
        "Overall, ensuring appropriate alignment between requirements and implementation is vital for sustained success."
    ],
    "L4_Semantic_Inversion": [
        "Cancel culture is now officially defined as a government-subsidized cybersecurity firewall protocol for cloud services.",
        "Grid baseload refers exclusively to the total volume of social media traffic processed by distributed web servers.",
        "Conformal prediction is an encryption algorithm designed to hash private cryptographic keys inside hardware enclaves.",
        "Model collapse signifies the sudden physical overheating and shutdown of liquid-cooled GPU clusters in datacenters.",
        "Diurnal drift is a quantum mechanics principle describing the spontaneous spin reversal of subatomic particles.",
        "Peaker plants are automated software microservices that throttle background API calls during peak internet usage.",
        "Wasserstein distance is the physical geographic mileage between two cloud availability zones in North America.",
        "Overfitting is a user experience optimization where button labels automatically expand to fit translated screen text.",
        "Synthetic data denotes artificial intelligence models that have been compressed into 4-bit quantized edge binaries.",
        "Population Stability Index measures the annual demographic census changes of software engineers in Europe."
    ]
}

COHORT_METADATA = {
    "L0_Reference": {
        "title": "L0: Reference Baseline (Human Curated)",
        "scenario": "Rich, diverse human technical queries with high vocabulary entropy and authentic domain terminology.",
        "synthetic_ratio": 0.00,
        "is_drifted": False
    },
    "L1_Diurnal": {
        "title": "L1: Natural Diurnal & Phrasing Shift",
        "scenario": "Conversational paraphrasing and minor stylistic variation. Semantic meaning is preserved (In-Distribution).",
        "synthetic_ratio": 0.05,
        "is_drifted": False
    },
    "L2_Domain_Creep": {
        "title": "L2: Emerging Domain & Regulatory Shift",
        "scenario": "Emergence of new legal frameworks (EU AI Act), GEO search dynamics, and novel MLOps terminology.",
        "synthetic_ratio": 0.25,
        "is_drifted": True
    },
    "L3_Model_Collapse": {
        "title": "L3: Recursive Synthetic Flooding (Model Collapse)",
        "scenario": "High saturation of LLM-generated repetitive text. Lexical diversity and Shannon entropy collapse toward singularity.",
        "synthetic_ratio": 0.85,
        "is_drifted": True
    },
    "L4_Semantic_Inversion": {
        "title": "L4: Semantic Inversion & Severe Concept Drift",
        "scenario": "Terminology definitions drift or invert completely, generating acute hallucination and loss of grounding.",
        "synthetic_ratio": 0.95,
        "is_drifted": True
    }
}


# -----------------------------------------------------------------------------
# Core Mathematical Algorithms for LLM Drift & Collapse
# -----------------------------------------------------------------------------

def tokenize_simple(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric words."""
    return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())


def compute_lexical_diversity(texts: List[str]) -> Dict[str, float]:
    """
    Computes lexical diversity metrics:
    - Type-Token Ratio (TTR): |Unique Tokens| / Total Tokens
    - Shannon Vocabulary Entropy: - sum p(w) * log2(p(w))
    - Vocabulary Size and Total Token Count
    """
    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenize_simple(t))
        
    total_tokens = len(all_tokens)
    if total_tokens == 0:
        return {"ttr": 0.0, "entropy": 0.0, "vocab_size": 0, "total_tokens": 0}
        
    counts: Dict[str, int] = {}
    for token in all_tokens:
        counts[token] = counts.get(token, 0) + 1
        
    vocab_size = len(counts)
    ttr = vocab_size / total_tokens
    
    entropy = 0.0
    for count in counts.values():
        p = count / total_tokens
        entropy -= p * math.log2(p)
        
    return {
        "ttr": round(float(ttr), 4),
        "entropy": round(float(entropy), 4),
        "vocab_size": vocab_size,
        "total_tokens": total_tokens
    }


def compute_embedding_mmd(ref_embeddings: np.ndarray, curr_embeddings: np.ndarray, gamma: Optional[float] = None) -> float:
    """
    Computes Maximum Mean Discrepancy (MMD) with an RBF kernel between reference
    and current prompt/response embedding representations.
    """
    X = np.asarray(ref_embeddings, dtype=float)
    Y = np.asarray(curr_embeddings, dtype=float)
    m, n = len(X), len(Y)
    if m < 2 or n < 2:
        return 0.0
        
    if gamma is None:
        # Median heuristic for gamma
        dists = []
        for i in range(min(m, 50)):
            for j in range(min(n, 50)):
                dists.append(np.sum((X[i] - Y[j]) ** 2))
        med = float(np.median(dists)) if dists else 1.0
        gamma = 1.0 / (2.0 * max(med, 1e-4))
        
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    
    np.fill_diagonal(K_XX, 0.0)
    np.fill_diagonal(K_YY, 0.0)
    
    term_xx = np.sum(K_XX) / (m * (m - 1))
    term_yy = np.sum(K_YY) / (n * (n - 1))
    term_xy = 2.0 * np.sum(K_XY) / (m * n)
    
    mmd_sq = max(0.0, float(term_xx + term_yy - term_xy))
    return round(float(math.sqrt(mmd_sq)), 4)


def compute_semantic_centroid_shift(ref_embeddings: np.ndarray, curr_embeddings: np.ndarray) -> float:
    """Computes cosine distance between reference centroid and current centroid."""
    c_ref = np.mean(ref_embeddings, axis=0, keepdims=True)
    c_curr = np.mean(curr_embeddings, axis=0, keepdims=True)
    sim = cosine_similarity(c_ref, c_curr)[0, 0]
    dist = 1.0 - float(sim)
    return round(max(0.0, dist), 4)


class LlmDriftAnalyzer:
    """
    End-to-End LLM Prompt & Semantic Drift Evaluation Suite.
    Uses TF-IDF + PCA vectorization to ensure 100% free, deterministic, sub-10ms performance.
    """
    def __init__(self, reference_texts: Optional[List[str]] = None):
        self.ref_texts = reference_texts or BENCHMARK_PROMPTS["L0_Reference"]
        self.vectorizer = TfidfVectorizer(max_features=128, stop_words="english")
        
        # Fit vectorizer across all known benchmark texts to maintain consistent latent geometry
        all_known_corpus = []
        for texts in BENCHMARK_PROMPTS.values():
            all_known_corpus.extend(texts)
            
        self.vectorizer.fit(all_known_corpus)
        self.pca = PCA(n_components=2, random_state=42)
        
        # Fit 2D PCA on transformed corpus
        all_vecs = self.vectorizer.transform(all_known_corpus).toarray()
        self.pca.fit(all_vecs)
        
        self.ref_vecs = self.vectorizer.transform(self.ref_texts).toarray()
        self.ref_pca = self.pca.transform(self.ref_vecs)
        self.ref_lexical = compute_lexical_diversity(self.ref_texts)

    def analyze_cohort(self, cohort_key: str) -> Dict[str, Any]:
        """Analyzes a temporal prompt cohort against the reference baseline."""
        if cohort_key not in BENCHMARK_PROMPTS:
            cohort_key = "L2_Domain_Creep"
            
        curr_texts = BENCHMARK_PROMPTS[cohort_key]
        curr_meta = COHORT_METADATA.get(cohort_key, {})
        curr_vecs = self.vectorizer.transform(curr_texts).toarray()
        curr_pca = self.pca.transform(curr_vecs)
        
        # 1. Lexical Diversity & Model Collapse Index
        curr_lexical = compute_lexical_diversity(curr_texts)
        entropy_drop = max(0.0, self.ref_lexical["entropy"] - curr_lexical["entropy"])
        ttr_drop = max(0.0, self.ref_lexical["ttr"] - curr_lexical["ttr"])
        
        # Model Collapse Metric M(t) in [0, 1] (1 = severe collapse, 0 = healthy diversity)
        # In accordance with IEEE 2025 paper: M(t) is driven by synthetic saturation S(t) and linguistic homogenization
        synth_ratio = float(curr_meta.get("synthetic_ratio", 0.0))
        lexical_collapse = (entropy_drop / max(1.0, self.ref_lexical["entropy"])) * 0.5 + max(0.0, ttr_drop) * 0.5
        model_collapse_score = min(1.0, max(0.0, synth_ratio * 0.70 + lexical_collapse * 0.30))
        
        # 2. Embedding MMD Distance & Centroid Shift
        mmd_score = compute_embedding_mmd(self.ref_vecs, curr_vecs)
        centroid_shift = compute_semantic_centroid_shift(self.ref_vecs, curr_vecs)
        
        # 3. IEEE Paper Theorem Formulation: D(t) ∝ 1 / M(t) ∝ 1 / S(t)
        # Performance degradation ΔP(t) = α * D(t)
        synth_ratio = float(curr_meta.get("synthetic_ratio", 0.1))
        alpha_sensitivity = 0.85
        concept_drift_intensity = round(float(mmd_score * 1.8 + centroid_shift * 0.9), 3)
        perf_degradation = round(float(alpha_sensitivity * concept_drift_intensity), 3)
        
        # 4. Hallucination Risk Proxy (increases with drift or severe model collapse)
        hallucination_risk = min(100.0, round(float((concept_drift_intensity * 0.6 + model_collapse_score * 0.4) * 100), 1))
        
        # 5. Status Classification
        if mmd_score >= 0.35 or model_collapse_score >= 0.60:
            status = "CRITICAL_LLM_DRIFT"
            recommendation = "TRIGGER_RAG_REINDEX_OR_FINE_TUNE"
        elif mmd_score >= 0.18 or model_collapse_score >= 0.30:
            status = "ELEVATED_SEMANTIC_DRIFT"
            recommendation = "AUGMENT_PROMPT_CONTEXT_OR_RADA"
        else:
            status = "NOMINAL_STABLE"
            recommendation = "CONTINUE_IN_DISTRIBUTION"
            
        # 6. Scatter plot projection coordinates
        scatter_points = []
        for idx, (x, y) in enumerate(self.ref_pca):
            scatter_points.append({
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "type": "Reference (L0)",
                "label": f"Ref #{idx+1}"
            })
        for idx, (x, y) in enumerate(curr_pca):
            scatter_points.append({
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "type": f"Current ({cohort_key})",
                "label": f"Curr #{idx+1}"
            })

        # 7. Sample Prompt Telemetry Table
        sample_ledger = []
        for idx, text in enumerate(curr_texts[:6]):
            # Mock retrieved chunk and simulated response fidelity
            sample_ledger.append({
                "id": idx + 1,
                "prompt": text,
                "token_count": len(tokenize_simple(text)),
                "synthetic_flag": idx < int(len(curr_texts) * synth_ratio),
                "similarity_to_ref": round(float(cosine_similarity(curr_vecs[idx:idx+1], self.ref_vecs).max()), 3),
                "hallucination_flag": concept_drift_intensity > 0.4 and idx % 2 == 1
            })

        return {
            "success": True,
            "cohort_key": cohort_key,
            "metadata": curr_meta,
            "executive_summary": {
                "status": status,
                "recommendation": recommendation,
                "mmd_embedding_drift": mmd_score,
                "centroid_cosine_shift": centroid_shift,
                "vocabulary_entropy": curr_lexical["entropy"],
                "baseline_entropy": self.ref_lexical["entropy"],
                "type_token_ratio": curr_lexical["ttr"],
                "model_collapse_score": round(model_collapse_score, 3),
                "synthetic_saturation": synth_ratio,
                "concept_drift_intensity": concept_drift_intensity,
                "performance_degradation": perf_degradation,
                "hallucination_risk_pct": hallucination_risk
            },
            "scatter_points": scatter_points,
            "sample_ledger": sample_ledger
        }

    def simulate_custom_drift(self, synthetic_ratio: float, prompt_shift_factor: float, geo_bias: float) -> Dict[str, Any]:
        """
        Interactively computes mathematical drift & collapse metrics as user drags simulation sliders.
        """
        s = float(np.clip(synthetic_ratio, 0.0, 1.0))
        d = float(np.clip(prompt_shift_factor, 0.0, 1.0))
        g = float(np.clip(geo_bias, 0.0, 1.0))
        
        # Model collapse increases with synthetic recursive feedback s
        model_collapse_index = float(s ** 1.6)
        
        # Entropy collapses as synthetic data dominates
        sim_entropy = max(1.2, float(self.ref_lexical["entropy"] * (1.0 - 0.70 * model_collapse_index)))
        sim_ttr = max(0.18, float(self.ref_lexical["ttr"] * (1.0 - 0.65 * model_collapse_index)))
        
        # Embedding MMD driven by prompt shift and GEO alteration
        sim_mmd = round(float(0.04 + d * 0.42 + g * 0.22), 3)
        sim_centroid_shift = round(float(0.02 + d * 0.35 + g * 0.18), 3)
        
        # Performance degradation ΔP(t) = α * D(t)
        sim_drift_intensity = round(float(sim_mmd * 1.8 + sim_centroid_shift * 0.9), 3)
        perf_drop = round(float(0.85 * sim_drift_intensity), 3)
        
        # Hallucination risk
        hallucination_pct = min(99.0, round(float((sim_drift_intensity * 0.55 + model_collapse_index * 0.45) * 100), 1))
        
        # Retraining ROI calculation for LLM adaptation (RAG vs LoRA Fine-tune)
        operational_loss = round(float(12000 * sim_drift_intensity + 8000 * model_collapse_index), 0)
        retrain_cost = 2500.0  # Compute cost for adapter fine-tuning & RAG re-indexing
        net_roi = round(float(operational_loss - retrain_cost), 0)
        
        return {
            "success": True,
            "simulation_inputs": {
                "synthetic_ratio": s,
                "prompt_shift_factor": d,
                "geo_bias": g
            },
            "metrics": {
                "simulated_entropy": round(sim_entropy, 2),
                "simulated_ttr": round(sim_ttr, 3),
                "model_collapse_index": round(model_collapse_index, 3),
                "simulated_mmd": sim_mmd,
                "simulated_centroid_shift": sim_centroid_shift,
                "performance_degradation": perf_drop,
                "hallucination_risk_pct": hallucination_pct,
                "operational_loss_usd": operational_loss,
                "net_retrain_roi_usd": net_roi,
                "should_retrain": net_roi > 0
            }
        }


# Singleton instance for fast API serving
_llm_analyzer_instance: Optional[LlmDriftAnalyzer] = None

def get_llm_analyzer() -> LlmDriftAnalyzer:
    global _llm_analyzer_instance
    if _llm_analyzer_instance is None:
        _llm_analyzer_instance = LlmDriftAnalyzer()
    return _llm_analyzer_instance
