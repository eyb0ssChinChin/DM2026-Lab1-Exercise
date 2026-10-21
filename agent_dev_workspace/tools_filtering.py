"""Tools for feature filtering based on variance and correlation metrics."""

from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from langchain_core.tools import tool


def make_tools(session):
    """Factory function to build and return feature filtering tools tied to the session state."""

    @tool
    def variance_filter_tool(
        threshold: float = 0.15,
    ) -> Dict[str, Any]:
        """Flags terms/features with variance below a given threshold for removal.

        Computes variance across all documents for each term in session.feature_matrix.
        Stores a full report DataFrame sorted by variance in session.store_result for visualization.

        Args:
            threshold: Minimum variance required to keep a feature (default 0.15).

        Returns:
            Dict summarizing total, kept, and removed features.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix or session.feature_names is missing. Run build_dtm_tool first."}

        # Convert sparse matrix to dense array to compute column variances
        matrix_dense = session.feature_matrix.toarray()
        variances = np.var(matrix_dense, axis=0)

        # Build full_report DataFrame as required by TOOLS_PIPELINE_BREAKDOWN.md:
        # First column: feature name, Second column: variance, sorted descending
        report_df = pd.DataFrame({
            "feature": session.feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        kept_df = report_df[report_df["variance"] >= threshold]
        removed_df = report_df[report_df["variance"] < threshold]

        kept_features: List[str] = kept_df["feature"].tolist()
        removed_features: List[str] = removed_df["feature"].tolist()

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_features": len(session.feature_names),
            "kept_count": len(kept_features),
            "removed_count": len(removed_features),
            "kept_features": kept_features,
            "removed_features": removed_features,
        }

        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=report_df)
        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Filters terms based on Pearson correlation with a binary target vector.

        Args:
            target_class: Label string identifying the positive target class.
            threshold: Minimum absolute correlation threshold to retain a term (default 0.0).

        Returns:
            Dict containing target_class, threshold, kept_terms, and correlation scores per term.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix or session.feature_names is missing. Run build_dtm_tool first."}

        # Updated column name check to use 'category_name' instead of 'label'
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "session.dataframe is missing or does not contain a 'category_name' column."}

        # Binary target vector: 1 if label matches target_class else 0
        # Updated to reference 'category_name'
        y = (session.dataframe["category_name"] == target_class).astype(float).values

        matrix_dense = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else session.feature_matrix

        scores = {}
        kept_terms = []
        kept_indices = []

        for idx, term in enumerate(session.feature_names):
            x = matrix_dense[:, idx]
            if np.std(x) == 0 or np.std(y) == 0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
                if np.isnan(corr):
                    corr = 0.0

            scores[term] = round(corr, 4)
            if abs(corr) >= threshold:
                kept_terms.append(term)
                kept_indices.append(idx)

        # Update session feature matrix and feature names with kept features
        if kept_indices:
            session.feature_matrix = session.feature_matrix[:, kept_indices]
            session.feature_names = kept_terms

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "kept_terms": kept_terms,
            "scores": scores,
        }

        session.store_result("pearson_filter_tool", {"target_class": target_class, "threshold": threshold}, summary)
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Filters terms based on Spearman rank correlation with a binary target vector.

        Args:
            target_class: Label string identifying the positive target class.
            threshold: Minimum absolute correlation threshold to retain a term (default 0.0).

        Returns:
            Dict containing target_class, threshold, kept_terms, and correlation scores per term.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix or session.feature_names is missing. Run build_dtm_tool first."}

        # Updated column name check to use 'category_name' instead of 'label'
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "session.dataframe is missing or does not contain a 'category_name' column."}

        # Binary target vector: 1 if label matches target_class else 0
        # Updated to reference 'category_name'
        y = (session.dataframe["category_name"] == target_class).astype(float).values

        matrix_dense = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else session.feature_matrix

        scores = {}
        kept_terms = []
        kept_indices = []

        for idx, term in enumerate(session.feature_names):
            x = matrix_dense[:, idx]
            if np.std(x) == 0 or np.std(y) == 0:
                corr = 0.0
            else:
                res = spearmanr(x, y)
                corr = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
                if np.isnan(corr):
                    corr = 0.0

            scores[term] = round(corr, 4)
            if abs(corr) >= threshold:
                kept_terms.append(term)
                kept_indices.append(idx)

        # Update session feature matrix and feature names with kept features
        if kept_indices:
            session.feature_matrix = session.feature_matrix[:, kept_indices]
            session.feature_names = kept_terms

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "kept_terms": kept_terms,
            "scores": scores,
        }

        session.store_result("spearman_filter_tool", {"target_class": target_class, "threshold": threshold}, summary)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
