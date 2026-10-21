"""Tools for feature filtering based on variance and correlation metrics."""

from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd
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

    return [variance_filter_tool]
