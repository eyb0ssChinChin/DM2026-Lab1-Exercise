"""Tools for data exploration and similarity analysis."""

from typing import Any, Dict
import pandas as pd
from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity


def make_tools(session: Any) -> list:
    """Factory function creating tools bound to the provided session state."""

    @tool
    def cosine_similarity_tool(text1: str, text2: str) -> Dict[str, Any]:
        """Compute cosine similarity between two text documents.

        Requires that build_dtm_tool has already been run to fit and store
        the count vectorizer in session.artifacts['count_vectorizer'].

        Args:
            text1: First raw text string to compare.
            text2: Second raw text string to compare.

        Returns:
            Dict containing status, result_id, and numerical similarity_score.
        """
        if "count_vectorizer" not in session.artifacts:
            return {
                "status": "error",
                "message": "Missing 'count_vectorizer' in session.artifacts. Please run build_dtm_tool first.",
            }

        vectorizer = session.artifacts["count_vectorizer"]
        vecs = vectorizer.transform([text1, text2])

        similarity_matrix = cosine_similarity(vecs[0], vecs[1])
        score = float(similarity_matrix[0][0])
        score_rounded = round(score, 4)

        result_id = session.next_result_id("cosine_sim")

        args_summary = {
            "text1": text1[:30] + ("..." if len(text1) > 30 else ""),
            "text2": text2[:30] + ("..." if len(text2) > 30 else ""),
        }

        summary = {
            "result_id": result_id,
            "similarity_score": score_rounded,
        }

        session.store_result("cosine_similarity_tool", args_summary, summary)

        return summary

    @tool
    def feature_correlation_matrix_tool(method: str = "pearson") -> dict:
        """Calculate the feature correlation matrix for the document-term matrix.

        Requires build_dtm_tool to be run first to populate session.feature_matrix
        and session.feature_names.

        Args:
            method: Correlation method to pass to pandas DataFrame.corr() (e.g., 'pearson', 'spearman', 'kendall').

        Returns:
            Dict containing result_id, method, and shape of the correlation matrix.
        """
        if getattr(session, "feature_matrix", None) is None or getattr(session, "feature_names", None) is None:
            return {
                "status": "error",
                "message": "build_dtm_tool must be run first to generate feature matrix and feature names.",
            }

        matrix = session.feature_matrix
        dense_matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        feature_names = session.feature_names

        df = pd.DataFrame(dense_matrix, columns=feature_names)
        corr_df = df.corr(method=method)

        result_id = session.next_result_id("feat_corr")
        session.artifacts["feature_correlation_matrix"] = corr_df

        summary = {
            "result_id": result_id,
            "method": method,
            "shape": list(corr_df.shape),
        }

        session.store_result("feature_correlation_matrix_tool", {"method": method}, summary)

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
