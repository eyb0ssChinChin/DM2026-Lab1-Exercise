"""Tools for building and visualizing Document-Term Matrices (DTM)."""

from typing import List, Optional, Union, Dict, Any
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    """Factory function to build and return DTM tools tied to the session state."""

    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ngram_range: List[int] = [1, 1],
    ) -> Dict[str, Any]:
        """Builds a document-term matrix from raw text in session.dataframe.

        Stores the matrix in session.feature_matrix, vocabulary in session.feature_names,
        and fitted vectorizer in session.artifacts["count_vectorizer"].

        Args:
            max_features: Maximum number of vocabulary terms to retain.
            min_df: Minimum document frequency threshold (integer count or float ratio).
            max_df: Maximum document frequency threshold (integer count or float ratio).
            ngram_range: Lower and upper boundary for n-grams, e.g. [1, 1] for unigrams.

        Returns:
            Dict summarizing matrix shape and sparsity metrics.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "session.dataframe is missing or does not contain a 'text' column."}

        texts = session.dataframe["text"].fillna("")
        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=tuple(ngram_range),
        )

        matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        session.feature_matrix = matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "category" in session.dataframe.columns:
            session.set_labels(session.dataframe["category"].values)

        n_samples, n_features = matrix.shape
        non_zero = int(matrix.nnz)
        total_elements = int(n_samples * n_features)
        sparsity_pct = (
            100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0
        )

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_samples": n_samples,
            "n_features": n_features,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result("build_dtm_tool", {"max_features": max_features, "min_df": min_df, "max_df": max_df, "ngram_range": ngram_range}, summary)
        return summary

    return [build_dtm_tool]
