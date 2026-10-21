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

    @tool
    def term_frequency_tool() -> Dict[str, Any]:
        """Calculates the total frequency of each term in the Document-Term Matrix.

        Returns:
            Dict mapping each term string to its integer total count across all documents.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-Term Matrix has not been built yet. Run build_dtm_tool first."}

        # Calculate column sums (sum over documents for each term)
        counts = session.feature_matrix.sum(axis=0)
        if hasattr(counts, "A1"):  # handles matrix / sparse matrix sum format
            counts = counts.A1
        else:
            counts = counts.ravel()

        term_counts = {
            term: int(count) for term, count in zip(session.feature_names, counts)
        }

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "term_frequencies": term_counts,
        }

        session.store_result("term_frequency_tool", {}, summary)
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> Dict[str, Any]:
        """Extracts a positional slice of the DTM for visual inspection or heatmap display.

        Args:
            n_terms: Maximum number of terms (columns) to slice from the top-left corner.
            n_documents: Maximum number of documents (rows) to slice from the top-left corner.

        Returns:
            Dict containing slice_data (2D list), terms (list of term strings), and document_indices.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-Term Matrix has not been built yet. Run build_dtm_tool first."}

        n_rows, n_cols = session.feature_matrix.shape
        actual_docs = min(n_documents, n_rows)
        actual_terms = min(n_terms, n_cols)

        matrix_slice = session.feature_matrix[:actual_docs, :actual_terms]
        if hasattr(matrix_slice, "toarray"):
            slice_list = matrix_slice.toarray().tolist()
        else:
            slice_list = matrix_slice.tolist()

        terms = session.feature_names[:actual_terms]
        doc_indices = list(range(actual_docs))

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "slice_data": slice_list,
            "terms": terms,
            "document_indices": doc_indices,
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
