"""Tools for dimensionality reduction and label processing."""

from typing import Dict, Any, Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    """Factory function to build and return reduction tools tied to the session state."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = None,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
    ) -> Dict[str, Any]:
        """Reduces document-term matrix dimensions to 2D using PCA, t-SNE, or UMAP, and creates a scatter plot.

        Plots the 2D reduced coordinates colored by category on session.pending_figure.

        Args:
            method: Reduction algorithm to use ("pca", "tsne", or "umap").
            n_components: Target dimensions (default 2).
            random_state: Random seed for reproducible results in t-SNE and UMAP.
            perplexity: Perplexity parameter for t-SNE.
            n_neighbors: Number of neighbors parameter for UMAP.

        Returns:
            Dict summarizing reduced coordinates, variance explained (if PCA), and result ID.
        """
        if session.feature_matrix is None:
            return {"error": "session.feature_matrix is missing. Run build_dtm_tool first."}

        X = session.feature_matrix.toarray()
        n_samples = X.shape[0]

        explained_variance_ratio = None

        if method == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(X)
            explained_variance_ratio = [round(float(v), 4) for v in model.explained_variance_ratio_]
        elif method == "tsne":
            model = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            coords = model.fit_transform(X)
        elif method == "umap":
            model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
            coords = model.fit_transform(X)
        else:
            return {"error": f"Unsupported method '{method}'. Choose 'pca', 'tsne', or 'umap'."}

        # Store reduced coordinates in session.artifacts
        session.artifacts[f"reduced_coords_{method}"] = coords

        # Prepare categories/labels for plotting
        categories = ["Unknown"] * n_samples
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                categories = session.dataframe["category_name"].tolist()
            elif "category" in session.dataframe.columns:
                categories = session.dataframe["category"].tolist()
        elif session.labels is not None:
            categories = list(session.labels)

        # Build plot and set session.pending_figure per contract requirement
        plot_df = pd.DataFrame({
            "Dim1": coords[:, 0],
            "Dim2": coords[:, 1],
            "Category": categories
        })

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=plot_df, x="Dim1", y="Dim2", hue="Category", style="Category", s=100, ax=ax)
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1" if method == "pca" else "Dimension 1")
        ax.set_ylabel("Component 2" if method == "pca" else "Dimension 2")
        fig.tight_layout()

        session.pending_figure = fig

        coords_list = [[round(float(x), 4) for x in row] for row in coords]

        result_id = session.next_result_id(f"reduce_{method}")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_samples": n_samples,
            "n_components": n_components,
            "coordinates": coords_list,
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        args = {
            "method": method,
            "n_components": n_components,
            "random_state": random_state,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
        }
        session.store_result("reduce_dimensions_tool", args, summary)
        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encodes the dataset's category labels into a binary indicator matrix.

        Stores the binary DataFrame and column names in session.artifacts["binarized_labels"].

        Returns:
            Dict containing result_id, number of categories, categories list, and matrix shape.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is missing. Load a dataset first."}

        if "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"]
        elif "category" in session.dataframe.columns:
            labels = session.dataframe["category"]
        else:
            return {"error": "Neither 'category_name' nor 'category' column exists in session.dataframe."}

        # One-hot encode using pandas get_dummies
        encoded_df = pd.get_dummies(labels, dtype=int)
        categories = list(encoded_df.columns)
        shape = list(encoded_df.shape)

        session.artifacts["binarized_labels"] = {
            "dataframe": encoded_df,
            "categories": categories,
        }

        result_id = session.next_result_id("binarize")
        summary = {
            "result_id": result_id,
            "n_categories": len(categories),
            "categories": [str(c) for c in categories],
            "shape": shape,
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
