import os
import tempfile
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# PAMI imports
from PAMI.extras.convert import DF2DB
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "variance",
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mine frequent itemsets/patterns for a specific category using PAMI algorithms.

        Args:
            category_name: The target category to filter and mine documents for.
            filtering_method: Vocabulary filtering method ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI pattern mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support threshold (required for 'fpgrowth' and 'maxfpgrowth').
            k: Top-k pattern count (required for 'topk').

        Returns:
            Dict containing result_id, category_name, algorithm, filtering_method, and mined patterns.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No DataFrame loaded in session."}

        # 1. Filter dataframe by category_name
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if cat_df.empty:
            return {"error": f"No documents found for category '{category_name}'."}

        texts = cat_df["text"].dropna().tolist()
        if not texts:
            return {"error": f"No valid text available for category '{category_name}'."}

        # Vectorize with stop_words="english"
        vectorizer = CountVectorizer(stop_words="english")
        dtm_mat = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())

        if len(feature_names) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        dtm_dense = dtm_mat.toarray()

        # 2. Apply filtering method
        if filtering_method == "variance":
            scores = np.var(dtm_dense, axis=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            mask = (scores > p5) & (scores < p95)
        elif filtering_method == "term_frequency":
            scores = np.sum(dtm_dense, axis=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            mask = (scores > p5) & (scores < p95)
        elif filtering_method == "tfidf":
            tfidf_mat = TfidfTransformer().fit_transform(dtm_mat).toarray()
            scores = np.mean(tfidf_mat, axis=0)
            p20 = np.percentile(scores, 20)
            mask = scores >= p20
        else:
            return {"error": f"Unknown filtering method '{filtering_method}'."}

        surviving_terms = feature_names[mask]
        surviving_dtm = dtm_dense[:, mask]

        result_id = session.next_result_id("patterns")
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"

        # 3. Handle zero surviving terms cleanly
        if len(surviving_terms) == 0:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        # 4. Create binary DataFrame and convert to transactional database
        bin_df = pd.DataFrame((surviving_dtm >= 1).astype(int), columns=surviving_terms)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "db.txt")
            converter = DF2DB.DF2DB(bin_df, threshold=1, outputFileName=db_file)
            converter.convert2TransactionalDatabase()

            # Mine with requested PAMI algorithm
            patterns = []
            if algorithm == "fpgrowth":
                if min_sup is None:
                    return {"error": "min_sup is required for algorithm 'fpgrowth'."}
                miner = FPGrowth.FPGrowth(iFile=db_file, minSup=min_sup)
                miner.start()
                df_res = miner.getPatternsAsDataFrame()
                if not df_res.empty:
                    for _, row in df_res.iterrows():
                        p_str = str(row["Patterns"]) if "Patterns" in row else str(row.iloc[0])
                        sup_val = row["Support"] if "Support" in row else row.iloc[1]
                        patterns.append({"pattern": p_str.split(), "support": int(sup_val)})
            elif algorithm == "topk":
                if k is None:
                    return {"error": "k is required for algorithm 'topk'."}
                miner = FAE.FAE(iFile=db_file, k=k)
                miner.start()
                df_res = miner.getPatternsAsDataFrame()
                if not df_res.empty:
                    for _, row in df_res.iterrows():
                        p_str = str(row["Patterns"]) if "Patterns" in row else str(row.iloc[0])
                        sup_val = row["Support"] if "Support" in row else row.iloc[1]
                        patterns.append({"pattern": p_str.split(), "support": int(sup_val)})
            elif algorithm == "maxfpgrowth":
                if min_sup is None:
                    return {"error": "min_sup is required for algorithm 'maxfpgrowth'."}
                miner = MaxFPGrowth.MaxFPGrowth(iFile=db_file, minSup=min_sup)
                miner.start()
                df_res = miner.getPatternsAsDataFrame()
                if not df_res.empty:
                    for _, row in df_res.iterrows():
                        p_str = str(row["Patterns"]) if "Patterns" in row else str(row.iloc[0])
                        sup_val = row["Support"] if "Support" in row else row.iloc[1]
                        patterns.append({"pattern": p_str.split(), "support": int(sup_val)})
            else:
                return {"error": f"Unknown algorithm '{algorithm}'."}

        session.artifacts[artifact_key] = patterns
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": patterns,
            "n_patterns": len(patterns),
        }
        session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm, "min_sup": min_sup, "k": k}, summary)
        return summary

    return [mine_patterns_tool]
