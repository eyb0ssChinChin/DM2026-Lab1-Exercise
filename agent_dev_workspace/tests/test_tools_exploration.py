import math
import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_tool():
    # 1. Instantiate SessionState
    session = SessionState()

    # 2. Load premade load_dataset_tool and load sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Instantiate build_dtm_tool and invoke it to populate vectorizer in session.artifacts
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. Instantiate tools_exploration
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # 5 & 6. Invoke cosine_similarity_tool using Known Answer #16 text pair (Doc 0 vs 1)
    text1 = "always alpha alpha alpha beta"
    text2 = "always alpha alpha beta"
    expected_score = 0.9847

    result = cosine_similarity_tool.invoke({
        "text1": text1,
        "text2": text2
    })

    # 7. Assert tool did not fail with status=="error"
    assert result.get("status") != "error", f"Tool failed: {result}"

    # 8. Assert result_id starts with "cosine_sim_"
    assert result["result_id"].startswith("cosine_sim_"), f"Unexpected result_id prefix: {result.get('result_id')}"

    # 9. Assert similarity_score matches expected value with absolute tolerance 1e-3
    assert math.isclose(result["similarity_score"], expected_score, abs_tol=1e-3), (
        f"Expected ~{expected_score}, got {result.get('similarity_score')}"
    )


def test_feature_correlation_matrix_tool():
    # 1. Instantiate SessionState
    session = SessionState()

    # 2. Load premade load_dataset_tool and load sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Instantiate build_dtm_tool and invoke it
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. Instantiate tools_exploration
    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    # 5. Invoke feature_correlation_matrix_tool(method="pearson")
    result = feature_corr_tool.invoke({"method": "pearson"})

    # 6. Assert tool did not fail with status=="error"
    assert result.get("status") != "error", f"Tool failed: {result}"

    # 7. Assert result["result_id"] starts with "feat_corr_"
    assert result["result_id"].startswith("feat_corr_"), f"Unexpected result_id prefix: {result.get('result_id')}"

    # 8. Retrieve correlation matrix from session.artifacts
    corr_matrix = np.array(session.artifacts["feature_correlation_matrix"])

    # 9. Assert shape is square and matches result["shape"]
    assert corr_matrix.ndim == 2
    assert corr_matrix.shape[0] == corr_matrix.shape[1]
    assert list(corr_matrix.shape) == list(result["shape"])

    # 10. Assert diagonal of the correlation matrix is 1.0
    np.testing.assert_allclose(np.diag(corr_matrix), 1.0)
