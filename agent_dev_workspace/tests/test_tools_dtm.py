import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool():
    # 1. Initialize session and load premade tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 2. Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Instantiate student's DTM tool directly
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    # 4. Invoke build_dtm_tool
    res = build_dtm_tool.invoke({})

    # 5. Assert known answer #4 from TEST_FIXTURE.md
    assert res["n_features"] == 5
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert pytest.approx(res["sparsity_pct"], abs=1e-3) == 47.5

    # Check session state updates
    assert session.feature_matrix is not None
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool():
    # 1. Initialize session and load premade tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 2. Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Instantiate tools and get build_dtm_tool and term_frequency_tool
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    # 4. Call build_dtm_tool first to populate state
    build_dtm_tool.invoke({})

    # 5. Call term_frequency_tool
    res = term_frequency_tool.invoke({})

    # 6. Assert term_frequencies dictionary
    assert res["term_frequencies"] == {"alpha": 6, "always": 7, "beta": 3, "delta": 4, "gamma": 7}


def test_dtm_heatmap_tool():
    # 1. Initialize session and load premade tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 2. Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Instantiate tools and get build_dtm_tool and dtm_heatmap_tool
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    # 4. Call build_dtm_tool first to populate state
    build_dtm_tool.invoke({})

    # 5. Call dtm_heatmap_tool
    res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    # 6. Assert slice_data key
    assert res["slice_data"] == [[3, 1], [2, 1], [1, 1]]
