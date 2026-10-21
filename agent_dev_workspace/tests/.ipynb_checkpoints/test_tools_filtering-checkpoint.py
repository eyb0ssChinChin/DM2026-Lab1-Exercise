import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
    # 1. Initialize session and load premade load_dataset_tool
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 2. Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Build DTM matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. Invoke variance_filter_tool with threshold=0.15
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    res = variance_filter_tool.invoke({"threshold": 0.15})

    # 5. Assert known answer #5 from TEST_FIXTURE.md
    assert res["kept_count"] == 4
    assert res["removed_count"] == 1
    assert set(res["kept_features"]) == {"alpha", "beta", "delta", "gamma"}
    assert res["removed_features"] == ["always"]

    # Verify full report DataFrame stored in session
    stored = session.results_store[res["result_id"]]
    full_report = stored["full_report"]
    assert full_report.columns[0] == "feature"
    assert full_report.columns[1] == "variance"

    # Verify calculated variance values against TEST_FIXTURE.md table
    variance_map = dict(zip(full_report["feature"], full_report["variance"]))
    assert pytest.approx(variance_map["alpha"], abs=1e-4) == 1.1875
    assert pytest.approx(variance_map["gamma"], abs=1e-4) == 1.109375
    assert pytest.approx(variance_map["delta"], abs=1e-4) == 0.25
    assert pytest.approx(variance_map["beta"], abs=1e-4) == 0.234375
    assert pytest.approx(variance_map["always"], abs=1e-4) == 0.109375
