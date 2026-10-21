import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools


def _setup_session_and_dtm():
    session = SessionState()
    
    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    
    # Build DTM using build_dtm_tool from tools_dtm
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return session


def test_reduce_dimensions_pca():
    session = _setup_session_and_dtm()
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ])
    expected_evr = [0.7532, 0.1965]

    coords = np.array(res["coordinates"])
    evr = res["explained_variance_ratio"]

    assert np.allclose(coords, expected_coords, atol=1e-3)
    assert np.allclose(evr, expected_evr, atol=1e-3)
    assert session.pending_figure is not None


def test_reduce_dimensions_tsne():
    session = _setup_session_and_dtm()
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })

    coords = np.array(res["coordinates"])
    
    # Check shape (8 documents, 2 components)
    assert coords.shape == (8, 2)
    # Check that exact duplicates (rows 5 and 6) map to identical coordinates
    assert np.allclose(coords[5], coords[6], atol=1e-3)
    # Check that pending figure is set
    assert session.pending_figure is not None


def test_reduce_dimensions_umap():
    session = _setup_session_and_dtm()
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })

    coords = np.array(res["coordinates"])
    
    # Check shape (8 documents, 2 components)
    assert coords.shape == (8, 2)
    # Check that pending figure is set
    assert session.pending_figure is not None
