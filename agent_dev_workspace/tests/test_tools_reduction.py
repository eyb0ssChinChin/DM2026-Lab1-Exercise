import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools


def setup_fixture_session():
    """Helper to initialize a session loaded with sample_fixture.csv and build a basic DTM."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM using build_dtm_tool from tools_dtm
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return session


def test_reduce_dimensions_pca():
    session = setup_fixture_session()
    reduction_tools = make_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({"method": "pca", "n_components": 2})
    
    # Check return structure
    assert res is not None
    assert "coordinates" in res or "explained_variance_ratio" in res or "result_id" in res
    
    # Check known answer #7 coordinates
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])
    expected_var = [0.7532, 0.1965]
    
    if "coordinates" in res:
        actual_coords = np.array(res["coordinates"])
        np.testing.assert_allclose(actual_coords, expected_coords, atol=1e-3)
        
    if "explained_variance_ratio" in res:
        np.testing.assert_allclose(res["explained_variance_ratio"], expected_var, atol=1e-3)
        
    # Verify plot requirement
    assert session.pending_figure is not None


def test_reduce_dimensions_tsne():
    session = setup_fixture_session()
    reduction_tools = make_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})
    
    assert res is not None
    assert "coordinates" in res
    actual_coords = np.array(res["coordinates"])
    
    # Check shape (8 documents, 2 components)
    assert actual_coords.shape == (8, 2)
    # Check that duplicate documents (rows 5 and 6) produce close coordinates
    np.testing.assert_allclose(actual_coords[5], actual_coords[6], atol=1e-2)
        
    # Verify plot requirement
    assert session.pending_figure is not None


def test_reduce_dimensions_umap():
    session = setup_fixture_session()
    reduction_tools = make_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    
    assert res is not None
    assert "coordinates" in res
    actual_coords = np.array(res["coordinates"])
    
    # Check shape (8 documents, 2 components)
    assert actual_coords.shape == (8, 2)
    # Check that duplicate documents (rows 5 and 6) produce close coordinates
    np.testing.assert_allclose(actual_coords[5], actual_coords[6], atol=1.0)
        
    # Verify plot requirement
    assert session.pending_figure is not None


def test_binarize_labels_tool():
    session = setup_fixture_session()
    reduction_tools = make_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    
    res = binarize_tool.invoke({})
    assert res is not None
