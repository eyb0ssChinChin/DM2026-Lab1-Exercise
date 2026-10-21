"""Tests for mine_patterns_tool in tools_patterns.py."""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


@pytest.fixture
def setup_fixture():
    """Sets up a fresh session and loads sample_fixture.csv via load_dataset_tool."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")
    
    return session, mine_patterns_tool


def test_mine_patterns_zero_surviving_terms(setup_fixture):
    """Variance and term_frequency filtering should yield 0 terms and handle cleanly."""
    session, mine_patterns_tool = setup_fixture
    
    for category in ["catA", "catB"]:
        for method in ["variance", "term_frequency"]:
            res = mine_patterns_tool.invoke({
                "category_name": category,
                "filtering_method": method,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })
            assert isinstance(res, dict)
            assert "result_id" in res
            assert res.get("patterns") == [] or "note" in res


def test_mine_patterns_fpgrowth_tfidf(setup_fixture):
    """Test fpgrowth with tfidf filtering on catA and catB."""
    session, mine_patterns_tool = setup_fixture
    
    # catA, min_sup=3 -> {'alpha'}: 3
    res_a3 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 3
    })
    patterns_a3 = res_a3.get("patterns", [])
    assert len(patterns_a3) == 1
    assert set(patterns_a3[0]["itemset"]) == {"alpha"} or set(patterns_a3[0]["items"]) == {"alpha"} or patterns_a3[0].get("pattern") == ["alpha"] or "alpha" in str(patterns_a3[0])
    assert patterns_a3[0]["support"] == 3
    
    # catA, min_sup=4 -> 0 patterns
    res_a4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert len(res_a4.get("patterns", [])) == 0
    
    # catB, min_sup=4 -> {'gamma'}: 4
    res_b4 = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    patterns_b4 = res_b4.get("patterns", [])
    assert len(patterns_b4) == 1
    assert patterns_b4[0]["support"] == 4


def test_mine_patterns_topk_tfidf(setup_fixture):
    """Test topk (FAE) algorithm with tfidf filtering."""
    session, mine_patterns_tool = setup_fixture
    
    for k_val in [1, 2, 3]:
        # catA -> exact 1 pattern ({'alpha'}, support 3)
        res_a = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        patterns_a = res_a.get("patterns", [])
        assert len(patterns_a) == 1
        assert patterns_a[0]["support"] == 3
        
        # catB -> exact 1 pattern ({'gamma'}, support 4)
        res_b = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        patterns_b = res_b.get("patterns", [])
        assert len(patterns_b) == 1
        assert patterns_b[0]["support"] == 4


def test_mine_patterns_maxfpgrowth_tfidf(setup_fixture):
    """Test maxfpgrowth returns 0 patterns cleanly for both categories."""
    session, mine_patterns_tool = setup_fixture
    
    for category in ["catA", "catB"]:
        for min_sup in [1, 4]:
            res = mine_patterns_tool.invoke({
                "category_name": category,
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": min_sup
            })
            assert isinstance(res, dict)
            assert "result_id" in res
            assert len(res.get("patterns", [])) == 0
