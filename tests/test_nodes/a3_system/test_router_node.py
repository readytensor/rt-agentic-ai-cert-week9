from nodes.a3_nodes import route_from_reviewer
from consts import (
    NEEDS_REVISION,
    TLDR_GENERATOR,
    TITLE_GENERATOR,
    REFERENCES_GENERATOR,
)


def test_route_from_reviewer_returns_end_when_no_revision_needed():
    state = {
        NEEDS_REVISION: False,
    }
    result = route_from_reviewer(state)
    assert result == "end"


def test_route_from_reviewer_returns_end_when_key_missing():
    state = {}  # No NEEDS_REVISION key — should default to False
    result = route_from_reviewer(state)
    assert result == "end"


def test_route_from_reviewer_routes_to_revision_dispatcher():
    state = {
        NEEDS_REVISION: True,
    }
    result = route_from_reviewer(state)
    assert result == [TLDR_GENERATOR, TITLE_GENERATOR, REFERENCES_GENERATOR]
