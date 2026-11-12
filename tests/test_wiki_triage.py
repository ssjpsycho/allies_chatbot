import hashlib
import json

import pytest

from allies_bot.audit import PageSnapshot, WikiSnapshot
from allies_bot.wiki_format import preservation_checks
from allies_bot.wiki_triage import inspect_page, triage


def page(source: str, page_id: int = 999) -> PageSnapshot:
    return PageSnapshot(
        id=page_id, title="Example", url=f"https://wiki.test/page/{page_id}",
        book="Rules", chapter=None, updated_at="today", html=source,
        content_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )


def test_mastery_levels_are_row_headers_not_column_headers() -> None:
    source = (
        '<table id="mastery"><tr><td>Mastery Level 1</td><td>+3 Force</td></tr>'
        '<tr><td>Mastery Level 2</td><td>+6 Force</td></tr></table>'
    )
    proposal, after, review = inspect_page(page(source))
    assert review.outcome == "repair_proposed"
    assert '<th scope="row">Mastery Level 1</th><td>+3 Force</td>' in after
    assert 'scope="col"' not in after
    assert all(proposal.checks.values())
    assert all(preservation_checks(source, after).values())
    assert inspect_page(page(after))[1] == after


@pytest.mark.parametrize("title_id", [350, 351, 376])
def test_minion_stat_cards_do_not_gain_false_column_headers(title_id: int) -> None:
    source = (
        '<table><tr><td>Warrior</td><td>ML:3</td></tr>'
        '<tr><td>Tier:1</td><td>1 Magnitude</td></tr>'
        '<tr><td>Force</td><td>3</td></tr></table>'
    )
    _, after, _ = inspect_page(page(source, title_id))
    assert '<th scope="row">Warrior</th><td>ML:3</td>' in after
    assert '<th scope="row">Force</th><td>3</td>' in after
    assert 'scope="col"' not in after


def test_numeric_axis_tables_keep_all_values() -> None:
    source = (
        '<table><tr><td>Might</td><td>&lt; 14</td><td>14</td></tr>'
        '<tr><td>Force Value</td><td>3</td><td>4</td></tr></table>'
    )
    _, after, _ = inspect_page(page(source, 24))
    assert '<th scope="col">&lt; 14</th>' in after
    assert '<th scope="row">Force Value</th>' in after
    assert all(preservation_checks(source, after).values())
    # No guessing the same interpretation on unrelated pages.
    _, other, review = inspect_page(page(source, 999))
    assert other == source
    assert review.outcome == "decision_needed"


def test_merged_tables_left_intact_with_specific_recommendation() -> None:
    source = (
        '<table><tr><td colspan="2">Archon</td></tr>'
        '<tr><td>d12</td><td># of items</td></tr>'
        '<tr><td>1-5</td><td>no items</td></tr></table>'
    )
    _, after, review = inspect_page(page(source, 378))
    assert after == source
    assert review.outcome == "decision_needed"
    assert "second rows" in review.items[0].recommendation
    assert "Archon" in review.items[0].evidence


def test_long_prose_and_formulas_not_rewritten() -> None:
    source = '<p>' + "This is prose. " * 120 + '</p><p><strong>Force = 3 + Might</strong></p>'
    _, after, review = inspect_page(page(source))
    assert after == source
    assert review.outcome == "retain_current_structure"
    assert {"editorial_optional", "keep_emphasis"} == {item.kind for item in review.items}


def test_conflicting_existing_scopes_not_overwritten() -> None:
    source = (
        '<table><tr><th scope="col">Mastery Level 1</th><td>Force +3</td></tr>'
        '<tr><td>Mastery Level 2</td><td>Force +6</td></tr></table>'
    )
    _, after, review = inspect_page(page(source))
    assert '<th scope="col">Mastery Level 1</th>' in after
    assert review.outcome == "repair_and_decisions"


def test_header_labels_matched_exactly_without_inventing_text() -> None:
    source = (
        '<table><tr><td>Action</td><td>AP Cost</td><td>Additional Cost</td></tr>'
        '<tr><td>Initiate</td><td>2 AP</td><td>2 Exertion</td></tr></table>'
    )
    _, after, review = inspect_page(page(source))
    assert 'scope="col"' in after
    assert "column headers" in review.items[0].recommendation
    assert all(preservation_checks(source, after).values())


def test_all_flagged_pages_accounted_for_in_offline_review(tmp_path) -> None:
    a = page('<p>' + 'Long prose. ' * 150 + '</p>', 1)
    b = page('<table><tr><td>Mastery Level 1</td><td>+3</td></tr>'
             '<tr><td>Mastery Level 2</td><td>+6</td></tr></table>', 2)
    c = page('<p>No issue.</p>', 3)
    snapshot = WikiSnapshot(captured_at="today", base_url="https://wiki.test", pages=[a, b, c])
    out = tmp_path / "review"
    result = triage(snapshot, out)
    assert {p.page_id for p in result} == {1, 2}
    data = json.loads((out / "triage.json").read_text())
    assert data["pages_scanned"] == 3
    assert data["flagged_pages"] == 2
    assert sum(data["outcomes"].values()) == 2
    assert "color-scheme:dark" in (out / "index.html").read_text()
    assert (out / "pages/2/preview.html").exists()
    assert not (out / "pages/3").exists()
