import json

import pytest

from allies_bot.ingest import html_to_text, sync_state_changed
from allies_bot.knowledge import chunk_document


def test_each_table_value_retains_caption_column_and_row():
    source = (
        "<h2>Items</h2><table><caption>Archon</caption>"
        '<tr><th scope="col">d12</th><th scope="col"># of items</th></tr>'
        '<tr><th scope="row">1-5</th><td>2</td></tr></table>'
    )
    text = html_to_text(source)
    assert "Table: Archon; Row: 1-5; Column: # of items\t2" in text
    assert "## Items" in text


def test_long_table_cells_repeat_context_in_every_chunk():
    source = (
        '<table><caption>Heal</caption><tr><th>Name</th><th>Effect</th></tr>'
        '<tr><th scope="row">Holy Revive</th><td>' + "Restore Resolve. " * 120
        + "</td></tr></table>"
    )
    chunks = chunk_document("Heal Family", html_to_text(source), chunk_size=400)
    assert len(chunks) > 2
    for text, _ in chunks:
        assert len(text) <= 400
        if "Restore Resolve." in text:
            assert "Table: Heal; Row: Holy Revive; Column: Effect" in text


def test_enemy_title_row_is_not_mistaken_for_column_header():
    source = (
        '<table><tr><td colspan="2">Regular Enemy</td></tr>'
        '<tr><th scope="col">d12</th><th scope="col"># of items</th></tr>'
        '<tr><td>1-5</td><td>1</td></tr></table>'
    )
    assert "Table: Regular Enemy; Row: 1-5; Column: # of items\t1" in html_to_text(source)


def test_unlabelled_table_does_not_invent_headers():
    text = html_to_text("<table><tr><td>3</td><td>7</td></tr><tr><td>12</td><td>14</td></tr></table>")
    assert "Column 1\t3" in text
    assert "Column 2\t7" in text
    assert "Column: 3" not in text


def test_rowspan_and_colspan_headers_follow_logical_columns():
    source = (
        '<table><tr><th rowspan="2">Name</th><th colspan="2">Stats</th></tr>'
        '<tr><th>Cost</th><th>Range</th></tr>'
        '<tr><th scope="row" rowspan="2">Manna</th><td>3</td><td>Self</td></tr>'
        '<tr><td>7</td><td>Self</td></tr></table>'
    )
    text = html_to_text(source)
    assert "Row: Manna; Column: Stats / Cost\t7" in text
    assert "Row: Manna; Column: Stats / Range\tSelf" in text


def test_nested_tables_do_not_erase_outer_text():
    text = html_to_text(
        "<table><tr><td>Before<table><tr><td>Inside</td></tr></table>After</td></tr></table>"
    )
    assert all(word in text for word in ("Before", "Inside", "After"))


def test_script_markup_cannot_inject_table_structure():
    text = html_to_text("<script><table><tr><td>Hidden</td></tr></table></script><p>Visible</p>")
    assert text == "Visible"


def test_implicit_row_headers_and_blank_cells_are_not_filled():
    text = html_to_text(
        "<table><tr><th>Holy Revive</th><td>5</td></tr>"
        "<tr><th>Revive</th><td></td></tr></table>"
    )
    assert "Row: Holy Revive; Column 2\t5" in text
    assert "Row: Revive; Column 2\t(empty)" in text


def test_caption_entities_and_inline_blocks_preserve_word_boundaries():
    text = html_to_text(
        "<table><caption>Rank &amp; Honor</caption><tr><td>First<br>Second<p>Third</p>Fourth"
        "</td></tr></table>"
    )
    assert "Column 1\tFirst Second Third Fourth" in text
    assert "Table: Rank & Honor;" in text


def test_table_context_does_not_leak_into_following_table_or_prose():
    text = html_to_text(
        "<table><caption>One</caption><tr><td>First</td></tr></table><p>Between</p>"
        "<table><tr><td>Second</td></tr></table>"
    )
    assert "Table; Row: Second; Column 1\tSecond" in text
    assert "\n\nBetween\n\n" in text


def test_section_group_titles_replace_rather_than_accumulate():
    text = html_to_text(
        '<table><caption>Main</caption><tr><td colspan="2">Group A</td></tr>'
        '<tr><td>A</td><td>1</td></tr><tr><td colspan="2">Group B</td></tr>'
        '<tr><td>B</td><td>2</td></tr></table>'
    )
    assert "Table: Main / Group B; Row: B; Column 2\t2" in text
    assert "Group A / Group B" not in text


def test_changed_ingestion_forces_refresh_even_when_wiki_is_unchanged(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"wiki_fingerprint": "same"}))
    assert sync_state_changed(state, "same")


def test_unclosed_table_and_unsupported_spans_fail_explicitly():
    with pytest.raises(ValueError, match="Unclosed"):
        html_to_text("<table><tr><td>Rule</td></tr>")
    with pytest.raises(ValueError, match="Unsupported"):
        html_to_text('<table><tr><td rowspan="0">Rule</td></tr></table>')


def test_encounter_row_identifier_survives_without_column_headers():
    source = (
        "<table><caption>Tactical, Experienced Players</caption>"
        "<tr><td>Trivial Encounter</td><td>" + "Fraction below 1.0 " * 120
        + "</td></tr></table>"
    )
    chunks = chunk_document("Encounters", html_to_text(source), chunk_size=400)
    for text, _ in chunks:
        if "Fraction below" in text:
            assert "Table: Tactical, Experienced Players; Row: Trivial Encounter" in text


def test_minion_stat_card_keeps_role_and_level_on_each_stat():
    text = html_to_text(
        '<table><tr><th scope="row">Warrior</th><td>ML:12</td></tr>'
        '<tr><th scope="row">Force</th><td>7</td></tr></table>'
    )
    assert "Table: Warrior (ML:12); Row: Force; Column 2\t7" in text
