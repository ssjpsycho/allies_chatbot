import asyncio
import hashlib
import json
import sys

import pytest

from allies_bot import wiki_format
from allies_bot.audit import PageSnapshot, WikiSnapshot
from allies_bot.wiki_format import (
    Edit,
    apply_edits,
    preservation_checks,
    propose_page,
    write_preview,
)


def page(source: str) -> PageSnapshot:
    return PageSnapshot(
        id=43, title="Heal Family", url="https://wiki.test/books/rules/page/heal",
        book="Character Creation & Advancement", chapter="Spiritual Effects",
        updated_at="2026-11-12", html=source,
        content_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )


PROGRESSION = (
    '<table id="progress"><tr><td>Progression</td><td>Tier:1</td><td>Tier:2</td></tr>'
    '<tr><td>1</td><td><strong>Laying on Hands</strong></td>'
    '<td><strong>Sanctified Laying on Hands</strong></td></tr></table>'
)


def test_heading_repairs_preserve_all_existing_anchors_and_words() -> None:
    source = (
        PROGRESSION + '<h4 id="empty"></h4>'
        '<p id="visual"><strong>Laying on Hands</strong></p>'
        '<h4 id="empty2"></h4><p>Restore twice the amount given.</p>'
        '<h4 id="manna">Manna</h4><h6 id="nested">Variants</h6>'
        '<p>Keep <a href="#empty">this</a> and <img src="/image.png" alt="image">.</p>'
    )
    proposal, after = propose_page(page(source))
    assert proposal.status == "proposed"
    assert '<span id="empty"></span>' in after
    assert '<h2 id="visual"><strong>Laying on Hands</strong></h2>' in after
    assert '<h2 id="manna">Manna</h2><h3 id="nested">Variants</h3>' in after
    assert all(proposal.checks.values())
    assert all(preservation_checks(source, after).values())
    assert apply_edits(source, proposal.edits) == after
    assert propose_page(page(after))[1] == after


def test_table_headers_gain_scope_without_changing_cells() -> None:
    source = (
        '<table><tr><td></td><td>Class</td><td>Allotment</td><td>Effect</td></tr>'
        '<tr><td id="name"><strong>Manna</strong></td><td>Manifesting</td>'
        '<td>3</td><td>Restores 40 Endurance</td></tr></table>'
    )
    proposal, after = propose_page(page(source))
    assert '<th scope="col">Class</th>' in after
    assert '<th id="name" scope="row"><strong>Manna</strong></th>' in after
    assert "<td>3</td>" in after
    assert all(proposal.checks.values())
    assert propose_page(page(after))[1] == after


@pytest.mark.parametrize("source", [
    ('<table><tr><td colspan="2">Class</td><td>Effect</td></tr>'
     '<tr><td>Name</td><td>Type</td><td>Damage</td></tr></table>'),
    '<table><tr><td>Class</td><td>Effect</td></tr><tr><td>Name</td></tr></table>',
    '<table><tr><td><strong>Mastery Level 1</strong></td><td>+24 Passion</td></tr></table>',
])
def test_ambiguous_or_merged_tables_are_not_guessed(source: str) -> None:
    proposal, after = propose_page(page(source))
    assert source == after
    assert proposal.review_notes


def test_arbitrary_bold_paragraphs_not_promoted() -> None:
    source = '<p id="bold"><strong>Important rule</strong></p><p>Rule details.</p>'
    proposal, after = propose_page(page(source))
    assert after == source
    assert any("manual review" in note for note in proposal.review_notes)


def test_escaped_entities_unicode_comments_and_attributes_preserved() -> None:
    source = (
        '<!-- Keep this -->\n<H4 ID="title" class=\'p2\' style="color: red">'
        'Faith &amp; Hope</H4>\n<p title="a &quot;quote&quot;">'
        '12\u201d &nbsp; &lt; 14 <a href="/page#anchor">text</a></p>'
    )
    proposal, after = propose_page(page(source))
    assert '<h2 ID="title" class=\'p2\' style="color: red">' in after
    assert "Faith &amp; Hope</h2>" in after
    assert '<!-- Keep this -->' in after
    assert all(proposal.checks.values())


def test_unbalanced_html_is_blocked_not_silently_repaired() -> None:
    source = '<h4 id="title">Title</h4><p><strong>Unclosed'
    proposal, after = propose_page(page(source))
    assert proposal.status == "blocked"
    assert not proposal.edits
    assert after == source
    assert not proposal.checks["balanced_html"]


def test_self_closing_non_void_heading_is_blocked() -> None:
    source = '<h4 id="x"/><p>Text</p>'
    proposal, after = propose_page(page(source))
    assert proposal.status == "blocked"
    assert after == source


def test_repeated_scope_attributes_not_introduced() -> None:
    source = (
        '<table><tr><th scope="col">Class</th><th scope="col">Effect</th></tr>'
        '<tr><th scope="row">Manna</th><td>Heal</td></tr></table>'
    )
    proposal, after = propose_page(page(source))
    assert source == after
    assert not proposal.edits


def test_preview_preserves_original_newline_bytes(tmp_path) -> None:
    source = '<h4 id="x">Title</h4>\r\n<p>Keep these lines.</p>\r\n'
    p = page(source)
    snapshot = WikiSnapshot(captured_at="2026-11-12", base_url="https://wiki.test", pages=[p])
    output = tmp_path / "previews"
    write_preview(snapshot, output)
    assert (output / "pages/43/before.html").read_bytes() == source.encode()
    assert (output / "pages/43/after.html").read_bytes() == source.replace("h4", "h2").encode()


def test_apply_edits_rejects_stale_or_overlapping_ranges() -> None:
    with pytest.raises(ValueError, match="mismatched"):
        apply_edits("abc", [Edit(start=0, end=1, before="x", after="a", reason="test")])
    with pytest.raises(ValueError, match="Overlapping"):
        apply_edits("abc", [
            Edit(start=0, end=2, before="ab", after="a", reason="test"),
            Edit(start=1, end=3, before="bc", after="b", reason="test"),
        ])


def test_preview_is_offline_sandboxed_and_contains_all_pages(tmp_path) -> None:
    p = page('<h4 id="x">Title</h4><script>alert(1)</script><img src="https://remote.test/x">')
    snapshot = WikiSnapshot(captured_at="2026-11-12", base_url="https://wiki.test", pages=[p])
    out = tmp_path / "previews"
    plan = write_preview(snapshot, out)
    assert len(plan.pages) == 1
    assert plan.mode == "preview_only"
    assert (out / "pages/43/before.html").read_text() == p.html
    preview = (out / "pages/43/preview.html").read_text()
    assert 'sandbox=""' in preview
    assert "Content-Security-Policy" in preview
    assert "<script>" not in preview  # source is confined to escaped srcdoc
    assert "&lt;script&gt;" in preview
    assert "color-scheme:dark" in preview
    assert "background:#0d1117" in preview
    assert (out / "index.html").exists()
    assert json.loads((out / "plan.json").read_text())["pages"][0]["checks"]["idempotent"]
    with pytest.raises(FileExistsError):
        write_preview(snapshot, out)


def test_offline_cli_never_contacts_services(tmp_path, monkeypatch) -> None:
    p = page("<h4>Title</h4>")
    snapshot = WikiSnapshot(captured_at="2026-11-12", base_url="https://wiki.test", pages=[p])
    source = tmp_path / "snapshot.json"
    source.write_text(snapshot.model_dump_json())
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "wiki_format", "--snapshot", str(source), "--output", str(output),
    ])

    async def forbidden(*_args):
        raise AssertionError("Offline preview contacted BookStack.")

    monkeypatch.setattr(wiki_format, "capture_wiki", forbidden)
    asyncio.run(wiki_format.main())
    assert (output / "index.html").exists()
