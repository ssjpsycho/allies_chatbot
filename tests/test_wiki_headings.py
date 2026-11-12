import asyncio
import hashlib
import json

import httpx
import pytest

from allies_bot import wiki_headings
from allies_bot.audit import AuditSettings, PageSnapshot, WikiSnapshot
from allies_bot.wiki_format import propose_page, write_preview
from allies_bot.wiki_headings import (
    apply_headings,
    heading_html,
    protect_heading_links,
    table_edits_html,
    table_html,
    verify_compatible_save,
)
from allies_bot.wiki_layout import propose_layout
from allies_bot.wiki_triage import propose_reviewed_page

SOURCE = (
    '<h4 id="title">Title</h4>'
    '<table><tr><td>Class</td><td>Effect</td></tr>'
    '<tr><td>Type</td><td>Rule</td></tr></table>'
)


def page() -> PageSnapshot:
    return PageSnapshot(
        id=1, title="Example", url="https://wiki.test/books/rules/page/example",
        book="Rules", chapter=None, updated_at="2026-11-12",
        html=SOURCE, content_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
    )


def test_heading_selection_never_applies_table_edits() -> None:
    proposal, _ = propose_page(page())
    assert any("table headers" in e.reason for e in proposal.edits)
    result = heading_html(SOURCE, proposal)
    assert '<h2 id="title">Title</h2>' in result
    assert table_html(result) == table_html(SOURCE)
    assert "<th" not in result


def test_table_selection_keeps_heading_levels_and_cell_content() -> None:
    proposal, _ = propose_page(page())
    result = table_edits_html(SOURCE, proposal)
    assert '<h4 id="title">Title</h4>' in result
    assert '<th scope="col">Class</th>' in result
    assert "<td>Rule</td>" in result


@pytest.mark.parametrize("scenario", [
    "success", "tables", "reviewed_tables", "layouts", "id_change", "stale", "concurrent", "markdown",
    "server_changed", "write_error",
])
def test_live_apply_checks_and_backups(tmp_path, monkeypatch, scenario: str) -> None:
    source = SOURCE
    undecided = '<table><tr><td colspan="2">Group</td></tr><tr><td>1</td><td>2</td></tr></table>'
    if scenario == "reviewed_tables":
        source = SOURCE.replace(">Class<", ">Tier<").replace(
            ">Effect<", ">Maximum Social Attribute Values<",
        ) + undecided
    specimen = page().model_copy(update={
        "html": source, "content_sha256": hashlib.sha256(source.encode()).hexdigest(),
    })
    if scenario == "layouts":
        source = '<h5 id="bkmrk-"><img src="/picture"></h5>'
        specimen = specimen.model_copy(update={
            "id": 102, "html": source, "content_sha256": hashlib.sha256(source.encode()).hexdigest(),
        })
    snapshot = WikiSnapshot(captured_at="now", base_url="https://wiki.test", pages=[specimen])
    preview = tmp_path / "preview"
    if scenario == "layouts":
        write_preview(snapshot, preview, proposer=propose_layout)
    elif scenario == "reviewed_tables":
        write_preview(snapshot, preview, proposer=propose_reviewed_page)
    else:
        write_preview(snapshot, preview)
    backup = tmp_path / "backup"
    original = {
        "id": 1, "name": "Example", "html": source, "updated_at": "2026-11-12",
        "revision_count": 1, "book_id": 1, "chapter_id": 0, "slug": "example",
        "priority": 1, "tags": [], "markdown": "", "editor": "wysiwyg",
    }
    state = original.copy()
    writes = []
    reads = 0

    async def get(_client, path):
        nonlocal reads
        reads += 1
        data = state.copy()
        if scenario == "stale":
            data["html"] += "<p>Other edit</p>"
        elif scenario == "concurrent" and reads == 2:
            data["revision_count"] = 2
        elif scenario == "markdown":
            data["markdown"] = "# Title"
        return httpx.Response(200, json=data)

    async def put(_client, path, *, json):
        assert (backup / f"{specimen.id}-before.json").exists()
        assert (backup / f"{specimen.id}-intended.json").exists()
        writes.append(json)
        request = httpx.Request("PUT", "https://wiki.test" + path)
        if scenario == "write_error":
            return httpx.Response(503, request=request)
        state["html"] = json["html"]
        if scenario == "id_change":
            state["html"] = state["html"].replace('id="title"', 'id="title-0"')
        if scenario == "server_changed":
            state["html"] += "<p>Unexpected</p>"
        return httpx.Response(200, request=request, json=state)

    monkeypatch.setattr(wiki_headings, "get_bookstack", get)
    monkeypatch.setattr(httpx.AsyncClient, "put", put)
    settings = AuditSettings(bookstack_base_url="https://wiki.test", bookstack_token="test")
    if scenario == "server_changed":
        with pytest.raises(ValueError, match="verification failed"):
            asyncio.run(apply_headings(preview / "plan.json", backup, settings))
    elif scenario == "write_error":
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(apply_headings(
                preview / "plan.json", backup, settings, tables=scenario == "tables",
                reviewed_tables=scenario == "reviewed_tables",
            ))
        assert len(writes) == 1
    else:
        asyncio.run(apply_headings(
            preview / "plan.json", backup, settings, tables=scenario == "tables",
            reviewed_tables=scenario == "reviewed_tables",
            approved_layouts=scenario == "layouts",
        ))
    events = [json.loads(line) for line in (backup / "journal.jsonl").read_text().splitlines()]
    statuses = {event["status"] for event in events}
    if scenario in {"success", "id_change", "tables", "reviewed_tables", "layouts"}:
        assert "verified" in statuses
        assert set(writes[0]) == {"html", "changelog"}
        if scenario == "layouts":
            assert '<div id="bkmrk-"><img src="/picture"></div>' == state["html"]
        elif scenario == "reviewed_tables":
            assert '<th scope="col">Tier</th>' in state["html"]
            assert '<h4 id="title">' in state["html"]
            assert table_html(state["html"])[-1] == undecided
        elif scenario == "tables":
            assert '<th scope="col">Class</th>' in state["html"]
            assert '<h4 id="title">' in state["html"]
        else:
            assert table_html(state["html"]) == table_html(SOURCE)
        assert json.loads((backup / f"{specimen.id}-before.json").read_text()) == original
    elif scenario in {"stale", "concurrent", "markdown"}:
        assert not writes
        assert any(status.startswith("skipped") for status in statuses)
    elif scenario == "server_changed":
        assert "verification_failed" in statuses
        assert "completed" not in statuses
    else:
        assert "write_failed_or_uncertain" in statuses


@pytest.mark.parametrize("tamper", ["edit", "hash", "wrong_mode"])
def test_reviewed_apply_rejects_unapproved_plan(tmp_path, tamper: str) -> None:
    source = '<table><tr><td>Mastery Level 1</td><td>Rule</td></tr>' \
             '<tr><td>Mastery Level 2</td><td>Other</td></tr></table>'
    specimen = page().model_copy(update={
        "html": source, "content_sha256": hashlib.sha256(source.encode()).hexdigest(),
    })
    preview = tmp_path / "preview"
    write_preview(
        WikiSnapshot(captured_at="now", base_url="https://wiki.test", pages=[specimen]),
        preview, proposer=propose_reviewed_page,
    )
    path = preview / "plan.json"
    plan = json.loads(path.read_text())
    if tamper == "edit":
        plan["pages"][0]["edits"][0]["after"] = "<td>"
    elif tamper == "hash":
        plan["pages"][0]["after_sha256"] = "0" * 64
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="verified source|preservation failed"):
        asyncio.run(apply_headings(
            path, tmp_path / "backup",
            AuditSettings(bookstack_base_url="https://wiki.test", bookstack_token="test"),
            reviewed_tables=tamper != "wrong_mode", tables=tamper == "wrong_mode",
        ))
    assert not (tmp_path / "backup").exists()


def test_compatibility_bookmarks_survive_heading_id_change() -> None:
    source = '<blockquote><h3 id="bkmrk-a%2C">A, title</h3></blockquote>'
    intended, names = protect_heading_links(source)
    assert names == {"bkmrk-a%2C", "bkmrk-a,"}
    saved = intended.replace('id="bkmrk-a%2C"', 'id="bkmrk-a%2C-0"')
    assert verify_compatible_save(intended, saved, names)
    assert protect_heading_links(intended)[0] == intended
    assert not verify_compatible_save(intended, saved.replace('name="bkmrk-a,"', 'name="lost"'), names)
    assert not verify_compatible_save(intended, saved.replace("A, title", "Changed"), names)
    assert not verify_compatible_save(intended, saved.replace("<h3", "<h2"), names)


def test_bookmark_collisions_rejected() -> None:
    with pytest.raises(ValueError, match="collision"):
        protect_heading_links('<a name="bkmrk-title"></a><h2 id="bkmrk-title">Title</h2>')
    with pytest.raises(ValueError, match="collision"):
        protect_heading_links('<p id="bkmrk-a,">Other</p><h2 id="bkmrk-a%2C">Title</h2>')


def test_whitespace_bookmark_variants_are_not_duplicated() -> None:
    intended, names = protect_heading_links('<h2 id="bkmrk-title%C2%A0">Title</h2>')
    assert names == {"bkmrk-title%C2%A0"}
    assert intended.count('name="') == 1
    _, unicode_names = protect_heading_links('<h2 id="bkmrk-%E2%80%9Ctitle">Title</h2>')
    assert unicode_names == {"bkmrk-%E2%80%9Ctitle"}
    _, quoted_names = protect_heading_links('<h2 id="bkmrk-%22title%22">Title</h2>')
    assert quoted_names == {"bkmrk-%22title%22"}
