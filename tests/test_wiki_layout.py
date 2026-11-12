import hashlib

import pytest

from allies_bot.audit import PageSnapshot
from allies_bot.wiki_layout import propose_layout


def specimen(pid: int, source: str) -> PageSnapshot:
    return PageSnapshot(
        id=pid, title="Example", url="https://wiki.test/page/example",
        book="Rules", chapter=None, updated_at="now", html=source,
        content_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )


@pytest.mark.parametrize("pid", [102, 377, 378])
def test_approved_layout_preserves_content_and_is_idempotent(pid):
    if pid == 102:
        source = '<h5 id="bkmrk-"><a name="bkmrk-"></a><a href="/picture">' \
                 '<img src="/image" width="131" height="418"></a></h5>'
    else:
        titles = (
            ["For each Elohim", "Average, Non-tactical Players", "Tactical, Experienced Players",
             "For each unholy Elohim", "For each Unholy Elohim",
             "Average, Non-tactical Players", "Tactical, Experienced Players"]
            if pid == 377 else ["Archon", "Regular Enemy", "Other"]
        )
        source = "".join(
            f'<table id="table-{i}"><tbody><tr><td colspan="2"><strong>{title}</strong>'
            '</td></tr><tr><td>d12</td><td># of items</td></tr>'
            '<tr><td>1-5</td><td>1</td></tr></tbody></table>'
            for i, title in enumerate(titles)
        )
    page = specimen(pid, source)
    proposal, after = propose_layout(page)
    assert proposal.edits
    assert all(proposal.checks.values())
    rerun, twice = propose_layout(specimen(pid, after))
    assert twice == after
    assert not rerun.edits
    if pid == 102:
        assert after == source.replace("<h5", "<div").replace("</h5>", "</div>")
    elif pid == 377:
        assert after.count("<caption ") == 7
        assert after.index("<caption ") < after.index("<tbody>")
    else:
        assert after.count('scope="col"') == 4
        assert after.split('<table id="table-2">')[1] == source.split('<table id="table-2">')[1]


def test_unapproved_page_and_changed_source_are_rejected():
    with pytest.raises(ValueError, match="not a supported"):
        propose_layout(specimen(21, "<p>Herald</p>"))
    with pytest.raises(ValueError, match="table count"):
        propose_layout(specimen(377, "<table></table>"))
