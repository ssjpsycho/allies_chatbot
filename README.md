# Allies of Majesty Discord Bot

This bot answers Discord slash-command questions using only the Allies of Majesty BookStack wiki and the local EPUB. It is restricted to the Discord channel IDs you explicitly configure.

## How the project works

The project has two related workflows: a code workflow and a knowledge workflow.

### Code workflow

1. You edit the project locally in VS Code.
2. You run tests and lint checks locally.
3. Git commits the code changes and pushes them to the `main` branch of GitHub.
4. Railway watches the GitHub repository. A push to `main` starts a new deployment.
5. Railway builds the Docker image, installs the Python package, and starts the Discord bot.

Railway deploys the application code. It does not automatically copy your local `.env`, EPUB, or Python virtual environment. Railway secrets and variables must be configured separately in the Railway service.

### Knowledge workflow

The wiki and EPUB are not sent to Discord and are not committed to Git. They are converted into searchable knowledge in Qdrant:

1. The importer reads public BookStack pages through the BookStack API and extracts text from the EPUB.
2. Each source document is split into smaller overlapping chunks so relevant passages can be retrieved later.
3. OpenAI's embedding model converts each chunk into a numerical vector.
4. Qdrant stores the vectors together with the original text, source title, and wiki URL.
5. When a user asks `/ask`, the bot embeds the question, searches Qdrant, and retrieves relevant source passages.
6. The configured chat model receives the question, recent conversation context, and retrieved source passages.
7. The bot returns a source-grounded answer in Discord with citations.

The chat model generates the wording of an answer; it does not replace the source data. The embedding model is used for search, while Qdrant is the searchable store. Changing the chat model does not require re-indexing. Changing the embedding model requires rebuilding or migrating the Qdrant collection and re-indexing the sources.

### End-to-end workflow

```text
VS Code edit
   |
   v
Git commit and push to GitHub -----> Railway builds and runs the Discord bot

BookStack wiki + local EPUB
   |
   v
Importer -> chunks -> OpenAI embeddings -> Qdrant
                                    ^
                                    |
Discord question -> question embedding -> retrieval -> chat model -> cited answer
```

The daily GitHub Actions workflow checks whether BookStack page metadata changed. If it did, the workflow re-indexes the wiki in Qdrant and commits a small sync-state file to GitHub. That commit triggers Railway to redeploy the code. The workflow does not re-index the local EPUB because the EPUB is not available in GitHub Actions; run the EPUB command locally when the book changes.

## 1. Create accounts and credentials

1. Create a Discord application at https://discord.com/developers/applications, add a Bot, and copy its token.
2. In **OAuth2 > URL Generator**, select `bot` and `applications.commands`. Under Bot Permissions select `View Channels`, `Send Messages`, `Read Message History`, and `Use Application Commands`. Open the generated URL to add the bot to your server.
3. Enable Discord Developer Mode, right-click each allowed Discord channel, and choose **Copy Channel ID**.
4. Create a Qdrant Cloud cluster at https://cloud.qdrant.io/, then copy its cluster URL and API key.
5. Create an OpenAI API key at https://platform.openai.com/api-keys.
6. In your BookStack admin area, create an API token. Enter it as `token_id:token_secret`.

## 2. Configure locally

1. Copy `.env.example` to `.env`.
2. Fill in every blank value. Set `ALLOWED_CHANNEL_IDS` to comma-separated IDs, for example `123456789,987654321`.
3. `EPUB_PATH` already points to `../Allies of Majesty Chronicles vol 1.epub`, the file in this workspace.
4. Install dependencies:

   ```sh
   python3 -m pip install '.[dev]'
   ```

5. Index both sources. This costs embedding API usage and may take several minutes. The importer slows down and retries automatically if BookStack rate-limits requests, and batches both embeddings and Qdrant uploads for large source collections:

   ```sh
   python3 -m allies_bot.ingest --wiki --epub
   ```

6. Start the bot:

   ```sh
   python3 -m allies_bot.bot
   ```

7. In an allowed channel, run `/ask` and ask a question. The bot remembers the latest eight turns per server/channel/user in Qdrant, so follow-up questions can refer to earlier messages. Backend calls have timeouts and return a retry message if a service is unavailable. Long answers are split across Discord messages; `/sources` shows the configured sources.

## 3. Re-index sources

Run these commands from the project directory with the project environment active:

```sh
cd "/Users/benjaminh/Documents/Allies/AI Tools/allies-discord-bot"
source .venv/bin/activate
```

After adding or changing wiki pages, refresh the wiki only:

```sh
python3 -m allies_bot.ingest --wiki
```

After replacing or changing the EPUB, refresh both sources:

```sh
python3 -m allies_bot.ingest --wiki --epub
```

Re-indexing updates the existing Qdrant collection using stable chunk IDs, so it is safe to run again. It uses embedding API credits and may take several minutes. Use the current Qdrant, OpenAI, and BookStack credentials in `.env`. Do not run this from Railway unless the EPUB is available there; run it locally, then leave the bot deployed on Railway.

### How content is prepared for retrieval

- HTML is converted to Markdown-like text (headings, lists and tables are kept; scripts and styles are dropped).
- Table values carry their caption/group title, row label and column label in
  their own text. Explicit row/column headers and row/column spans are respected;
  unlabelled columns use positional labels instead of invented rule names.
  Blank cells stay explicitly empty, rather than inheriting adjacent rules.
- Long table cells repeat that context in every chunk. Table context that cannot
  fit the chunk budget, unsupported spans and unclosed tables stop ingestion
  explicitly rather than silently dropping associations or content.
- Pages are chunked by heading section. Every chunk begins with a `Page > Heading > Subheading` line, so it makes sense on its own when embedded. Tiny sections are merged with the next one.
- Each chunk stores `source_id` and `heading` in Qdrant. After a re-index, chunks that no longer exist (edited or deleted pages, and chunks from the old chunking scheme) are deleted for the source kind being indexed, so `--wiki` prunes wiki chunks and `--epub` prunes EPUB chunks.
- Chunks also store their per-source `chunk_index`. Explicit named-page requests
  (including `Minion Level 12`) retrieve every indexed chunk of that source in
  order, rather than only the nearest vector matches. Source-label lookups and
  direct damage-rule scans paginate to completion; explicit damage matches are
  not discarded by the normal top-k result cap.

Because chunk text changed, run `python3 -m allies_bot.ingest --wiki --epub` once to apply this to existing data.
The daily sync state also records an ingestion version, so an extractor upgrade
triggers a wiki refresh even when the source page metadata has not changed.
Image-only pages have no indexable text; this importer does not perform OCR.

### Order eligibility and answer coverage

Order-scoped questions include the requested Order's rule page, not just
character-creation questions. Unrequested other-Order Discipline sources, hybrid
Order options and Song families are excluded from standard single-Order options;
explicit questions about those options retain the evidence needed to discuss
their restrictions. Retrieved weapon permissions also bring in their full source,
including Character Creation Notes. Answers must distinguish `Me` (Messenger)
from `MS` (Ministering Spirit), and must not recommend weapons unavailable to
the requested Order.

Minion/enemy intent uses whole words, so generic Minion questions can retrieve
enemy stat pages while unrelated words do not accidentally enable enemy sources.
Named-page stat answers receive a checklist of every retrieved table group,
including multi-Order and bestial cards, to avoid summarising only a few examples.
Broad numeric Minion-stat requests are rendered directly from every source card
without a model summary, preserving exact values and avoiding omitted variants.
Equipment/ability slots are excluded from that numeric projection; conflicting
source values stop the answer explicitly. Specific-type and comparison questions
still use the evidence-grounded answer path.
These safeguards improve evidence coverage; they do not prove that every indirect
rule or eligibility exception in the game has been found. Generated answers still
need source-grounded regression checks, not just expected-page retrieval ranks.

### Data quality report

```sh
python3 -m allies_bot.quality --wiki                      # chunk stats and flagged pages (no OpenAI/Qdrant calls)
python3 -m allies_bot.quality --wiki --show-page "Page"   # print the chunks of one page
python3 -m allies_bot.quality --queries questions.json    # retrieval hit rate against the live index
```

`questions.json` is a list such as `[{"question": "How does Flame Song work?", "expected": "Flame Family"}]`; `expected` is matched against the retrieved page titles. Run it before and after re-indexing to measure the change.

### Read-only bulk audit and structured pilot

This separate tool **does not update BookStack, Qdrant, or the deployed bot**.
It uses BookStack GET requests only, with no OpenAI calls or embedding costs.
It needs only `BOOKSTACK_BASE_URL` and `BOOKSTACK_TOKEN` from `.env`.
Run from the project directory with the environment active:

```sh
PYTHONPATH=src python -m allies_bot.audit --output /path/outside/repository/wiki-audit
```

Choose a **new output directory**; existing files are never overwritten.
The directory contains:

- `snapshot.json`: page HTML, book/chapter hierarchy, source URLs, capture time, update times, and content hashes. This is a sequential read, not a transactional backup.
- `audit.json`: findings across every accessible page, including missing headings, irregular tables, unresolved links, unmatched progression entries, and potential prose/table rule discrepancies.
- `records.json`: source-linked Order, Spiritual Effect, and Song review records from the corresponding chapters of **Character Creation & Advancement**. Guidance and unrecognised sections remain reference records.
- `report.html`: local, readable findings, priority review items, all records, and a Resolve-damage candidate sweep for the original Ministering Spirit question.

Reproduce the audit without any service calls:

```sh
PYTHONPATH=src python -m allies_bot.audit \
  --snapshot /path/outside/repository/wiki-audit/snapshot.json \
  --output /path/outside/repository/wiki-audit-rerun
```

The pilot preserves entire source sections and table cells (including cell spans),
and links progression variants to their tier and position only when table alignment
is explicit. It does not merge unrelated sections or infer missing rules.
Empty HTML headings do not erase section context. A bold paragraph immediately
following an empty heading is recovered as a section title only if its entire
text matches an extracted progression entry; the recovery is flagged and its
original source anchor is retained.
Each extracted field contains exact quotes with evidence IDs, or an explicit
`unknown` status. **All records require review**: a damage/resistance mention is
not proof that an ability damages that attribute, and chapter classification is
not proof of Order eligibility. Tier variants may share a description; do not
apply base values to every variant.

The Resolve sweep examines every Spiritual Effect candidate rather than a top-k
vector result, but it is a review queue, not a complete or eligibility-validated
answer. Weapon rules, social actions, disciplines, lore, and EPUB content are
outside the structured pilot. Review and test records before using them in
production. Unresolved anchors are absent from API content; rendered-page
validity is not confirmed. Reports contain wiki content, so keep them outside
source control and share them only with authorised reviewers.

### Wiki formatting previews (no live edits)

To propose formatting repairs to **all accessible wiki pages**, without
changing any wording, mechanics, page titles, locations, links or existing IDs:

```sh
PYTHONPATH=src python -m allies_bot.wiki_format \
  --output /path/outside/repository/wiki-format-previews
```

This command uses only BookStack GET requests and requires only the BookStack
settings. It never writes to BookStack, Qdrant or deployment services.
Use `--snapshot /path/to/snapshot.json` to work entirely offline.
The output directory must be new and should be outside source control.

Open `index.html` in a browser to review every page. The review display uses dark
mode, including both preview panes, without changing the saved wiki HTML.
Each page has a sandboxed
side-by-side preview (remote media and scripts are blocked), the exact HTML
diff, manual-review notes, and preservation checks. The output also includes
`snapshot.json`, a machine-readable `plan.json` with source hashes and exact
edit ranges, and original/proposed HTML under `pages/<page-id>/`.
The original HTML files are backups, not sandboxed previews; review through
`index.html` rather than opening those raw files directly.

Proposals normalize heading levels below BookStack's page title, replace empty
headings with anchor-preserving spans, recover clearly identified progression
labels as headings, and mark unambiguous table row/column headers with `th`
and `scope`. Existing inline styles are retained, so some semantic improvements
may look identical. Wording, attributes other than added header scope, table
cell text/spans and IDs are checked for preservation, and each proposed result
must be idempotent.

Ambiguous bold text, headingless prose, nested/merged/irregular tables and
malformed markup are flagged rather than guessed. An unchanged page is not
necessarily structurally ideal. This does not fix existing broken links or
rewrite rules. **There is no apply mode**: review and approval come first.
Before any later live editing, current page hashes must be checked against the
approved plan so intervening edits cannot be overwritten.

After explicit approval, heading-only proposals can be applied separately:

```sh
PYTHONPATH=src python -m allies_bot.wiki_headings \
  --plan /path/to/wiki-format-previews/plan.json \
  --backup /path/outside/repository/wiki-heading-backups \
  --apply-approved-headings
```

This is a live BookStack update command; table proposals are excluded.
For separately approved table-header proposals, use `--apply-approved-tables`
instead of `--apply-approved-headings`, with the corresponding reviewed plan
and a new backup directory. Table mode applies only the proposed `th`/`scope`
edits; it preserves cell text and spans and does not change heading levels.
The same bookmark protection, concurrency checks, backups and verification apply.
It saves private full-page API backups and intended HTML before writing,
skips pages changed since review and Markdown-editor pages, rechecks just before
each PUT, and verifies saved HTML, metadata, anchors, text and unchanged tables.
Writes are never retried automatically. Failures stop the run and are recorded
in a durable journal; inspect backups before recovery or retry.
BookStack does not document atomic conditional updates, so avoid simultaneous
wiki edits during this operation. The pre-write check reduces but cannot eliminate
the concurrency window. Qdrant is not updated directly; the existing daily wiki
sync may later re-index the changed pages.
BookStack may regenerate heading anchor IDs during an HTML save, even when IDs
are supplied explicitly. The heading writer adds empty named bookmarks inside
headings to preserve their original fragment targets, rejecting conflicting
bookmarks. It permits server-generated heading ID changes only when all other
intended structure, text, attributes and bookmarks survive verification.
Tables must remain byte-for-byte unchanged. Re-submitting a backup through the
same API may also regenerate IDs:
do not assume that an API rollback restores anchors byte-for-byte, and inspect
the recorded before/after responses before taking further action.

A bounded test on the home page confirmed that empty named bookmarks
(`<a name="old-fragment"></a>`) inside the affected headings survive repeated
API saves on the currently deployed instance, even while nested heading IDs
change. Browser checks confirmed that both old and newer fragments reach their
intended sections. Percent-encoded IDs also needed decoded bookmark variants.
The heading writer now uses this compatibility approach. Decoded aliases are
added only for restricted ASCII spellings (including commas) that survive
BookStack's name-attribute normalization. Whitespace, non-ASCII and quote aliases
would be encoded back into duplicate bookmarks, so only their original encoded
form is added. Compatibility with later editor saves or BookStack upgrades must
be rechecked. Retain backups and verify rendered links after bulk updates.

### Focused triage of manual-review pages

After automatic heading and table repairs, triage the remaining review notes
from a fresh snapshot:

```sh
PYTHONPATH=src python -m allies_bot.wiki_triage \
  --snapshot /path/to/fresh-wiki-format-previews/snapshot.json \
  --output /path/outside/repository/wiki-manual-review
```

This command is **offline and preview-only**. Open the dark-mode `index.html`
for a page-by-page disposition, focused decision list, and links to before/after
previews. `triage.json` accounts for every flagged page, while `plan.json`
contains new repair proposals with source hashes and exact edit ranges.
No API writes or model calls are made.

The triage recognises reviewed, explicit source patterns: Mastery Level rows,
stat-card labels, numeric lookup axes and specific existing column labels.
It proposes row/column semantics without adding text, moving values, changing
cell spans or changing headings. Page-specific interpretations are deliberately
limited to the reviewed page IDs; this is not a general rule inference engine.
All proposed patches must pass content/attribute preservation and idempotence
checks. They need separate approval and are not accepted by the original
heading/table apply modes. After approval, apply only the concrete triage patches
with the dedicated mode:

```sh
PYTHONPATH=src python -m allies_bot.wiki_headings \
  --plan /path/outside/repository/wiki-manual-review/plan.json \
  --backup /path/outside/repository/wiki-reviewed-tables-backup \
  --apply-approved-reviewed-tables
```

This mode independently regenerates the reviewed proposals from their snapshot,
rejects altered patches, and uses the same backups, stale/concurrency checks,
bookmark protection and saved-content verification as the other apply modes.
Undecided sections are left unchanged even on pages with other approved repairs.

Long prose is not an error merely because it has no section headings, and bold
formulas/list labels remain emphasis. Those pages are recommended for retention,
with optional editorial restructuring explicitly separated from formatting.
Mixed-section tables, merged group titles and media-only headings retain their
original markup with source-backed recommendations for a human decision.
An outcome of `retain_current_structure` is a recommendation, not a claim that
the page has been rewritten or that every possible structural issue is resolved.

The separately approved layout decisions use `wiki_layout.propose_layout` and
the writer's `--apply-approved-layouts` mode with a matching fresh snapshot and
plan. This mode is restricted to Epic Blessings (image heading to ordinary block),
Balancing Encounters (existing title rows to captions, leaving data rows intact),
and Generating Items for Enemies (second-row column headers). Herald's intentional
Social Attributes grouping is retained. Layout plans are regenerated before
writing and use the same private backups, concurrency and saved-content checks.

## 4. Deploy to Railway

1. Create a GitHub repository from this project and push it.
2. At https://railway.app/, choose **New Project > Deploy from GitHub Repo** and select the repository.
3. Add every value from `.env` under Railway **Variables**. Never commit `.env`.
   Set `CHAT_MODEL` to the answer model you want, such as `gpt-4.1`, and set `EMBEDDING_MODEL` to `text-embedding-3-small`.
4. Deploy. Railway reads `railway.json`, builds the Dockerfile, and keeps the bot process running.
5. Railway runs the bot from the already indexed Qdrant collection. Re-index locally whenever the EPUB changes or wiki content needs a refresh, then verify the Railway bot is still running.

## 5. Automatic daily wiki sync

The repository includes a GitHub Actions workflow at `.github/workflows/wiki-sync.yml`. It runs daily at 03:17 UTC and can also be started manually from the GitHub **Actions** tab. It checks BookStack page metadata first; if nothing changed, it exits without embedding or committing anything. If the wiki changed, it updates Qdrant, commits `.wiki-sync-state.json`, and pushes to `main`, which triggers Railway's GitHub deployment.

Add these GitHub repository secrets under **Settings > Secrets and variables > Actions**:

- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `BOOKSTACK_BASE_URL`
- `BOOKSTACK_TOKEN`

The workflow does not need `DISCORD_TOKEN` because it never starts the bot. It does not re-index the local EPUB; run the full local command manually after changing that file:

```sh
python3 -m allies_bot.ingest --wiki --epub
```

## Security and behavior

- The bot accepts commands only in `ALLOWED_CHANNEL_IDS`; Discord permissions provide a second layer of control.
- It combines semantic retrieval with exact-term retrieval, which helps named mechanics such as `Endurance`, `damage`, and `Stops` appear even when a question is phrased broadly.
- For character-building questions, it checks Order and discipline eligibility and distinguishes Spiritual Effects from Songs before recommending options.
- Character-building responses use Discord-compatible headings and bullets rather than Markdown tables.
- Normal character-building answers exclude enemy and corruption pages; those sources are included only when the question explicitly asks about unholy, corrupted, enemy, or evil options.
- It retrieves source excerpts before answering and includes source links when the material came from the wiki.
- Do not place bot, OpenAI, Qdrant, or BookStack tokens in source control or Discord messages.

## Open in VS Code

Open this folder directly in VS Code. The `code .` command works only after
the VS Code shell command has been installed on your PATH.
