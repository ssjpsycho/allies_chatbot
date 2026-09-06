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
