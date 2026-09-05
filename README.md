# Allies of Majesty Discord Bot

This bot answers Discord slash-command questions using only the Allies of Majesty BookStack wiki and the local EPUB. It is restricted to the Discord channel IDs you explicitly configure.

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

## 3. Deploy to Railway

1. Create a GitHub repository from this project and push it.
2. At https://railway.app/, choose **New Project > Deploy from GitHub Repo** and select the repository.
3. Add every value from `.env` under Railway **Variables**. Never commit `.env`.
4. Deploy. Railway reads `railway.json`, builds the Dockerfile, and keeps the bot process running.
5. Run source indexing from your computer whenever the EPUB changes or wiki content needs a refresh. Automated scheduled sync can be added after the first successful deployment.

## Security and behavior

- The bot accepts commands only in `ALLOWED_CHANNEL_IDS`; Discord permissions provide a second layer of control.
- It retrieves source excerpts before answering and includes source links when the material came from the wiki.
- Do not place bot, OpenAI, Qdrant, or BookStack tokens in source control or Discord messages.

## Open in VS Code

Open this folder directly in VS Code. The `code .` command works only after
the VS Code shell command has been installed on your PATH.
