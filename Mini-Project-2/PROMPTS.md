# PROMPTS.md

This is a record of how I built the voice-activated Library App together with an AI coding assistant (Claude, by Anthropic).

**A note on how this was written.** I actually worked in Turkish and English, in many short messages, with terminal output pasted in between, one step at a time. For this file I translated my prompts into English, merged the small follow-up messages that belonged together, and removed the logs. So it is not a word-for-word transcript, but the order of the steps and the decisions behind them are the real ones.

---

## 1. Setting the scene

> I already have my Week 4 Flask Library App with `Book`, `DVD`, `LibraryCollection` and eight REST endpoints. I want to make it voice-controlled with Azure Speech Services. Every endpoint should work from a spoken command, and after each command the app should say out loud what happened, including the errors (404 and 409). Please don't change the Week 4 API or its status codes. Before writing any code, tell me how you would structure this.

**What came out of it.** A plan with two layers: the Week 4 API stays exactly as it was, and a thin voice layer goes on top. The browser listens to the microphone, the server parses the command and creates the spoken answer.

## 2. Keeping the secrets safe

> The Azure key and region must live in a local `.env` file, and `.env` must be in `.gitignore` so it never reaches GitHub. Please also add a `.env.example` with fake values and load everything with `python-dotenv`. The browser must never get to see my real key.

**What came out of it.** `.env`, `.env.example`, the `.gitignore` entry and `load_dotenv()` at the top of `app.py`. The browser only receives a short-lived token (see step 5).

## 3. Turning a sentence into a command

> Write a separate module, `voice_parser.py`, that takes the text Azure heard and turns it into a command: an intent, an item id, some fields and some filters. It should not know anything about Flask or Azure, so I can test and explain it easily. One sentence style per endpoint, for example "Create item, type book, title Dune, author Frank Herbert, pages 412", "Check out item 3", "List available books" and "Update item 3, pages 250".
>
> Speech recognition makes predictable mistakes, so please handle spoken numbers ("item three", "two hundred twenty"), "item for" meaning item four, "D V D" meaning DVD, and both "220 pages" and "pages 220".
>
> One more thing: decide the intent only from the part of the sentence before the first field word. Otherwise a title like "The Return of the King" would be understood as a return command.

**What came out of it.** `normalize`, `extract_item_id`, `extract_fields`, `detect_intent`, `extract_filters` and `parse_command`.

## 4. What the librarian says

> Now create `voice_speech.py`, which builds the sentence the librarian speaks. It gets the parsed command, the HTTP status and the response body. Please use the wording from the assignment, like "Successfully created book 'The Catcher in the Rye' with ID 3." and "Sorry, item 3 is already checked out." I need one sentence for each intent and each status that matters (200, 201, 204, 400, 404, 409). A long list should be cut to five items, and nothing should sound strange when read aloud, for example underscores.

**What came out of it.** `speech_for()` and its small helpers, plus a friendly help sentence for commands the app does not understand.

## 5. The voice part of the backend

> Add three endpoints to `app.py`. First, `GET /api/speech-token`, which trades my key for a short-lived Azure token. Second, `POST /api/speak`, which turns a sentence into WAV audio with Azure text-to-speech, keeps it in memory and caches repeated sentences. Third, `POST /api/voice`, which parses the