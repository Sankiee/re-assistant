# RE Assistant — Royal Enfield Troubleshooting Assistant

Chat over the official Royal Enfield service & owner's manuals using text,
voice, or images. Powered by GPT-4o, ChromaDB, and Sarvam AI.

## Tech stack

- **Backend:** Python · FastAPI · ChromaDB · LangChain · OpenAI · Sarvam
- **Frontend:** Next.js 14 (App Router) · TypeScript · Tailwind CSS

## Setup

### 1. Clone and install

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd ../frontend
npm install
```

### 2. Add API keys

Edit `backend/.env`:

```
OPENAI_API_KEY=your_key_here
SARVAM_API_KEY=your_key_here
```

`SARVAM_API_KEY` is optional — voice endpoints will degrade gracefully if
it's missing.

### 3. Add manuals

Drop PDFs into `backend/data/manuals/`:

- `ALL_NEW_CLASSIC_350_SERVICE_MANUAL_EURO_V.pdf`
- `Himalayan_Engine_Manual.pdf`

The owner's manuals for Classic 350 and Himalayan are downloaded
automatically from royalenfield.com during ingestion. The Meteor 350 and
Bullet 350 owner's manuals live on manualslib.com (auth required) — see
the `NOTE` in `backend/modules/ingest.py` to add them manually.

### 4. Run ingestion (one time only)

```bash
cd backend && python modules/ingest.py
```

This embeds the PDFs into per-model ChromaDB collections under
`backend/vectorstore/`. Idempotent — already-populated collections are
skipped on rerun.

### 5. Start the app

```bash
bash start.sh
```

Open <http://localhost:3000>.

The script starts the backend (`uvicorn` on `:8000`) and the frontend
(`next dev` on `:3000`) together and shuts both down on Ctrl+C.

## Features

- Text troubleshooting grounded in the official RE manuals
- Voice input/output via Sarvam STT (`saarika:v2`) + TTS (`bulbul:v2`)
- Image-based diagnosis via GPT-4o vision (`/chat/image`)
- Combined voice + image (`/voice/image-chat`)
- Per-conversation memory with rolling summarization and 30-day expiry
- Strict guardrails — never hallucinates beyond manual content; declines
  comparisons, pricing, and other brands with a polite redirect

## Models supported

- **Classic 350** — J-platform, Euro V
- **Himalayan** — LS410, Euro IV / BS6
- **Meteor 350** — J-platform, shared service manual
- **Bullet 350** — J-platform, shared service manual

The canonical list is served by `GET /models` so the frontend stays in
sync with the backend.

## Project layout

```
re-assistant/
├── start.sh                   one-command launcher
├── backend/
│   ├── run.sh                 backend launcher (uvicorn)
│   ├── main.py                FastAPI app + endpoints
│   ├── requirements.txt
│   ├── .env / .env.example
│   ├── data/
│   │   ├── manuals/           PDFs go here
│   │   └── conversations/     per-chat JSON files
│   ├── vectorstore/           ChromaDB persistent storage
│   └── modules/
│       ├── ingest.py          PDF → chunks → embeddings
│       ├── retrieval.py       per-model vector search
│       ├── llm.py             GPT-4o + guardrails
│       ├── memory.py          conversation persistence + summarization
│       ├── voice.py           Sarvam STT + TTS
│       └── vision.py          GPT-4o vision + query builder
└── frontend/
    ├── app/                   pages, layout, error, loading
    ├── components/            ModelSelector, ChatWindow, MessageBubble,
    │                          InputBar, MicButton, ImageUpload, …
    └── lib/                   typed API client, model registry, helpers
```

## API surface

| Method | Path                              | Purpose                             |
| ------ | --------------------------------- | ----------------------------------- |
| GET    | `/health`                         | Liveness check                      |
| GET    | `/models`                         | Supported models                    |
| POST   | `/ingest`                         | Re-ingest all manuals               |
| POST   | `/chat`                           | Text query                          |
| POST   | `/chat/image`                     | Image query                         |
| POST   | `/voice/transcribe`               | STT only                            |
| POST   | `/voice/synthesize`               | TTS only                            |
| POST   | `/voice/chat`                     | Voice round-trip                    |
| POST   | `/voice/image-chat`               | Voice + image combined              |
| POST   | `/conversations`                  | Create a conversation               |
| GET    | `/conversations`                  | List conversations (optional model) |
| GET    | `/conversations/{id}`             | Full conversation detail            |
| DELETE | `/conversations/{id}`             | Delete a conversation               |

Interactive docs: <http://localhost:8000/docs>.

## Costs heads-up

- OpenAI embeddings on first ingest (~$0.02 per million tokens) — the
  Classic 350 service manual alone is ~95 MB.
- Each chat call: 1 GPT-4o completion.
- Each `/chat/image` call: GPT-4o vision (image describe) + GPT-4o
  (query build) + GPT-4o (final answer) ≈ 3 calls.
- Voice round-trips additionally hit Sarvam STT + TTS per request.
