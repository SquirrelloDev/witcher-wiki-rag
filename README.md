# Witcher World Encyclopedia – Local RAG Chatbot

A locally running chatbot that answers questions about the universe of *The Witcher* using **Retrieval-Augmented Generation (RAG)**. Knowledge is scraped from the Polish [Wiedźmin Wiki](https://wiedzmin.fandom.com/wiki/Wied%C5%BAmin_Wiki), stored in a vector database, and passed as context to a local Llama 3 model, so answers are grounded in the source articles instead of the model's own (often hallucinated) knowledge.

> Originally created as a university project for the *Natural Language Processing* course (Computer Science, Społeczna Akademia Nauk, Łódź 2026).

## Why RAG?

Large language models have a fixed knowledge cutoff and tend to invent answers when they don't know something. This project shows the difference between a plain LLM and a RAG-backed one:

- **`/ask_raw`** – the question goes straight to the LLM (no retrieval).
- **`/ask`** – the question is embedded, the most relevant wiki fragments are retrieved, and the LLM is instructed to answer **only** from that context. If the context has no answer, it replies *"Moja baza danych nie zawiera odpowiedzi na to pytanie"* ("My database does not contain the answer to this question") instead of making something up.

## Features

- Automatic knowledge ingestion from a MediaWiki-based wiki (by category, with duplicate skipping)
- Persistent vector database with semantic search
- Questions in Polish and English; answers in Polish
- Source attribution: each answer shows which wiki articles were used
- Side-by-side comparison of answers with and without RAG
- Simple chat-style web UI
- Fully local – no paid external APIs
- Modular: swap the wiki URL and category names to build a bot for a different universe

## How it works

1. The user types a question in the web UI, which sends it as an HTTP POST (JSON) to the backend.
2. The backend turns the question into a vector with the `multilingual-e5-small` embedding model.
3. ChromaDB returns the 3 nearest text chunks (with article-title metadata).
4. The chunks are injected into a prompt template with a strict "don't make things up" instruction.
5. Llama 3 (8B), served by Ollama, generates the answer.
6. The answer and its sources are displayed in the chat window.

**Data ingestion** (`db_populate.py`): articles are discovered by walking wiki categories and subcategories (skipping duplicates), downloaded via the MediaWiki API, cleaned of HTML with BeautifulSoup, articles shorter than 200 characters are skipped, and the text is split with LangChain's `RecursiveCharacterTextSplitter` into chunks of 800 characters with a 100-character overlap. Chunks are stored in ChromaDB together with their article name.

## Tech stack

| Layer | Tools |
|---|---|
| Data source | Wiedźmin Wiki (Fandom), MediaWiki API, BeautifulSoup |
| Vector database | ChromaDB (persistent, file-system storage) |
| Embeddings | `intfloat/multilingual-e5-small` (Hugging Face) |
| LLM | Meta Llama 3 8B via Ollama |
| Backend | Python, FastAPI, LangChain |
| Frontend | HTML, vanilla JavaScript (SPA), Tailwind CSS |

## Requirements

To run the project locally you need:

- **Python 3.10+** (and `pip`)
- **[Ollama](https://ollama.com/)** installed and running, with the Llama 3 8B model pulled:
  ```bash
  ollama pull llama3
  ```
- **Python packages**: FastAPI with Uvicorn, LangChain (`langchain-core`, `langchain-community`, `langchain-huggingface`, `langchain-text-splitters`), ChromaDB, `sentence-transformers`, BeautifulSoup4 and `requests`, all listed in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```
- **Internet connection** – to scrape the wiki, download the embedding model on first run, and load Tailwind CSS from its CDN in the frontend.
- **Hardware**: enough RAM to run an 8B-parameter model locally (roughly 8–16 GB). A GPU is optional but makes responses noticeably faster; the project was developed to work without one.

## Getting started

```bash
# 1. Clone the repository
git clone https://github.com/SquirrelloDev/witcher-wiki-rag.git
cd witcher-wiki-rag

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure Ollama is running and the model is available
ollama pull llama3

# 5. Build the vector database from the wiki (run once; creates ./baza_wiedzmin)
python db_populate.py

# 6. Start the backend (http://127.0.0.1:8000)
python main.py

# 7. Open index.html in your browser and start asking questions
#    (the frontend talks to http://127.0.0.1:8000)
```

> **Note on build time:** `db_populate.py` deliberately waits between requests to the wiki (0.5 s between category requests and 2 s after each article). This rate limiting is a safeguard against getting your IP address banned by the wiki's servers, so please don't remove it. As a result, building the full database can take a long time (the article limit in `populateDb()` is set to 10,000); lower `limit` for a quicker first run.

## API

Both endpoints accept `POST` requests with a JSON body containing the question, e.g. `{"q": "Who was Geralt's master?"}`. The response contains the question, the answer, and the list of source articles.

| Endpoint | Description |
|---|---|
| `/ask` | RAG pipeline: retrieval + grounded answer with sources |
| `/ask_raw` | Plain LLM answer without retrieval (for comparison) |

## Example results

Three test questions were used to compare the plain model with the RAG version:

| Question | Without RAG | With RAG |
|---|---|---|
| Who was Geralt of Rivia's master? | Hallucinated the same character as his own master | Correct, matches the source |
| Where and how did Geralt meet Zoltan Chivay in Novigrad? (asked in English) | Invented place names that don't exist in the city | Initially mixed sources; correct once the question mentioned *The Witcher 3* |
| Did Vernon Roche open his own brewery after losing the war with Nilfgaard? (false premise) | Muddled answer with factual errors | Correctly stated that the database has no answer |

## Known limitations

- Llama 3 8B sometimes drops the requested answer language when given long, foreign-language context.
- The model has a fixed context window, so the number of retrieved chunks is limited (3).
- Vague questions can retrieve fragments from the wrong articles; adding context (e.g. the game title) helps.
- Data is fetched through the live wiki API, so it depends on the wiki's availability and current content.

## Possible improvements

- Use a model with a larger context window
- Move the whole pipeline to GPU for faster responses
- Add a loading animation to the UI
- Ingest wiki XML exports instead of using the REST API (when available)

## Adapting to another universe

Change the wiki API URL and the list of categories (`mainCats`) in `db_populate.py`, then re-run it to rebuild the vector database.
