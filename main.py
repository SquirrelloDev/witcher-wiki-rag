from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Wiedźmin RAG API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

db = Chroma(
    persist_directory="./baza_wiedzmin",
    embedding_function=embeddings_model
)

llm = Ollama(model="llama3")

prompt_template = """
Jesteś ekspertem od uniwersum Wiedźmina. Odpowiedz na pytanie użytkownika wykorzystując WYŁĄCZNIE poniższy kontekst. 
Jeśli w kontekście nie ma odpowiedzi, powiedz: "Moja baza danych nie zawiera odpowiedzi na to pytanie." Nie zmyślaj.
Odpowiadaj ZAWSZE w języku polskim.

Kontekst:
{ctx}

Pytanie użytkownika: {question}

Odpowiedź:
"""
prompt_rag = PromptTemplate(template=prompt_template, input_variables=["ctx", "question"])

prompt_template_raw = """
Jesteś asystentem AI. Odpowiedz na poniższe pytanie użytkownika najlepiej jak potrafisz, korzystając ze swojej wbudowanej wiedzy.
Odpowiadaj ZAWSZE w języku polskim.

Pytanie użytkownika: {question}

Odpowiedź:
"""
prompt_raw = PromptTemplate(template=prompt_template_raw, input_variables=["question"])

class Request(BaseModel):
    q: str

@app.post("/ask")
async def ask_rag(req: Request):
    dbResults = db.similarity_search(req.q, k=3)
    print("got results", dbResults)
    builtCTX = "\n\n".join([doc.page_content for doc in dbResults])
    print("context", builtCTX)
    prompt = prompt_rag.format(ctx=builtCTX, question=req.q)
    promptResp = llm.invoke(prompt)
    print("odpowiedz_llm", promptResp)
    return {
        "pytanie": req.q,
        "odpowiedz": promptResp,
        "znalezione_zrodla": [doc.metadata.get("zrodlo", "Nieznane") for doc in dbResults]
    }


@app.post("/ask_raw")
async def ask_raw_llm(req: Request):
    prompt = prompt_raw.format(question=req.q)

    promptResp = llm.invoke(prompt)

    return {
        "pytanie": req.q,
        "odpowiedz": promptResp,
        "znalezione_zrodla": ["Brak - Odpowiedź wygenerowana z pamięci modelu"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)