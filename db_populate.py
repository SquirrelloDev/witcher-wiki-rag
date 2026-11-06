import requests
import time
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

db = Chroma(
    persist_directory="./baza_wiedzmin",
    embedding_function=embeddings_model
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ".", " "]
)

def getArticles(kategorie_startowe, limit=50):
    """
    Algorytm BFS przeszukujący kategorie i podkategorie z pomijaniem duplikatów.
    """
    api_url = "https://wiedzmin.fandom.com/api.php"

    initCategories = list(kategorie_startowe)
    visitedCats = set()
    uniqueArticles = set()

    print("\nSkanuję kategorii w poszukiwaniu unikalnych artykułów")

    while initCategories and len(uniqueArticles) < limit:
        cat = initCategories.pop(0)

        if cat in visitedCats:
            continue

        visitedCats.add(cat)
        print(f"Skanowanie: Kategoria:{cat}")

        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Kategoria:{cat}",
            "cmlimit": 50,
            "cmtype": "page|subcat",
            "format": "json"
        }

        try:
            response = requests.get(api_url, params=params)
            data = response.json()
            members = data.get('query', {}).get('categorymembers', [])

            for item in members:
                # Is article
                if item['ns'] == 0:
                    uniqueArticles.add(item['title'])
                # Is subcat
                elif item['ns'] == 14:
                    subCatName = item['title'].replace("Kategoria:", "")
                    if subCatName not in visitedCats:
                        initCategories.append(subCatName)

            time.sleep(0.5)

        except Exception as e:
            print(f"Błąd przy kategorii {cat}: {e}")

    return list(uniqueArticles)[:limit]


def getArticleContent(title):
    api_url = "https://wiedzmin.fandom.com/api.php"
    params = {
        "action": "parse", "format": "json", "page": title,
        "prop": "text", "disabletoc": True
    }
    try:
        response = requests.get(api_url, params=params)
        data = response.json()
        if "error" in data: return None
        rawHTML = data["parse"]["text"]["*"]
        soup = BeautifulSoup(rawHTML, "html.parser")
        for tag in soup.find_all(['table', 'script', 'style', 'nav', 'sup', 'div', 'img']):
            if tag.name == 'div' and not tag.get('class'): continue
            if tag.name == 'div' and 'mw-parser-output' in tag.get('class', []): continue
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except:
        return None


def populateDb():
    mainCats = ["Postacie", "Bestiariusz", "Geografia", "Organizacje", "Rasy", "Magia"]
    limit = 10000

    articlesResult = getArticles(mainCats, limit=limit)
    print(f"\n Znaleziono {len(articlesResult)} unikalnych artykułów.\n")

    for index, tytul in enumerate(articlesResult, start=1):
        print(f"[{index}/{len(articlesResult)}] Ściągam tekst dla: {tytul}...")

        rawText = getArticleContent(tytul)

        if not rawText or len(rawText) < 200:
            print("   -> Pomijam (artykuł zbyt krótki).")
            continue

        chunks = text_splitter.split_text(rawText)
        metadata = [{"zrodlo": tytul} for _ in chunks]

        db.add_texts(texts=chunks, metadatas=metadata)
        print(f"   -> Dodano {len(chunks)} wektorów do bazy.")

        time.sleep(2)

    print("\nZakończono")


if __name__ == "__main__":
    populateDb()