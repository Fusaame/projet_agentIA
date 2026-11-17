import chromadb
import ollama
from pypdf import PdfReader


def get_embedding(texte: str, is_query: bool = False) -> list[float]:
  prefixe = "search_query: " if is_query else "search_document: "
  response = ollama.embeddings(
      model="nomic-embed-text",
      prompt=f"{prefixe}{texte}",
  )
  return response["embedding"]


# 1. Extraction brute du PDF
reader = PdfReader("ING4_rules.pdf")
texte_complet = "\n".join(
    page.extract_text() for page in reader.pages if page.extract_text()
)

# 2. Découpage en chunks de 1000 caractères
chunks = []
chunk_now = ""
for ligne in texte_complet.split("\n"):
  if len(chunk_now) + len(ligne) > 1000:
    chunks.append(chunk_now.strip())
    chunk_now = ligne
  else:
    chunk_now += " " + ligne

if chunk_now:
  chunks.append(chunk_now.strip())

# 3. Stockage dans une collection dédiée au test PDF
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="ing4_rules_pdf_test", metadata={"hnsw:space": "cosine"}
)

vecteurs = [get_embedding(c) for c in chunks]
ids = [f"pdf_{i}" for i in range(len(chunks))]
metadatas = [{"source": "ING4_rules.pdf"} for _ in chunks]

collection.add(
    ids=ids, documents=chunks, embeddings=vecteurs, metadatas=metadatas
)
print(f"Indexation terminée : {collection.count()} chunks créés.\n")

# 4. Affichage des derniers chunks (là où se trouvent les règles R1-R4)
for i in range(max(0, len(chunks) - 3), len(chunks)):
  print(f"=== CHUNK {ids[i]} ===")
  print(chunks[i])
  print()
