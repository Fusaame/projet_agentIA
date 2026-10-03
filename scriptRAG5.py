import chromadb
import ollama

# 1. Initialiser le client ChromaDB avec persistance locale sur disque
client = chromadb.PersistentClient(path="./chroma_db")

# 2. Créer ou récupérer une collection
"""
Distance de 0.0 : vecteurs identiques (sens identique à 100 %).

Distance entre 0.20 et 0.45 : forte proximité sémantique. Les deux scores obtenus (0.3785 et 0.4075) se situent exactement dans cette zone de pertinence.

Le chunk de cuisine a été complètement écarté du Top 2 car sa distance était bien plus élevée (probablement supérieure à 0.65 ou 0.70).
"""
collection = client.get_or_create_collection(
    name="document_rag_cosine",
    metadata={"hnsw:space": "cosine"}  # <-- Force le calcul de similarité cosinus
)

def get_embedding(texte: str, is_query:bool = False) -> list[float]:
    """Génère l'embedding via Ollama avec le bon préfixe Nomic."""
    prefixe = "search_query: " if is_query else "search_document: "
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=f"{prefixe}{texte}",
    )
    return response["embedding"]

# 3. Nos données d'exemple (textes, métadonnées, identifiants)
chunks = [
    "Guide ANSSI : il est recommandé de segmenter les réseaux et d'isoler les environnements critiques.",
    "L'authentification multifacteur (MFA) renforce l'accès aux interfaces d'administration.",
    "Pour préparer des pâtes au poulet, faites dorer les escalopes avant d'ajouter la crème."
]

ids = ["doc_1", "doc_2", "doc_3"]
metadatas = [
    {"source": "guide_anssi.pdf", "theme": "reseau"},
    {"source": "guide_anssi.pdf", "theme": "authentification"},
    {"source": "recette.txt", "theme": "cuisine"}
]

vecteurs = [get_embedding(chunk) for chunk in chunks]

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=vecteurs,
    metadatas=metadatas
)

print(f"Nombre d'éléments dans la base : {collection.count()}")

# 6. test de recherche par similarité (Retrieval)
question = "Comment sécuriser les connexions administrateur ?"
vecteur_ask = get_embedding(question, is_query=True)

resultats = collection.query(
    query_embeddings=[vecteur_ask],
    n_results=2 # Top 2 des résultats les plus proches
)

print("\n--- Résultat de la recherche ---")
for i, (doc, dist) in enumerate(zip(resultats["documents"][0], resultats["distances"][0])):
    print(f"Top {i+1} [Distance: {dist:.4f}] : {doc}")