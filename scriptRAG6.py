from pypdf import PdfReader
from scriptRAG5 import get_embedding
import chromadb

# 1. Charger le fichier
anssi = PdfReader('guide_anssi.pdf')

# 2. Extraire le texte de toutes les pages dans une chaîne
texte_complet = ""
for page in anssi.pages:
    texte_page = page.extract_text()
    if texte_page:
        texte_complet += texte_page + "\n"

# 3. On regarde si ça a été découpé correctement avec 1 saut de ligne
paragraphes = texte_complet.split("\n")
print(len(paragraphes))

chunks = []
chunk_now = ""

for ligne in paragraphes:
    # Si ajouter la ligne dépasse 1000 caractères, on valide le chunk
    if len(chunk_now) + len(ligne) > 1000:
        chunks.append(chunk_now)
        chunk_now = ligne
    else:
        # Sinon on empile la ligne (ajoute un espace ou un saut de ligne)
        chunk_now += "" + ligne

if chunk_now:
    chunks.append(chunk_now)

#print(f"Nombre de chunks : {len(chunks)}")
#print("--- Premier chunk ---")
#print(chunks[0])

# Génération de la liste des 80 vecteurs (is_query=False par défaut)
vecteurs = [get_embedding(chunk) for chunk in chunks]

# Génération des identifiants uniques
ids = [f"anssi_{i}" for i in range(len(chunks))]

# Métadonnées simples
metadatas = [{"source": "guide_anssi.pdf"} for _ in chunks]
# Connexion avec persistance locale
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="guide_anssi_complet", metadata={"hnsw:space": "cosine"}
)

# Insertion massive des 80 éléments
collection.add(
    ids=ids, documents=chunks, embeddings=vecteurs, metadatas=metadatas
)

print(f"Base mise à jour ! Nombre total d'éléments : {collection.count()}")

################################ Affichage Database ################################

# 3. Récupération des 5 premiers morceaux
donnees = collection.get(limit=5)

for i in range(len(donnees["ids"])):
    print(f"--- CHUNK ID : {donnees['ids'][i]} ---")
    print(f"Source : {donnees['metadatas'][i]['source']}")
    print(f"Contenu :\n{donnees['documents'][i]}\n")
    print("=" * 60)


