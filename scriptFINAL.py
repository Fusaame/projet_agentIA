import chromadb
import ollama
from scriptRAG5 import get_embedding

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("guide_anssi_complet")

question = (
    "Quelles sont les formations recommandées pour les équipes opérationnelles ?"
)

# --- À COMPLÉTER ---
# a. Génère le vecteur de la question avec get_embedding(..., is_query=True)
# b. Fais la requête dans collection.query(..., n_results=3)
# c. Concatène les documents récupérés dans une variable `contexte`
# d. Envoie le tout à ollama.generate(model="llama3.1", prompt=...)

# 4. Retrieval: Recup des 3 chunks les plus proches
vecteur_q = get_embedding(question, is_query=True)
resultat = collection.query(query_embeddings=[vecteur_q], n_results=3)

# Assemblage du contexte
contexte = "\n\n".join(resultat["documents"][0])

# 5. Generation: Prompt structure pour Llama 3.1
prompt = f"""Tu es un expert en cybersécurité. Réponds à la question posée en t'appuyant STRICTEMENT sur le contexte fourni ci-dessous. Si l'information n'est pas présente dans le texte, indique-le clairement.
CONTEXTE : 
{contexte}

QUESTION :
{question}

"""
print("--- Generation de la reponse avec Llama 3.1... ---")
response = ollama.generate(model="llama3.1", prompt=prompt)

print("\n--- REPONSE DU RAG ---")
print(response["response"])




