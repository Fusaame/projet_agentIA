####################### AGENT IA : Retrieval-Augmented Generation  #######################

import chromadb
import ollama

def get_embedding(texte: str, is_query: bool = False) -> list[float]:
  prefixe = "search_query: " if is_query else "search_document: "
  response = ollama.embeddings(
      model="nomic-embed-text",
      prompt=f"{prefixe}{texte}",
  )
  return response["embedding"]


# Connexion à la base
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="guide_anssi_complet")

print("Agent IA ANSSI Ready ! Pose tes questions (tape q ou exit pour quitter).\n")

while True:
  question = input("Vous > ").strip()
  if not question:
    continue
  if question.lower() in ["q", "quit", "exit"]:
    break

  # Retrieval
  vecteur_q = get_embedding(question, is_query=True)
  resultats = collection.query(
      query_embeddings=[vecteur_q],
      n_results=3
  )

  # Récupération sécurisée du contexte

  # Extrait la liste des textes trouvés (renvoie [[]] par sécurité si la clé 'documents' est absente)
  # Le [0] permet de récupérer la liste de chunks de notre unique question
  docs = resultats.get("documents", [[]])[0]
  # Fusionne les chunks avec un double saut de ligne pour créer un bloc de texte lisible
  # Si aucun chunk n'a été trouvé, attribue un texte par défaut au lieu de laisser la variable vide
  contexte = "\n\n".join(docs) if docs else "Aucun extrait trouvé."

  # Ajout du 'f' devant le prompt triple-quote
  prompt = f"""Tu es un expert en cybersécurité.

Sers-toi du CONTEXTE ci-dessous pour répondre précisément à la QUESTION. 
Si le contexte permet de répondre, appuie-toi dessus.
Si la question est générale (ex: définition d'un concept cyber comme DICP ou OWASP) et absente du texte, réponds avec tes connaissances de base en le précisant brièvement.

CONTEXTE : 
{contexte}

QUESTION :
{question}
"""

  # Modèle en minuscules : llama3.1
  response = ollama.generate(
      model="llama3.1",
      prompt=prompt
  )
# response['response'] extrait le texte généré par le modèle depuis le dictionnaire renvoyé par la fonction ollama.generate()
  print(f"\nAgent : {response['response']}\n" + "-" * 50)