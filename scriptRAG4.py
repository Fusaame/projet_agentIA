# On veut vectoriser une liste de texte
"""
    Transforme une liste de segments textuels (chunks) en représentations vectorielles (embeddings).

    Fonctionnement du pipeline :
    1. Itération sur chaque segment de texte brut.
    2. Ajout du préfixe 'search_document:' recommandé par le modèle Nomic pour optimiser l'indexation.
    3. Appel au modèle local 'nomic-embed-text' via Ollama pour projeter le texte dans un espace à 768 dimensions.
    4. Structuration des données sous forme de dictionnaire associant l'identifiant, le contenu original et le vecteur dense.

    Paramètres :
        chunks (list[str]) : Liste des morceaux de texte issus du découpage (chunking) d'un document.

    Retour :
        list[dict] : Liste d'objets prêts pour l'indexation vectorielle, formatés comme suit :
            [
                {
                    "id": int,              # Identifiant numérique unique du chunk
                    "texte": str,           # Texte brut d'origine (utilisé pour le contexte LLM)
                    "vecteur": list[float]  # Coordonnées sémantiques (longueur : 768) pour la recherche de similarité
                },
                ...
            ]
    """
import ollama


def vectoriser_chunks(chunks: list[str]) -> list[dict]:
    """
        Prend en entrée une liste de morceaux de texte
        et retourne une structure contenant le texte et son embedding.
    """
    documents_vectorises = []

    for i, chunk in enumerate(chunks):
        texte_encode = f"search_document: {chunk}"

        response = ollama.embeddings(
            model='nomic-embed-text',
            prompt=texte_encode
        )

        documents_vectorises.append({
            "id": i,
            "text": chunk,
            "vecteur": response["embedding"]
        })

    return documents_vectorises


# petits test rapide pour comprendre notre fonction
chunks_test = [
    "Guide ANSSI : durcissement des systèmes d'exploitation.",
    "L'authentification multifacteur réduit fortement le risque d'intrusion.",
    "Les architectures microservices nécessitent une surveillance réseau active."
]

resultat = vectoriser_chunks(chunks_test)
print(f"Nombre d'éléments indexés : {len(resultat)}")
print(f"Premier élément : id={resultat[0]['id']}, taille vecteur={len(resultat[0]['vecteur'])}")
