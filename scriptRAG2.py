import ollama

# Texte a vectoriser
texte = "L'ANSSI recommande de sécuriser les accès réseau."

# Appel à Ollama pour générer l'embedding
response = ollama.embeddings(
    model="nomic-embed-text",
    prompt=texte
)

# recuperation du vecteur
vecteur = response["embedding"]

# Le resultat
print(f"Type de l'objet : {type(vecteur)}")
print(f"Dimension du vecteur (longueur de la liste) : {len(vecteur)}")
print(f"Appercu des 5 premieres coordonnées : {vecteur[5]}")