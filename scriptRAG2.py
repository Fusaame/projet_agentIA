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
# Tu demandes le 6e élément (index 5) :
print(f"Aperçu : {vecteur[5]}")

# Pourquoi ce nombre et à quoi sert-il ?
# Son index (l'emplacement) : vecteur[5] correspond au 6ᵉ axe (car les index débutent à 0 en Python) parmi les 768 dimensions créées par le modèle nomic-embed-text.
# Sa valeur (le poids) : Ce nombre traduit l'intensité d'une caractéristique sémantique ou grammaticale abstraite identifiée par le réseau de neurones dans la phrase « L'ANSSI recommande de sécuriser les accès réseau. ».
# L'interprétation : Prise isolément, cette 6ᵉ valeur n'a aucun sens pour un humain (elle ne veut pas dire « réseau » ou « ANSSI »). C'est la combinaison globale des 768 coordonnées qui forme l'« empreinte numérique » du sens de la phrase.
# Chaque phrase (ou chunk) est représentée par un seul vecteur, et ce vecteur possède 768 coordonnées (ce qu'on appelle un espace à 768 dimensions).