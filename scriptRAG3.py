# Les résultats confirment la théorie : 0,6278 pour les deux phrases de sécurité contre 0,5534 avec la cuisine. Le modèle capte bien que les phrases 1 et 2 partagent le même univers sémantique.


import math
import ollama

def cos_sim(a, b):
    # On fait le produit scalaire
    dot_product = sum(x * y for x, y in zip(a, b))
    # Normes euclidienne
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot_product / (norm_a * norm_b)

def get_embedding(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response['embedding']

# Trois phrases de test
phrase1 = "A critical vulnerability was detected on the server."
phrase2 = "A major security flaw affects our infrastructure."
phrase3 = "I love cooking a good chicken and pasta recipe."

v1 = get_embedding(phrase1)
v2 = get_embedding(phrase2)
v3 = get_embedding(phrase3)

similarite = cos_sim(v1, v2)
similarite_no = cos_sim(v1, v3)

print(f"Similarité entre phrase 1 et phrase 2 (meme sujet) : {similarite:.4f}")
print(f"Similarité entre prase 1 et phrase 3 (pas le meme sujet) : {similarite_no:.4f}")