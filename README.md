# Projet RAG Local & Agent IA — Documentation Technique

Ce référentiel technique documente la conception, l'initialisation et la mise en œuvre pas-à-pas d'une architecture **RAG** (*Retrieval-Augmented Generation*) 100 % locale, autonome et gratuite, articulée autour d'un pipeline Python, du moteur d'inférence **Ollama** et de la base vectorielle persistante **ChromaDB**.

---

## 1. Objectif du Projet

L'objectif principal est de construire un système d'interrogation documentaire intelligent pour exploiter des documents techniques spécialisés (notamment le `guide_anssi.pdf`, `Data-Centric_Model_for_Architectures_Vulnerabilities_Analysis.pdf`, et `Projet_IA_Agent_SystemDesign.pdf`).

Le système résout les limites inhérentes aux modèles de langage (LLM) :
1. **Absence de connaissances privées** : Un LLM généraliste ne connaît pas les documents internes non publics.
2. **Hallucinations** : En contraignant le LLM à répondre exclusivement à partir d'extraits sourcés, les réponses deviennent vérifiables et auditables.
3. **Confidentialité & Coût** : Exécution intégrale sur la machine locale sans appel API payant ni envoi de données vers des serveurs tiers.

---

## 2. Initialisation & Versioning Git / GitHub

Pour assurer un suivi rigoureux du code, le projet a été initialisé en ligne de commande et lié à un dépôt distant GitHub.

### Procédure suivie en ligne de commande
```bash
# 1. Se placer dans le dossier racine
cd "/Users/dejesuscabral/project/ESIEE IT/ING4/PROJET_TRANSVERSE"

# 2. Initialiser Git localement et définir la branche principale
git init
git branch -M main

# 3. Indexer et commiter les premiers fichiers
git add .
git commit -m "Initial commit"

# 4. Lier le dépôt distant (URL du repo GitHub créé vide)
git remote add origin https://github.com/Fusaame/projet_agentIA.git

# 5. Pousser la branche locale
git push -u origin main
```

> **Remarque technique (Erreur de rejet `fetch first`) :**  
> Si le dépôt GitHub distant est créé avec un fichier préexistant (`README`, `.gitignore` ou licence), un rejet de type non-fast-forward survient. La solution consiste soit à fusionner les historiques distants (`git pull origin main --allow-unrelated-histories --no-rebase`), soit à recréer un dépôt GitHub strictement vierge pour y pousser le code local sans conflit.

---

## 3. Architecture Globale du Pipeline RAG

Le pipeline complet s'articule en 5 étapes standard :

```
┌───────────────────────┐
│ 1. Extraction PDF     │  Lecture et extraction du texte brut depuis les PDF sources
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ 2. Chunking           │  Découpage en segments textuels digestes avec recouvrement (overlap)
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ 3. Embeddings         │  Projection des textes en vecteurs denses (768 dimensions via Ollama)
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ 4. Vector Store       │  Stockage persistant (ChromaDB) et recherche par distance cosinus
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ 5. Génération LLM     │  Injection des segments récupérés dans le prompt système du LLM
└───────────────────────┘
```

---

## 4. Détail des Scripts Développés

### Script 1 : Extraction du Texte PDF (`scriptRAG1.py`)
* **Rôle** : Lire les documents sources (`.pdf`) présents dans le projet et extraire les flux de texte brut nécessaires à la chaîne de traitement.
* **Point d'attention** : La qualité de l'extraction (préservation des retours à la ligne, élimination des en-têtes et pieds de page répétés) conditionne la qualité des chunks ultérieurs.

---

### Script 2 : Découverte des Embeddings avec Ollama (`test_embedding.py` / Script 2)
* **Rôle** : Comprendre ce qu'est un embedding et valider la communication entre Python et le runtime local Ollama avec le modèle `nomic-embed-text`.
* **Fonctionnement** :
  ```python
  import ollama

  texte = "L'ANSSI recommande de sécuriser les accès réseau."
  reponse = ollama.embeddings(
      model="nomic-embed-text",
      prompt=texte
  )
  vecteur = reponse["embedding"]
  ```
* **Résultat obtenu** :
  * Type d'objet : `list[float]`
  * Dimension du vecteur : `768` (longueur invariante, quelle que soit la longueur du texte d'entrée).
  * Coordonnées décimales représentant la signature sémantique du texte.

---

### Script 3 : Mesure Mathématique de Similarité Cosinus (`Script 3`)
* **Rôle** : Valider expérimentalement la capacité du modèle à regrouper des phrases proches en sens et à écarter les phrases hors-sujet.
* **Formule mathématique** :
  $$\text{similarité}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$
* **Expérience comparative menée (FR vs EN)** :
  * **En Français** :
    * Phrase 1 (*Vulnérabilité critique*) vs Phrase 2 (*Faille majeure*) : **0,6278**
    * Phrase 1 (*Vulnérabilité critique*) vs Phrase 3 (*Recette de pâtes*) : **0,5534**
    * Écart ($\Delta$) : **0,0744**
  * **En Anglais** :
    * Phrase 1 (*Critical vulnerability*) vs Phrase 2 (*Security flaw*) : **0,6639**
    * Phrase 1 (*Critical vulnerability*) vs Phrase 3 (*Chicken pasta recipe*) : **0,3410**
    * Écart ($\Delta$) : **0,3229** (séparation 4,3 fois plus nette).

---

### Script 4 : Fonction d'Indexation par Chunks (`scriptRAG4.py`)
* **Rôle** : Structurer une liste de morceaux de texte en un tableau d'objets prêts à être consommés par la base vectorielle.
* **Code documenté** :
```python
import ollama

def vectoriser_chunks(chunks: list[str]) -> list[dict]:
    """
    Transforme une liste de segments textuels (chunks) en représentations vectorielles (embeddings).

    Fonctionnement :
    1. Itération sur chaque segment de texte brut.
    2. Ajout du préfixe 'search_document:' recommandé par le modèle Nomic.
    3. Appel au modèle local 'nomic-embed-text' via Ollama (espace à 768 dimensions).
    4. Structuration sous forme de dictionnaire associant id, texte original et vecteur.
    """
    documents_vectorises = []
    
    for index, chunk in enumerate(chunks):
        texte_a_encoder = f"search_document: {chunk}"
        reponse = ollama.embeddings(
            model="nomic-embed-text",
            prompt=texte_a_encoder
        )
        documents_vectorises.append({
            "id": index,
            "texte": chunk,
            "vecteur": reponse["embedding"]
        })
        
    return documents_vectorises
```

---

### Script 5 : Persistance et Recherche avec ChromaDB (`scriptRAG5.py`)
* **Rôle** : Stocker durablement sur le disque dur les vecteurs calculés, les textes et les métadonnées dans ChromaDB, puis exécuter une requête en langage naturel pour récupérer les meilleurs passages.
* **Code appliqué** :
```python
import chromadb
import ollama

# Initialisation du client persistant local sur disque
client = chromadb.PersistentClient(path="./chroma_db")

# Création d'une collection explicitement configurée en distance cosinus
collection = client.get_or_create_collection(
    name="documents_rag_cosine",
    metadata={"hnsw:space": "cosine"}
)

def get_embedding(texte: str, is_query: bool = False) -> list[float]:
    prefixe = "search_query: " if is_query else "search_document: "
    reponse = ollama.embeddings(
        model="nomic-embed-text",
        prompt=f"{prefixe}{texte}"
    )
    return reponse["embedding"]

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

# Insertion dans la base
vecteurs = [get_embedding(chunk) for chunk in chunks]
collection.add(
    ids=ids,
    documents=chunks,
    embeddings=vecteurs,
    metadatas=metadatas
)

# Interrogation par l'utilisateur
question = "Comment sécuriser les connexions administrateur ?"
vecteur_question = get_embedding(question, is_query=True)

resultats = collection.query(
    query_embeddings=[vecteur_question],
    n_results=2
)

print("\n--- Résultat de la recherche ---")
for i, (doc, dist) in enumerate(zip(resultats["documents"][0], resultats["distances"][0])):
    print(f"Top {i+1} [Distance: {dist:.4f}] : {doc}")
```

* **Résultats obtenus à l'exécution** :
  * `Top 1 [Distance: 0.3785]` : Guide ANSSI : segmentation réseau et isolation.
  * `Top 2 [Distance: 0.4075]` : Authentification multifacteur (MFA).
  * Le document sur la cuisine a été complètement éliminé de la réponse.

---

## 5. Points Techniques & Résolution de Problèmes

### Clé de dictionnaire Ollama : `'embedding'` vs `'embeddings'`
* **`ollama.embeddings(...)`** : Traite un texte unitaire. La clé retournée est **au singulier** : `reponse["embedding"]` (renvoie une liste de 768 flottants). L'accès à `reponse["embeddings"]` lève un `KeyError: 'embeddings'`.
* **`ollama.embed(...)`** : Traite un lot de plusieurs textes (*batch*). La clé retournée est **au pluriel** : `reponse["embeddings"]` (renvoie une liste de listes de flottants).

### Préfixes recommandés par le modèle `nomic-embed-text`
Pour maximiser l'écart entre les passages indexés et les requêtes utilisateurs :
* Pour les chunks du document à enregistrer : `f"search_document: {chunk}"`
* Pour la question posée par l'utilisateur : `f"search_query: {question}"`

### Rôle et multiplicité des Métadonnées
Les métadonnées sont attachées à **chaque chunk** et non au fichier tout entier :
* Deux chunks d'un même document peuvent partager la même source (`source: "guide_anssi.pdf"`) mais avoir des thèmes, sections ou pages différentes (`page: 4` vs `page: 18`).
* Elles permettent la traçabilité des réponses et le **filtrage ciblé** dans les requêtes via l'argument `where={"source": "guide_anssi.pdf"}`.

---

## 6. Grille d'Évaluation : Comment Savoir si une Info est Bonne ?

Dans un moteur de recherche vectoriel configuré en **distance cosinus**, la distance est inversement proportionnelle à la similarité :
$$\text{Distance Cosinus} = 1 - \text{Similarité Cosinus}$$

| Plage de Distance Cosinus | Niveau de Pertinence | Interprétation RAG |
|---|---|---|
| **0.00 à 0.20** | **Identique ou quasi-paraphrase** | Passage extrêmement pertinent, citation directe ou reformulation très proche de la question. |
| **0.20 à 0.45** | **Forte corrélation sémantique** | **Zone idéale de récupération RAG**. Le passage traite exactement du sujet ciblé avec un vocabulaire adapté. |
| **0.45 à 0.60** | **Corrélation moyenne / périphérique** | Le passage aborde le même domaine général (ex. la sécurité informatique au sens large), mais ne répond pas forcément précisément à la question. |
| **> 0.60** | **Hors-sujet / Bruit** | Passage sans rapport avec la question posée (ex. document de cuisine face à une question de cybersécurité). À éliminer des prompts LLM. |

---

## 7. Paramètres de Configuration de ChromaDB

La configuration de l'espace métrique détermine comment ChromaDB calcule la proximité entre les vecteurs :

| Métrique (`hnsw:space`) | Nom Mathématique | Formule de Distance | Cas d'Usage Recommandé |
|---|---|---|---|
| **`"cosine"`** | **Distance Cosinus** *(Recommandé)* | $1 - \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$ | **Standard absolu pour le RAG et le traitement de texte (NLP)**. Évalue la proximité sémantique en ignorant les variations de longueur brute du vecteur. Résultats bornés et lisibles entre 0 et 2. |
| **`"l2"`** | Distance Euclidienne au carré *(Défaut)* | $\sum_{i=1}^n (A_i - B_i)^2$ | Déconseillé avec des modèles d'embedding non-normalisés (distances brutes dépassant 300+). Réservé aux espaces géométriques euclidiens ou aux descripteurs d'images normalisés. |
| **`"ip"`** | Produit Scalaire (*Inner Product*) | $1 - (\vec{A} \cdot \vec{B})$ | Utilisé uniquement si tous les vecteurs sont strictement normalisés à une norme euclidienne de 1 ($\|\vec{v}\| = 1$). Offre un gain minime de performance de calcul en grande échelle. |

---

## 8. Prochaines Étapes du Projet

1. **Intégration PDF -> Chunks réels** : Remplacer les 3 phrases d'exemple par le texte extrait du `guide_anssi.pdf` découpé avec une fonction de chunking (avec overlap).
2. **Étape 5 (Génération)** : Transmettre les chunks récupérés à un modèle de chat local (ex. `mistral` ou `llama3` via `ollama.chat`) avec un prompt d'instruction pour générer la réponse finale sourcée.