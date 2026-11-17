import chromadb
import ollama

def get_embedding(texte: str, is_query: bool = False) -> list[float]:
  prefixe = "search_query: " if is_query else "search_document: "
  response = ollama.embeddings(
      model="nomic-embed-text",
      prompt=f"{prefixe}{texte}",
  )
  return response["embedding"]

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("ing4_rules_pdf_test")

question = r"""Est-ce que ma structure est conforme à la règle R1 et aux règles de confidentialité ?

    \begin{tikzpicture}[node distance=2.5cm, auto]
        % Définition des Enclaves et de leurs niveaux de protection (e)
        % E1 : Niveau Secret, Crypt(E1) = True
        \node[draw, rectangle, rounded corners, minimum width=3cm, minimum height=2cm, label=below:{Enclave $e_1$ ($e = \text{Secret}$, $\text{Crypt}=\text{True}$)}] (E1) {};
        
        % E2 : Niveau Public (vide), Crypt(E2) = False (Enclave de transit non fiable)
        \node[draw, rectangle, rounded corners, right of=E1, node distance=4.5cm, minimum width=3cm, minimum height=2cm, label=below:{Enclave $e_2$ ($e = \emptyset$, $\text{Crypt}=\text{False}$)}] (E2) {};
        
        % E3 : Niveau Secret, Crypt(E3) = True
        \node[draw, rectangle, rounded corners, right of=E2, node distance=4.5cm, minimum width=3cm, minimum height=2cm, label=below:{Enclave $e_3$ ($e = \text{Secret}$, $\text{Crypt}=\text{True}$)}] (E3) {};

        % Définition des Fonctions (fid, Ec, ftype, c_level)
        % f1 : Création de la donnée avec niveau Secret dans E1
        \node[draw, fill=purple!15, at=(E1.center)] (f1) {$^C f_1 : \text{Secret}$};
        
        % f2 : Transit dans E2 (donnée chiffrée, donc niveau effectif = vide)
        \node[draw, fill=green!15, at=(E2.center)] (f2) {$^T f_2 : \perp$};
        
        % f3 : Lecture de la donnée déchiffrée dans E3
        \node[draw, fill=yellow!20, at=(E3.center)] (f3) {$^R f_3 : \perp$};

        % Flux de données et mécanismes cryptographiques (Cas 3 de la Figure 1 / Règle R3)
        \draw[->, thick] (f1) -- node[above, font=\footnotesize] {$\text{Ciph}(\text{data})$} node[below, font=\footnotesize] {$\text{Action}=\text{ciphered}$} (f2);
        \draw[->, thick] (f2) -- node[above, font=\footnotesize] {$\text{Deciph}(\text{data})$} (f3);
    \end{tikzpicture}
"""


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


################################# CONCLUSION test 1 avec un pdf sur un schéma correct #################################
"""
L'expérimentation menée sur l'ingestion brute du document ING4_rules.pdf démontre les limites d'une approche vectorielle appliquée directement à des fichiers PDF complexes. Bien que le pipeline RAG aboutisse à un verdict final correct sur le cas testé, l'analyse détaillée du raisonnement met en évidence trois biais critiques :
Dégradation sémantique à l'extraction : Le parseur PDF aplatit les formules mathématiques et les règles d'inférence (R init à R4), supprimant des opérateurs logiques essentiels (notamment le critère de violation entre $E_{ef}$ et $e$) et altérant la pertinence des vecteurs générés.
Bruit syntaxique lors de la recherche (retrieval) : L'injection brute de code graphique TikZ dans la requête dilue l'information utile, empêchant le modèle d'embedding (nomic-embed-text) de remonter systématiquement les règles pertinentes dans les premiers résultats.
Hallucination compensatoire du LLM : Faute de récupérer un contexte formel complet, llama3.1 s'appuie sur les commentaires présents dans la requête ou invente des justifications plausibles plutôt que d'appliquer rigoureusement les règles du modèle.
Pour fiabiliser l'audit automatique, la chaîne d'ingestion doit évoluer vers le traitement direct des fichiers sources .tex nettoyés (suppression du bruit de mise en page, découpage sémantique par règle et préservation des macros logiques), associé à des schémas d'entrée dépourvus d'annotations explicatives.
"""