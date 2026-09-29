# Ici on veut charger un fichier pdf
# Afficher le nombre de pages
# Extraire les informations d'une page

from pypdf import PdfReader

# Charger un fichier
anssi = PdfReader("guide_anssi.pdf")

# Nb de pages du fichier
nb_pages = len(anssi.pages)
print("Pages: ", nb_pages)

# 3. Récupérer une page spécifique (index base 0 : page 1 = index 0)
page = anssi.pages[2]
print(page)

# Extraire du texte brut de cette page
texte = page.extract_text()
print(texte)