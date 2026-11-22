"""Application de classification des fèves de cacao (modèle A7 : CNN multi-échelle + Transformer + ANFIS)."""
from pathlib import Path

import streamlit as st
from PIL import Image

from modele import CATEGORIES, REGLES, analyser, charger_modele, preparer_image, superposer_gradcam

DOSSIER_EXEMPLES = Path(__file__).resolve().parent / "exemples"

st.set_page_config(page_title="Classer une fève de cacao", page_icon="🫘", layout="centered")

# ---------- Style ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,700&display=swap');

:root {
    --fond: #FBF7F2;
    --pulpe: #F3E9DD;
    --encre: #3E2723;
    --cacao: #8B4513;
    --feve-claire: #D2A679;
    --gris: #8A7F78;
}
.stApp .stMarkdown, .stApp .stMarkdown p, .stApp .stMarkdown div, .stApp .stMarkdown span,
.stApp label, .stApp .stButton p, .stApp [data-testid="stCaptionContainer"] {
    font-family: 'Bricolage Grotesque', system-ui, sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 880px; padding-top: 3rem; padding-bottom: 3rem; }

.stApp .titre { font-size: 2.6rem; font-weight: 700; line-height: 1.1; letter-spacing: -0.02em; margin: 0 0 0.6rem 0; }
.stApp .stMarkdown .chapeau { font-size: 1.1rem; color: var(--gris); max-width: 34rem; line-height: 1.55; margin-bottom: 2rem; }

.stApp .verdict-nom { font-size: 3rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1; margin: 0.2rem 0 0.4rem 0; }
.stApp .verdict-desc { font-size: 1.05rem; margin-bottom: 0.2rem; }
.stApp .stMarkdown .confiance { color: var(--gris); font-size: 0.95rem; margin-bottom: 1.6rem; }

.nuancier { position: relative; height: 14px; border-radius: 7px; margin: 0.4rem 0 0.5rem 0;
            background: linear-gradient(90deg, var(--feve-claire) 0%, var(--cacao) 50%, var(--encre) 100%); }
.repere { position: absolute; top: -7px; width: 4px; height: 28px; border-radius: 2px;
          background: var(--encre); box-shadow: 0 0 0 3px var(--fond); transform: translateX(-2px); }
.nuancier-legende { display: flex; justify-content: space-between; font-size: 0.82rem; color: var(--gris); margin-bottom: 1.6rem; }

.proba { display: grid; grid-template-columns: 7.5rem 1fr 3rem; align-items: center; gap: 0.7rem; margin: 0.35rem 0; font-size: 0.95rem; }
.proba-fond { background: var(--pulpe); border-radius: 4px; height: 8px; overflow: hidden; }
.proba-barre { height: 8px; border-radius: 4px; }
.proba-valeur { text-align: right; font-variant-numeric: tabular-nums; }

.regle { margin-top: 1.6rem; padding: 0.9rem 1.1rem; background: var(--pulpe); border-radius: 10px;
         border-left: 4px solid var(--cacao); font-size: 0.98rem; line-height: 1.5; }
.stApp .regle-titre { font-weight: 600; margin-bottom: 0.2rem; }
.stApp .stMarkdown .alerte { margin-top: 1rem; font-size: 0.92rem; color: var(--cacao); }

.stApp .stMarkdown .pied { margin-top: 3.5rem; padding-top: 1.2rem; border-top: 1px solid var(--pulpe);
        font-size: 0.85rem; color: var(--gris); line-height: 1.6; }

[data-testid="stFileUploaderDropzone"] { background: var(--pulpe); border: 1.5px dashed var(--feve-claire); border-radius: 14px; }
.stButton > button { border-radius: 999px; border: 1px solid var(--feve-claire); background: transparent; color: var(--encre); }
.stButton > button:hover { border-color: var(--cacao); color: var(--cacao); }
.stButton > button:focus-visible { outline: 2px solid var(--cacao); outline-offset: 2px; }
[data-testid="stImage"] img { border-radius: 18px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Chargement du modèle…")
def modeles():
    return charger_modele()


modele, modele_gradcam, modele_flou = modeles()

# ---------- En-tête ----------
st.markdown('<div class="titre">Quelle est la catégorie de cette fève ?</div>', unsafe_allow_html=True)
st.markdown('<div class="chapeau">Déposez la photo d\'une fève de cacao. Le modèle indique sa catégorie '
            'et montre sur quoi il s\'appuie pour décider.</div>', unsafe_allow_html=True)

# ---------- Choix de l'image ----------
fichier = st.file_uploader("Photo de la fève (JPG ou PNG)", type=["jpg", "jpeg", "png"])

st.caption("Pas de photo sous la main ? Essayez une fève du jeu de test :")
colonnes = st.columns(3)
for k, colonne in enumerate(colonnes):
    if colonne.button(f"Exemple {k + 1}", width="stretch"):
        st.session_state["exemple"] = DOSSIER_EXEMPLES / f"exemple_{k + 1}.jpg"

if fichier is not None:
    image = Image.open(fichier)
elif "exemple" in st.session_state:
    image = Image.open(st.session_state["exemple"])
else:
    image = None

# ---------- Résultat ----------
if image is not None:
    image_carree, tableau = preparer_image(image)
    proba, classe, cam, regle, force_regle = analyser(tableau, modele_gradcam, modele_flou, modele)
    categorie = CATEGORIES[classe]

    st.write("")
    gauche, droite = st.columns([1, 1.25], gap="large")

    with gauche:
        voir_attention = st.toggle("Afficher les zones regardées par le modèle")
        affichage = superposer_gradcam(image_carree, cam) if voir_attention else image_carree.resize((480, 480), Image.LANCZOS)
        st.image(affichage, width=340)
        if voir_attention:
            st.caption("Plus la zone est dorée, plus elle a compté dans la décision (Grad-CAM).")

    with droite:
        # Position sur le nuancier : fève claire (catégorie 3) -> brun-rouge (1) -> sombre et violacée (2)
        position = (proba[2] * 0 + proba[0] * 0.5 + proba[1] * 1) * 100
        lignes_proba = "".join(
            f'<div class="proba"><span>{CATEGORIES[k]["nom"]}</span>'
            f'<div class="proba-fond"><div class="proba-barre" style="width:{proba[k] * 100:.1f}%; '
            f'background:{CATEGORIES[k]["couleur"]}"></div></div>'
            f'<span class="proba-valeur">{proba[k] * 100:.0f} %</span></div>'
            for k in (2, 0, 1))
        texte_regle = REGLES.get(regle, f"Combinaison de critères propre à la règle {regle}.")

        st.markdown(f"""
            <div class="verdict-nom" style="color:{categorie['couleur'] if classe != 2 else '#A9753F'}">{categorie['nom']}</div>
            <div class="verdict-desc">{categorie['description'].capitalize()}</div>
            <div class="confiance">Confiance du modèle : {proba[classe] * 100:.0f} %</div>
            <div class="nuancier"><div class="repere" style="left:{position:.1f}%"></div></div>
            <div class="nuancier-legende"><span>claire</span><span>brun-rouge</span><span>sombre et violacée</span></div>
            {lignes_proba}
            <div class="regle"><div class="regle-titre">Règle floue appliquée (règle {regle})</div>{texte_regle}</div>
        """, unsafe_allow_html=True)

        if proba[classe] < 0.70:
            st.markdown('<div class="alerte">Prédiction incertaine : cette fève ressemble à deux catégories à la fois. '
                        'Un contrôle visuel est conseillé.</div>', unsafe_allow_html=True)

# ---------- Pied de page ----------
st.markdown("""
<div class="pied">
Modèle : architecture complète CNN multi-échelle + Transformer + ANFIS, F1 macro de 95,6 ± 0,5 % sur l'ensemble de test.<br>
Le modèle a appris sur des photos de 60 × 60 pixels prises sur fond clair : une photo très différente peut donner un résultat moins fiable.<br>
Projet de Master 2 Intelligence Artificielle — classification des fèves de cacao.
</div>
""", unsafe_allow_html=True)