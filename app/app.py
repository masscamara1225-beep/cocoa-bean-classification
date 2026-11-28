"""Analyse de fèves de cacao : classification en trois catégories avec le modèle A7
(CNN multi-échelle + Transformer + ANFIS)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

from modele import (CATEGORIES, REGLES, analyser, charger_modele, espaces_couleur,
                    predire_lot, preparer_image, superposer_gradcam)

DOSSIER_EXEMPLES = Path(__file__).resolve().parent / "exemples"
SEUIL_INCERTITUDE = 0.70
ORDRE = (2, 0, 1)        # du plus clair au plus sombre : catégorie 3, catégorie 1, catégorie 2

st.set_page_config(page_title="Analyse de fèves de cacao", page_icon="🫘", layout="wide")

# ---------- Style ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,700&display=swap');
:root { --surface: #F5F4F1; --trait: #E7E4DE; --texte: #2E2A27; --gris: #77706A; --feuille: #4F7A55;
        --cacao: #8B4513; --encre: #3E2723; --feve-claire: #D2A679; }
.stApp .stMarkdown, .stApp .stMarkdown p, .stApp .stMarkdown div, .stApp .stMarkdown span,
.stApp label, .stApp [data-testid="stCaptionContainer"], .stApp button p {
    font-family: 'Bricolage Grotesque', system-ui, sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1360px; padding: 2.2rem 3rem 3rem 3rem; }
@media (max-width: 640px) { .block-container { padding: 1.2rem 1rem 2rem 1rem; } }

.stApp .titre { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; }
.stApp .stMarkdown .sous-titre { color: var(--gris); font-size: 1rem; margin: 0.15rem 0 1.6rem 0; }

.stApp .verdict-nom { font-size: 2.5rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.05; margin: 0.1rem 0 0.15rem 0; }
.stApp .stMarkdown .verdict-court { font-size: 1.15rem; color: var(--gris); font-weight: 400; }
.stApp .verdict-desc { font-size: 1rem; margin-top: 0.3rem; }
.stApp .stMarkdown .confiance { color: var(--gris); font-size: 0.95rem; margin: 0.2rem 0 1.3rem 0; }

.nuancier { position: relative; height: 12px; border-radius: 6px; margin: 0.3rem 0 0.4rem 0;
            background: linear-gradient(90deg, var(--feve-claire) 0%, var(--cacao) 50%, var(--encre) 100%); }
.repere { position: absolute; top: -7px; width: 4px; height: 26px; border-radius: 2px;
          background: var(--texte); box-shadow: 0 0 0 3px #FFFFFF; transform: translateX(-2px); }
.nuancier-legende { display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--gris); margin-bottom: 1.3rem; }

.proba { display: grid; grid-template-columns: minmax(9rem, 16rem) 1fr 3rem; align-items: center; gap: 0.7rem; margin: 0.3rem 0; font-size: 0.93rem; }
.proba-fond { background: var(--surface); border-radius: 4px; height: 8px; overflow: hidden; }
.proba-barre { height: 8px; border-radius: 4px; }
.proba-valeur { text-align: right; font-variant-numeric: tabular-nums; }

.regle { padding: 0.9rem 1.1rem; background: var(--surface); border-radius: 10px; line-height: 1.5; }
.stApp .regle-titre { font-weight: 600; margin-bottom: 0.2rem; }
.stApp .stMarkdown .alerte { margin-top: 0.8rem; font-size: 0.92rem; color: #9A4A12; }
.stApp .stMarkdown .vide { padding: 1.6rem 1.8rem; background: var(--surface); border-radius: 12px; line-height: 1.7; }
.stApp .stMarkdown .vide-titre { font-weight: 600; font-size: 1.1rem; margin-bottom: 0.3rem; }
.stApp .stMarkdown .vide ol { margin: 0; padding-left: 1.2rem; color: var(--gris); }

.repartition { display: flex; height: 34px; border-radius: 8px; overflow: hidden; margin: 0.6rem 0 0.8rem 0; }
.repartition div { display: flex; align-items: center; justify-content: center; font-size: 0.85rem; font-weight: 600; }

.stApp .stMarkdown .pied { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--trait); font-size: 0.82rem; color: var(--gris); line-height: 1.6; }

[data-testid="stFileUploaderDropzone"] { background: var(--surface); border: 1.5px dashed #CFCAC2; border-radius: 12px; }
.stButton > button, .stDownloadButton > button { border-radius: 999px; }
[data-testid="stImage"] img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Chargement du modèle…")
def modeles():
    return charger_modele()


modele, modele_gradcam, modele_flou = modeles()


def couleur_texte(classe):
    """Couleur du texte de la catégorie (le beige de la catégorie 3 est foncé pour rester lisible)."""
    return "#A9753F" if classe == 2 else CATEGORIES[classe]["couleur"]


def etiquette(classe):
    """Nom de la catégorie suivi de sa description courte, ex. « Catégorie 3 (claire) »."""
    return f'{CATEGORIES[classe]["nom"]} ({CATEGORIES[classe]["court"]})'


def barres_probabilites(proba):
    return "".join(
        f'<div class="proba"><span>{etiquette(k)}</span>'
        f'<div class="proba-fond"><div class="proba-barre" style="width:{proba[k] * 100:.1f}%; '
        f'background:{CATEGORIES[k]["couleur"]}"></div></div>'
        f'<span class="proba-valeur">{proba[k] * 100:.0f} %</span></div>' for k in ORDRE)


def figure_espaces(tableau):
    """Fonctionnalité 3 du sujet : la fève dans les espaces RGB, HSV et LAB, canal par canal."""
    espaces = espaces_couleur(tableau)
    fig, axes = plt.subplots(3, 3, figsize=(6, 6.3), facecolor="white")
    for ligne, (nom_espace, (image_convertie, noms_canaux)) in enumerate(espaces.items()):
        for col in range(3):
            ax = axes[ligne, col]
            ax.imshow(image_convertie[..., col], cmap="gray", vmin=0, vmax=255)
            ax.set_title(noms_canaux[col], fontsize=9, color="#2E2A27")
            ax.set_xticks([]); ax.set_yticks([])
            for bord in ax.spines.values():
                bord.set_visible(False)
        axes[ligne, 0].set_ylabel(nom_espace, fontsize=11, color="#2E2A27", fontweight="bold")
    fig.tight_layout()
    return fig


def message_vide(titre, etapes):
    """Panneau affiché tant que rien n'est analysé : explique ce que l'on peut faire."""
    liste = "".join(f"<li>{e}</li>" for e in etapes)
    st.markdown(f'<div class="vide"><div class="vide-titre">{titre}</div><ol>{liste}</ol></div>',
                unsafe_allow_html=True)


# ---------- En-tête ----------
st.markdown('<div class="titre">Analyse de fèves de cacao</div>', unsafe_allow_html=True)
st.markdown('<div class="sous-titre">Classification en trois catégories à partir de photos de fèves.</div>',
            unsafe_allow_html=True)

saisie, resultats_zone = st.columns([1.15, 2], gap="large")

# ---------- Colonne de gauche : ce que l'on analyse ----------
with saisie:
    mode = st.segmented_control("Que voulez-vous analyser ?", ["Une fève", "Un lot de fèves"], default="Une fève")

    if mode == "Un lot de fèves":
        fichiers = st.file_uploader("Photos des fèves du lot (une fève par photo)",
                                    type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if st.button("Essayer avec un lot d'exemple (30 fèves)", type="primary"):
            st.session_state["lot_exemple"] = True
        lot_exemple = st.session_state.get("lot_exemple", False)
    else:
        fichier = st.file_uploader("Photo de la fève (JPG ou PNG)", type=["jpg", "jpeg", "png"])
        exemple = st.pills("Ou choisissez une fève du jeu de test", ["Exemple 1", "Exemple 2", "Exemple 3"])

# ---------- Colonne de droite : résultats ----------
with resultats_zone:

    # ===== Mode 1 : une fève =====
    if mode != "Un lot de fèves":
        if fichier is not None:
            image = Image.open(fichier)
        elif exemple:
            image = Image.open(DOSSIER_EXEMPLES / f"exemple_{exemple[-1]}.jpg")
        else:
            image = None

        if image is None:
            message_vide("Analyser une fève", [
                "Déposez la photo d'une fève à gauche, ou cliquez sur un exemple.",
                "Le modèle affiche sa catégorie et ses probabilités.",
                "Les onglets montrent la fève dans les espaces de couleur et ce qui a guidé la décision.",
                "Pour un échantillon entier, choisissez « Un lot de fèves » en haut à gauche.",
            ])
        else:
            image_carree, tableau = preparer_image(image)
            proba, classe, cam, regle, _ = analyser(tableau, modele_gradcam, modele_flou, modele)
            categorie = CATEGORIES[classe]

            onglet_resultat, onglet_couleurs, onglet_explication = st.tabs(
                ["Résultat", "Espaces de couleur", "Explication du modèle"])

            with onglet_resultat:
                gauche, droite = st.columns([1, 1.4], gap="large")
                with gauche:
                    st.image(image_carree.resize((400, 400), Image.LANCZOS), width="stretch")
                    st.caption("Image recadrée puis réduite à 60 × 60 pixels avant l'analyse.")
                with droite:
                    position = (proba[0] * 0.5 + proba[1] * 1.0) * 100
                    st.markdown(f"""
<div class="verdict-nom" style="color:{couleur_texte(classe)}">{categorie['nom']}
<span class="verdict-court">({categorie['court']})</span></div>
<div class="verdict-desc">{categorie['description'].capitalize()}</div>
<div class="confiance">Confiance du modèle : {proba[classe] * 100:.0f} %</div>
<div class="nuancier"><div class="repere" style="left:{position:.1f}%"></div></div>
<div class="nuancier-legende"><span>claire</span><span>brun-rouge</span><span>sombre et violacée</span></div>
{barres_probabilites(proba)}
""", unsafe_allow_html=True)
                    if proba[classe] < SEUIL_INCERTITUDE:
                        st.markdown('<div class="alerte">Résultat incertain : la fève ressemble à deux catégories. '
                                    'Un contrôle visuel est conseillé.</div>', unsafe_allow_html=True)

            with onglet_couleurs:
                st.caption("Chaque canal est affiché en niveaux de gris : clair = valeur élevée, sombre = valeur faible. "
                           "La teinte H est tournée de 180° pour ne pas couper les rouges en deux.")
                st.pyplot(figure_espaces(tableau), width=560)

            with onglet_explication:
                gauche, droite = st.columns([1, 1.4], gap="large")
                with gauche:
                    st.image(superposer_gradcam(image_carree, cam, taille=400), width="stretch")
                    st.caption("Zones dorées : parties de la fève qui ont le plus compté dans la décision (Grad-CAM).")
                with droite:
                    texte_regle = REGLES.get(regle, f"Combinaison de critères propre à la règle {regle}.")
                    st.markdown(f'<div class="regle" style="border-left:4px solid {categorie["couleur"]}">'
                                f'<div class="regle-titre">Règle floue appliquée (règle {regle})</div>'
                                f'{texte_regle}</div>', unsafe_allow_html=True)
                    st.caption("La couche neuro-floue du modèle décide à l'aide de règles « SI … ALORS ». "
                               "Les trois règles principales correspondent chacune à une catégorie.")

    # ===== Mode 2 : un lot de fèves =====
    else:
        if fichiers:
            sources = [(f.name, Image.open(f)) for f in fichiers]
        elif lot_exemple:
            sources = [(p.name, Image.open(p)) for p in sorted((DOSSIER_EXEMPLES / "lot").glob("*.jpg"))]
        else:
            sources = []

        if not sources:
            message_vide("Analyser un lot de fèves", [
                "Déposez plusieurs photos à gauche (une fève par photo), ou essayez le lot d'exemple.",
                "Le modèle classe chaque fève et donne la répartition du lot.",
                "Les fèves pour lesquelles il hésite sont mises de côté pour un contrôle à l'œil.",
                "Les résultats se téléchargent en CSV.",
            ])
        else:
            preparees = [preparer_image(image) for _, image in sources]
            probas = predire_lot([tableau for _, tableau in preparees], modele)
            classes_predites = probas.argmax(axis=1)
            confiances = probas.max(axis=1)
            n = len(sources)

            st.markdown(f"#### Répartition du lot ({n} fèves)")
            segments = ""
            for k in ORDRE:
                part = (classes_predites == k).mean() * 100
                if part > 0:
                    texte = "#2E2A27" if k == 2 else "#FFFFFF"
                    segments += (f'<div style="width:{part}%; background:{CATEGORIES[k]["couleur"]}; color:{texte}">'
                                 f'{part:.0f} %</div>')
            st.markdown(f'<div class="repartition">{segments}</div>', unsafe_allow_html=True)

            synthese = pd.DataFrame([{
                "Catégorie": etiquette(k),
                "Nombre de fèves": int((classes_predites == k).sum()),
                "Part du lot (%)": round((classes_predites == k).mean() * 100, 1),
                "Confiance moyenne (%)": f"{confiances[classes_predites == k].mean() * 100:.1f}"
                                         if (classes_predites == k).any() else "—",
            } for k in ORDRE])
            st.dataframe(synthese, hide_index=True, width="stretch")

            incertaines = [i for i in range(n) if confiances[i] < SEUIL_INCERTITUDE]
            if incertaines:
                st.markdown(f"#### Fèves à vérifier à l'œil ({len(incertaines)})")
                st.caption(f"Le modèle est confiant à moins de {SEUIL_INCERTITUDE:.0%} pour ces fèves (sous chaque fève : catégorie prédite et confiance).")
                st.image([preparees[i][0].resize((120, 120)) for i in incertaines], width=96,
                         caption=[f"{CATEGORIES[classes_predites[i]]['nom'][-1]} ({confiances[i] * 100:.0f} %)"
                                  for i in incertaines])
            else:
                st.caption("Aucune fève incertaine : le modèle est confiant à plus de 70 % pour toutes les fèves du lot.")

            st.markdown("#### Fèves par catégorie")
            for k in ORDRE:
                indices = [i for i in range(n) if classes_predites[i] == k]
                if indices:
                    st.markdown(f'<span style="color:{couleur_texte(k)}; font-weight:600">{etiquette(k)}</span> '
                                f'— {len(indices)} fèves', unsafe_allow_html=True)
                    st.image([preparees[i][0].resize((120, 120)) for i in indices], width=70)

            resultats = pd.DataFrame({
                "Fichier": [nom for nom, _ in sources],
                "Catégorie prédite": [CATEGORIES[c]["nom"] for c in classes_predites],
                "Confiance (%)": (confiances * 100).round(1),
                **{f"Probabilité {CATEGORIES[k]['nom']} (%)": (probas[:, k] * 100).round(1) for k in ORDRE},
            })
            st.download_button("Télécharger les résultats (CSV)", resultats.to_csv(index=False).encode("utf-8"),
                               file_name="resultats_lot_feves.csv", mime="text/csv")

# ---------- Pied de page ----------
st.markdown("""
<div class="pied">
Modèle : CNN multi-échelle + Transformer + ANFIS (F1 macro de 95,6 ± 0,5 % sur l'ensemble de test).<br>
Les catégories 1, 2 et 3 sont celles du jeu de données fourni ; les mots entre parenthèses résument les différences visuelles observées entre elles.<br>
Le modèle a appris sur des photos de 60 × 60 pixels sur fond clair : une photo prise dans d'autres conditions peut donner un résultat moins fiable.
</div>
""", unsafe_allow_html=True)
