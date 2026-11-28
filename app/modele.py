"""Chargement du modèle A7 et fonctions de prédiction pour l'application."""
from pathlib import Path

import numpy as np
import keras
import tensorflow as tf
from keras import layers, ops
from matplotlib import colormaps
from PIL import Image

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
CHEMIN_MODELE = DOSSIER_PROJET / "models" / "A7_architecture_complete.keras"


# Couches personnalisées (identiques au notebook 6)
class AjoutPosition(layers.Layer):
    def build(self, forme):
        self.positions = self.add_weight(shape=(1, forme[1], forme[2]), initializer="random_normal", name="positions")

    def call(self, x):
        return x + self.positions


class CoucheANFIS(layers.Layer):
    def __init__(self, nb_regles=12, nb_sorties=3, **kwargs):
        super().__init__(**kwargs)
        self.nb_regles = nb_regles
        self.nb_sorties = nb_sorties

    def build(self, forme):
        d = forme[-1]
        self.centres = self.add_weight(shape=(self.nb_regles, d), initializer="random_normal", name="centres")
        self.log_sigmas = self.add_weight(shape=(self.nb_regles, d), initializer="zeros", name="log_sigmas")
        self.A = self.add_weight(shape=(self.nb_regles, d, self.nb_sorties), initializer="glorot_uniform", name="A")
        self.b = self.add_weight(shape=(self.nb_regles, self.nb_sorties), initializer="zeros", name="b")

    def forces(self, x):
        ecart = (ops.expand_dims(x, 1) - self.centres) / ops.exp(self.log_sigmas)
        return ops.softmax(-0.5 * ops.sum(ops.square(ecart), axis=-1), axis=-1)

    def call(self, x):
        w = self.forces(x)
        y_regles = ops.einsum("bd,rdk->brk", x, self.A) + self.b
        return ops.sum(ops.expand_dims(w, -1) * y_regles, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(nb_regles=self.nb_regles, nb_sorties=self.nb_sorties)
        return config


# Description des catégories et des règles floues (résultats de l'étape 8)
CATEGORIES = {
    0: {"nom": "Catégorie 1", "court": "brun-rouge", "description": "fève brun-rouge, aux caractéristiques intermédiaires", "couleur": "#8B4513"},
    1: {"nom": "Catégorie 2", "court": "sombre et violacée", "description": "fève sombre et violacée, à surface contrastée", "couleur": "#3E2723"},
    2: {"nom": "Catégorie 3", "court": "claire", "description": "fève claire et orangée, à surface lisse", "couleur": "#D2A679"},
}
REGLES = {
    9: "SI la fève est claire, orangée, peu saturée et à surface lisse, ALORS catégorie 3.",
    5: "SI la fève est sombre, violacée et à surface contrastée, ALORS catégorie 2.",
    11: "SI la fève ne présente aucun critère marqué (ni claire, ni violacée), ALORS catégorie 1.",
}


def charger_modele():
    modele = keras.models.load_model(CHEMIN_MODELE,
                                     custom_objects={"CoucheANFIS": CoucheANFIS, "AjoutPosition": AjoutPosition})
    # Sous-modèle pour Grad-CAM : dernière carte de convolution (avant le passage aux tokens)
    position_reshape = [i for i, c in enumerate(modele.layers) if isinstance(c, layers.Reshape)][0]
    derniere_conv = modele.layers[position_reshape - 1].input
    modele_gradcam = keras.Model(modele.inputs, [derniere_conv, modele.output])
    # Sous-modèle pour les règles floues : entrées de la couche ANFIS
    modele_flou = keras.Model(modele.inputs, modele.get_layer("normalisation_floue").output)
    return modele, modele_gradcam, modele_flou


def preparer_image(image):
    """Recadre l'image en carré (centre), puis la réduit à 60 x 60 pixels comme le jeu d'entraînement."""
    image = image.convert("RGB")
    cote = min(image.size)
    gauche, haut = (image.width - cote) // 2, (image.height - cote) // 2
    carre = image.crop((gauche, haut, gauche + cote, haut + cote))
    petite = carre.resize((60, 60), Image.LANCZOS)
    return carre, np.asarray(petite, dtype="float32")


def analyser(tableau, modele_gradcam, modele_flou, modele):
    """Probabilités, carte Grad-CAM et règle floue dominante pour une image 60 x 60."""
    x = tf.convert_to_tensor(tableau[None])
    with tf.GradientTape() as enregistreur:
        cartes, proba = modele_gradcam(x, training=False)
        classe = int(tf.argmax(proba[0]))
        score = proba[:, classe]
    gradients = enregistreur.gradient(score, cartes)
    poids = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(cartes * poids[:, None, None, :], axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)

    entrees = modele_flou(x, training=False)
    forces = ops.convert_to_numpy(modele.get_layer("anfis").forces(entrees))[0]
    regle = int(forces.argmax()) + 1
    return proba[0].numpy(), classe, cam, regle, float(forces.max())


def espaces_couleur(tableau):
    """Les 9 canaux de la fève (60 x 60) dans RGB, HSV (teinte tournée, comme à l'étape 3) et LAB."""
    import cv2
    rgb = tableau.astype("uint8")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(int) + 90) % 180
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return {
        "RGB": (rgb, ["R (rouge)", "G (vert)", "B (bleu)"]),
        "HSV": (hsv, ["H (teinte)", "S (saturation)", "V (valeur)"]),
        "LAB": (lab, ["L* (luminosité)", "a* (vert-rouge)", "b* (bleu-jaune)"]),
    }


def predire_lot(tableaux, modele):
    """Probabilités pour plusieurs fèves d'un coup."""
    return modele.predict(np.stack(tableaux), verbose=0)


def superposer_gradcam(image_carree, cam, taille=480):
    """Superpose la carte Grad-CAM (teintes chaudes, transparente là où l'attention est faible)."""
    fond = image_carree.resize((taille, taille), Image.LANCZOS).convert("RGBA")
    cam_grande = np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize((taille, taille), Image.BICUBIC)) / 255
    couleurs = colormaps["YlOrBr"](cam_grande)
    couleurs[..., 3] = np.clip(cam_grande, 0, 1) * 0.75          # transparence proportionnelle à l'attention
    calque = Image.fromarray((couleurs * 255).astype("uint8"), mode="RGBA")
    return Image.alpha_composite(fond, calque).convert("RGB")
