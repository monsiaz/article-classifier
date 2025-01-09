import os
import random
import subprocess
import pandas as pd
import time

# -------------------------------------------------------------------
# GLOBAL PARAMETERS (English comments; docstrings in French)
# -------------------------------------------------------------------
TRAIN_CSV = "/Users/simonazoulay/Codes/sample_test_relevance.csv"
TARGET_CSV = "/Users/simonazoulay/Codes/articles_to_classify.csv"
OLLAMA_MODEL = "llama3.1"  # or "llama3.1:8b"
NB_OUI = 8
NB_NON = 8

# -------------------------------------------------------------------
# 1) BUILD FEW-SHOT EXAMPLES
# -------------------------------------------------------------------
def construire_few_shot():
    """
    Lit le CSV d'entraînement (Title, Société, Pertinent, Commentaire).
    Sélectionne NB_OUI échantillons "Oui" et NB_NON échantillons "Non".
    Retourne le texte (prompt) ou "" en cas d'erreur.
    """
    if not os.path.exists(TRAIN_CSV):
        print(f"[ERREUR] Le fichier d'entraînement n'existe pas : {TRAIN_CSV}")
        return ""

    try:
        df_train = pd.read_csv(TRAIN_CSV, sep=",", quotechar='"', on_bad_lines="skip")
    except Exception as e:
        print("[ERREUR] Impossible de lire le CSV d'entraînement :", e)
        return ""

    # Check required columns
    for col in ["Title", "Société", "Pertinent", "Commentaire"]:
        if col not in df_train.columns:
            print(f"[ERREUR] Colonne manquante : {col} dans {TRAIN_CSV}")
            return ""

    # Split the data into "Oui" and "Non"
    df_oui = df_train[df_train["Pertinent"].str.strip().str.lower() == "oui"]
    df_non = df_train[df_train["Pertinent"].str.strip().str.lower() == "non"]

    if len(df_oui) == 0 and len(df_non) == 0:
        print("[ERREUR] Aucune donnée Oui/Non dans le CSV d'entraînement.")
        return ""

    # Random sampling
    oui_sample = df_oui.sample(min(NB_OUI, len(df_oui)), random_state=random.randint(0, 999999))
    non_sample = df_non.sample(min(NB_NON, len(df_non)), random_state=random.randint(0, 999999))

    examples = []
    for _, row in oui_sample.iterrows():
        examples.append((row["Title"], row["Société"], row["Commentaire"], "OUI"))
    for _, row in non_sample.iterrows():
        examples.append((row["Title"], row["Société"], row["Commentaire"], "NON"))

    # Shuffle the combined list
    random.shuffle(examples)

    # Prompt in French
    prompt = (
        "Tu es un classifieur d'articles. "
        "Tu dois répondre UNIQUEMENT par «OUI» ou «NON» pour indiquer :\n\n"

        "«OUI» si ce sont des articles relatifs à :\n"
        "- la stratégie\n"
        "- aux finances de l'entreprise\n"
        "- aux salariés / syndicats\n"
        "- les fusions/acquisitions\n"
        "- les concurrents\n"
        "- le directoire\n"
        "- etc.\n\n"

        "«NON» si ce sont des articles relatifs à :\n"
        "- des promotions\n"
        "- des offres commerciales\n"
        "- des faits divers\n"
        "- du sport ou événements sportifs\n\n"

        "Voici quelques exemples :\n\n"
    )

    index_example = 1
    for titre, societe, commentaire, label in examples:
        prompt += f"Exemple {index_example}:\n"
        prompt += f"Titre : \"{titre}\"\n"
        prompt += f"Entreprise : {societe}\n"
        prompt += f"Commentaire : {commentaire}\n"
        prompt += f"Pertinent : {label}\n\n"
        index_example += 1

    prompt += (
        "Fin des exemples.\n\n"
        "Maintenant, analyse l'article ci-dessous et réponds UNIQUEMENT par «OUI» ou «NON».\n"
    )

    return prompt

# -------------------------------------------------------------------
# 2) CLASSIFY ONE ARTICLE
# -------------------------------------------------------------------
def classifier_article(bloc_few_shot, titre, societe):
    """
    Construit le prompt final (bloc_few_shot + article),
    appelle Ollama via STDIN,
    renvoie "Oui" ou "Non".
    """
    final_prompt = (
        bloc_few_shot
        + "-----\n"
        + f"Titre : \"{titre}\"\n"
        + f"Entreprise : {societe}\n"
        + "Pertinent : "
    )

    cmd = ["ollama", "run", OLLAMA_MODEL]

    try:
        result = subprocess.run(
            cmd,
            input=final_prompt,
            text=True,
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("[ERREUR] Échec lors de l'appel à Ollama.")
        print(e.stderr)
        return "Non"

    # Raw output
    sortie = result.stdout.strip().lower()

    # Simple detection of "oui" or "non"
    if "oui" in sortie:
        return "Oui"
    elif "non" in sortie:
        return "Non"
    else:
        return "Non"

# -------------------------------------------------------------------
# 3) MAIN SCRIPT
# -------------------------------------------------------------------
def main():
    """
    Lit le CSV cible (TARGET_CSV),
    pour chaque ligne, construit un few-shot prompt, appelle classifier_article,
    met à jour Pertinent, et écrit le CSV à la fin.
    Affiche la durée totale en fin.
    """
    print("[INFO] Début du script. Classification via Ollama (few-shot).")

    start_time = time.time()

    # Check if target CSV exists
    if not os.path.exists(TARGET_CSV):
        print(f"[ERREUR] Le fichier à classifier n'existe pas : {TARGET_CSV}")
        return

    # Read the target CSV
    try:
        df_target = pd.read_csv(TARGET_CSV, sep=",", quotechar='"', on_bad_lines="skip")
    except Exception as e:
        print("[ERREUR] Impossible de lire le CSV cible :", e)
        return

    for col in ["Title", "Société", "Pertinent"]:
        if col not in df_target.columns:
            print(f"[ERREUR] Colonne manquante : {col} dans {TARGET_CSV}")
            return

    print(f"[DEBUG] Nombre d'articles à classifier : {len(df_target)}")

    # For each article
    for idx in range(len(df_target)):
        titre = str(df_target.at[idx, "Title"])
        societe = str(df_target.at[idx, "Société"])

        # Build few-shot
        bloc = construire_few_shot()
        if not bloc:
            print("[ERREUR] Impossible de construire le few-shot.")
            df_target.at[idx, "Pertinent"] = "Non"
            continue

        # Classify
        label = classifier_article(bloc, titre, societe)
        df_target.at[idx, "Pertinent"] = label

        print(f"[DEBUG] idx={idx}, Titre='{titre[:60]}...' => {label}")

    # Save the updated CSV
    df_target.to_csv(TARGET_CSV, index=False)
    print(f"[INFO] Classification terminée. CSV mis à jour : {TARGET_CSV}")

    end_time = time.time()
    duree = end_time - start_time
    print(f"[INFO] Durée d'exécution : {duree:.2f} secondes")


if __name__ == "__main__":
    main()
