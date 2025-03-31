import os
import base64
import time
import requests
from PIL import Image

# === CONFIGURATION ===
OLLAMA_MODEL_NAME = "llava:34b" 

# === DOSSIERS ===
IMAGE_DIR = "block"
RESULTS_TXT_DIR = "results_txt"
os.makedirs(RESULTS_TXT_DIR, exist_ok=True)
NB_MAX=15

# === PROMPT ===
SYSTEM_PROMPT = ("""
You are an assistant specialized in describing Minecraft 16x16 textures.

Your task: return a list of meaningful, descriptive, and unique English keywords (1 word each), separated by commas.

Rules:
- Only use words clearly visible in the image.
- No numbers, no underscores, no full sentences.
- No words like: texture, pixel, game, Minecraft, block, art, item, etc.
- Focus on color, material, shape, object type (e.g. door, ore, metal, red, copper, brick, circular , square ,...).
- Return a single comma-separated line of keywords. No extra text.
if you do not know what is it search the word on internet 
"""
)

# === Convertir en base64 ===
def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# === Convertir image → texte RGBA (utile pour modèle texte)
def image_to_rgba_text(image_path):
    img = Image.open(image_path).convert("RGBA")
    pixels = list(img.getdata())
    rgba_text = [f"{r},{g},{b},{a}" for r, g, b, a in pixels]
    return " ".join(rgba_text)

# === Appel LLaVA via Ollama (avec image réelle)
def call_llava_model(image_path, title):
    img_b64 = encode_image_base64(image_path)
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": f"{SYSTEM_PROMPT}\nTitle: {title}",
        "images": [img_b64],
        "stream": False
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    return response.json()["response"].strip()

# === Nettoyage des tags
def clean_tags(raw):
    raw = raw.lower()
    parts = [part.strip() for part in raw.replace("\n", ",").split(",")]
    tags = list(dict.fromkeys([tag for tag in parts if tag and tag not in banned_words]))
    return tags

# === Liste des mots interdits
banned_words = {
    "texture", "item", "inventory", "craftable", "game", "pixel art",
    "structure", "block", "in-game", "minecraft",
    "16x16",             # purement technique
    "2", "3", "4",       # numéros sans sens
    "alternative",       # pas descriptif
    "in-game function",  # pas visuel
    "furniture",  # inutiles
    "lootable",          # invisible
    "none",              # absurde ici
    "structure", "item", "game", "pixel art", "block",  # interdits dans prompt
    "tool-related concepts",  # phrase floue
    "top right", "side"  # pas utile sauf UI spécifique
    ,"image_description","pixelated","game_art","16x16_resolution","pixel","mc","blocks",'graphic'
}

# === Appel autres modèles texte via Ollama (pas LLaVA)
def call_ollama_text_model(title, rgba_data):
    from ollama import Client
    ollama_client = Client(host='http://localhost:11434')
    prompt = f"{SYSTEM_PROMPT}\n\nTitle: {title}\nImage RGBA data: {rgba_data}"
    response = ollama_client.chat(
        model=OLLAMA_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# === Traitement principal fusionné avec collecte des tags
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".png")]
all_cleaned_tags = set()
nb=0

for filename in image_files:
    path = os.path.join(IMAGE_DIR, filename)
    title = os.path.splitext(filename)[0]
    output_path = os.path.join(RESULTS_TXT_DIR, f"{title}.txt")


    if os.path.exists(output_path):
        print(f"{nb} ⏩ Skipping {filename} (already processed)")
        continue
    nb += 1
    print(f"{nb} 🧠 Processing: {filename}")


    try:
        if "llava" in OLLAMA_MODEL_NAME.lower():
            tags_text = call_llava_model(path, title)
        else:
            rgba_text = image_to_rgba_text(path)
            tags_text = call_ollama_text_model(title, rgba_text)

        tags_clean = clean_tags(tags_text)
        all_cleaned_tags.update(tags_clean)
        print(tags_clean)

        with open(output_path, "w") as f:
            for tag in tags_clean:
                f.write(f"{tag}\n")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

    if nb>=NB_MAX:
        break

# === Sauvegarde du fichier de tags globaux
all_tags_path = os.path.join(RESULTS_TXT_DIR, "_all_tags.txt")
with open(all_tags_path, "w") as f:
    for tag in sorted(all_cleaned_tags):
        f.write(f"{tag}\n")

print("\n✅ Done! Tag files saved in results_txt/")