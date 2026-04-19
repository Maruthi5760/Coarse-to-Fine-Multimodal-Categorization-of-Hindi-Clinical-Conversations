import os


BASE_DIR = "/content/drive/MyDrive/multimodal_project"

EXCEL_PATH = os.path.join(BASE_DIR, "dataset.xlsx")
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
AUDIO_DIR = os.path.join(BASE_DIR, "audios")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_features_v2")

# Save checkpoints/test indices directly in the main project folder
MODEL_SAVE_DIR = BASE_DIR 

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- HYPERPARAMETERS ---
MAX_TEXT_LEN = 128
MAX_FRAMES = 30
MAX_AUDIO_FRAMES = 150   # mel frames (≈1.5 s of audio at hop 160)
N_MELS = 80
BATCH_SIZE = 16
EPOCHS = 120
LR = 2e-5
PATIENCE = 20

# --- CLASS MAPPINGS ---
LABEL_FIXES = {
    "Acknowlege": "Acknowledge", 
    "Acknowlegde": "Acknowledge",
    "Oos(question)": "Oos(Question)",
    "Ask For Opinion": "Ask For Opinions",
}

MERGE_MAP = {
    "Inform": "Inform", "Explain": "Explain", "Advise": "Advise",
    "Acknowledge": "Acknowledge", "Agree": "Acknowledge", "Confirm": "Acknowledge",
    "Doubt": "Doubt",
    "Care": "Social", "Comfort": "Social", "Praise": "Social", 
    "Taunt": "Social", "Joke": "Social", "Emphasize": "Social",
    "Ask For Help": "Request", "Ask For Opinions": "Request", "Invite": "Request",
    "Refuse": "Negative", "Warn": "Negative", "Prevent": "Negative", "Complain": "Negative",
    "Greet": "Ritual", "Introduce": "Ritual", "Thank": "Ritual", "Apologise": "Ritual", "Leave": "Ritual",
    "Plan": "Plan", "Arrange": "Plan",
    "Oos": "Oos", "Oos(Question)": "Oos",
}

def normalize_label(s):
    s = str(s).strip().title()
    return LABEL_FIXES.get(s, s)

def merged_label(s):
    return MERGE_MAP.get(normalize_label(s), normalize_label(s))