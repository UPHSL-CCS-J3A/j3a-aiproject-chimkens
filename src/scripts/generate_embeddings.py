"""
generate_embeddings.py

Generates hybrid vector embeddings for music tracks by combining:
1. Text embeddings (track name, artist, genre, etc.)
2. Audio features (energy, danceability, tempo, etc.)

Run this script once to generate embeddings for all tracks in your datasets.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
from tqdm import tqdm
import unicodedata

print("Loading sentence transformer model...")
# Using a lightweight, fast model that works offline
model = SentenceTransformer('all-MiniLM-L6-v2')

import os

# Resolve root data directory robustly
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(root_dir, 'data')

print("Loading datasets...")
# Load all CSV files from data folder
dataset_files = [
    os.path.join(data_dir, 'high_popularity_spotify_data.csv'),
    os.path.join(data_dir, 'low_popularity_spotify_data.csv'),
    os.path.join(data_dir, 'spotify_dataset.csv')
]

dfs = []
for file in dataset_files:
    if os.path.exists(file):
        print(f"  Loading {file}...")
        df = pd.read_csv(file)
        dfs.append(df)
    else:
        print(f"  Warning: {file} not found, skipping...")

if not dfs:
    print(f"ERROR: No datasets found! Make sure CSV files are in the {data_dir} folder.")
    exit(1)

# Combine all datasets
df = pd.concat(dfs, ignore_index=True)

# Normalize column names by coalescing
col_mappings = {
    'track_artist': 'artists',
    'track_album_name': 'album_name',
    'playlist_genre': 'track_genre'
}

for target, source in col_mappings.items():
    if source in df.columns:
        if target in df.columns:
            # Coalesce: fill target NaNs with source values
            df[target] = df[target].fillna(df[source])
            df.drop(columns=[source], inplace=True)
        else:
            # Just rename if target doesn't exist
            df.rename(columns={source: target}, inplace=True)

print(f"Total tracks loaded: {len(df)}")

# Remove duplicates
df.drop_duplicates(subset=['track_id'], inplace=True)
# IMPORTANT: Reset index after removing duplicates
df.reset_index(drop=True, inplace=True)
print(f"After removing duplicates: {len(df)}")

# Fill missing values appropriately
# For text columns, fill with empty string
text_columns = ['track_name', 'track_artist', 'track_album_name', 'playlist_genre', 'playlist_name']
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].fillna('')

# For numeric columns (audio features), fill with 0
audio_features = ['energy', 'danceability', 'valence', 'tempo', 'acousticness', 
                  'instrumentalness', 'speechiness', 'liveness']

# Ensure all audio features exist
for feat in audio_features:
    if feat not in df.columns:
        print(f"Warning: '{feat}' column not found, using 0 as default")
        df[feat] = 0
    else:
        # Fill NaN with 0 for numeric columns
        df[feat] = df[feat].fillna(0)

print("\nGenerating embeddings...")

# Known artists for nationality detection (imported from local_library.py concept)
KNOWN_ARTISTS = {
    'korean': ['bts', 'blackpink', 'twice', 'newjeans', 'stray kids', 'ive', 'le sserafim', 
               'nct', 'seventeen', 'txt', 'tomorrow x together', 'enhypen', 'itzy', 'aespa', 
               'red velvet', 'exo', 'got7', 'ateez', 'the boyz', 'monsta x', 'super junior',
               'girls generation', 'snsd', 'bigbang', 'shinee', 'mamamoo', 'everglow'],
    'japanese': ['yoasobi', 'ado', 'kenshi yonezu', 'official hige dandism', 'king gnu', 
                 'back number', 'radwimps', 'mrs. green apple', 'yorushika', 'eve',
                 'one ok rock', 'babymetal', 'perfume'],
    'filipino': ['ben&ben', 'moira dela torre', 'zack tabudlo', 'sb19', 'sarah geronimo',
                 'bamboo', 'rivermaya', 'eraserheads', 'parokya ni edgar']
}

def detect_script(text):
    """Detect the Unicode script of text (Hangul, Hiragana, Katakana, etc.)"""
    if not text:
        return None
    
    # Check for Korean (Hangul)
    if any('\uac00' <= char <= '\ud7a3' for char in text):
        return 'korean'
    
    # Check for Japanese (Hiragana or Katakana)
    if any('\u3040' <= char <= '\u309f' for char in text):  # Hiragana
        return 'japanese'
    if any('\u30a0' <= char <= '\u30ff' for char in text):  # Katakana
        return 'japanese'
    
    return None

def infer_artist_nationality(artist_name):
    """Infer artist nationality from name"""
    if not artist_name:
        return None
    
    artist_lower = artist_name.lower()
    
    # Check against known artists
    for nationality, artists in KNOWN_ARTISTS.items():
        if any(known in artist_lower for known in artists):
            return nationality
    
    # Check Unicode script
    script = detect_script(artist_name)
    if script:
        return script
    
    return None

def detect_language_hint(track_name, artist_name):
    """Detect language hints from track/artist names"""
    # Check track name script
    track_script = detect_script(track_name)
    if track_script:
        return f"{track_script} language"
    
    # Check artist name script
    artist_script = detect_script(artist_name)
    if artist_script:
        return f"{artist_script} language"
    
    return None

# Helper functions for enhanced descriptions
def get_mood_description(row):
    """Convert audio features to mood words"""
    valence = row.get('valence', 0.5)
    energy = row.get('energy', 0.5)
    
    moods = []
    if valence > 0.7 and energy > 0.7:
        moods.append("happy energetic upbeat")
    elif valence < 0.3 and energy < 0.4:
        moods.append("sad melancholic slow")
    elif valence > 0.7 and energy < 0.4:
        moods.append("calm peaceful happy")
    elif valence < 0.3 and energy > 0.7:
        moods.append("angry aggressive intense")
    elif energy > 0.8:
        moods.append("high energy intense")
    elif energy < 0.2:
        moods.append("chill relaxed acoustic")
        
    if row.get('danceability', 0) > 0.7:
        moods.append("danceable groovy")
        
    return " ".join(moods)

def get_tempo_category(tempo):
    if tempo < 80: return "slow tempo"
    if tempo < 120: return "moderate tempo"
    return "fast tempo"

def create_text_description(row):
    text_parts = []
    
    # Basic info
    if row.get('track_name'): text_parts.append(f"Song: {row['track_name']}")
    
    # Artist with nationality hint
    if row.get('track_artist'):
        artist = row['track_artist']
        text_parts.append(f"Artist: {artist}")
        
        # Add nationality if detected
        nationality = infer_artist_nationality(artist)
        if nationality:
            text_parts.append(f"{nationality} artist")
    
    if row.get('track_album_name'): text_parts.append(f"Album: {row['track_album_name']}")
    if row.get('playlist_genre'): text_parts.append(f"Genre: {row['playlist_genre']}")
    
    # Language hint
    lang_hint = detect_language_hint(row.get('track_name', ''), row.get('track_artist', ''))
    if lang_hint:
        text_parts.append(lang_hint)
    
    # Enhanced info
    mood = get_mood_description(row)
    if mood: text_parts.append(f"Mood: {mood}")
    
    if row.get('tempo'):
        text_parts.append(f"Tempo: {get_tempo_category(row['tempo'])}")
        
    if row.get('track_year'):
        text_parts.append(f"Year: {row['track_year']}")
        
    return ". ".join(text_parts)

print("Generating text descriptions...")
texts = [create_text_description(row) for _, row in df.iterrows()]

print("Encoding text embeddings (Batch Mode)...")
# Batch encoding is much faster
text_embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

# Normalize audio features to 0-1 range
scaler = MinMaxScaler()
audio_normalized = scaler.fit_transform(df[audio_features])

# Combine embeddings (Balanced Weighting)
# Text: 0.7, Audio: 0.3
print("Combining embeddings...")
text_weight = 0.7
audio_weight = 0.3

# Normalize text embeddings first
text_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
text_embeddings = text_embeddings / text_norms

embeddings_list = []
for i in range(len(df)):
    # Combine
    hybrid = np.concatenate([
        text_embeddings[i] * text_weight,
        audio_normalized[i] * audio_weight
    ])
    embeddings_list.append(hybrid)

embeddings_array = np.array(embeddings_list)

print(f"\nEmbedding shape: {embeddings_array.shape}")
print(f"  - Text dimensions: 384")
print(f"  - Audio dimensions: 8")
print(f"  - Total dimensions: {embeddings_array.shape[1]}")

# Save embeddings and metadata
output_data = {
    'embeddings': embeddings_array,
    'track_ids': df['track_id'].tolist(),
    'scaler': scaler,  # Save scaler for future use
    'audio_features': audio_features,
    'text_weight': text_weight,
    'audio_weight': audio_weight
}

output_file = os.path.join(data_dir, 'embeddings.pkl')
print(f"\nSaving embeddings to {output_file}...")
with open(output_file, 'wb') as f:
    pickle.dump(output_data, f)

print("✅ Done! Embeddings generated successfully.")
print(f"   File: {output_file}")
print(f"   Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
print("\nYou can now use the enhanced local library for better recommendations!")
