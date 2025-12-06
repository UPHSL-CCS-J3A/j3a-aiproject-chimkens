# Chim-eTunes: Personalized Music Recommendation Chatbot

## Overview
With millions of songs available on streaming platforms, users often struggle to find new music they enjoy. Manually searching for songs can be overwhelming and time-consuming, leading to lower engagement and satisfaction.  

**Chim-eTunes** solves this by providing **personalized music recommendations** through a **chat interface**, taking into account each user’s preferences, moods, and listening history.

---

## 🎯 Problem Description

With millions of songs available on streaming platforms, users face significant challenges:
- *Information Overload*: Overwhelming volume of available music makes discovery difficult
- *Time-Consuming Search*: Manual browsing through playlists and catalogs is inefficient
- *Limited Personalization*: Generic recommendations don't account for nuanced preferences (mood, context, specific artists)
- *Poor User Experience*: Lack of conversational, intuitive interfaces for music discovery

These issues lead to reduced user engagement, missed discovery opportunities, and overall dissatisfaction with music streaming experiences.

---

## 💡 Proposed Solution

*Chim-eTunes* is an intelligent desktop application that provides *personalized music recommendations* through a natural, conversational chat interface. The system:

- 🎵 *Understands Natural Language*: Processes complex queries like "sad songs by Adele" or "upbeat K-pop for workouts"
- 🧠 *Semantic Search*: Uses AI embeddings to find music based on meaning, not just keywords
- 🎨 *Mood-Aware Recommendations*: Analyzes audio features (valence, energy, tempo) to match emotional context
- 📊 *Hybrid Architecture*: Combines local library intelligence with Spotify API integration
- 🎭 *Conversational AI*: Powered by Groq LLM for natural, context-aware interactions

---

## PEAS Model

| **Component** | **Description** |
|----------------|-----------------|
| **Performance Measure** |• User satisfaction and positive interactions (e.g., adding to playlists).<br> • Recommendation accuracy (semantic similarity scores).<br>• User satisfaction (successful playlist additions).<br>• Response relevance (intent parsing accuracy).<br>• System responsiveness (query processing time). |
| **Environment** |• *Virtual*: a chat-based interface connected to a music database or streaming service. Desktop application with GUI.<br>• *Data-Driven*: 85,000+ track database with embeddings.<br>• *API-Connected*: Spotify Web API for playlist management<br>• *User Context*: Chat history, preferences, session data. |
| **Actuators** |• Sends text-based recommendations for songs, playlists, or artists.<br>• Display text-based recommendations.<br>• Create/modify Spotify playlists.<br>• Show track details and audio features.<br>• Update UI elements (typing indicators, theme switching). | |
| **Sensors** |• Collects user inputs such as text messages, moods, favorite genres, listening history, and ratings.<br>• User text input (natural language queries).<br>• Spotify authentication state.<br>• Local music library metadata.<br>• Audio feature data (valence, energy, danceability, etc.).<br>• User interaction history. |

---

## AI Concepts Used

| **Concept** | **Description** |
|--------------|-----------------|
| **Intelligent Agent Type** | **Goal-based agent**<br> • *Goal*: Maximize user satisfaction by providing relevant music recommendations.<br>• *Rationality*: Selects actions (recommendations) that best achieve the goal given available information.<br>• *Autonomy*: Operates independently once initialized, learning from user interactions.|
| **Search / Optimization Strategy** | **Semantic Search (Vector Space Model)**<br>• Uses *similarity scoring* (e.g., cosine similarity) to find songs closely matching user taste.<br>• Uses *sentence-transformers* (all-MiniLM-L6-v2) to generate 384-dimensional embeddings.<br>• Combines text metadata (70%) with audio features (30%) for hybrid representations.<br>• Employs *cosine similarity* to find nearest neighbors in embedding space.<br>• Optimizes for both semantic relevance and audio feature matching.<br><br> **Local Search Optimization**<br>• Filters results by artist, genre, or mood constraints.<br>• Applies audio feature thresholds (e.g., high valence for "happy" queries).<br>• Randomizes within top results to provide variety.|
| **Learning / Decision Component** | **Natural Language Understanding (NLU)**<br> • *Intent Classification*: Groq LLM (Llama-based) parses user queries into structured intents.<br> • *Entity Extraction*: Identifies artists, genres, moods, and actions from free-form text.<br> • *Context Awareness*: Maintains conversation history for follow-up queries.<br><br>**Recommendation Systems**<br> • *Content-Based Filtering*: Matches song features to user preferences.<br> • *Hybrid Approach*: Combines semantic similarity with audio feature analysis.<br> • *Collaborative Signals*: Uses popularity scores and playlist co-occurrence. |

---

## System Architecture / Flowchart

### Flow

<img width="480" height="673" alt="image" src="https://github.com/user-attachments/assets/f4d2b295-faf9-4298-853b-358d4f67c9af" />


### Processing Flow

1. *Input Processing*
   - User enters natural language query
   - Text sent to Groq LLM for intent parsing
   - Extracts: intent type, entities (artist, genre, mood), parameters

2. *Recommendation Generation*
   - Query converted to embedding vector (384-dim)
   - Cosine similarity computed against 85K+ track embeddings
   - Results filtered by constraints (artist, audio features)
   - Top matches ranked and returned

3. *Response Formatting*
   - Recommendations formatted as conversational text
   - Track details displayed with audio features
   - Playlist actions executed via Spotify API

4. *UI Update*
   - Chat interface updated with bot response
   - Typing indicator shown during processing
   - Theme-aware color coding for messages

---

## Datasets Used
**Public Spotify Dataset:**  
- [Spotify Music Dataset on Kaggle](https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset) 
- https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset
- https://drive.google.com/drive/folders/1l5mQLOQwb6DREUV7MmpyITWKoUSqz8-p?usp=drive_link
- https://www.canva.com/design/DAG6Q5MZ0hM/F93Usbp8sW6pgxYP558YHg/edit?utm_content=DAG6Q5MZ0hM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

