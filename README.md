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
| **Performance Measure** | User satisfaction, engagement (clicks, listens, likes), and positive interactions (e.g., adding to playlists). |
| **Environment** | Virtual — a chat-based interface connected to a music database or streaming service. |
| **Actuators** | Sends text-based recommendations for songs, playlists, or artists. |
| **Sensors** | Collects user inputs such as text messages, moods, favorite genres, listening history, and ratings. |

---

## AI Concepts Used

| **Concept** | **Description** |
|--------------|-----------------|
| **Intelligent Agent Type** | *Goal-based agent* — aims to recommend music that aligns with user preferences. |
| **Search / Optimization Strategy** | Uses **similarity scoring** (e.g., cosine similarity) to find songs closely matching user taste. |
| **Learning / Decision Component** | Implements **recommendation systems** using:<br> - Content-Based Filtering (matches song features to user preferences)<br> - Collaborative Filtering (finds songs liked by users with similar tastes) |

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

## Possible Datasets
**Public Spotify Dataset:**  
- [Spotify Music Dataset on Kaggle](https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset) 
- https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset

