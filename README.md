# Chim-eTunes: Personalized Music Recommendation Chatbot

## Overview
With millions of songs available on streaming platforms, users often struggle to find new music they enjoy. Manually searching for songs can be overwhelming and time-consuming, leading to lower engagement and satisfaction.  

**Chim-eTunes** solves this by providing **personalized music recommendations** through a **chat interface**, taking into account each user’s preferences, moods, and listening history.

---

## Problem Description
Finding new music that fits a listener’s unique taste can be difficult due to the massive volume of available songs. Users spend excessive time browsing playlists or searching manually, which reduces enjoyment and discovery potential.

---

## Proposed Solution
Chim-eTunes recommends songs, playlists, or artists that align with a user’s mood and taste. Through natural conversation, it helps users:
- Discover new music effortlessly  
- Get recommendations based on mood, genre, or favorite artists  
- Stay engaged through personalized and interactive responses  

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

[User Input]  
⬇️  
[Text Processing / Mood Analysis]  
⬇️  
[Recommendation Engine]  
⬇️  
[Music Database]  
⬇️  
[Chatbot Response]

### Steps
1. **User Input** — User provides input such as mood, genre, or favorite artist.  
2. **Text Processing / NLP** — Extracts mood, genre, and artist preferences using natural language processing.  
3. **Recommendation Engine** — Matches user preferences with the song database using filtering algorithms.  
4. **Music Database** — Stores songs and metadata (artist, genre, mood, tempo, etc.).  
5. **Chatbot Response** — Sends personalized music recommendations back to the user.

---

## Possible Datasets
**Public Spotify Dataset:**  
[Spotify Music Dataset on Kaggle](https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset)


