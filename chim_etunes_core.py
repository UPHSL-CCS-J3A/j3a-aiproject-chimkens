"""
chim_etunes_core.py

Core logic for Chim-eTunes Music Bot
Contains:
- SpotifyClient: Handles Spotify API interactions
- IntentParser: Handles Groq LLM interactions
- ChatBot: Orchestrates the logic
"""

import os
import re
import json
import time
import random
from groq import Groq
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from local_library import LocalMusicLibrary

load_dotenv()

# ---------- Config ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:3000/callback")

if not (GROQ_API_KEY and SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET):
    raise RuntimeError("Please set GROQ_API_KEY, SPOTIPY_CLIENT_ID, and SPOTIPY_CLIENT_SECRET environment variables.")

SPOTIFY_SCOPE = "playlist-read-private playlist-read-collaborative playlist-modify-private playlist-modify-public user-read-private"

def get_sp_oauth():
    # For desktop, we can use a cache handler to persist tokens automatically
    # This creates a .cache file in the current directory
    return SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope=SPOTIFY_SCOPE,
        open_browser=True # Allow auto-opening browser for login
    )

def parse_playlist_id(link):
    if not link: return None
    m = re.search(r"playlist/([A-Za-z0-9]+)", link)
    if m: return m.group(1)
    return link if re.fullmatch(r"[A-Za-z0-9]+", link) else None

# ---------- Classes ----------

class SpotifyClient:
    """Encapsulates all Spotify API interactions."""
    def __init__(self, token_info):
        self.sp = spotipy.Spotify(auth=token_info['access_token'])

    def current_user(self):
        return self.sp.current_user()

    def search(self, q, type, limit=10):
        return self.sp.search(q=q, type=type, limit=limit)

    def create_playlist(self, user_id, name, description="Created by Chim-eken"):
        return self.sp.user_playlist_create(user=user_id, name=name, public=False, description=description)

    def add_tracks(self, playlist_id, uris):
        return self.sp.playlist_add_items(playlist_id, uris)

    def remove_tracks(self, playlist_id, uris):
        return self.sp.playlist_remove_all_occurrences_of_items(playlist_id, uris)

    def get_playlist_tracks(self, playlist_id, limit=100):
        results = self.sp.playlist_items(playlist_id, additional_types=['track'], limit=limit)
        items = results.get('items', [])
        while results.get('next'):
            results = self.sp.next(results)
            items.extend(results.get('items', []))
        
        tracks = []
        for it in items:
            track = it.get('track')
            if track and track.get('id'):
                tracks.append(track)
        return tracks

    def get_recommendations(self, seed_artists=None, seed_tracks=None, seed_genres=None, limit=10):
        return self.sp.recommendations(seed_artists=seed_artists, seed_tracks=seed_tracks, seed_genres=seed_genres, limit=limit)

    def get_artist_top_tracks(self, artist_id):
        return self.sp.artist_top_tracks(artist_id)

    def get_audio_features(self, track_ids):
        return self.sp.audio_features(track_ids)
    
    def get_album_tracks(self, album_id):
        """Get tracks from an album"""
        return self.sp.album_tracks(album_id)

    def get_user_playlists(self, limit=50):
        """Get current user's playlists"""
        results = self.sp.current_user_playlists(limit=limit)
        playlists = []
        for item in results.get('items', []):
            playlists.append({
                'name': item['name'],
                'id': item['id'],
                'tracks': item['tracks']['total']
            })
        return playlists


class IntentParser:
    """Handles interactions with the Groq LLM for intent parsing and chat generation."""
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def parse(self, history_text):
        prompt = f"""
You are an assistant that extracts user intent for a music chatbot.
Here is the conversation history:
\"\"\"{history_text}\"\"\"

IMPORTANT CONTEXT HANDLING: If the user uses pronouns or references like "that", "it", "them", "those songs", look at the conversation history to resolve what they're referring to:
- If the previous bot message showed track details, use that track name
- If the previous bot message showed recommendations, use the seed track from that recommendation
- Example:
  User: "do you know die with a smile"
  Bot: "**Die With A Smile** by Lady Gaga, Bruno Mars..."
  User: "give me songs like that"
  -> Extract: seed_track: "Die With A Smile", seed_artist: "Lady Gaga"

Based on the LATEST user message (and using previous context if needed), return a JSON object with fields:
- "intent": one of ["create_playlist", "show_playlists", "analyze_playlist", "add_to_playlist", "remove_from_playlist", "get_recommendations", "read_playlist", "get_artist_details", "get_track_details", "get_album_details", "casual_chat"]
  * "create_playlist": when user wants to CREATE a new playlist (e.g., "create a playlist called My Vibes", "make a playlist with these songs", "create a workout playlist")
  * "show_playlists": when user wants to see a LIST of all their playlists (e.g., "show my playlists", "list my playlists", "what playlists do I have")
  * "read_playlist": when user wants to see tracks IN a specific playlist by name OR URL (e.g., "show me my workout playlist", "read my chill playlist", "show tracks in my favorites", or provides a Spotify playlist URL/link)
  * "edit_playlist": when user wants to rename a playlist (e.g., "rename my playlist to X", "change playlist name to Y")
  * "analyze_playlist": when user wants recommendations based on a specific playlist (e.g., "analyze my workout playlist", "songs like my chill playlist")
  * "get_recommendations": when user wants recommendations based on a specific genre, mood, or artist (e.g., "recommend 5 pop songs", "give me 10 k-pop songs", "give me songs by adele", "sad metal rock", "pop", "rock metal")
  * "get_artist_details": ONLY when user explicitly asks for artist information/details (e.g., "tell me about adele", "who is taylor swift", "artist info for drake", "Do you know munimuni", )
  * "add_to_playlist": when user wants to ADD songs to an EXISTING playlist (e.g., "add FAKE LOVE to a playlist", "add these songs to playlist", "put this in a playlist"). ALWAYS extract this intent even without a URL - the bot will ask for the URL. Do NOT respond with casual chat about creating playlists.
    → IMPORTANT: Extract track names to "tracks_to_add" field when user mentions specific songs!
  * "get_album_details": when user wants to get album details (e.g., "tell me about album 'album name'", "who is taylor swift", "artist info for drake")
- "tracks_to_remove": (array of strings) track names to remove from playlist if specified
  * "remove Hacking to the Gate" -> ["Hacking to the Gate"]
  * "remove grace by fujii kaze" -> ["grace"] (and artist: "fujii kaze")
- "playlist_url": (string) the spotify playlist URL/URI if provided, else empty string
- "artist": (string) artist name for adding/removing, else empty string. If user says "add them" or "add these songs", leave empty.
- "limit": (integer) number of songs requested. Extract from phrases like:
  * "give me 5 songs" -> 5
  * "recommend 20 tracks" -> 20
  * "show me 15 k-pop songs" -> 15
  * If no number specified, use 10 as default
- "playlist_name": (string) name to create playlist if user requested creation
- "seed_artist": (string) artist name to use as seed for recommendations. IMPORTANT: Normalize artist names to their full, proper form:
  * "give me songs by adele" -> seed_artist: "Adele"
  * "21 pilots", "twenty 1 pilots", "tøp" -> "twenty one pilots"
  * "billie eyelash" -> "billie eilish"
  * "ariana" -> "ariana grande"
  * "weeknd" -> "the weeknd"
  * "posty" -> "post malone"
  * IMPORTANT: Adjective forms of genres (e.g., "metallic", "jazzy", "rocky") are mood descriptors, NOT artist names
  * "metalic", "metallic" -> mood descriptor (NOT "Metallica" the band)
  * Be flexible with typos and abbreviations, use your best judgment to match to the correct artist name.
- "seed_track": (string) track name to use as seed for recommendations.
  * "songs like house of gold" -> seed_track: "House of Gold" (NOT seed_artist)
  * "songs similar to the ghost of you by mcr" -> seed_track: "The Ghost of You", seed_artist: "My Chemical Romance"
  * "songs like love by michael buble" -> seed_track: "L.O.V.E", seed_artist: "Michael Buble"
- "tracks_to_add": (array of strings) track names to add to playlist if specified. ALWAYS extract track names when user mentions specific songs.
  * "add FAKE LOVE and euphoria to a playlist" -> ["FAKE LOVE", "euphoria"]
  * "add Break the Silence and Waste it on me into a playlist" -> ["Break the Silence", "Waste it on me"]
  * "add Hacking to the Gate - symphonic ver. by いとうかなこ" -> ["Hacking to the Gate - symphonic ver."]
  * "add these songs" -> [] (use last_recs from session)
  * "add songs by BTS" -> [] (use artist field instead)
- "add_mode": (string) "artist" (default) or "similar" (if user wants to add songs SIMILAR to the playlist content)
- "search_query": (string) for MULTI-WORD descriptive queries like "love songs", "workout music", "sad songs", etc.
  NOTE: For SINGLE-WORD genres (e.g., "pop", "jazz", "emo", "rock"), use target_genre instead!
  IMPORTANT: If user says "love songs by arctic monkeys", set "search_query": "love songs" AND "seed_artist": "Arctic Monkeys".
  * "sad opm" -> "sad opm" (combined mood + nationality)
  * "happy k-pop" -> "happy k-pop"
  * "chill japanese songs" -> "chill japanese songs"
  * "energetic russian music" -> "energetic russian music"
  For combined mood + nationality queries, use search_query and leave target_genre empty.
- "target_genre": (string) for genre names. Do NOT include region prefixes here, EXCEPT for these specific known genres:
  * "Korean pop" or "kpop" or "k-pop" or "korean" -> "korean"
  * "Filipino pop" or "filipino" or "tagalog" or "opm" or "pinoy" -> "filipino"
  * "Japanese pop" or "jpop" or "j-pop" or "japanese" -> "japanese"
  * "Japanese rock" or "j-rock" -> "j-rock"
  * "Chinese pop" or "cpop" or "c-pop" or "mandarin" or "chinese" -> "chinese"
  * "Russian pop" or "russian" -> "russian"
  * For other genres, use the base genre (e.g. "french jazz" -> "jazz")
- "target_playlist": (string) name or ID of playlist to show, read, or analyze. Examples:
  * "show my playlists" -> "" (intent: show_playlists)
  * "show me my workout playlist" -> "workout" (intent: read_playlist)
  * "analyze my chill vibes playlist" -> "chill vibes" (intent: analyze_playlist)
  * "recommend songs like my workout playlist" -> "workout" (intent: analyze_playlist)
- "needs_playlist_url": (boolean) true if user wants to add to playlist but didn't provide a URL
  * "add FAKE LOVE to a playlist" -> true
  * "add these songs to my playlist" -> true (no URL given)
  * "add these songs to [playlist URL]" -> false (URL provided)
- "album_name": (string) album name if user is asking about an album
- "new_playlist_name": (string) new name for playlist if user wants to rename it

Only return valid JSON. Example:
{{"intent":"add_to_playlist","playlist_url":"","tracks_to_add":["Break the Silence","Waste it on me"],"artist":"","limit":10,"playlist_name":"","seed_artist":"","seed_track":"","add_mode":"","search_query":"","target_genre":""}}
"""
        resp = self.client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        content = resp.choices[0].message.content
        try:
            js = content
            start = js.find("{")
            end = js.rfind("}") + 1
            js = js[start:end]
            return json.loads(js)
        except Exception:
            return {"intent":"casual_chat"}

    def generate_response(self, history_text, extra_instructions="", context=None):
        # Build context-aware instructions
        context_info = ""  
        if context:
            if context.get('spotify_logged_in'):
                context_info += "\n- User IS logged into Spotify. You CAN create playlists, add songs to playlists, and manage their library."
            else:
                context_info += "\n- User is NOT logged into Spotify. You can only provide recommendations as text lists."
        
            if context.get('last_recs'):
                context_info += f"\n- You just recommended {len(context['last_recs'])} songs. User can add these to a playlist if they're logged in."
    
        music_guide = "You are Chim-eken, a friendly music recommendation bot. Keep responses warm and conversational, but gently guide users to ask for music recommendations. Examples: 'Try asking for sad songs, songs by Adele, or energetic workout music!' Be helpful and stay focused on music discovery."
        prompt = f"{music_guide}\n\n{context_info}\n\n{history_text}\n\n{extra_instructions}\nReply naturally as Chim-eken."
        resp = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return resp.choices[0].message.content

class ChatBot:
    """Orchestrates the conversation and business logic."""
    def __init__(self, spotify_client, intent_parser, local_library=None):
        self.sp = spotify_client
        self.parser = intent_parser
        self.local = local_library

    def handle_message(self, user_text, history, session_data):
        # 1. Update history
        history.append({"role": "user", "content": user_text})
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])

        # 2. Parse intent
        parsed = self.parser.parse(history_text)
        intent = parsed.get("intent", "casual_chat")
        
        # 3. Execute logic
        reply_text = ""
        reply_items = []
        
        try:
            if intent == "create_playlist":
                reply_text, reply_items = self._handle_create_playlist(parsed, session_data)
            elif intent == "add_to_playlist":
                reply_text, reply_items = self._handle_add_to_playlist(parsed, session_data)
            elif intent == "remove_from_playlist":
                reply_text, reply_items = self._handle_remove_from_playlist(parsed)
            elif intent == "read_playlist":
                reply_text, reply_items = self._handle_read_playlist(parsed)
            elif intent == "get_recommendations":
                reply_text, reply_items = self._handle_recommendations(parsed, session_data)
            elif intent == "get_artist_details":
                reply_text, reply_items = self._handle_artist_details(parsed)
            elif intent == "get_track_details":
                reply_text, reply_items = self._handle_track_details(parsed)
            elif intent == "show_playlists":
                reply_text, reply_items = self._handle_show_playlists()
            elif intent == "analyze_playlist":
                playlist_name = parsed.get("target_playlist", "")
                reply_text, reply_items = self._handle_analyze_playlist(playlist_name)
            elif intent == "edit_playlist":
                reply_text, reply_items = self._handle_edit_playlist(parsed)
            elif intent == "get_album_details":
                reply_text, reply_items = self._handle_album_details(parsed)
            else:
                context = {
                    'spotify_logged_in': self.sp is not None,
                    'last_recs': session_data.get('last_recs', [])
                }
                reply_text = self.parser.generate_response(history_text, context=context)
        except Exception as e:
            reply_text = f"Sorry, something went wrong: {e}"
            print(f"DEBUG: Error handling message: {e}")

        # 4. Update history & return
        history.append({"role": "assistant", "content": reply_text})
        # Return images if available
        if 'reply_images' in locals() and reply_images:
            return reply_text, reply_items + reply_images, history
        else:
            return reply_text, reply_items, history

    # --- Handlers ---

    def _handle_create_playlist(self, parsed, session_data):
        if not self.sp: return "You need to log in with Spotify to create playlists (Local mode is read-only).", []
        
        name = parsed.get("playlist_name") or "Chim-eken Playlist"
        me = self.sp.current_user()
        new_pl = self.sp.create_playlist(me['id'], name)

        session_data['last_created_playlist_url'] = new_pl['external_urls']['spotify']
        
        uris = []
        artist = parsed.get("artist")
        tracks_to_add = parsed.get("tracks_to_add", [])
        limit = int(parsed.get("limit", 0) or 10)

        # Priority 1: Search for specific tracks if provided
        if tracks_to_add:
            for track_name in tracks_to_add:
                res = self.sp.search(q=track_name, type="track", limit=1)
                if res['tracks']['items']:
                    uris.append(res['tracks']['items'][0]['uri'])

        # Priority 2: Search by artist
        elif artist:
            for a in [x.strip() for x in artist.split(",")]:
                res = self.sp.search(q=f"artist:{a}", type="track", limit=limit)
                uris.extend([t['uri'] for t in res['tracks']['items']])

        # Priority 3: Use last recommendations
        elif not uris:
            uris = session_data.get('last_recs', [])
        
        # ADD THESE LINES BACK:
        if uris:
            self.sp.add_tracks(new_pl['id'], uris)
            return f"Created playlist '{name}' with {len(uris)} tracks.", [new_pl['external_urls']['spotify']]
        
        return f"Created playlist '{name}'.", [new_pl['external_urls']['spotify']]

    def _handle_add_to_playlist(self, parsed, session_data):
        if not self.sp: return "You need to log in with Spotify to modify playlists.", []
        
        pl_url = parsed.get("playlist_url")
        
        # Check if user is referring to "the playlist you created" or "that playlist"
        if not pl_url and session_data.get('last_created_playlist_url'):
            pl_url = session_data['last_created_playlist_url']
        
        pid = parse_playlist_id(pl_url)
        
        # Extract specific tracks to add
        tracks_to_add = parsed.get("tracks_to_add", [])
        
        if not pid:
            # If user specified tracks, store them for later
            if tracks_to_add:
                session_data['pending_tracks'] = tracks_to_add
                return f"I'll add {', '.join(tracks_to_add)} for you. Please provide your playlist link (right-click on a playlist in Spotify → Share → Copy link to playlist).", []
            elif session_data.get('last_recs'):
                return "Sure! I have some songs ready to add. Please provide your playlist link (right-click on a playlist in Spotify → Share → Copy link to playlist).", []
            else:
                return "I'd be happy to add songs to your playlist! First, tell me what kind of songs you want, then share your playlist link.", []

        uris = []
        add_mode = parsed.get("add_mode")
        limit = int(parsed.get("limit", 0) or 10)
        artist = parsed.get("artist")

        if add_mode == "similar":
            tracks = self.sp.get_playlist_tracks(pid)
            if not tracks: return "Playlist is empty, cannot find similar songs.", []
            
            seed_artists = [t['artists'][0]['id'] for t in tracks[:5] if t.get('artists')]
            seed_tracks = [t['id'] for t in tracks[:5] if t.get('id')]
            
            try:
                recs = self.sp.get_recommendations(
                    seed_artists=seed_artists[:2] if seed_artists else None,
                    seed_tracks=seed_tracks[:3] if seed_tracks else None,
                    limit=limit
                )
                uris = [t['uri'] for t in recs['tracks']]
            except Exception:
                pass
            
            if uris:
                self.sp.add_tracks(pid, uris)
                return f"Added {len(uris)} similar tracks.", []
            return "Could not find similar tracks.", []

        else: # Default mode
            # Priority 1: Search for specific tracks if provided
            if tracks_to_add or session_data.get('pending_tracks'):
                track_list = tracks_to_add or session_data.get('pending_tracks', [])
                for track_name in track_list:
                    # Try exact search first
                    res = self.sp.search(q=track_name, type="track", limit=1)
                    
                    # If not found, try without special characters
                    if not res['tracks']['items']:
                        simplified = track_name.replace(" - ", " ").replace(".", "")
                        res = self.sp.search(q=simplified, type="track", limit=1)
                    
                    if res['tracks']['items']:
                        uris.append(res['tracks']['items'][0]['uri'])
                
                # Clear pending tracks
                if 'pending_tracks' in session_data:
                    del session_data['pending_tracks']
            
            # Priority 2: Search by artist
            elif artist:
                for a in [x.strip() for x in artist.split(",")]:
                    res = self.sp.search(q=f"artist:{a}", type="track", limit=limit)
                    uris.extend([t['uri'] for t in res['tracks']['items']])
            
            # Priority 3: Use last recommendations
            elif not uris:
                uris = session_data.get('last_recs', [])
            
            if uris:
                self.sp.add_tracks(pid, uris)
                return f"Added {len(uris)} track(s) to the playlist.", []
            return "No tracks found to add.", []

    def _handle_remove_from_playlist(self, parsed):
        if not self.sp: return "You need to log in with Spotify to modify playlists.", []
        
        pl_url = parsed.get("playlist_url")
        
        # Check for last created playlist
        if not pl_url and session_data.get('last_created_playlist_url'):
            pl_url = session_data['last_created_playlist_url']
        
        pid = parse_playlist_id(pl_url)
        if not pid: return "Invalid playlist link.", []
        
        tracks_to_remove = parsed.get("tracks_to_remove", [])
        artist = parsed.get("artist")
        
        # Get all tracks in playlist
        all_tracks = self.sp.get_playlist_tracks(pid)
        uris_to_remove = []
        
        # Priority 1: Remove specific tracks by name
        if tracks_to_remove:
            for track_name in tracks_to_remove:
                for track in all_tracks:
                    if track_name.lower() in track['name'].lower():
                        # If artist also specified, check artist match
                        if artist and artist.lower() not in track['artists'][0]['name'].lower():
                            continue
                        uris_to_remove.append(track['uri'])
        
        # Priority 2: Remove by artist only
        elif artist:
            for track in all_tracks:
                if artist.lower() in track['artists'][0]['name'].lower():
                    uris_to_remove.append(track['uri'])
        
        if not uris_to_remove:
            return "No matching tracks found to remove.", []
        
        self.sp.remove_tracks(pid, uris_to_remove)
        return f"Removed {len(uris_to_remove)} track(s) from the playlist.", []

    def _handle_read_playlist(self, parsed):
        if not self.sp: return "You need to log in with Spotify to read playlists.", []
        
        pl_url = parsed.get("playlist_url")
        playlist_name = parsed.get("target_playlist", "")
        
        pid = None
        pl_name = ""
        
        # Try URL first
        if pl_url:
            pid = parse_playlist_id(pl_url)
        
        # If no URL, search by name
        if not pid and playlist_name:
            playlists = self.sp.get_user_playlists()
            for pl in playlists:
                if playlist_name.lower() in pl['name'].lower():
                    pid = pl['id']
                    pl_name = pl['name']
                    break
         
        if not pid:
            return "Please provide a valid playlist link or name.", []
    
        tracks = self.sp.get_playlist_tracks(pid)
        items = [f"{t['name']} by {t['artists'][0]['name']}" for t in tracks]
        # Get playlist details to extract image

        return f"📋 Playlist: {pl_name} ({len(tracks)} tracks)", items

    def _handle_recommendations(self, parsed, session_data):
        # Support Local Mode
        if not self.local: return "Local library not loaded. Please ensure embeddings are generated.", []
        
        limit = int(parsed.get("limit", 0) or 10)
        seed_artist = parsed.get("seed_artist")
        seed_track = parsed.get("seed_track")
        search_query = parsed.get("search_query")
        target_genre = parsed.get("target_genre")

        # --- Smart Context Reset ---
        # Create a unique signature for this request
        current_signature = f"{seed_artist}|{seed_track}|{search_query}|{target_genre}"
        last_signature = session_data.get('last_signature')
        
        if current_signature != last_signature:
            # New topic! Clear the history so we start fresh
            session_data['shown_track_ids'] = set()
            session_data['last_signature'] = current_signature
            # Clear locked seed track
            if 'locked_seed_id' in session_data:
                del session_data['locked_seed_id']

        recs = []
        msg = ""

        # --- Local Mode Logic ---
        if search_query:
            res = self.local.search(search_query, limit=limit * 10)
            recs = res['tracks']['items']
            
            # If semantic search returns poor results for mood queries, use audio features
            if recs and 'similarity_score' in recs[0] and recs[0]['similarity_score'] < 0.45:
                # Check if this is a mood query
                mood_keywords = {
                    'energetic': {'energy': (0.7, 1.0), 'valence': (0.5, 1.0)},
                    'happy': {'valence': (0.7, 1.0), 'energy': (0.5, 1.0)},
                    'sad': {'valence': (0.0, 0.3), 'energy': (0.0, 0.4)},
                    'calm': {'energy': (0.0, 0.4), 'valence': (0.4, 0.8)},
                    'angry': {'energy': (0.7, 1.0), 'valence': (0.0, 0.4)},
                    'chill': {'energy': (0.0, 0.4)},
                    'upbeat': {'energy': (0.6, 1.0), 'valence': (0.6, 1.0)}
                }
                
                query_lower = search_query.lower()
                for mood, features in mood_keywords.items():
                    if mood in query_lower:
                        # Filter by audio features
                        filtered_df = self.local.df.copy()
                        for feature, (min_val, max_val) in features.items():
                            filtered_df = filtered_df[(filtered_df[feature] >= min_val) & (filtered_df[feature] <= max_val)]
                        
                        if not filtered_df.empty:
                            sample = filtered_df.sample(n=min(limit * 10, len(filtered_df))) # Get more for filtering
                            recs = self.local._to_spotify_format(sample)
                        break
            
            # --- NEW: Filter by Artist if provided ---
            if seed_artist:
                # Filter recs to only include tracks by the seed_artist
                # We need to be fuzzy because "Arctic Monkeys" might be "Arctic Monkeys" or "The Arctic Monkeys"
                filtered_recs = []
                for t in recs:
                    # Check all artists on the track
                    t_artists = [a['name'].lower() for a in t['artists']]
                    if any(seed_artist.lower() in art for art in t_artists):
                        filtered_recs.append(t)
                
                if filtered_recs:
                    recs = filtered_recs
                    msg = f"🎵 Here are some {search_query} by {seed_artist}:"
                else:
                    # Fallback: Filter artist's tracks by mood using audio features
                    # First, get all tracks by this artist
                    artist_tracks_res = self.local.search(seed_artist, type='track', limit=50)
                    artist_tracks = artist_tracks_res['tracks']['items']
                    
                    if artist_tracks:
                        # Map mood keywords to audio feature ranges
                        mood_features = {
                            'love': {'valence': (0.4, 0.8), 'energy': (0.3, 0.7)},  # Romantic, emotional
                            'romantic': {'valence': (0.4, 0.8), 'energy': (0.3, 0.7)},
                            'sad': {'valence': (0.0, 0.3), 'energy': (0.0, 0.4)},
                            'happy': {'valence': (0.7, 1.0), 'energy': (0.5, 1.0)},
                            'energetic': {'energy': (0.7, 1.0), 'valence': (0.5, 1.0)},
                            'dance': {'danceability': (0.6, 1.0), 'energy': (0.6, 1.0)},
                            'chill': {'energy': (0.0, 0.4)},
                            'calm': {'energy': (0.0, 0.4), 'valence': (0.4, 0.8)},
                        }
                        
                        # Extract mood from search_query
                        query_lower = search_query.lower()
                        matched_features = None
                        for mood_key, features in mood_features.items():
                            if mood_key in query_lower:
                                matched_features = features
                                break
                        
                        if matched_features:
                            # Filter artist's tracks by audio features
                            # Get track IDs to look up in dataframe
                            track_ids = [t['id'] for t in artist_tracks]
                            df = self.local.df
                            artist_df = df[df['track_id'].isin(track_ids)]
                            
                            # Apply feature filters
                            for feature, (min_val, max_val) in matched_features.items():
                                if feature in artist_df.columns:
                                    artist_df = artist_df[(artist_df[feature] >= min_val) & (artist_df[feature] <= max_val)]
                            
                            if not artist_df.empty:
                                # Convert back to Spotify format
                                sample = artist_df.sample(n=min(limit * 10, len(artist_df)))
                                recs = self.local._to_spotify_format(sample)
                                msg = f"🎵 Here are some {search_query} by {seed_artist}:"
                            else:
                                # No tracks match the mood criteria, return top tracks
                                recs = artist_tracks
                                msg = f"I couldn't find specific '{search_query}' matches for {seed_artist}, but here are their top tracks:"
                        else:
                            # No mood mapping found, return top tracks
                            recs = artist_tracks
                            msg = f"I couldn't find specific '{search_query}' matches for {seed_artist}, but here are their top tracks:"
                    else:
                        recs = []
                        msg = f"I don't have any songs by {seed_artist} in my library."
            else:
                msg = f"🎵 Here are some {search_query} tracks I found for you:"

        elif target_genre:
            # Map nationality queries to script detection
            script_map = {
                'russian': 'russian',
                'chinese': 'chinese',
                'mandarin': 'chinese',
                'c-pop': 'chinese',
                'japanese': 'japanese',
                'j-pop': 'japanese',
                'korean': 'korean',
                'k-pop': 'korean',
                'filipino': 'filipino',
                'tagalog': 'filipino',
                'opm': 'filipino',
                'pinoy': 'filipino'
            }
            
            target_script = script_map.get(target_genre.lower())
            
            if target_script:
                # HYBRID APPROACH: Combine script detection + known artists
                
                # Method 1: Script detection (finds songs with native characters)
                script_tracks = self.local.search_by_script(target_script, limit=limit * 5)
                
                # Method 2: Known artists (finds K-pop/J-pop/etc. with English titles)
                artist_tracks = []
                from local_library import KNOWN_ARTISTS
                
                # Map script to known artists
                script_to_genre = {
                    'korean': 'k-pop',
                    'japanese': 'j-pop',
                    'chinese': 'c-pop',  # You'll need to add this to KNOWN_ARTISTS
                    'russian': 'russian',  # You'll need to add this to KNOWN_ARTISTS
                    'filipino': 'opm',
                    'tagalog': 'opm'
                }
                
                genre_key = script_to_genre.get(target_script)
                if genre_key and genre_key in KNOWN_ARTISTS:
                    # Search for tracks by known artists
                    all_tracks = []
                    for artist in KNOWN_ARTISTS[genre_key][:20]:  # Check top 20 artists
                        artist_res = self.local.search(artist, type='track', limit=5)
                        if artist_res['tracks']['items']:
                            for track in artist_res['tracks']['items']:
                                track_artist = track['artists'][0]['name'].lower()
                                # Better matching: check if artist name matches the track artist
                                # Split by separators and check each part
                                artist_parts = [p.strip() for p in track_artist.replace(',', ';').split(';')]
                                if any(artist.lower() == part or artist.lower() in part.split() for part in artist_parts):
                                    all_tracks.append(track)
                    
                    # Remove duplicates
                    seen_ids = set()
                    for track in all_tracks:
                        if track['id'] not in seen_ids:
                            seen_ids.add(track['id'])
                            artist_tracks.append(track)
                
                # Combine both methods and remove duplicates
                combined_tracks = []
                seen_ids = set()
                
                # Add script-detected tracks first
                for track in script_tracks:
                    if track['id'] not in seen_ids:
                        seen_ids.add(track['id'])
                        combined_tracks.append(track)
                
                # Add artist-based tracks
                for track in artist_tracks:
                    if track['id'] not in seen_ids:
                        seen_ids.add(track['id'])
                        combined_tracks.append(track)
                
                # Shuffle and limit
                if combined_tracks:
                    random.shuffle(combined_tracks)
                    recs = combined_tracks[:limit * 10]  # Get more for filtering
                    msg = f"🎧 Here are some {target_genre} tracks I found:"
                else:
                    recs = []
                    genre_names = {
                        'russian': 'Russian',
                        'chinese': 'Chinese/Mandarin',
                        'japanese': 'Japanese',
                        'korean': 'Korean',
                        'filipino': 'Filipino',
                        'tagalog': 'Filipino',
                        'opm': 'Filipino',
                        'pinoy': 'Filipino'
                    }
                    msg = f"I don't have any {genre_names.get(target_script, target_genre)} music in my library. 😔"
            else:
                # Regular genre search (pop, rock, jazz, etc.)
                res = self.local.get_recommendations(seed_genres=[target_genre], limit=limit * 10)
                recs = res['tracks']
                
                if recs:
                    msg = f"🎧 Check out these {target_genre} recommendations:"

        elif seed_track:
            # Search track then get similar
            
            target_track = None
            
            # 1. Check if we have a locked seed ID from previous turn
            if 'locked_seed_id' in session_data:
                # Retrieve the specific track directly
                locked_id = session_data['locked_seed_id']
                track_row = self.local.df[self.local.df['track_id'] == locked_id]
                if not track_row.empty:
                    # Convert row to dict format expected by code below
                    target_track = self.local._to_spotify_format(track_row)[0]
            
            # 2. If no locked ID, search for it
            if not target_track:
                # Use text search to find the exact track first
                res = self.local.search(seed_track, type='track', limit=5)
                
                if res['tracks']['items']:
                    if seed_artist:
                        # Filter by artist if provided
                        for t in res['tracks']['items']:
                            if seed_artist.lower() in t['artists'][0]['name'].lower():
                                target_track = t
                                break
                    
                    # Fallback to first result if no artist match or no artist provided
                    if not target_track:
                        target_track = res['tracks']['items'][0]
                    
                    # Lock this track ID for future "give me more" requests
                    if target_track:
                        session_data['locked_seed_id'] = target_track['id']

            if target_track:
                tid = target_track['id']
                track_name = target_track['name']
                artist_name = target_track['artists'][0]['name']
                
                # Get recommendations using this track as seed
                r = self.local.get_recommendations(seed_tracks=[tid], limit=limit * 10)
                recs = r['tracks']
                msg = f"🎶 Based on '{track_name}' by {artist_name}, you might also like:"
            else:
                msg = f"Hmm, I don't have '{seed_track}' right now. 😔 Try asking for a different song or artist!"
        elif seed_artist:
            # Local search for artist
            res = self.local.search(seed_artist, type='artist', limit=1)
            if res['artists']['items']:
                # Just return top tracks for that artist from local DB
                res = self.local.search(seed_artist, type='track', limit=limit * 10)
                recs = res['tracks']['items']
                msg = f"🎤 Here are some tracks by {seed_artist}:"
            else:
                msg = f"I don't have any songs by {seed_artist} at the moment. 🎵 How about trying a different artist or genre?"
        else:
            # Exploratory query - user asking for general recommendations
            # Return a diverse sample from popular genres
            msg = "🎵 Here's a diverse selection you might enjoy! Try asking for specific genres like 'k-pop', 'rock', 'jazz', or moods like 'happy', 'sad', 'energetic':"
            
            # Get a mix from different genres
            popular_genres = ['pop', 'rock', 'hip-hop', 'electronic', 'indie', 'jazz']
            available_genres = [g for g in popular_genres if not self.local.df[self.local.df['playlist_genre'].str.lower() == g].empty]
            
            if available_genres:
                # Sample 2-3 tracks from each available genre
                tracks_per_genre = max(2, (limit * 10) // len(available_genres))
                for genre in available_genres[:3]:  # Limit to 3 genres for variety
                    genre_df = self.local.df[self.local.df['playlist_genre'].str.lower() == genre]
                    sample = genre_df.sample(n=min(tracks_per_genre, len(genre_df)))
                    recs.extend(self.local._to_spotify_format(sample))
                
                # Shuffle and limit
                random.shuffle(recs)
            else:
                # Fallback: random popular tracks
                sample = self.local.df.sample(n=min(limit * 10, len(self.local.df)))
                recs = self.local._to_spotify_format(sample)
                msg = "🎵 Here are some random tracks from the library:"
        
        # --- Global Filtering of Shown Tracks ---
        if recs:
            shown_ids = session_data.get('shown_track_ids', set())
            recs = [t for t in recs if t['id'] not in shown_ids]
            
            # Slice to limit AFTER filtering
            if len(recs) > limit:
                recs = recs[:limit]
            
            if 'shown_track_ids' not in session_data:
                session_data['shown_track_ids'] = set()
            session_data['shown_track_ids'].update([t['id'] for t in recs])

        if recs:
            session_data['last_recs'] = [t['uri'] for t in recs]
            # Format items with similarity scores if available
            items = []
            for t in recs:
                track_info = f"{t['name']} by {t['artists'][0]['name']}"
                if 'similarity_score' in t:
                    # Normalize score to 0-1 range (cap at 1.0 for display)
                    # Since we add text boost, scores can exceed 1.0
                    normalized_score = min(t['similarity_score'], 1.0)
                    similarity_pct = int(normalized_score * 100)
                    track_info += f" ({similarity_pct}% match)"
                items.append(track_info)
            return msg, items
        else:
            # No results found - provide helpful message
            if search_query or target_genre:
                query_term = search_query or target_genre
                
                # Special message for culture-specific genres
                if target_genre in ['opm', 'k-pop', 'j-pop']:
                    genre_names = {'opm': 'OPM/Filipino', 'k-pop': 'K-pop', 'j-pop': 'J-pop'}
                    return f"I don't have any {genre_names.get(target_genre, target_genre)} artists in my collection yet. 😔 Try asking for pop, rock, hip-hop, or jazz instead!", []
                
                return f"Sorry, I don't have any '{query_term}' songs right now. 😕 Want to try a different genre or mood? I've got pop, rock, k-pop, jazz, and more!", []
            return msg, []

    def _handle_artist_details(self, parsed):
        if not self.sp: return "Artist details only available in Spotify mode.", []
        # ... (rest same)
        artist = parsed.get("artist")
        if not artist: return "Please specify an artist.", []
        
        res = self.sp.search(q=f"artist:{artist}", type="artist", limit=1)
        if not res['artists']['items']: return f"Artist '{artist}' not found.", []
        
        art = res['artists']['items'][0]
        top = self.sp.get_artist_top_tracks(art['id'])
        top_names = [t['name'] for t in top['tracks'][:5]]
        
        details = (f"**{art['name']}**\n"
                   f"Followers: {art['followers']['total']:,}\n"
                   f"Popularity: {art['popularity']}/100\n"
                   f"Genres: {', '.join(art['genres'])}\n"
                   f"Top Tracks: {', '.join(top_names)}")
        
        imgs = [art['images'][0]['url']] if art['images'] else []
        return details, imgs
    
    def _handle_edit_playlist(self, parsed):
        if not self.sp: 
            return "You need to log in with Spotify to edit playlists.", []
    
        pl_url = parsed.get("playlist_url")
        new_name = parsed.get("new_playlist_name")
    
        if not pl_url or not new_name:
            return "Please provide the playlist link and the new name.", []
        
        pid = parse_playlist_id(pl_url)
        if not pid:
            return "Invalid playlist link.", []
        
        try:
            self.sp.sp.playlist_change_details(pid, name=new_name)
            return f"Playlist renamed to '{new_name}'.", []
        except Exception as e:
            return f"Failed to rename playlist: {e}", []

    def _handle_album_details(self, parsed):
        """Get album details and track list"""
        if not self.sp: 
            return "Album details only available in Spotify mode. Please log in to Spotify.", []
        
        album_name = parsed.get("album_name") or parsed.get("seed_track", "")
        artist = parsed.get("seed_artist", "")
        
        if not album_name:
            return "Please specify an album name.", []
        
        try:
            # Search for album
            search_query = f"album:{album_name}"
            if artist:
                search_query += f" artist:{artist}"
            
            res = self.sp.search(q=search_query, type="album", limit=1)
            
            if not res['albums']['items']:
                return f"Album '{album_name}' not found.", []
            
            album = res['albums']['items'][0]
            album_id = album['id']
            
            # Get album tracks
            tracks_res = self.sp.get_album_tracks(album_id)
            track_list = [f"{i+1}. {t['name']}" for i, t in enumerate(tracks_res['items'])]
            
            # Format album details
            release_date = album.get('release_date', 'Unknown')
            total_tracks = album.get('total_tracks', 0)
            artists = ', '.join([a['name'] for a in album['artists']])
            
            details = (f"**{album['name']}** by {artists}\n"
                    f"Released: {release_date}\n"
                    f"Total Tracks: {total_tracks}\n"
                    f"Popularity: {album.get('popularity', 'N/A')}/100\n\n"
                    f"Track List:\n" + "\n".join(track_list[:15]))  # Show first 15 tracks
            
            if total_tracks > 15:
                details += f"\n... and {total_tracks - 15} more tracks"
            
            # Return album cover image
            imgs = [album['images'][0]['url']] if album['images'] else []
            
            return details, imgs
            
        except Exception as e:
            print(f"Error fetching album details: {e}")
            return f"Failed to get details for '{album_name}'. Please try again.", []

    def _handle_show_playlists(self):
        """Show user's Spotify playlists"""
        if not self.sp:
            return "Please log in to Spotify first to view your playlists.", []
        
        try:
            playlists = self.sp.get_user_playlists()
            
            if not playlists:
                return "You don't have any playlists yet. Create one to get started!", []
        
            response = "Here are your playlists:"
            items = []
            for pl in playlists:
                items.append(f"{pl['name']} ({pl['tracks']} tracks)")
            
            return response, items 
        except Exception as e:
            return f"Failed to fetch playlists: {e}", []
    
    def _handle_analyze_playlist(self, playlist_name):
        """Analyze playlist and recommend similar songs"""
        if not self.sp:
            return "Please log in to Spotify first.", []
        
        if not self.local:
            return "Local library not loaded yet.", []
        
        try:
            # Get user's playlists
            playlists = self.sp.get_user_playlists()
            
            # Find matching playlist
            target_playlist = None
            for pl in playlists:
                if playlist_name.lower() in pl['name'].lower() or playlist_name == pl['id']:
                    target_playlist = pl
                    break
            
            if not target_playlist:
                return f"Playlist '{playlist_name}' not found. Use 'show my playlists' to see available playlists.", []
            
            # Get tracks from playlist
            tracks = self.sp.get_playlist_tracks(target_playlist['id'])
            
            if not tracks:
                return f"Playlist '{target_playlist['name']}' is empty.", []
            
            # Analyze tracks (get average audio features)
            track_names = [f"{t['name']} by {t['artists'][0]['name']}" for t in tracks[:5]]
            
            # Use first track as seed for recommendations
            seed_track = f"{tracks[0]['name']} {tracks[0]['artists'][0]['name']}"
            
            # Get recommendations
            search_results = self.local.search(seed_track, type='track', limit=10)
            results = search_results.get('tracks', {}).get('items', [])

            if not results:
                return f"Couldn't find similar songs to your playlist.", []

            response = f"Based on your playlist '{target_playlist['name']}' (which includes {', '.join(track_names[:3])}...), here are some recommendations:"

            items = []
            for track in results:
                items.append(f"{track['name']} by {track['artists'][0]['name']}")
            
            return response, items
            
        except Exception as e:
            return f"Failed to analyze playlist: {e}", []

    def _handle_track_details(self, parsed):
        seed_track = parsed.get('seed_track', '')
        artist = parsed.get('artist', '')
        
        # 1. Try Local Library First
        if self.local and self.local.is_ready:
            # Search locally
            res = self.local.search(seed_track, type='track', limit=5)
            if res['tracks']['items']:
                target = None
                # Filter by artist if provided
                if artist:
                    for t in res['tracks']['items']:
                        if artist.lower() in t['artists'][0]['name'].lower():
                            target = t
                            break
                else:
                    target = res['tracks']['items'][0]
                
                if target:
                    # Get audio features from local DF
                    df = self.local.df
                    mask = df['track_name'].str.lower() == target['name'].lower()
                    if artist:
                        mask &= df['track_artist'].str.lower().str.contains(artist.lower())
                    
                    matches = df[mask]
                    if not matches.empty:
                        row = matches.iloc[0]
                        
                        # Use raw values if available for better display
                        dance = row.get('raw_danceability', row.get('danceability', 0))
                        energy = row.get('raw_energy', row.get('energy', 0))
                        tempo = row.get('raw_tempo', row.get('tempo', 0))
                        
                        # Format nicely
                        dance_str = f"{dance:.2f}"
                        energy_str = f"{energy:.2f}"
                        tempo_str = f"{int(tempo)}" if tempo > 10 else f"{tempo:.2f}" # Tempo is usually > 50 BPM

                        info = (f"**{row['track_name']}** by {row['track_artist']}\n"
                                f"Danceability: {dance_str}\n"
                                f"Energy: {energy_str}\n"
                                f"Tempo: {tempo_str} BPM\n"
                                f"(Source: Local Library)")
                        return info, []

        # 2. Fallback to Spotify
        if not self.sp: return "Track not found in local library. Log in to Spotify to search there.", []
        
        query = f"track:{seed_track} artist:{artist}".strip()
        if not query: return "Please specify a track.", []
        
        try:
            res = self.sp.search(q=query, type="track", limit=1)
            if not res['tracks']['items']: return "Track not found.", []
            
            t = res['tracks']['items'][0]
            feats = self.sp.get_audio_features([t['id']])
            if feats and feats[0]:
                f = feats[0]
                info = (f"**{t['name']}** by {t['artists'][0]['name']}\n"
                        f"Danceability: {f['danceability']}\n"
                        f"Energy: {f['energy']}\n"
                        f"Tempo: {f['tempo']} BPM\n"
                        f"(Source: Spotify)")
                return info, []
            return f"Found {t['name']}, but no features available.", []
        except Exception as e:
            print(f"Spotify API Error: {e}")
            return f"Found {seed_track}, but couldn't get details from Spotify right now. (API Error)", []

    
