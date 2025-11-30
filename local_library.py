"""
local_library.py

Handles music recommendations using local CSV datasets when Spotify is unavailable.
Enhanced with vector embeddings for semantic similarity search.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import os

# Known artists for culture-specific genres (for filtering misclassified tracks)
KNOWN_ARTISTS = {
    'k-pop': [
        'bts', 'blackpink', 'twice', 'newjeans', 'stray kids', 'ive', 'le sserafim', 
        'nct', 'seventeen', 'txt', 'tomorrow x together', 'enhypen', 'itzy', 'aespa', 
        'red velvet', 'exo', 'got7', 'ateez', 'the boyz', 'monsta x', 'super junior',
        'girls generation', 'snsd', 'bigbang', 'shinee', 'mamamoo', 'everglow',
        'loona', 'dreamcatcher', '(g)i-dle', 'gidle', 'kep1er', 'nmixx', 'ikon',
        'winner', 'day6', 'stayc', 'viviz', 'weeekly', 'fromis_9', 'oh my girl',
        'apink', 'cosmic girls', 'wjsn', 'cravity', 'treasure', 'astro', 'pentagon',
        'sf9', 'the rose', 'n.flying', 'onewe', 'oneus', 'ab6ix', 'cix', 'drippin'
    ],
    'j-pop': [
        'yoasobi', 'ado', 'kenshi yonezu', 'official hige dandism', 'king gnu', 
        'back number', 'radwimps', 'mrs. green apple', 'yorushika', 'eve',
        'amazarashi', 'bump of chicken', 'one ok rock', 'my first story',
        'lisa', 'reol', 'zutomayo', 'aimyon', 'fujii kaze', 'higedan',
        'perfume', 'babymetal', 'scandal', 'band-maid', 'silent siren',
        'arashi', 'exile', 'generations', 'sandaime j soul brothers'
    ],
    'opm': [
        'ben&ben', 'moira dela torre', 'zack tabudlo', 'sb19', 'sarah geronimo',
        'regine velasquez', 'bamboo', 'rivermaya', 'eraserheads', 'parokya ni edgar',
        'sponge cola', 'silent sanctuary', 'callalily', 'orange and lemons',
        'moonstar88', 'kamikazee', 'hale', 'sandwich', 'up dharma down',
        'ben&ben', 'iv of spades', 'unique salonga', 'juan karlos', 'this band',
        'the juans', 'arthur nery', 'mrld', 'adie', 'dionela'
    ]
}


class LocalMusicLibrary:
    def __init__(self):
        self.df = None
        self.features = ['energy', 'danceability', 'valence', 'tempo', 'acousticness', 'instrumentalness', 'speechiness', 'liveness']
        self.scaler = MinMaxScaler()
        self.model = NearestNeighbors(n_neighbors=20, algorithm='brute', metric='cosine')
        self.is_ready = False
        
        # Embedding-related attributes
        self.embeddings = None
        self.embedding_model = None
        self.embeddings_ready = False
        
        self._load_data()
        self._load_embeddings()

    def _load_data(self):
        try:
            # Try loading from datasets folder first
            files = [
                'datasets/high_popularity_spotify_data.csv',
                'datasets/low_popularity_spotify_data.csv',
                'datasets/spotify_dataset.csv'
            ]
            
            # Fallback to root folder if datasets folder doesn't exist
            if not any(os.path.exists(f) for f in files):
                files = [
                    'high_popularity_spotify_data.csv',
                    'low_popularity_spotify_data.csv',
                    'spotify_dataset.csv'
                ]
            
            dfs = []
            for f in files:
                if os.path.exists(f):
                    print(f"Loading {f}...")
                    dfs.append(pd.read_csv(f))
            
            if not dfs:
                print("No CSV datasets found.")
                return

            self.df = pd.concat(dfs, ignore_index=True)
            
            # Normalize column names by coalescing
            # Handle track_genre -> playlist_genre mapping
            if 'track_genre' in self.df.columns and 'playlist_genre' in self.df.columns:
                # Both exist: merge them (prefer playlist_genre, fill with track_genre)
                self.df['playlist_genre'] = self.df['playlist_genre'].fillna(self.df['track_genre'])
                self.df.drop(columns=['track_genre'], inplace=True)
            elif 'track_genre' in self.df.columns:
                # Only track_genre exists: rename it
                self.df.rename(columns={'track_genre': 'playlist_genre'}, inplace=True)
            
            # Handle other mappings
            col_mappings = {
                'track_artist': 'artists',
                'track_album_name': 'album_name'
            }
            
            for target, source in col_mappings.items():
                if source in self.df.columns:
                    if target in self.df.columns:
                        self.df[target] = self.df[target].fillna(self.df[source])
                        self.df.drop(columns=[source], inplace=True)
                    else:
                        self.df.rename(columns={source: target}, inplace=True)
            
            # Fill text columns with empty string
            text_columns = ['track_name', 'track_artist', 'track_album_name', 'playlist_genre', 'playlist_name']
            for col in text_columns:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna('')
            
            # Filter out tracks with empty name or artist
            mask = (self.df['track_name'].str.strip() != '') & (self.df['track_artist'].str.strip() != '')
            self.df = self.df[mask]
            
            # Normalize artist separators for better deduplication
            # Fixes "Drake, 21 Savage" vs "Drake;21 Savage"
            self.df['track_artist'] = self.df['track_artist'].str.replace(';', ', ')
            
            # Drop duplicates by Name + Artist (keep first)
            self.df.drop_duplicates(subset=['track_name', 'track_artist'], inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            
            # Fill numeric columns with 0
            for col in self.features:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna(0)
            
            # Save raw features before normalization for display
            for col in self.features:
                self.df[f'raw_{col}'] = self.df[col]

            # Normalize features
            self.df[self.features] = self.scaler.fit_transform(self.df[self.features])
            
            # Fit model
            self.model.fit(self.df[self.features])
            self.is_ready = True
            print(f"Local library loaded with {len(self.df)} tracks.")
            
        except Exception as e:
            print(f"Error loading local library: {e}")

    def _load_embeddings(self):
        """Load pre-computed embeddings if available"""
        try:
            embedding_file = 'datasets/embeddings.pkl'
            if not os.path.exists(embedding_file):
                print("Embeddings not found. Run 'python generate_embeddings.py' to create them.")
                print("Using basic audio feature search for now.")
                return
            
            print("Loading embeddings...")
            with open(embedding_file, 'rb') as f:
                data = pickle.load(f)
            
            self.embeddings = data['embeddings']
            self.embedding_track_ids = data['track_ids']
            
            # Load the sentence transformer model for query encoding
            print("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.embeddings_ready = True
            print(f"✅ Embeddings loaded! Enhanced semantic search enabled.")
            
        except Exception as e:
            print(f"Could not load embeddings: {e}")
            print("Falling back to audio feature search.")

    def search(self, query, type='track', limit=10):
        if not self.is_ready: return {'tracks': {'items': []}}
        
        query = query.lower()
        results = []
        
        if type == 'track':
            # Check if query is an artist name (exact match in our database)
            is_artist_query = self.df['track_artist'].str.lower().str.contains(query, na=False, regex=False).any()
            
            # Use semantic search for general queries, but text matching for artist names
            if self.embeddings_ready and not is_artist_query:
                results = self._semantic_search(query, limit)
                if results:
                    return {'tracks': {'items': results}}
            
            # Fallback to text matching (or use directly for artist queries)
            mask = self.df['track_name'].str.lower().str.contains(query, na=False) | \
                   self.df['track_artist'].str.lower().str.contains(query, na=False)
            matches = self.df[mask].head(limit)
            results = self._to_spotify_format(matches)
            
        elif type == 'artist':
            mask = self.df['track_artist'].str.lower().str.contains(query, na=False)
            matches = self.df[mask].drop_duplicates(subset=['track_artist']).head(limit)
            results = [{'id': 'local', 'name': row['track_artist'], 'genres': [row['playlist_genre']], 'followers': {'total': 0}, 'popularity': 50, 'images': []} for _, row in matches.iterrows()]
            return {'artists': {'items': results}}

        return {'tracks': {'items': results}}

    def _semantic_search(self, query, limit=10):
        """Perform semantic search using embeddings"""
        try:
            # Encode the query
            query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
            
            # Normalize
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            
            # Create hybrid query embedding (70% text, 30% zeros for audio)
            # This matches the weighting used in generate_embeddings.py
            text_weight = 0.7
            audio_weight = 0.3
            
            query_hybrid = np.concatenate([
                query_norm * text_weight,
                np.zeros(8) * audio_weight  # No audio features for text query
            ])
            
            # Calculate cosine similarity
            similarities = cosine_similarity([query_hybrid], self.embeddings)[0]
            
            # Get larger pool of top matches (3x limit) to allow for variety
            pool_size = limit * 3
            top_indices = np.argsort(similarities)[::-1][:pool_size]
            
            # Randomly sample from the top pool, but weight towards top results
            # We'll just shuffle the top pool for now to give "different" results
            np.random.shuffle(top_indices)
            top_indices = top_indices[:limit]
            
            # Get corresponding tracks and their similarity scores
            top_track_ids = [self.embedding_track_ids[i] for i in top_indices]
            top_similarities = [similarities[i] for i in top_indices]
            
            # Filter out low-quality matches (below 40% similarity)
            MIN_SIMILARITY = 0.40
            filtered_ids = []
            filtered_sims = []
            for tid, sim in zip(top_track_ids, top_similarities):
                if sim >= MIN_SIMILARITY:
                    filtered_ids.append(tid)
                    filtered_sims.append(sim)
            
            if not filtered_ids:
                return []  # No good matches found
            
            matches = self.df[self.df['track_id'].isin(filtered_ids)].copy()
            
            # Sort by similarity score (preserve order of top_track_ids)
            # Create a mapping of track_id to its position in top_track_ids
            id_to_order = {track_id: idx for idx, track_id in enumerate(top_track_ids)}
            matches['_sort_order'] = matches['track_id'].map(id_to_order)
            matches = matches.sort_values('_sort_order').drop('_sort_order', axis=1)
            
            return self._to_spotify_format(matches, similarities=top_similarities)
            
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    def get_recommendations(self, seed_artists=None, seed_tracks=None, seed_genres=None, limit=10):
        if not self.is_ready: return {'tracks': []}
        
        # 1. Prioritize Exact Genre Match
        # If the user asks for a specific genre that exists in our DB, use it directly!
        if seed_genres:
            genre = seed_genres[0].lower()
            # Check for exact match in playlist_genre
            genre_df = self.df[self.df['playlist_genre'].str.lower() == genre]
            
            # If not found, try replacing spaces with hyphens or vice versa (common mismatch)
            if genre_df.empty:
                alt_genre = genre.replace(' ', '-')
                genre_df = self.df[self.df['playlist_genre'].str.lower() == alt_genre]
            
            if not genre_df.empty:
                # For culture-specific genres, sample MORE initially to account for filtering
                # This ensures we get enough tracks after whitelist filtering
                sample_size = limit
                if genre in KNOWN_ARTISTS:
                    # Sample 3x more to ensure we have enough after filtering
                    sample_size = min(limit * 3, len(genre_df))
                
                sample = genre_df.sample(n=min(sample_size, len(genre_df)))
                tracks = self._to_spotify_format(sample)
                
                # Apply artist whitelist filtering for culture-specific genres
                tracks = self._filter_by_known_artists(tracks, genre)
                
                # Trim to requested limit after filtering
                tracks = tracks[:limit]
                
                return {'tracks': tracks}

        # 2. Try embedding-based recommendations (for Tracks)
        if self.embeddings_ready and seed_tracks:
            results = self._embedding_based_recommendations(seed_tracks, limit)
            if results:
                return {'tracks': results}
        
        # 3. Semantic Search for Genre (Fallback)
        # If exact genre wasn't found, try semantic search (e.g. "sad songs" might not be a genre but is a concept)
        if seed_genres and self.embeddings_ready:
            genre_query = seed_genres[0]
            results = self._semantic_search(genre_query, limit)
            if results:
                return {'tracks': results}
        
        # 4. Fallback to Audio Features (if no embeddings)
        target_features = None
        
        if seed_tracks:
            seed_row = self.df[self.df['track_id'].isin(seed_tracks)]
            if not seed_row.empty:
                target_features = seed_row[self.features].mean().values.reshape(1, -1)
        
        if target_features is not None:
            distances, indices = self.model.kneighbors(target_features, n_neighbors=limit+1)
            neighbor_indices = indices[0]
            matches = self.df.iloc[neighbor_indices]
            return {'tracks': self._to_spotify_format(matches)}

        return {'tracks': []}

    def _embedding_based_recommendations(self, seed_track_ids, limit=10):
        """Get recommendations using embeddings"""
        try:
            # Find seed tracks in embeddings
            seed_indices = []
            for track_id in seed_track_ids:
                if track_id in self.embedding_track_ids:
                    idx = self.embedding_track_ids.index(track_id)
                    seed_indices.append(idx)
            
            if not seed_indices:
                return []
            
            # Average seed embeddings
            seed_embedding = np.mean(self.embeddings[seed_indices], axis=0).reshape(1, -1)
            
            # Find similar tracks
            similarities = cosine_similarity(seed_embedding, self.embeddings)[0]
            
            # Get top matches (excluding seeds)
            top_indices = np.argsort(similarities)[::-1]
            
            # Filter out seed tracks
            filtered_indices = [i for i in top_indices if i not in seed_indices][:limit]
            
            # Get tracks and their similarity scores
            top_track_ids = [self.embedding_track_ids[i] for i in filtered_indices]
            top_similarities = [similarities[i] for i in filtered_indices]
            
            matches = self.df[self.df['track_id'].isin(top_track_ids)].copy()
            
            # Sort by similarity (preserve order of top_track_ids)
            id_to_order = {track_id: idx for idx, track_id in enumerate(top_track_ids)}
            matches['_sort_order'] = matches['track_id'].map(id_to_order)
            matches = matches.sort_values('_sort_order').drop('_sort_order', axis=1)
            
            return self._to_spotify_format(matches, similarities=top_similarities)
            
        except Exception as e:
            print(f"Embedding-based recommendation error: {e}")
            return []

    def _filter_by_known_artists(self, tracks, genre):
        """Filter tracks by known artists for culture-specific genres"""
        if genre not in KNOWN_ARTISTS:
            return tracks  # No filtering needed for this genre
        
        known = KNOWN_ARTISTS[genre]
        filtered = []
        
        for track in tracks:
            artist_lower = track['artists'][0]['name'].lower()
            # Check if any known artist name is in the track's artist
            if any(known_artist in artist_lower for known_artist in known):
                filtered.append(track)
        
        # If filtering removed everything, return original (better than nothing)
        # But print a warning
        if not filtered:
            print(f"Warning: No known {genre} artists found in results. Dataset may have incorrect labels.")
            return tracks
        
        return filtered

    def _to_spotify_format(self, df, similarities=None):
        tracks = []
        for i, (idx, row) in enumerate(df.iterrows()):
            track = {
                'id': row['track_id'],
                'name': row['track_name'],
                'uri': row['track_id'],
                'artists': [{'name': row['track_artist'], 'id': 'local'}],
                'album': {'name': row['track_album_name'], 'images': []},
                'external_urls': {'spotify': ''}
            }
            
            # Add similarity score if available
            if similarities is not None and i < len(similarities):
                track['similarity_score'] = float(similarities[i])
            
            tracks.append(track)
        return tracks

    def get_track_by_id(self, track_id):
        if not self.is_ready: return None
        row = self.df[self.df['track_id'] == track_id]
        if row.empty: return None
        return self._to_spotify_format(row)[0]
