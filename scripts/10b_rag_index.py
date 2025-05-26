import json
import os
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import pickle
from sentence_transformers import SentenceTransformer
import re
from collections import defaultdict
import time

@dataclass
class RetrievalResult:
    chunk_id: str
    chunk_type: str
    play_title: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    rank: int

class ShakespeareRetriever:
    def __init__(self, 
                 chunks_dir: str = "retrieval/chunks",
                 index_dir: str = "retrieval/index",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Shakespeare retrieval system.
        
        Args:
            chunks_dir: Directory containing processed chunks
            index_dir: Directory to store FAISS index and embeddings
            model_name: SentenceTransformer model for embeddings
        """
        self.chunks_dir = Path(chunks_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Storage for chunks and embeddings
        self.chunks = []
        self.chunk_embeddings = None
        self.faiss_index = None
        self.chunk_id_to_idx = {}
        
        # Load glossary for query enhancement
        self.glossary = self.load_glossary()
        
        # Query processing patterns
        self.query_patterns = self.compile_query_patterns()
        
    def load_glossary(self) -> Dict[str, str]:
        """Load Shakespeare glossary for query enhancement."""
        try:
            glossary_path = Path("data/glossary/glossary.json")
            with open(glossary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Glossary not found - query enhancement disabled")
            return {}
    
    def compile_query_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for query classification."""
        return {
            'quote_query': re.compile(r'\b(quote|famous|said|says|line)\b', re.IGNORECASE),
            'character_query': re.compile(r'\b(character|who is|relationship|family)\b', re.IGNORECASE),
            'plot_query': re.compile(r'\b(what happens|plot|story|summary|scene)\b', re.IGNORECASE),
            'theme_query': re.compile(r'\b(theme|meaning|about|represents|symbolize)\b', re.IGNORECASE),
            'factual_query': re.compile(r'\b(when|where|year|setting|category|written)\b', re.IGNORECASE),
            'play_specific': re.compile(r'\b(hamlet|macbeth|romeo|othello|lear|julius|caesar|tempest)\b', re.IGNORECASE)
        }
    
    def load_chunks(self) -> List[Dict]:
        """Load processed chunks from JSON file."""
        chunks_file = self.chunks_dir / "all_chunks.json"
        
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        print(f"Loaded {len(chunks_data)} chunks")
        return chunks_data
    
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Create embeddings for a list of texts."""
        print(f"Creating embeddings for {len(texts)} texts...")
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=True)
            embeddings.extend(batch_embeddings)
            
            if i % (batch_size * 10) == 0:
                print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} embeddings")
        
        return np.array(embeddings, dtype=np.float32)
    
    def build_index(self, force_rebuild: bool = False):
        """Build or load FAISS index from chunks."""
        embeddings_file = self.index_dir / "embeddings.npy"
        index_file = self.index_dir / "faiss_index.bin"
        chunks_file = self.index_dir / "chunks_indexed.pkl"
        mapping_file = self.index_dir / "chunk_mapping.pkl"
        
        # Check if index already exists
        if not force_rebuild and all(f.exists() for f in [embeddings_file, index_file, chunks_file, mapping_file]):
            print("Loading existing FAISS index...")
            self.load_index()
            return
        
        print("Building new FAISS index...")
        
        # Load chunks
        chunks_data = self.load_chunks()
        self.chunks = chunks_data
        
        # Prepare texts for embedding
        chunk_texts = []
        for chunk in chunks_data:
            # Combine content with key metadata for better retrieval
            text = chunk['content']
            if chunk['metadata']:
                # Add important metadata to the embedding text
                if 'speakers' in chunk['metadata']:
                    text += f" Characters: {', '.join(chunk['metadata']['speakers'])}"
                if 'theme_name' in chunk['metadata']:
                    text += f" Theme: {chunk['metadata']['theme_name']}"
                if 'quote' in chunk['metadata']:
                    text += f" Quote: {chunk['metadata']['quote']}"
            
            chunk_texts.append(text)
        
        # Create embeddings
        self.chunk_embeddings = self.create_embeddings(chunk_texts)
        
        # Build FAISS index
        print("Building FAISS index...")
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.chunk_embeddings)
        self.faiss_index.add(self.chunk_embeddings)
        
        # Create chunk ID to index mapping
        self.chunk_id_to_idx = {chunk['chunk_id']: idx for idx, chunk in enumerate(chunks_data)}
        
        # Save everything
        self.save_index()
        
        print(f"FAISS index built with {len(chunks_data)} chunks")
    
    def save_index(self):
        """Save FAISS index and related data."""
        np.save(self.index_dir / "embeddings.npy", self.chunk_embeddings)
        faiss.write_index(self.faiss_index, str(self.index_dir / "faiss_index.bin"))
        
        with open(self.index_dir / "chunks_indexed.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(self.index_dir / "chunk_mapping.pkl", 'wb') as f:
            pickle.dump(self.chunk_id_to_idx, f)
        
        print("Index saved successfully")
    
    def load_index(self):
        """Load existing FAISS index and related data."""
        self.chunk_embeddings = np.load(self.index_dir / "embeddings.npy")
        self.faiss_index = faiss.read_index(str(self.index_dir / "faiss_index.bin"))
        
        with open(self.index_dir / "chunks_indexed.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        
        with open(self.index_dir / "chunk_mapping.pkl", 'rb') as f:
            self.chunk_id_to_idx = pickle.load(f)
        
        print(f"Index loaded successfully with {len(self.chunks)} chunks")
    
    def enhance_query(self, query: str) -> str:
        """Enhance query by expanding archaic words using glossary."""
        enhanced_query = query
        
        # Replace archaic words with modern equivalents
        words = query.lower().split()
        enhanced_words = []
        
        for word in words:
            # Remove punctuation for lookup
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.glossary:
                enhanced_words.append(f"{word} {self.glossary[clean_word]}")
            else:
                enhanced_words.append(word)
        
        if enhanced_words != words:
            enhanced_query = " ".join(enhanced_words)
            print(f"Enhanced query: {enhanced_query}")
        
        return enhanced_query
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """Classify query type and extract relevant information."""
        query_info = {
            'original_query': query,
            'query_types': [],
            'plays_mentioned': [],
            'confidence_boosts': {}
        }
        
        # Check query patterns
        for pattern_name, pattern in self.query_patterns.items():
            if pattern.search(query):
                query_info['query_types'].append(pattern_name)
        
        # Extract mentioned plays
        play_patterns = {
            'hamlet': r'\b(hamlet)\b',
            'macbeth': r'\b(macbeth)\b',
            'romeo_and_juliet': r'\b(romeo|juliet)\b',
            'othello': r'\b(othello)\b',
            'king_lear': r'\b(lear|king lear)\b',
            'julius_caesar': r'\b(julius|caesar)\b',
            'tempest': r'\b(tempest)\b'
        }
        
        for play, pattern in play_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                query_info['plays_mentioned'].append(play)
        
        # Set confidence boosts based on query types
        if 'quote_query' in query_info['query_types']:
            query_info['confidence_boosts']['quote'] = 0.3
        if 'character_query' in query_info['query_types']:
            query_info['confidence_boosts']['character_relationship'] = 0.2
            query_info['confidence_boosts']['factual'] = 0.1
        if 'plot_query' in query_info['query_types']:
            query_info['confidence_boosts']['summary_scene'] = 0.2
            query_info['confidence_boosts']['scene'] = 0.1
        if 'theme_query' in query_info['query_types']:
            query_info['confidence_boosts']['theme'] = 0.3
        if 'factual_query' in query_info['query_types']:
            query_info['confidence_boosts']['factual'] = 0.2
        
        return query_info
    
    def get_quote_popularity_boost(self, quote: str, speaker: str) -> float:
        """Apply popularity boost to famous quotes."""
        # Define famous quotes with popularity scores
        famous_quotes = {
            "to be, or not to be": 0.5,  # Most famous
            "something is rotten in the state": 0.4,
            "alas, poor yorick": 0.4,
            "get thee to a nunnery": 0.3,
            "the lady doth protest too much": 0.3,
            "brevity is the soul of wit": 0.2,
            "frailty, thy name is woman": 0.2,
            "to thine own self be true": 0.2,
            "there are more things in heaven and earth": 0.2,
            "what a piece of work is a man": 0.1,
        }
        
        quote_lower = quote.lower()
        for famous_quote, boost in famous_quotes.items():
            if famous_quote in quote_lower:
                return boost
        
        # Additional boost for Hamlet's quotes (main character)
        if speaker.lower() == "hamlet":
            return 0.05
            
        return 0.0

    def search(self, query: str, top_k: int = 10, min_score: float = 0.3) -> List[RetrievalResult]:
        """
        Search for relevant chunks based on query.
        
        Args:
            query: User query string
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of RetrievalResult objects
        """
        if self.faiss_index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Enhance and classify query
        enhanced_query = self.enhance_query(query)
        query_info = self.classify_query(query)
        
        print(f"Query types detected: {query_info['query_types']}")
        if query_info['plays_mentioned']:
            print(f"Plays mentioned: {query_info['plays_mentioned']}")
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([enhanced_query])
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding, min(top_k * 3, len(self.chunks)))
        
        # Process results
        results = []
        seen_chunks = set()
        
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if score < min_score:
                continue
            
            chunk = self.chunks[idx]
            chunk_id = chunk['chunk_id']
            
            # Avoid duplicates
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            
            # Apply query-type specific boosts
            adjusted_score = float(score)
            chunk_type = chunk['chunk_type']
            
            if chunk_type in query_info['confidence_boosts']:
                adjusted_score += query_info['confidence_boosts'][chunk_type]
            
            # Boost if play is specifically mentioned
            if query_info['plays_mentioned']:
                play_title_lower = chunk['play_title'].lower()
                for mentioned_play in query_info['plays_mentioned']:
                    if mentioned_play.replace('_', ' ') in play_title_lower:
                        adjusted_score += 0.2
                        break
            
            # Apply quote popularity boost for quote chunks
            if chunk_type == 'quote' and 'quote' in chunk['metadata']:
                quote_text = chunk['metadata']['quote']
                speaker = chunk['metadata'].get('speaker', '')
                popularity_boost = self.get_quote_popularity_boost(quote_text, speaker)
                adjusted_score += popularity_boost
                
                # Special handling for "famous quotes" queries
                if any(word in query.lower() for word in ['famous', 'popular', 'well-known', 'best']):
                    adjusted_score += popularity_boost * 0.5  # Extra boost for fame queries
            
            result = RetrievalResult(
                chunk_id=chunk_id,
                chunk_type=chunk_type,
                play_title=chunk['play_title'],
                content=chunk['content'],
                metadata=chunk['metadata'],
                similarity_score=adjusted_score,
                rank=rank + 1
            )
            
            results.append(result)
        
        # Re-sort by adjusted score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks and limit results
        for i, result in enumerate(results[:top_k]):
            result.rank = i + 1
        
        return results[:top_k]
    
    def format_results(self, results: List[RetrievalResult], max_content_length: int = 500) -> str:
        """Format search results for display."""
        if not results:
            return "No relevant results found."
        
        formatted = f"Found {len(results)} relevant results:\n\n"
        
        for result in results:
            formatted += f"{'='*60}\n"
            formatted += f"Rank: {result.rank} | Score: {result.similarity_score:.3f}\n"
            formatted += f"Type: {result.chunk_type} | Play: {result.play_title}\n"
            formatted += f"{'='*60}\n"
            
            # Truncate content if too long
            content = result.content
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            formatted += f"{content}\n"
            
            # Add relevant metadata
            if result.metadata:
                metadata_str = []
                for key, value in result.metadata.items():
                    if key in ['act', 'scene', 'speaker', 'theme_name', 'character_1', 'character_2']:
                        metadata_str.append(f"{key}: {value}")
                
                if metadata_str:
                    formatted += f"\nMetadata: {', '.join(metadata_str)}\n"
            
            formatted += "\n"
        
        return formatted
    
    def interactive_search(self):
        """Interactive search interface."""
        print("\n" + "="*70)
        print("SHAKESPEARE CHATBOT - Interactive Search")
        print("="*70)
        print("Ask me about Shakespeare's plays, characters, quotes, themes, or plot!")
        print("Examples:")
        print("- 'What are famous quotes from Hamlet?'")
        print("- 'Tell me about the relationship between Hamlet and Claudius'")
        print("- 'What happens in Macbeth Act 3 Scene 2?'")
        print("- 'What are the main themes in Romeo and Juliet?'")
        print("\nType 'quit' to exit\n")
        
        while True:
            try:
                query = input("🎭 Ask Shakespeare: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Farewell! 👋")
                    break
                
                if not query:
                    continue
                
                print(f"\n🔍 Searching for: '{query}'")
                start_time = time.time()
                
                results = self.search(query)
                search_time = time.time() - start_time
                
                print(f"⏱️  Search completed in {search_time:.2f} seconds\n")
                print(self.format_results(results))
                
            except KeyboardInterrupt:
                print("\n\nFarewell! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

def main():
    """Main function to demonstrate the retrieval system."""
    # Initialize retriever
    retriever = ShakespeareRetriever()
    
    try:
        # Build or load index
        retriever.build_index()
        
        # Test with some example queries
        test_queries = [
            "What are famous quotes from Hamlet?",
            "Tell me about Macbeth's themes",
            "What happens in Romeo and Juliet?",
            "Who is Othello's relationship with Iago?"
        ]
        
        print("\n" + "="*70)
        print("TESTING RETRIEVAL SYSTEM")
        print("="*70)
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            results = retriever.search(query, top_k=3)
            print(f"Found {len(results)} results")
            
            if results:
                print(f"Top result: {results[0].chunk_type} from {results[0].play_title}")
                print(f"Score: {results[0].similarity_score:.3f}")
        
        # Start interactive mode
        retriever.interactive_search()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure you have run the chunking process first.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()