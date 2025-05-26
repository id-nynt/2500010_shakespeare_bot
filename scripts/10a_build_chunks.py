import json
import os
from pathlib import Path
from typing import Dict, List, Any
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class Chunk:
    chunk_id: str
    chunk_type: str  # 'scene', 'character_relationship', 'theme', 'quote', 'factual'
    play_title: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float] = None  # Will be populated later with embeddings

class ShakespeareChunker:
    def __init__(self, data_dir: str = "data", output_dir: str = "retrieval/chunks"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load glossary for reference
        self.glossary = self.load_glossary()
        
        self.chunks = []
        
    def load_glossary(self) -> Dict[str, str]:
        """Load the Shakespeare glossary for reference."""
        glossary_path = self.data_dir / "glossary" / "glossary.json"
        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Glossary not found at {glossary_path}")
            return {}
    
    def generate_chunk_id(self, content: str, chunk_type: str, play_title: str) -> str:
        """Generate a unique chunk ID based on content hash."""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        safe_title = play_title.lower().replace(' ', '_').replace(',', '').replace('.', '')
        return f"{chunk_type}_{safe_title}_{content_hash}"
    
    def process_dialogue_files(self):
        """Process dialogue files to create scene-level chunks."""
        dialogue_dir = self.data_dir / "processed" / "dialogue"
        
        if not dialogue_dir.exists():
            print(f"Dialogue directory not found: {dialogue_dir}")
            return
            
        for file_path in dialogue_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    play_data = json.load(f)
                
                play_title = play_data.get("title", "Unknown Play")
                
                for act in play_data.get("acts", []):
                    act_num = act.get("act", 0)
                    
                    for scene in act.get("scenes", []):
                        scene_num = scene.get("scene", 0)
                        location = scene.get("location", "Unknown location")
                        scene_id = scene.get("scene_id", f"{play_title}_{act_num}_{scene_num}")
                        
                        # Build scene content
                        scene_content = f"Play: {play_title}\n"
                        scene_content += f"Act {act_num}, Scene {scene_num}\n"
                        scene_content += f"Location: {location}\n\n"
                        scene_content += "Dialogue:\n"
                        
                        speakers = set()
                        for dialogue in scene.get("dialogues", []):
                            speaker = dialogue.get("speaker", "Unknown")
                            line = dialogue.get("line", "")
                            speakers.add(speaker)
                            scene_content += f"{speaker}: {line}\n"
                        
                        # Create chunk
                        chunk_id = self.generate_chunk_id(scene_content, "scene", play_title)
                        
                        chunk = Chunk(
                            chunk_id=chunk_id,
                            chunk_type="scene",
                            play_title=play_title,
                            content=scene_content,
                            metadata={
                                "act": act_num,
                                "scene": scene_num,
                                "location": location,
                                "scene_id": scene_id,
                                "speakers": list(speakers),
                                "dialogue_count": len(scene.get("dialogues", []))
                            }
                        )
                        
                        self.chunks.append(chunk)
                        
            except Exception as e:
                print(f"Error processing dialogue file {file_path}: {e}")
    
    def process_factual_data(self):
        """Process factual data to create play-level chunks."""
        factual_path = self.data_dir / "processed" / "factual" / "factual.json"
        
        if not factual_path.exists():
            print(f"Factual data not found: {factual_path}")
            return
            
        try:
            with open(factual_path, 'r', encoding='utf-8') as f:
                factual_data = json.load(f)
            
            for play in factual_data:
                play_title = play.get("title", "Unknown Play")
                
                # Create main factual chunk
                factual_content = f"Play: {play_title}\n"
                factual_content += f"Category: {play.get('category', 'Unknown')}\n"
                factual_content += f"Year: {play.get('year', 'Unknown')}\n"
                factual_content += f"Setting: {play.get('setting', 'Unknown')}\n\n"
                
                # Add main characters
                factual_content += "Main Characters:\n"
                for char in play.get("main_characters", []):
                    factual_content += f"- {char}\n"
                
                # Add character descriptions
                factual_content += "\nCharacter Descriptions:\n"
                for char_desc in play.get("character_descriptions", []):
                    name = char_desc.get("name", "Unknown")
                    desc = char_desc.get("description", "No description")
                    factual_content += f"- {name}: {desc}\n"
                
                chunk_id = self.generate_chunk_id(factual_content, "factual", play_title)
                
                chunk = Chunk(
                    chunk_id=chunk_id,
                    chunk_type="factual",
                    play_title=play_title,
                    content=factual_content,
                    metadata={
                        "category": play.get("category", "Unknown"),
                        "year": play.get("year", "Unknown"),
                        "setting": play.get("setting", "Unknown"),
                        "main_characters": play.get("main_characters", []),
                        "character_count": len(play.get("main_characters", []))
                    }
                )
                
                self.chunks.append(chunk)
                
                # Create individual theme chunks
                for theme_data in play.get("themes", []):
                    theme_name = theme_data.get("theme", "Unknown Theme")
                    theme_explanation = theme_data.get("theme_explanation", "No explanation")
                    
                    theme_content = f"Play: {play_title}\n"
                    theme_content += f"Theme: {theme_name}\n\n"
                    theme_content += f"Explanation: {theme_explanation}"
                    
                    theme_chunk_id = self.generate_chunk_id(theme_content, "theme", play_title)
                    
                    theme_chunk = Chunk(
                        chunk_id=theme_chunk_id,
                        chunk_type="theme",
                        play_title=play_title,
                        content=theme_content,
                        metadata={
                            "theme_name": theme_name,
                            "category": play.get("category", "Unknown")
                        }
                    )
                    
                    self.chunks.append(theme_chunk)
                    
        except Exception as e:
            print(f"Error processing factual data: {e}")
    
    def process_summaries(self):
        """Process summary files to create summary chunks."""
        summary_dir = self.data_dir / "processed" / "full_summary"
        
        if not summary_dir.exists():
            print(f"Summary directory not found: {summary_dir}")
            return
            
        for file_path in summary_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                
                play_title = summary_data.get("title", "Unknown Play")
                
                # Create play-level summary chunk
                play_summary_content = f"Play: {play_title}\n"
                play_summary_content += f"Play Summary: {summary_data.get('play_summary', 'No summary available')}"
                
                play_chunk_id = self.generate_chunk_id(play_summary_content, "summary_play", play_title)
                
                play_chunk = Chunk(
                    chunk_id=play_chunk_id,
                    chunk_type="summary_play",
                    play_title=play_title,
                    content=play_summary_content,
                    metadata={
                        "summary_level": "play"
                    }
                )
                
                self.chunks.append(play_chunk)
                
                # Create act and scene summary chunks
                for act in summary_data.get("acts", []):
                    act_num = act.get("act", 0)
                    act_summary = act.get("act_summary", "No summary available")
                    
                    # Act summary chunk
                    act_content = f"Play: {play_title}\n"
                    act_content += f"Act {act_num} Summary: {act_summary}"
                    
                    act_chunk_id = self.generate_chunk_id(act_content, "summary_act", play_title)
                    
                    act_chunk = Chunk(
                        chunk_id=act_chunk_id,
                        chunk_type="summary_act",
                        play_title=play_title,
                        content=act_content,
                        metadata={
                            "summary_level": "act",
                            "act": act_num
                        }
                    )
                    
                    self.chunks.append(act_chunk)
                    
                    # Scene summary chunks
                    for scene in act.get("scenes", []):
                        scene_num = scene.get("scene", 0)
                        location = scene.get("location", "Unknown location")
                        scene_summary = scene.get("scene_summary", "No summary available")
                        
                        scene_content = f"Play: {play_title}\n"
                        scene_content += f"Act {act_num}, Scene {scene_num}\n"
                        scene_content += f"Location: {location}\n"
                        scene_content += f"Scene Summary: {scene_summary}"
                        
                        scene_chunk_id = self.generate_chunk_id(scene_content, "summary_scene", play_title)
                        
                        scene_chunk = Chunk(
                            chunk_id=scene_chunk_id,
                            chunk_type="summary_scene",
                            play_title=play_title,
                            content=scene_content,
                            metadata={
                                "summary_level": "scene",
                                "act": act_num,
                                "scene": scene_num,
                                "location": location
                            }
                        )
                        
                        self.chunks.append(scene_chunk)
                        
            except Exception as e:
                print(f"Error processing summary file {file_path}: {e}")
    
    def process_quotes(self):
        """Process quotes to create individual quote chunks."""
        quotes_path = self.data_dir / "processed" / "quote" / "quote.json"
        
        if not quotes_path.exists():
            print(f"Quotes data not found: {quotes_path}")
            return
            
        try:
            with open(quotes_path, 'r', encoding='utf-8') as f:
                quotes_data = json.load(f)
            
            for play in quotes_data:
                play_title = play.get("title", "Unknown Play")
                
                for quote_data in play.get("famous_quotes", []):
                    quote = quote_data.get("quote", "")
                    speaker = quote_data.get("speaker", "Unknown")
                    act = quote_data.get("act", 0)
                    scene = quote_data.get("scene", 0)
                    explanation = quote_data.get("explanation", "No explanation")
                    
                    quote_content = f"Play: {play_title}\n"
                    quote_content += f"Quote: \"{quote}\"\n"
                    quote_content += f"Speaker: {speaker}\n"
                    quote_content += f"Act {act}, Scene {scene}\n\n"
                    quote_content += f"Explanation: {explanation}"
                    
                    quote_chunk_id = self.generate_chunk_id(quote_content, "quote", play_title)
                    
                    quote_chunk = Chunk(
                        chunk_id=quote_chunk_id,
                        chunk_type="quote",
                        play_title=play_title,
                        content=quote_content,
                        metadata={
                            "quote": quote,
                            "speaker": speaker,
                            "act": act,
                            "scene": scene
                        }
                    )
                    
                    self.chunks.append(quote_chunk)
                    
        except Exception as e:
            print(f"Error processing quotes: {e}")
    
    def process_character_relationships(self):
        """Process character relationships to create relationship chunks."""
        relationships_path = self.data_dir / "glossary" / "character_relationship.json"
        
        if not relationships_path.exists():
            print(f"Character relationships not found: {relationships_path}")
            return
            
        try:
            with open(relationships_path, 'r', encoding='utf-8') as f:
                relationships_data = json.load(f)
            
            for play_data in relationships_data:
                play_title = play_data.get("title", "Unknown Play")
                
                for rel_data in play_data.get("character_descriptions", []):
                    char1 = rel_data.get("character_1", "Unknown")
                    char2 = rel_data.get("character_2", "Unknown")
                    rel_type = rel_data.get("relationship_type", "Unknown")
                    description = rel_data.get("description", "No description")
                    
                    rel_content = f"Play: {play_title}\n"
                    rel_content += f"Characters: {char1} and {char2}\n"
                    rel_content += f"Relationship Type: {rel_type}\n"
                    rel_content += f"Description: {description}"
                    
                    rel_chunk_id = self.generate_chunk_id(rel_content, "relationship", play_title)
                    
                    rel_chunk = Chunk(
                        chunk_id=rel_chunk_id,
                        chunk_type="character_relationship",
                        play_title=play_title,
                        content=rel_content,
                        metadata={
                            "character_1": char1,
                            "character_2": char2,
                            "relationship_type": rel_type
                        }
                    )
                    
                    self.chunks.append(rel_chunk)
                    
        except Exception as e:
            print(f"Error processing character relationships: {e}")
    
    def save_chunks(self):
        """Save processed chunks to JSON files."""
        # Save all chunks to a single file
        chunks_data = []
        for chunk in self.chunks:
            chunk_dict = asdict(chunk)
            chunks_data.append(chunk_dict)
        
        all_chunks_path = self.output_dir / "all_chunks.json"
        with open(all_chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save chunks by type for easier debugging and analysis
        chunks_by_type = {}
        for chunk in self.chunks:
            chunk_type = chunk.chunk_type
            if chunk_type not in chunks_by_type:
                chunks_by_type[chunk_type] = []
            chunks_by_type[chunk_type].append(asdict(chunk))
        
        for chunk_type, type_chunks in chunks_by_type.items():
            type_path = self.output_dir / f"chunks_{chunk_type}.json"
            with open(type_path, 'w', encoding='utf-8') as f:
                json.dump(type_chunks, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        metadata = {
            "total_chunks": len(self.chunks),
            "chunks_by_type": {k: len(v) for k, v in chunks_by_type.items()},
            "processed_timestamp": datetime.now().isoformat(),
            "chunk_types": list(chunks_by_type.keys())
        }
        
        metadata_path = self.output_dir / "chunks_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata
    
    def process_all(self):
        """Process all data sources and create chunks."""
        print("Processing Shakespeare data into chunks...")
        
        print("1. Processing dialogue files...")
        self.process_dialogue_files()
        
        print("2. Processing factual data...")
        self.process_factual_data()
        
        print("3. Processing summaries...")
        self.process_summaries()
        
        print("4. Processing quotes...")
        self.process_quotes()
        
        print("5. Processing character relationships...")
        self.process_character_relationships()
        
        print("6. Saving chunks...")
        metadata = self.save_chunks()
        
        print("\nProcessing complete!")
        print(f"Total chunks created: {metadata['total_chunks']}")
        print("Chunks by type:")
        for chunk_type, count in metadata['chunks_by_type'].items():
            print(f"  - {chunk_type}: {count}")
        print(f"\nChunks saved to: {self.output_dir}")
        
        return metadata

def main():
    """Main function to run the chunking process."""
    chunker = ShakespeareChunker()
    metadata = chunker.process_all()
    
    # Display some example chunks
    print("\n" + "="*50)
    print("EXAMPLE CHUNKS:")
    print("="*50)
    
    chunk_types_to_show = ["scene", "quote", "theme", "character_relationship"]
    
    for chunk_type in chunk_types_to_show:
        matching_chunks = [chunk for chunk in chunker.chunks if chunk.chunk_type == chunk_type]
        if matching_chunks:
            example_chunk = matching_chunks[0]
            print(f"\n--- {chunk_type.upper()} CHUNK EXAMPLE ---")
            print(f"ID: {example_chunk.chunk_id}")
            print(f"Play: {example_chunk.play_title}")
            print(f"Content Preview: {example_chunk.content[:200]}...")
            print(f"Metadata: {example_chunk.metadata}")

if __name__ == "__main__":
    main()