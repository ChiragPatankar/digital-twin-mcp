from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
from pathlib import Path
import logging
from .llm.base import BaseLLM

Base = declarative_base()

logger = logging.getLogger(__name__)

class Memory(Base):
    """SQLAlchemy model for storing memories."""
    __tablename__ = "memories"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    content = Column(String)
    context = Column(JSON)
    importance = Column(Integer, default=0)  # 0-100 scale
    category = Column(String)
    embedding = Column(JSON)  # Store vector embeddings for similarity search

class MemoryManager:
    """Manages the digital twin's memory system."""
    def __init__(self, db_url: str = "sqlite:///digital_twin.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def add_memory(self, content: str, context: Dict[str, Any], 
                  importance: int = 50, category: str = "general") -> None:
        """Add a new memory to the system."""
        session = self.Session()
        try:
            memory = Memory(
                content=content,
                context=context,
                importance=importance,
                category=category
            )
            session.add(memory)
            session.commit()
        finally:
            session.close()
    
    def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the most recent memories."""
        session = self.Session()
        try:
            memories = session.query(Memory)\
                .order_by(Memory.timestamp.desc())\
                .limit(limit)\
                .all()
            return [{
                "id": m.id,
                "timestamp": m.timestamp,
                "content": m.content,
                "context": m.context,
                "importance": m.importance,
                "category": m.category
            } for m in memories]
        finally:
            session.close()
    
    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memories based on content similarity.
        This is a simple implementation that can be enhanced with vector similarity search.
        """
        session = self.Session()
        try:
            memories = session.query(Memory)\
                .filter(Memory.content.ilike(f"%{query}%"))\
                .order_by(Memory.importance.desc())\
                .limit(limit)\
                .all()
            return [{
                "id": m.id,
                "timestamp": m.timestamp,
                "content": m.content,
                "context": m.context,
                "importance": m.importance,
                "category": m.category
            } for m in memories]
        finally:
            session.close()
    
    def update_memory_importance(self, memory_id: int, new_importance: int) -> None:
        """Update the importance of a specific memory."""
        session = self.Session()
        try:
            memory = session.query(Memory).get(memory_id)
            if memory:
                memory.importance = new_importance
                session.commit()
        finally:
            session.close()

class MemorySystem:
    """Vectorized memory system for the digital twin."""
    
    def __init__(
        self,
        llm: BaseLLM,
        memory_dir: str = "memories",
        max_memories: int = 1000
    ):
        """Initialize the memory system.
        
        Args:
            llm: LLM instance for generating embeddings
            memory_dir: Directory to store memories
            max_memories: Maximum number of memories to keep
        """
        self.llm = llm
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.max_memories = max_memories
        self.memories: List[Dict[str, Any]] = []
        self._load_memories()
    
    def _load_memories(self) -> None:
        """Load memories from disk."""
        try:
            memory_file = self.memory_dir / "memories.json"
            if memory_file.exists():
                with open(memory_file, "r") as f:
                    self.memories = json.load(f)
                logger.info(f"Loaded {len(self.memories)} memories")
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
            self.memories = []
    
    def _save_memories(self) -> None:
        """Save memories to disk."""
        try:
            memory_file = self.memory_dir / "memories.json"
            with open(memory_file, "w") as f:
                json.dump(self.memories, f, indent=2)
            logger.info(f"Saved {len(self.memories)} memories")
        except Exception as e:
            logger.error(f"Error saving memories: {e}")
    
    async def add_memory(
        self,
        content: str,
        memory_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new memory.
        
        Args:
            content: Memory content
            memory_type: Type of memory (episodic, semantic, etc.)
            metadata: Additional metadata
        """
        try:
            # Generate embedding
            embedding = await self.llm.embed(content)
            
            # Create memory entry
            memory = {
                "content": content,
                "type": memory_type,
                "embedding": embedding,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            # Add to memory list
            self.memories.append(memory)
            
            # Trim if over limit
            if len(self.memories) > self.max_memories:
                self.memories = self.memories[-self.max_memories:]
            
            # Save to disk
            self._save_memories()
            
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise
    
    async def get_relevant_memories(
        self,
        query: str,
        limit: int = 5,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get memories relevant to the query.
        
        Args:
            query: Query text
            limit: Maximum number of memories to return
            memory_type: Optional filter by memory type
            
        Returns:
            List of relevant memories
        """
        try:
            # Generate query embedding
            query_embedding = await self.llm.embed(query)
            
            # Filter by type if specified
            memories = self.memories
            if memory_type:
                memories = [m for m in memories if m["type"] == memory_type]
            
            # Calculate similarities
            similarities = []
            for memory in memories:
                similarity = np.dot(query_embedding, memory["embedding"])
                similarities.append((similarity, memory))
            
            # Sort by similarity
            similarities.sort(reverse=True)
            
            # Return top matches
            return [m for _, m in similarities[:limit]]
            
        except Exception as e:
            logger.error(f"Error getting relevant memories: {e}")
            return []
    
    async def get_recent_memories(
        self,
        limit: int = 10,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get most recent memories.
        
        Args:
            limit: Maximum number of memories to return
            memory_type: Optional filter by memory type
            
        Returns:
            List of recent memories
        """
        try:
            # Filter by type if specified
            memories = self.memories
            if memory_type:
                memories = [m for m in memories if m["type"] == memory_type]
            
            # Sort by timestamp
            memories.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return memories[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent memories: {e}")
            return []
    
    async def get_memories_for_reflection(
        self,
        time_period: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get memories for reflection.
        
        Args:
            time_period: Optional time period (e.g., "1d", "1w", "1m")
            limit: Maximum number of memories to return
            
        Returns:
            List of memories for reflection
        """
        try:
            # Filter by time period if specified
            memories = self.memories
            if time_period:
                # TODO: Implement time period filtering
                pass
            
            # Sort by timestamp
            memories.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return memories[:limit]
            
        except Exception as e:
            logger.error(f"Error getting memories for reflection: {e}")
            return []
    
    async def import_memories(
        self,
        file_path: str,
        memory_type: str = "semantic"
    ) -> None:
        """Import memories from a file.
        
        Args:
            file_path: Path to the file
            memory_type: Type of memories to import
        """
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            # Split content into chunks
            chunks = self._split_into_chunks(content)
            
            # Add each chunk as a memory
            for chunk in chunks:
                await self.add_memory(
                    content=chunk,
                    memory_type=memory_type,
                    metadata={"source": file_path}
                )
            
        except Exception as e:
            logger.error(f"Error importing memories: {e}")
            raise
    
    def _split_into_chunks(
        self,
        text: str,
        max_chunk_size: int = 1000
    ) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Text to split
            max_chunk_size: Maximum chunk size
            
        Returns:
            List of text chunks
        """
        # Simple splitting by paragraphs
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            if current_size + paragraph_size > max_chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_size = paragraph_size
            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks 