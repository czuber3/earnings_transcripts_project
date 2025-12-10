from abc import ABC, abstractmethod
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import spacy

class BaseChunker(ABC):
    """Base class for text chunkers."""

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def chunk(
            self, 
            text: str, 
            max_chunk_size: int = 1000, 
            overlap: int = 200
        ) -> List[str]:
        """Chunks the text into smaller segments.

        Args:
            text (str): The input text to chunk.
            max_chunk_size (int): Maximum size of each chunk in characters.
            overlap (int): Number of overlapping characters between chunks.

        Returns:
            List[str]: A list of text chunks.
        """
        raise NotImplementedError("Subclasses must implement this method")

class RecursiveChunker:
    """A chunker that splits text using recursive character splitting."""

    def __init__(self):
        super().__init__()
    
    def chunk(
            self, 
            text: str, 
            max_chunk_size: int = 1000, 
            overlap: int = 200
        ) -> List[str]:
        """Chunks the text into smaller segments.

        Args:
            text (str): The input text to chunk.
            max_chunk_size (int): Maximum size of each chunk in characters.
            overlap (int): Number of overlapping characters between chunks.

        Returns:
            List[str]: A list of text chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)

class SemanticChunker:
    """A chunker that splits text based on semantic similarity using embeddings."""

    def __init__(self, embedding_model: str = "FinLang/finance-embeddings-investopedia"):
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
    
    def chunk(
            self, 
            text: str, 
            max_chunk_size: int = 1000, 
            overlap: int = 200, 
            similarity_threshold: int = 0.4
        ) -> List[str]:
        """Chunks the text into semantically similar segments.

        Args:
            text (str): The input text to chunk.
            max_chunk_size (int): Maximum size of each chunk in characters.
            overlap (int): Number of overlapping characters between chunks.

        Returns:
            List[str]: A list of text chunks.
        """
        # seperate the text into sentences
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        # compute embeddings and similarities for sentences
        embeddings = self.model.encode(sentences)
        similarities = [util.cos_sim(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]

        # Simple threshold-based segmentation
        paragraphs = []
        current = [sentences[0]]
        for i, sim in enumerate(similarities):
            if sim < similarity_threshold:
                paragraphs.append(" ".join(current))
                current = []
            current.append(sentences[i+1])
        paragraphs.append(" ".join(current))

        # Create a single text string from all paragraphs
        full_text = "\n\n".join(paragraphs)

        # Split the text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(full_text)