from abc import ABC, abstractmethod
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import spacy

from src.text_embedding.text_embedder import TextEmbedder
from .earnings_transcript_chunk import EarningsTranscriptChunk

class BaseChunker(ABC):
    """Base class for text chunkers."""

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def chunk(
        self, 
        text: str,
        ticker: str,
        year: int,
        quarter: int,
        max_chunk_size: int = 1000, 
        overlap: int = 200,
        **kwargs
    ) -> List[EarningsTranscriptChunk]:
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
        ticker: str,
        year: int,
        quarter: int,
        max_chunk_size: int = 1000, 
        overlap: int = 200,
        **kwargs
    ) -> List[EarningsTranscriptChunk]:
        """Chunks the text into smaller segments using recursive character
        chunking.

        Args:
            text (str): The text of the earnings transcript.
            ticker (str): The earnings transcript ticker.
            year (int): The year of the earnings transcript.
            quarter (int): The quarter of the earnings transcript.
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
        text_chunks = splitter.split_text(text)
        return [
            EarningsTranscriptChunk(
                text=text_chunk,
                ticker=ticker,
                year=year,
                quarter=quarter)
            for text_chunk in text_chunks
        ]

class SemanticChunker:
    """A chunker that splits text based on semantic similarity using embeddings."""

    def __init__(self, text_embedder: TextEmbedder):
        super().__init__()

        try: # spacy model to split sentences
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("en_core_web_sm Model not found. Run `python -m spacy download en_core_web_sm`")
            raise OSError
        
        self.text_embedder = text_embedder
    
    def chunk(
        self, 
        text: str,
        ticker: str,
        year: int,
        quarter: int,
        max_chunk_size: int = 1000, 
        overlap: int = 200,
        similarity_threshold: float = 0.4
    ) -> List[EarningsTranscriptChunk]:
        """Chunks the text into semantically similar segments.

        Args:
            text (str): The text of the earnings transcript.
            ticker (str): The earnings transcript ticker.
            year (int): The year of the earnings transcript.
            quarter (int): The quarter of the earnings transcript.
            max_chunk_size (int): Maximum size of each chunk in characters.
            overlap (int): Number of overlapping characters between chunks.
            similarity_threshold (float): Threshold used to structure text.

        Returns:
            List[str]: A list of text chunks.
        """
        # seperate the text into sentences
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        # compute embeddings and similarities for sentences
        text_embeddings = self.text_embedder.embed_texts(sentences)
        similarities = [
            util.cos_sim(text_embeddings[i], text_embeddings[i+1]) for i in range(len(text_embeddings)-1)
            ]

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
        text_chunks = splitter.split_text(text)
        return [
            EarningsTranscriptChunk(
                text=text_chunk,
                ticker=ticker,
                year=year,
                quarter=quarter)
            for text_chunk in text_chunks
        ]