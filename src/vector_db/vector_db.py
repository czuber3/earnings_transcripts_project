from typing import List, Optional

import chromadb
from chromadb.config import Settings

from src.chunking.earnings_transcript_chunk import EarningsTranscriptChunk

class VectorDB:
    """A vector database to store and query earnings transcripts text embeddings."""

    def __init__(self, vector_db_path: str = "data/chroma_db"):
        self.chroma_client = chromadb.PersistentClient(
            path=vector_db_path,
            settings=Settings()
        )

    def delete_collection(self, collection_name: str):
        """Deletes a collection in the vector database if it exists
        
        Args:
            collection_name (str): The name of the collection to delete.
        """
        try: # make sure it is 
            self.chroma_client.get_collection(collection_name)
            self.chroma_client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' deleted.")
        except chromadb.errors.NotFoundError:
            print(f"Collection '{collection_name}' does not exist.")
    
    def _get_or_create_collection(
        self, collection_name: str
    ) -> chromadb.api.models.Collection.Collection:
        """Retrieves a collection from the vector database. Or creates a new 
        one if it doesn't exist.

        Args:
            collection_name (str): The name of the collection to retrieve.
        
        Returns:
            chromadb.api.models.Collection.Collection: The retrieved collection.
        
        Raises:
            chromadb.errors.CollectionNotFoundError: If the collection does not exist.
        """
        return self.chroma_client.get_or_create_collection(collection_name) 
    
    def upload(
        self, 
        collection_name: str,
        earnings_transcript_chunks: List[EarningsTranscriptChunk], 
        embeddings: List[List[float]]
    ):
        """Adds texts and their corresponding embeddings to the database.

        Args:
            collection_name (str): The name of the collection to add the texts 
                to.
            earnings_transcript_chunks (List[EarningsTranscriptChunk]): A list 
                of EarningsTranscriptChunks.
            embeddings (List[List[float]]): A list of embeddings corresponding 
                to the texts.

        Raises:
            ValueError: If the lengths of texts and embeddings do not match.
        """
        if len(earnings_transcript_chunks) != len(embeddings):
            raise ValueError("The number of chunks and embeddings must match.")
        
        collection = self._get_or_create_collection(collection_name)

        chunk_ids, documents, metadatas = [], [], []
        for idx, earnings_transcript_chunk in enumerate(earnings_transcript_chunks):
            chunk_ids.append(
                f"{earnings_transcript_chunk.ticker}_{earnings_transcript_chunk.year}_{earnings_transcript_chunk.quarter}_{idx}"
            )
            documents.append(earnings_transcript_chunk.text)
            metadatas.append(
                {
                    "ticker": earnings_transcript_chunk.ticker,
                    "year": earnings_transcript_chunk.year,
                    "quarter": earnings_transcript_chunk.quarter
                }
            )

        collection.add(
            ids=chunk_ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def search(
        self,
        query_embedding: List[float],
        collection_name: str,
        ticker: Optional[str] = None,
        year: Optional[int] = None,
        quarter: Optional[int] = None,
        n_results: Optional[int] = 10
    ) -> List[EarningsTranscriptChunk]:
        """ Queries the given collection with the query embedding.
        
        Args:
            query_embedding (str): Embedding used to query the DB.
            colleciton_name (str): name of the collection to query.
            ticker (Optional(str)): ticker to filter to.
            year (Optional(str)): year to filter to.
            quarter (Optionl[str]): quarter to filter to.
            n_results (Optional[int]): number of results to return
        Returns:
            List of EarningsTranscriptChunks holding the resulting text and 
            metadatas.
        """
        collection = self._get_or_create_collection(collection_name)

        # build filter
        where = {"$and": []}
        if ticker is not None:
            where["$and"].append({"ticker": ticker})
        if year is not None:
            where["$and"].append({"year": year})
        if quarter is not None:
            where["$and"].append({"quarter": quarter})

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

        return [
            EarningsTranscriptChunk(
                text=results['documents'][0][idx],
                ticker=results['metadatas'][0][idx]['ticker'],
                quarter=results['metadatas'][0][idx]['quarter'],
                year=results['metadatas'][0][idx]['year']
            ) for idx in range(len(results['documents'][0]))
        ]