from typing import List

from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name: str = "FinLang/finance-embeddings-investopedia"):
        """Initializes the Embedder with a specified model.

        Args:
            model_name (str): The name of the pre-trained model to use for embeddings.
        """
        self.model = SentenceTransformer(model_name)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts.

        Args:
            texts (List[str]): A list of texts to embed.
        
        Returns:
            List[List[float]]: A list of embeddings corresponding to the input texts.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()