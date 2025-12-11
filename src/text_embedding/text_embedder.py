import torch
from typing import List, Optional

from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name: str = "FinLang/finance-embeddings-investopedia", device: Optional[str] = None):
        """Initializes the Embedder with a specified model and device.

        Args:
            model_name (str): The name of the pre-trained model to use for embeddings.
            device (Optional[str]): Torch device string, e.g. 'cuda' or 'cpu'. If
                None, the embedder will attempt to use GPU if available.
        """
        # Auto-detect device if not provided
        if device is None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        self.device = device
        # SentenceTransformer accepts a `device` argument
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts.

        Args:
            texts (List[str]): A list of texts to embed.

        Returns:
            List[List[float]]: A list of embeddings corresponding to the input texts.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()