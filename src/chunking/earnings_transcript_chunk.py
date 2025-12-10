from dataclasses import dataclass

@dataclass
class EarningsTranscriptChunk:
    """Class to hold text and metadata for earnings transcripts chunk"""
    text: str
    ticker: str
    quarter: str
    year: int