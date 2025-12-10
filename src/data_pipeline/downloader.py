from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Optional

import pandas as pd


class Downloader(ABC):
    """Abstract base class for dataset downloaders."""

    def __init__(self):
         pass

    def download_and_save_dataset(
        self,
        output_path: Path,
        quarters: Optional[list[int]] = None,
        years: Optional[list[int]] = None,
        tickers: Optional[list[str]] = None,

    ):
        """Downloads and filters the dataset based on the provided criteria.
        
        Args:
            output_path str: Path to save the filtered dataset.
            quarters (Optional[list[int]]): List of fiscal quarters to filter by.
            years (Optional[list[int]]): List of years to filter by.
            tickers (Optional[list[str]]): List of stock tickers to filter by.
        """
        if output_path.suffix != ".csv":
            raise ValueError("Output path must have a .csv extension.")

        # validate date filters before downloading dataset
        self._validate_date_filters(quarters=quarters, years=years)

        df = self._download_dataset()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_df = self._filter(df, quarters, years, tickers)
        filtered_df.to_csv(output_path, index=False)
    
    @abstractmethod
    def _validate_date_filters(self):
        """Abstract method to validate date filter criteria.
        
        Raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def _download_dataset(self) -> pd.DataFrame:
        """Abstract property to download the dataset.
        
        Raises NotImplementedError: If the property is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement this property")

    @abstractmethod
    def _filter(
        self,
        df: pd.DataFrame,
        quarters: Optional[list[int]] = None,
        years: Optional[list[int]] = None,
        tickers: Optional[list[str]] = None
    ) -> None:
        """Abstract method to filter the DataFrame based on quarters, years, and tickers.
        
        Args:
            df (pd.DataFrame): The DataFrame to filter.
            quarters (Optional[list[int]]): List of fiscal quarters to filter by.
            years (Optional[list[int]]): List of years to filter by.
            tickers (Optional[list[str]]): List of stock tickers to filter by.
        
        Raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")

class EarningsTranscriptsDownloader(Downloader):
    """Downloader for the earnings transcripts dataset."""

    def __init__(self):
        super().__init__()

    def _validate_date_filters(
        self,
        quarters: Optional[list[int]] = None,
        years: Optional[list[int]] = None,
    ):
        """Validates the quarters and years filters for the earnings transcripts dataset.
        Args:
            quarters (Optional[list[int]]): List of fiscal quarters to validate.
            years (Optional[list[int]]): List of years to validate.
        
        Raises:
            ValueError: If any quarter is not between 1 and 4, or if any year is not between 2005 and 2025.
        """
        if quarters is not None:
            if any(q not in [1, 2, 3, 4] for q in quarters):
                raise ValueError("Fiscal quarter values must be between 1 and 4.")
        
        if years is not None:
            if any(y < 2005 or y > 2025 for y in years):
                raise ValueError("Year values must be between 2005 and 2025.")

    def _download_dataset(self) -> pd.DataFrame:
        """Downloads the earnings transcripts dataset parquet file from huggingface.

        Returns:
            pd.DataFrame: The downloaded dataset as a DataFrame.
        """
        earnings_transcripts_df = pd.read_parquet(
            "hf://datasets/kurry/sp500_earnings_transcripts/parquet_files/part-0.parquet"
        )
        return earnings_transcripts_df

    def _filter(
        self,
        df: pd.DataFrame,
        quarters: Optional[list[int]] = None,
        years: Optional[list[int]] = None,
        tickers: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """Filters the DataFrame based on quarters, years, and tickers.

        Args:
            df (pd.DataFrame): The DataFrame to filter.
            quarters (Optional[list[int]]): List of fiscal quarters to filter by.
            years (Optional[list[int]]): List of years to filter by.
            tickers (Optional[list[str]]): List of stock tickers to filter by.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        if quarters is not None:
            df = df[df['quarter'].isin(quarters)]
        
        if years is not None:
            df = df[df['year'].isin(years)]
        
        if tickers is not None:
            df = df[df['symbol'].isin(tickers)]
        
        return df

class EarningsQADownloader(Downloader):
    """Downloader for the earnings qa dataset."""

    def __init__(self):
        super().__init__()

    def _validate_date_filters(
        self,
        quarters: Optional[list[int]] = None,
        years: Optional[list[int]] = None,
    ):
        """Validates the quarters and years filters for the earnings qa dataset.
        Args:
            quarters (Optional[list[int]]): List of fiscal quarters to validate.
            years (Optional[list[int]]): List of years to validate.

        Raises:
            ValueError: If any quarter is not between 1 and 4, or if any
                year is not between 2019 and 2023.
        """
        if quarters is not None:
            if any(q not in [1, 2, 3, 4] for q in quarters):
                raise ValueError("Fiscal quarter values must be between 1 and 4.")
        
        if years is not None:
            if any(y < 2019 or y > 2023 for y in years):
                raise ValueError("Year values must be between 2019 and 2023.")

    def _download_dataset(self) -> pd.DataFrame:
        """Downloads the earnings qa dataset parquet file from huggingface.

        Returns:
            pd.DataFrame: The downloaded dataset as a DataFrame.
        """
        earnings_qa_df = pd.read_json(
            "hf://datasets/lamini/earnings-calls-qa/filtered_predictions.jsonl", 
            lines=True
            )
        return earnings_qa_df

    def _filter(
        self,
        df: pd.DataFrame,
        quarters: Optional[list[int]] = None,
        years: Optional[list[int]] = None,
        tickers: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """Filters the DataFrame based on quarters, years, and tickers.

        Args:
            df (pd.DataFrame): The DataFrame to filter.
            quarters (Optional[list[int]]): List of fiscal quarters to filter by.
            years (Optional[list[int]]): List of years to filter by.
            tickers (Optional[list[str]]): List of stock tickers to filter by.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        if quarters is None: # Default to all quarters
            quarters = [1, 2, 3, 4]
        
        if years is None: # Default to all years
            years = [2019, 2020, 2021, 2022, 2023]

        # Create list of fiscal quarter strings to filter by
        fiscal_quarters = [f"{year}-Q{quarter}" for year in years for quarter in quarters]

        df = df[df['q'].isin(fiscal_quarters)]

        if tickers is not None:
            df = df[df['ticker'].isin(tickers)]
        return df