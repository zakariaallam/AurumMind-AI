"""Complete data ingestion pipeline for Phase 1."""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from ..utils.logging_config import setup_logging
from ..validation.data_validator import DataValidator
from .data_loader import DataLoader

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end data ingestion pipeline.
    
    Orchestrates loading, validation, and storage of market data.
    """
    
    def __init__(self, config_path: str = "configs/data.yaml"):
        """Initialize ingestion pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.loader = DataLoader(config_path)
        self.validator = DataValidator()
    
    def ingest_file(
        self,
        file_path: str,
        timeframe: Literal["1h", "4h", "1d"],
        source_name: str = "Unknown",
        file_format: Literal["csv", "parquet"] = "csv",
    ) -> tuple[pd.DataFrame, dict]:
        """Ingest a single data file through the complete pipeline.
        
        Args:
            file_path: Path to the data file
            timeframe: Timeframe of the data
            source_name: Name of the data provider
            file_format: Format of the file (csv or parquet)
            
        Returns:
            Tuple of (validated_dataframe, pipeline_report)
            
        Raises:
            ValueError: If validation fails with errors
        """
        logger.info(f"Starting ingestion pipeline for {file_path}")
        
        pipeline_report = {
            "file_path": file_path,
            "timeframe": timeframe,
            "source_name": source_name,
            "stages": {},
        }
        
        # Stage 1: Load data
        logger.info("Stage 1: Loading data")
        try:
            if file_format == "csv":
                df, metadata = self.loader.load_csv(file_path, timeframe, source_name)
            elif file_format == "parquet":
                df, metadata = self.loader.load_parquet(file_path, timeframe, source_name)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            pipeline_report["stages"]["loading"] = {
                "status": "success",
                "records_loaded": len(df),
            }
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            pipeline_report["stages"]["loading"] = {
                "status": "failed",
                "error": str(e),
            }
            raise
        
        # Stage 2: Validate data
        logger.info("Stage 2: Validating data")
        quality_report = self.validator.validate(df, metadata)
        
        pipeline_report["stages"]["validation"] = {
            "status": "success",
            "quality_score": quality_report.quality_score,
            "passed": quality_report.passed_validation,
            "issues_found": len(quality_report.issues),
        }
        
        # Stage 3: Save raw data
        logger.info("Stage 3: Saving raw data")
        try:
            raw_path = self.loader.save_raw_data(df, metadata)
            pipeline_report["stages"]["saving"] = {
                "status": "success",
                "output_path": str(raw_path),
            }
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            pipeline_report["stages"]["saving"] = {
                "status": "failed",
                "error": str(e),
            }
            raise
        
        # Stage 4: Save quality report
        logger.info("Stage 4: Saving quality report")
        report_path = raw_path.with_suffix(".quality_report.json")
        self.validator.save_report(quality_report, report_path)
        
        # Check if validation passed
        if not quality_report.passed_validation:
            logger.error("Data validation failed")
            logger.error(quality_report.get_summary())
            raise ValueError(
                f"Data validation failed with {len(quality_report.issues)} issues. "
                f"Quality score: {quality_report.quality_score:.2f}"
            )
        
        logger.info("Ingestion pipeline completed successfully")
        logger.info(quality_report.get_summary())
        
        pipeline_report["overall_status"] = "success"
        
        return df, pipeline_report
    
    def ingest_multiple_files(
        self,
        file_configs: list[dict],
    ) -> dict:
        """Ingest multiple files in sequence.
        
        Args:
            file_configs: List of file configuration dictionaries with keys:
                - file_path: Path to the file
                - timeframe: Timeframe (1h, 4h, 1d)
                - source_name: Data provider name
                - file_format: Format (csv or parquet)
                
        Returns:
            Dictionary with results for each file
        """
        results = {}
        
        for config in file_configs:
            file_path = config["file_path"]
            logger.info(f"Processing file: {file_path}")
            
            try:
                df, report = self.ingest_file(
                    file_path=config["file_path"],
                    timeframe=config["timeframe"],
                    source_name=config.get("source_name", "Unknown"),
                    file_format=config.get("file_format", "csv"),
                )
                
                results[file_path] = {
                    "status": "success",
                    "records": len(df),
                    "report": report,
                }
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results[file_path] = {
                    "status": "failed",
                    "error": str(e),
                }
        
        return results


def main() -> None:
    """Main entry point for running the ingestion pipeline."""
    setup_logging(level="INFO")
    
    logger.info("AurumMind AI - Data Ingestion Pipeline")
    logger.info("=" * 60)
    
    # Example usage
    pipeline = IngestionPipeline()
    
    # Add your data files here
    # Example:
    # file_configs = [
    #     {
    #         "file_path": "data/raw/XAUUSD_1h.csv",
    #         "timeframe": "1h",
    #         "source_name": "Dukascopy",
    #         "file_format": "csv",
    #     },
    # ]
    # results = pipeline.ingest_multiple_files(file_configs)
    
    logger.info("Ingestion pipeline ready. Configure files to process.")


if __name__ == "__main__":
    main()
