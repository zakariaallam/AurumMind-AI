"""Test the data ingestion pipeline with sample data.

Run this script after generating sample data with generate_sample_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aurummind_ai.ingestion.pipeline import IngestionPipeline
from aurummind_ai.utils.logging_config import setup_logging


def main():
    """Test ingestion pipeline with sample data."""
    setup_logging(level="INFO")
    
    print("AurumMind AI - Testing Data Ingestion Pipeline")
    print("=" * 70)
    print()
    
    # Initialize pipeline
    pipeline = IngestionPipeline(config_path="configs/data.yaml")
    
    # Define sample files to ingest
    sample_files = [
        {
            "file_path": "data/raw/samples/XAUUSD_1h_sample.csv",
            "timeframe": "1h",
            "source_name": "SampleData",
            "file_format": "csv",
        },
        {
            "file_path": "data/raw/samples/XAUUSD_4h_sample.csv",
            "timeframe": "4h",
            "source_name": "SampleData",
            "file_format": "csv",
        },
        {
            "file_path": "data/raw/samples/XAUUSD_1d_sample.csv",
            "timeframe": "1d",
            "source_name": "SampleData",
            "file_format": "csv",
        },
    ]
    
    # Check if sample files exist
    existing_files = []
    for config in sample_files:
        if Path(config["file_path"]).exists():
            existing_files.append(config)
        else:
            print(f"⚠ File not found: {config['file_path']}")
    
    if not existing_files:
        print()
        print("No sample files found. Please run:")
        print("  python scripts/generate_sample_data.py")
        print()
        return
    
    print(f"Found {len(existing_files)} sample files to process")
    print()
    
    # Process files
    results = pipeline.ingest_multiple_files(existing_files)
    
    # Print summary
    print()
    print("=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print()
    
    for file_path, result in results.items():
        status = result["status"]
        symbol = "✓" if status == "success" else "✗"
        
        print(f"{symbol} {Path(file_path).name}")
        
        if status == "success":
            records = result["records"]
            quality_score = result["report"]["stages"]["validation"]["quality_score"]
            print(f"  Records: {records}")
            print(f"  Quality Score: {quality_score:.2f}/100")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
        
        print()
    
    # Test with problematic data
    print("=" * 70)
    print("TESTING WITH PROBLEMATIC DATA")
    print("=" * 70)
    print()
    
    problematic_file = "data/raw/samples/XAUUSD_1h_with_issues.csv"
    
    if Path(problematic_file).exists():
        print(f"Processing: {problematic_file}")
        print("(This file contains intentional quality issues)")
        print()
        
        try:
            df, report = pipeline.ingest_file(
                file_path=problematic_file,
                timeframe="1h",
                source_name="SampleDataWithIssues",
                file_format="csv",
            )
            print("✓ File processed successfully (unexpected)")
        except ValueError as e:
            print("✗ Validation failed as expected")
            print(f"  Reason: {e}")
        print()
    
    print("=" * 70)
    print("Testing complete!")
    print()
    print("Check data/raw/ directory for ingested data and quality reports.")


if __name__ == "__main__":
    main()
