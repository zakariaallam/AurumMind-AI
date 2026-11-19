"""Generate comprehensive EDA report for XAUUSD data.

This script loads market data and generates a complete exploratory
data analysis report with statistics and visualizations.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from aurummind_ai.market_analysis.statistics import MarketStatistics
from aurummind_ai.market_analysis.visualization import MarketVisualizer
from aurummind_ai.utils.logging_config import setup_logging

setup_logging(level="INFO")


def generate_eda_report(
    data_path: str,
    output_dir: str = "reports/eda",
    timeframe: str = "1h",
) -> None:
    """Generate comprehensive EDA report.
    
    Args:
        data_path: Path to the CSV data file
        output_dir: Directory to save reports
        timeframe: Data timeframe (1h, 4h, 1d)
    """
    print("=" * 70)
    print("AurumMind AI - EDA Report Generation")
    print("=" * 70)
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"✓ Loaded {len(df)} records")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()
    
    # Generate comprehensive statistics
    print("Generating statistical analysis...")
    summary = MarketStatistics.calculate_summary_report(df)
    
    # Add additional analyses
    df["returns"] = MarketStatistics.calculate_returns(df, method="simple")
    df["log_returns"] = MarketStatistics.calculate_returns(df, method="log")
    df["volatility_20"] = MarketStatistics.calculate_volatility(df, window=20)
    df_with_candles = MarketStatistics.calculate_candle_features(df)
    
    # Detect outliers
    outliers = MarketStatistics.detect_outliers(
        df["returns"].dropna(), method="iqr", threshold=3.0
    )
    summary["outlier_analysis"] = {
        "total_outliers": int(outliers.sum()),
        "outlier_percentage": float(outliers.sum() / len(outliers) * 100),
    }
    
    print(f"✓ Statistical analysis complete")
    print()
    
    # Save summary report as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"eda_report_XAUUSD_{timeframe}_{timestamp}.json"
    report_path = output_path / report_filename
    
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary report: {report_path}")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    viz_dir = output_path / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Candlestick chart
    print("  - Candlestick chart")
    fig_candle = MarketVisualizer.plot_candlestick(
        df, title=f"XAUUSD {timeframe.upper()} - Candlestick Chart", show_volume=True
    )
    fig_candle.write_html(viz_dir / f"candlestick_{timeframe}.html")
    
    # 2. Price distribution
    print("  - Price distribution")
    fig_price_dist = MarketVisualizer.plot_price_distribution(df, price_col="close")
    fig_price_dist.write_html(viz_dir / f"price_distribution_{timeframe}.html")
    
    # 3. Returns distribution
    print("  - Returns distribution")
    fig_returns_dist = MarketVisualizer.plot_returns_distribution(df, price_col="close")
    fig_returns_dist.write_html(viz_dir / f"returns_distribution_{timeframe}.html")
    
    # 4. Volatility analysis
    print("  - Volatility analysis")
    fig_vol = MarketVisualizer.plot_volatility(df, window=20)
    fig_vol.write_html(viz_dir / f"volatility_{timeframe}.html")
    
    # 5. Price with MAs
    print("  - Price with moving averages")
    fig_ma = MarketVisualizer.plot_price_with_ma(df, ma_periods=[20, 50])
    fig_ma.write_html(viz_dir / f"price_with_ma_{timeframe}.html")
    
    # 6. Volume analysis
    print("  - Volume analysis")
    fig_volume = MarketVisualizer.plot_volume_analysis(df)
    fig_volume.write_html(viz_dir / f"volume_analysis_{timeframe}.html")
    
    # 7. Intraday patterns (only for hourly data)
    if timeframe == "1h":
        print("  - Intraday patterns")
        fig_intraday = MarketVisualizer.plot_intraday_patterns(df)
        fig_intraday.write_html(viz_dir / f"intraday_patterns_{timeframe}.html")
    
    print(f"✓ Visualizations saved to: {viz_dir}")
    print()
    
    # Print summary to console
    print("=" * 70)
    print("EDA SUMMARY")
    print("=" * 70)
    print()
    print(f"Dataset: XAUUSD {timeframe.upper()}")
    print(f"Records: {summary['general']['total_records']}")
    print(f"Period: {summary['general']['date_range']['start']} to {summary['general']['date_range']['end']}")
    print(f"Duration: {summary['general']['date_range']['duration_days']} days")
    print()
    
    print("-" * 70)
    print("PRICE STATISTICS (Close)")
    print("-" * 70)
    close_stats = summary["price_statistics"]["close"]
    print(f"Mean:     ${close_stats['mean']:.2f}")
    print(f"Std Dev:  ${close_stats['std']:.2f}")
    print(f"Range:    ${close_stats['min']:.2f} - ${close_stats['max']:.2f}")
    print(f"Skewness: {close_stats['skew']:.4f}")
    print(f"Kurtosis: {close_stats['kurtosis']:.4f}")
    print()
    
    print("-" * 70)
    print("RETURN STATISTICS")
    print("-" * 70)
    ret_stats = summary["return_statistics"]
    print(f"Mean Return:      {ret_stats['mean']*100:.4f}%")
    print(f"Volatility:       {ret_stats['std']*100:.4f}%")
    print(f"Positive Returns: {ret_stats['positive_returns_pct']:.2f}%")
    print(f"Negative Returns: {ret_stats['negative_returns_pct']:.2f}%")
    print(f"Skewness:         {ret_stats['skew']:.4f}")
    print(f"Kurtosis:         {ret_stats['kurtosis']:.4f}")
    print()
    
    print("-" * 70)
    print("CANDLE STATISTICS")
    print("-" * 70)
    candle_stats = summary["candle_statistics"]
    print(f"Avg Body:         ${candle_stats['avg_body']:.2f}")
    print(f"Avg Upper Wick:   ${candle_stats['avg_upper_wick']:.2f}")
    print(f"Avg Lower Wick:   ${candle_stats['avg_lower_wick']:.2f}")
    print(f"Avg Range:        ${candle_stats['avg_range']:.2f}")
    print(f"Bullish Candles:  {candle_stats['bullish_pct']:.2f}%")
    print()
    
    print("-" * 70)
    print("OUTLIER ANALYSIS")
    print("-" * 70)
    outlier_stats = summary["outlier_analysis"]
    print(f"Total Outliers:   {outlier_stats['total_outliers']}")
    print(f"Outlier %:        {outlier_stats['outlier_percentage']:.2f}%")
    print()
    
    print("=" * 70)
    print("✓ EDA Report Generation Complete")
    print("=" * 70)
    print()
    print(f"Reports saved to: {output_path}")
    print()
    print("Generated files:")
    print(f"  - {report_filename}")
    print(f"  - visualizations/*.html")


def main():
    """Main entry point."""
    # Check for sample data files
    sample_files = {
        "1h": "data/raw/samples/XAUUSD_1h_sample.csv",
        "4h": "data/raw/samples/XAUUSD_4h_sample.csv",
        "1d": "data/raw/samples/XAUUSD_1d_sample.csv",
    }
    
    for timeframe, file_path in sample_files.items():
        if Path(file_path).exists():
            print(f"\nProcessing {timeframe} data...")
            print()
            generate_eda_report(
                data_path=file_path,
                output_dir=f"reports/eda/{timeframe}",
                timeframe=timeframe,
            )
        else:
            print(f"⚠ File not found: {file_path}")
    
    print()
    print("=" * 70)
    print("All EDA reports generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
