"""Data quality report structures."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    """Single validation issue found in the data."""
    
    severity: str = Field(..., description="Severity: ERROR, WARNING, INFO")
    category: str = Field(..., description="Category of the issue")
    message: str = Field(..., description="Description of the issue")
    count: int = Field(default=1, description="Number of occurrences")
    examples: list[Any] = Field(default_factory=list, description="Example values")


class DataQualityReport(BaseModel):
    """Comprehensive data quality report.
    
    Contains all validation results, statistics, and detected issues.
    """
    
    # Metadata
    report_timestamp: datetime
    instrument: str
    timeframe: str
    data_source: str
    
    # Data statistics
    total_records: int
    date_range_start: datetime | None = None
    date_range_end: datetime | None = None
    
    # Validation results
    issues: list[ValidationIssue] = Field(default_factory=list)
    
    # Detailed statistics
    duplicate_timestamps: int = 0
    missing_values: dict[str, int] = Field(default_factory=dict)
    invalid_ohlc_relationships: int = 0
    negative_prices: int = 0
    zero_prices: int = 0
    suspicious_price_jumps: int = 0
    missing_candles: int = 0
    
    # Price statistics
    price_statistics: dict[str, float] = Field(default_factory=dict)
    volume_statistics: dict[str, float] = Field(default_factory=dict)
    
    # Overall assessment
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall data quality score (0-100)",
    )
    passed_validation: bool = False
    
    def add_issue(
        self,
        severity: str,
        category: str,
        message: str,
        count: int = 1,
        examples: list[Any] | None = None,
    ) -> None:
        """Add a validation issue to the report.
        
        Args:
            severity: Severity level (ERROR, WARNING, INFO)
            category: Category of the issue
            message: Description of the issue
            count: Number of occurrences
            examples: Example values demonstrating the issue
        """
        issue = ValidationIssue(
            severity=severity,
            category=category,
            message=message,
            count=count,
            examples=examples or [],
        )
        self.issues.append(issue)
    
    def calculate_quality_score(self) -> float:
        """Calculate overall data quality score based on issues found.
        
        Returns:
            Quality score from 0 to 100
        """
        base_score = 100.0
        
        # Deduct points for each issue based on severity
        for issue in self.issues:
            if issue.severity == "ERROR":
                base_score -= min(20, issue.count * 0.1)
            elif issue.severity == "WARNING":
                base_score -= min(10, issue.count * 0.05)
        
        self.quality_score = max(0.0, base_score)
        return self.quality_score
    
    def has_errors(self) -> bool:
        """Check if report contains any ERROR level issues."""
        return any(issue.severity == "ERROR" for issue in self.issues)
    
    def get_summary(self) -> str:
        """Generate a human-readable summary of the report."""
        lines = [
            f"Data Quality Report for {self.instrument} ({self.timeframe})",
            f"Generated: {self.report_timestamp}",
            f"Data Source: {self.data_source}",
            f"Total Records: {self.total_records}",
            f"Date Range: {self.date_range_start} to {self.date_range_end}",
            f"Quality Score: {self.quality_score:.2f}/100",
            f"Validation Status: {'PASSED' if self.passed_validation else 'FAILED'}",
            "",
            "Issues Found:",
        ]
        
        if not self.issues:
            lines.append("  No issues detected")
        else:
            for issue in self.issues:
                lines.append(f"  [{issue.severity}] {issue.category}: {issue.message}")
                if issue.count > 1:
                    lines.append(f"    Count: {issue.count}")
        
        return "\n".join(lines)
