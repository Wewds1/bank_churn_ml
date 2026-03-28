import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import pandas as pd
from src.data_prep import parse_income, check_for_leakage, check_duplicates_by_id



class TestParseIncome:
    """Tests for parse_income function."""
    
    def test_numeric_input(self):
        """Test parsing numeric input."""
        assert parse_income(45000) == 45000.0
        assert parse_income(45000.5) == 45000.5
    
    def test_thousands_suffix(self):
        """Test parsing K suffix."""
        assert parse_income("45k") == 45000.0
        assert parse_income("45K") == 45000.0
        assert parse_income("45.5K") == 45500.0
    
    def test_millions_suffix(self):
        """Test parsing M suffix."""
        assert parse_income("2M") == 2_000_000.0
        assert parse_income("1.5M") == 1_500_000.0
    
    def test_currency_symbols(self):
        """Test parsing with currency symbols."""
        assert parse_income("$45,000") == 45000.0
        assert parse_income("PHP 45k") == 45000.0
    
    def test_missing_values(self):
        """Test handling of missing values."""
        assert pd.isna(parse_income(np.nan))
        assert pd.isna(parse_income(None))
    
    def test_unparseable(self):
        """Test handling of unparseable values."""
        assert pd.isna(parse_income("invalid text"))
        assert pd.isna(parse_income(""))


class TestLeakageDetection:
    """Tests for leakage detection."""
    
    def test_perfect_leak_column(self):
        """Test detection of perfect leakage."""
        df = pd.DataFrame({
            'churned': [0, 0, 1, 1],
            'leak_col': [np.nan, np.nan, 'value1', 'value2']
        })
        
        results = check_for_leakage(df, 'churned', ['leak_col'])
        assert results['leak_col']['is_suspicious'] == True
    
    def test_non_leak_column(self):
        """Test that non-leak columns pass."""
        df = pd.DataFrame({
            'churned': [0, 0, 1, 1],
            'normal_col': [1, 2, 3, 4]
        })
        
        results = check_for_leakage(df, 'churned', ['normal_col'])
        assert results['normal_col']['is_suspicious'] == False


class TestDuplicateDetection:
    """Tests for duplicate detection."""
    
    def test_no_duplicates(self):
        """Test when no duplicates exist."""
        df = pd.DataFrame({'customer_id': ['A', 'B', 'C']})
        assert check_duplicates_by_id(df) == 0
    
    def test_with_duplicates(self):
        """Test when duplicates exist."""
        df = pd.DataFrame({'customer_id': ['A', 'A', 'B', 'B']})
        assert check_duplicates_by_id(df) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])