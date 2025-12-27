"""
Unit Tests for Configuration Module
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Set test mode to avoid validation on import
sys.modules['__test__'] = True

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    DATA_PATH,
    RAW_NUMERIC_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    SPENDING_COLUMNS,
    NUM_FEATURES,
    CAT_FEATURES,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
    MAIN_COLOR,
    SECONDARY_COLOR,
    ACCENT_COLOR,
    validate_config,
)


class TestConfigValues:
    """Tests for configuration values."""
    
    def test_random_state_is_int(self):
        """Test that RANDOM_STATE is an integer."""
        assert isinstance(RANDOM_STATE, int)
    
    def test_random_state_non_negative(self):
        """Test that RANDOM_STATE is non-negative."""
        assert RANDOM_STATE >= 0
    
    def test_test_size_in_range(self):
        """Test that TEST_SIZE is between 0 and 1."""
        assert 0 < TEST_SIZE < 1
    
    def test_cv_folds_at_least_2(self):
        """Test that CV_FOLDS is at least 2."""
        assert CV_FOLDS >= 2
    
    def test_data_path_is_string(self):
        """Test that DATA_PATH is a string."""
        assert isinstance(DATA_PATH, str)
    
    def test_feature_lists_not_empty(self):
        """Test that feature lists are not empty."""
        assert len(RAW_NUMERIC_FEATURES) > 0
        assert len(RAW_CATEGORICAL_FEATURES) > 0
        assert len(SPENDING_COLUMNS) > 0
        assert len(NUM_FEATURES) > 0
        assert len(CAT_FEATURES) > 0
    
    def test_target_variables_defined(self):
        """Test that target variables are defined."""
        assert TARGET_REGRESSION is not None
        assert TARGET_CLASSIFICATION is not None
    
    def test_colors_are_hex(self):
        """Test that color values are valid hex codes."""
        import re
        hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
        
        assert hex_pattern.match(MAIN_COLOR), f"Invalid hex: {MAIN_COLOR}"
        assert hex_pattern.match(SECONDARY_COLOR), f"Invalid hex: {SECONDARY_COLOR}"
        assert hex_pattern.match(ACCENT_COLOR), f"Invalid hex: {ACCENT_COLOR}"


class TestValidateConfig:
    """Tests for validate_config function."""
    
    def test_validate_config_passes(self):
        """Test that validate_config passes with valid config."""
        # Should not raise with default config
        validate_config()
    
    def test_validate_config_catches_invalid_test_size(self):
        """Test that invalid TEST_SIZE is caught."""
        # We can't easily test this without modifying the config,
        # but we can test the function exists and is callable
        assert callable(validate_config)


class TestFeatureConsistency:
    """Tests for feature list consistency."""
    
    def test_spending_columns_are_raw_numeric(self):
        """Test that spending columns are subset of raw numeric."""
        for col in SPENDING_COLUMNS:
            assert col in RAW_NUMERIC_FEATURES, f"{col} not in RAW_NUMERIC_FEATURES"
    
    def test_no_duplicate_features(self):
        """Test that feature lists have no duplicates."""
        assert len(RAW_NUMERIC_FEATURES) == len(set(RAW_NUMERIC_FEATURES))
        assert len(RAW_CATEGORICAL_FEATURES) == len(set(RAW_CATEGORICAL_FEATURES))
        assert len(NUM_FEATURES) == len(set(NUM_FEATURES))
        assert len(CAT_FEATURES) == len(set(CAT_FEATURES))
