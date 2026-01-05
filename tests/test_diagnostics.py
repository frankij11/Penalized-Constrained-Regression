"""
Test suite for diagnostics and summary report functionality.
"""
import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from penalized_constrained import (
    PenalizedConstrainedRegression,
    ModelDiagnostics,
    generate_summary_report,
    SummaryReport,
    CoefficientInfo,
    FitStatistics,
    hessian_standard_errors,
    bootstrap_confidence_intervals,
)


class TestModelDiagnostics:
    """Tests for ModelDiagnostics class."""

    @pytest.fixture
    def fitted_model(self):
        """Generate a fitted model with test data."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(50)

        model = PenalizedConstrainedRegression(
            bounds=[(-1, 0), (-1, 0)],
            feature_names=['LC', 'RC'],
            loss='sspe'
        )
        model.fit(X, y)
        return model, X, y

    def test_basic_diagnostics(self, fitted_model):
        """Test basic diagnostic computation."""
        model, X, y = fitted_model
        diag = ModelDiagnostics(model, X, y)

        # Check all attributes exist
        assert hasattr(diag, 'gdf')
        assert hasattr(diag, 'see')
        assert hasattr(diag, 'spe')
        assert hasattr(diag, 'mape')
        assert hasattr(diag, 'rmse')
        assert hasattr(diag, 'cv')
        assert hasattr(diag, 'r2')
        assert hasattr(diag, 'adj_r2')

        # Check reasonable values
        assert diag.gdf > 0
        assert diag.see > 0
        assert diag.spe > 0
        assert 0 <= diag.r2 <= 1

    def test_gdf_hu_method(self, fitted_model):
        """Test Hu's GDF method."""
        model, X, y = fitted_model
        diag = ModelDiagnostics(model, X, y, gdf_method='hu')

        assert diag.gdf_method == 'hu'
        assert diag.gdf > 0

    def test_gdf_gaines_method(self, fitted_model):
        """Test Gaines' GDF method."""
        model, X, y = fitted_model
        diag = ModelDiagnostics(model, X, y, gdf_method='gaines')

        assert diag.gdf_method == 'gaines'
        assert diag.gdf > 0

    def test_to_dict(self, fitted_model):
        """Test dictionary export."""
        model, X, y = fitted_model
        diag = ModelDiagnostics(model, X, y)

        d = diag.to_dict()

        assert 'r2' in d
        assert 'adj_r2' in d
        assert 'gdf' in d
        assert 'model_type' in d


class TestSummaryReport:
    """Tests for SummaryReport functionality."""

    @pytest.fixture
    def fitted_model(self):
        """Generate a fitted model with test data."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(50)

        model = PenalizedConstrainedRegression(
            bounds=[(-1, 0), (-1, 0)],
            feature_names=['LC', 'RC'],
            loss='sspe'
        )
        model.fit(X, y)
        return model, X, y

    def test_basic_summary_report(self, fitted_model):
        """Test basic summary report generation."""
        model, X, y = fitted_model
        report = generate_summary_report(model, X, y)

        assert isinstance(report, SummaryReport)
        assert report.model_spec is not None
        assert report.data_summary is not None
        assert len(report.coefficients) == 2
        assert report.fit_stats is not None
        assert report.constraints is not None

    def test_summary_with_hessian_ci(self, fitted_model):
        """Test summary with Hessian-based confidence intervals (default)."""
        model, X, y = fitted_model
        report = generate_summary_report(model, X, y, ci_method='hessian')

        assert report.ci_method == 'hessian'
        # Check CIs are computed
        for coef in report.coefficients:
            if coef.ci_lower is not None:
                assert coef.ci_lower < coef.value
                assert coef.ci_upper > coef.value

    def test_summary_without_ci(self, fitted_model):
        """Test summary without confidence intervals."""
        model, X, y = fitted_model
        report = generate_summary_report(model, X, y, ci_method='none')

        assert report.ci_method == 'none'
        for coef in report.coefficients:
            assert coef.ci_lower is None
            assert coef.ci_upper is None

    def test_full_summary_report(self, fitted_model):
        """Test full summary report with residual analysis."""
        model, X, y = fitted_model
        report = generate_summary_report(model, X, y, full=True)

        assert report.residuals is not None
        assert report.residuals.mean is not None
        assert report.residuals.std is not None
        assert report.residuals.residuals is not None
        assert report.residuals.y_pred is not None

    def test_report_datetime(self, fitted_model):
        """Test report datetime is recorded."""
        model, X, y = fitted_model
        before = datetime.now()
        report = generate_summary_report(model, X, y)
        after = datetime.now()

        assert report.report_datetime is not None
        assert before <= report.report_datetime <= after

    def test_fit_datetime(self, fitted_model):
        """Test model fit datetime is captured."""
        model, X, y = fitted_model
        report = generate_summary_report(model, X, y)

        assert report.model_spec.fit_datetime is not None
        assert report.model_spec.fit_duration_seconds is not None
        assert report.model_spec.fit_duration_seconds > 0

    def test_coefficient_info(self, fitted_model):
        """Test coefficient information structure."""
        model, X, y = fitted_model
        report = generate_summary_report(model, X, y)

        for coef in report.coefficients:
            assert isinstance(coef, CoefficientInfo)
            assert coef.name in ['LC', 'RC']
            assert np.isfinite(coef.value)
            assert coef.lower_bound == -1.0
            assert coef.upper_bound == 0.0

    def test_intercept_info(self, fitted_model):
        """Test intercept information."""
        model, X, y = fitted_model
        report = generate_summary_report(model, X, y)

        assert report.intercept is not None
        assert report.intercept.name == 'Intercept'
        assert np.isfinite(report.intercept.value)

    def test_fit_statistics(self, fitted_model):
        """Test fit statistics structure."""
        model, X, y = fitted_model
        report = generate_summary_report(model, X, y)

        stats = report.fit_stats
        assert isinstance(stats, FitStatistics)
        assert 0 <= stats.r2 <= 1
        assert stats.see > 0
        assert stats.spe > 0
        assert stats.gdf > 0


class TestSummaryReportExport:
    """Tests for summary report export methods."""

    @pytest.fixture
    def report(self):
        """Generate a summary report for export tests."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(50)

        model = PenalizedConstrainedRegression(
            bounds=[(-1, 0), (-1, 0)],
            feature_names=['LC', 'RC'],
            loss='sspe'
        )
        model.fit(X, y)
        return generate_summary_report(model, X, y, full=True)

    def test_to_dict(self, report):
        """Test dictionary export."""
        d = report.to_dict()

        assert 'model_specification' in d
        assert 'data_summary' in d
        assert 'coefficients' in d
        assert 'fit_statistics' in d
        assert 'constraints' in d
        assert 'report_datetime' in d
        assert 'ci_method' in d

    def test_to_dataframe(self, report):
        """Test DataFrame export."""
        df = report.to_dataframe()

        assert len(df) == 3  # 2 coefficients + intercept
        assert 'Parameter' in df.columns
        assert 'Value' in df.columns
        assert 'Status' in df.columns

    def test_to_excel(self, report):
        """Test Excel export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_summary.xlsx')
            report.to_excel(filepath)

            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0

            # Verify sheets - use context manager to ensure file is closed
            with pd.ExcelFile(filepath) as xl:
                assert 'Coefficients' in xl.sheet_names
                assert 'Fit Statistics' in xl.sheet_names
                assert 'Model Specification' in xl.sheet_names

    def test_to_html(self, report):
        """Test HTML export."""
        # Test returning string
        html = report.to_html()
        assert isinstance(html, str)
        assert '<html>' in html
        assert 'Penalized-Constrained Regression Summary' in html

        # Test writing to file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_summary.html')
            report.to_html(filepath)

            assert os.path.exists(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            assert '<html>' in content

    def test_print_summary(self, report, capsys):
        """Test console print output."""
        report.print_summary()
        captured = capsys.readouterr()

        assert 'PENALIZED-CONSTRAINED REGRESSION SUMMARY' in captured.out
        assert 'Model Specification' in captured.out
        assert 'Coefficients' in captured.out
        assert 'Fit Statistics' in captured.out


class TestSummaryReportPlotting:
    """Tests for summary report plotting functionality."""

    @pytest.fixture
    def report_with_residuals(self):
        """Generate a full summary report for plotting tests."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(50)

        model = PenalizedConstrainedRegression(
            bounds=[(-1, 0), (-1, 0)],
            feature_names=['LC', 'RC'],
            loss='sspe'
        )
        model.fit(X, y)
        return generate_summary_report(model, X, y, full=True)

    def test_plot_diagnostics(self, report_with_residuals):
        """Test diagnostic plot generation."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig = report_with_residuals.plot_diagnostics()

        assert fig is not None
        assert len(fig.axes) == 4  # 2x2 subplot

        plt.close(fig)

    def test_plot_save(self, report_with_residuals):
        """Test saving diagnostic plots."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'diagnostics.png')
            fig = report_with_residuals.plot_diagnostics(save_path=filepath)

            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0

            plt.close(fig)

    def test_plot_requires_full(self):
        """Test that plotting requires full=True."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(50)

        model = PenalizedConstrainedRegression(bounds=[(-1, 0), (-1, 0)])
        model.fit(X, y)

        # Basic report without residuals
        report = generate_summary_report(model, X, y, full=False)

        with pytest.raises(ValueError, match="Residual data not available"):
            report.plot_diagnostics()


class TestAutoFeatureNames:
    """Tests for automatic feature name generation."""

    def test_auto_generated_names(self):
        """Test auto-generated feature names from numpy array."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.array([-0.15, -0.07, 0.05]) + 4.5

        model = PenalizedConstrainedRegression()
        model.fit(X, y)

        assert model.feature_names_in_ is not None
        assert list(model.feature_names_in_) == ['X1_coef', 'X2_coef', 'X3_coef']
        assert model.named_coef_ is not None

    def test_dataframe_column_names(self):
        """Test feature names from DataFrame columns."""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(50, 2),
            columns=['learning_rate', 'rate_change']
        )
        y = np.random.randn(50)

        model = PenalizedConstrainedRegression()
        model.fit(X, y)

        assert model.feature_names_in_ is not None
        assert list(model.feature_names_in_) == ['learning_rate', 'rate_change']

    def test_user_provided_names(self):
        """Test user-provided feature names."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        model = PenalizedConstrainedRegression(feature_names=['LC', 'RC'])
        model.fit(X, y)

        assert list(model.feature_names_in_) == ['LC', 'RC']

    def test_feature_names_in_report(self):
        """Test that feature names appear correctly in report."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        model = PenalizedConstrainedRegression(feature_names=['alpha', 'beta'])
        model.fit(X, y)

        report = generate_summary_report(model, X, y)

        coef_names = [c.name for c in report.coefficients]
        assert 'alpha' in coef_names
        assert 'beta' in coef_names


class TestHessianStandardErrors:
    """Tests for Hessian-based standard error computation."""

    def test_hessian_se_basic(self):
        """Test basic Hessian SE computation."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(50)

        model = PenalizedConstrainedRegression()
        model.fit(X, y)

        se = hessian_standard_errors(model, X, y)

        assert len(se) == 3  # 2 coefficients + intercept
        assert all(np.isfinite(se) | np.isnan(se))  # Can be NaN if Hessian not invertible


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_basic(self):
        """Test basic bootstrap CI computation."""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(30)

        result = bootstrap_confidence_intervals(
            PenalizedConstrainedRegression,
            X, y,
            n_bootstrap=50,  # Small number for speed
            random_state=42
        )

        assert 'coef_mean' in result
        assert 'coef_std' in result
        assert 'coef_ci_lower' in result
        assert 'coef_ci_upper' in result
        assert 'intercept_mean' in result
        assert 'intercept_ci' in result

    def test_bootstrap_in_report(self):
        """Test bootstrap CIs in summary report."""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(30)

        model = PenalizedConstrainedRegression()
        model.fit(X, y)

        report = generate_summary_report(
            model, X, y,
            bootstrap=True,
            n_bootstrap=50,
            random_state=42
        )

        assert report.ci_method == 'bootstrap'
        assert report.bootstrap_results is not None

        # Check CIs are present
        for coef in report.coefficients:
            assert coef.ci_lower is not None
            assert coef.ci_upper is not None


class TestModelDiagnosticsSummary:
    """Tests for ModelDiagnostics.summary() method."""

    def test_summary_returns_report(self):
        """Test that summary() returns a SummaryReport."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(50)

        model = PenalizedConstrainedRegression()
        model.fit(X, y)

        diag = ModelDiagnostics(model, X, y)
        report = diag.summary()

        assert isinstance(report, SummaryReport)

    def test_summary_with_full(self):
        """Test summary() with full=True."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(50)

        model = PenalizedConstrainedRegression()
        model.fit(X, y)

        diag = ModelDiagnostics(model, X, y)
        report = diag.summary(full=True)

        assert report.residuals is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
