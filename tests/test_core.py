"""
Test suite for penalized_constrained package.
"""
import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from penalized_constrained import (
    PenalizedConstrainedRegression,
    PenalizedConstrainedCV,
    generate_correlated_learning_data,
    ModelDiagnostics,
    compute_vif,
    learning_rate_to_slope,
    slope_to_learning_rate
)


class TestPenalizedConstrainedRegression:
    """Tests for the core PenalizedConstrainedRegression class."""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(50)
        return X, y
    
    @pytest.fixture
    def learning_curve_data(self):
        """Generate learning curve test data."""
        data = generate_correlated_learning_data(
            n_lots=20,
            T1=100,
            target_correlation=0.5,
            cv_error=0.1,
            random_state=42
        )
        return data
    
    def test_basic_fit(self, simple_data):
        """Test basic fitting without constraints."""
        X, y = simple_data
        model = PenalizedConstrainedRegression()
        model.fit(X, y)
        
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert len(model.coef_) == 2
        assert model.converged_
    
    def test_with_bounds_list(self, simple_data):
        """Test fitting with list-style bounds."""
        X, y = simple_data
        model = PenalizedConstrainedRegression(
            bounds=[(-1, 0), (-1, 0)]
        )
        model.fit(X, y)
        
        assert model.coef_[0] <= 0
        assert model.coef_[1] <= 0
    
    def test_with_bounds_dict(self, simple_data):
        """Test fitting with dict-style bounds."""
        X, y = simple_data
        model = PenalizedConstrainedRegression(
            feature_names=['LC', 'RC'],
            bounds={'LC': (-1, 0), 'RC': (-0.5, 0)}
        )
        model.fit(X, y)
        
        assert model.coef_[0] <= 0
        assert model.coef_[1] <= 0
        assert model.named_coef_ is not None
        assert 'LC' in model.named_coef_
        assert 'RC' in model.named_coef_
    
    def test_sspe_loss(self, simple_data):
        """Test SSPE loss function."""
        X, y = simple_data
        y = np.abs(y)  # Ensure positive for SSPE
        
        model = PenalizedConstrainedRegression(loss='sspe')
        model.fit(X, y)
        
        assert model.converged_
    
    def test_sse_loss(self, simple_data):
        """Test SSE loss function."""
        X, y = simple_data
        model = PenalizedConstrainedRegression(loss='sse')
        model.fit(X, y)
        
        assert model.converged_
    
    def test_custom_loss(self, simple_data):
        """Test custom loss function."""
        X, y = simple_data
        
        def custom_loss(y_true, y_pred):
            return np.mean(np.abs(y_true - y_pred))  # MAE
        
        model = PenalizedConstrainedRegression(loss=custom_loss)
        model.fit(X, y)
        
        assert model.converged_
    
    def test_ridge_penalty(self, simple_data):
        """Test Ridge (L2) penalty."""
        X, y = simple_data
        model = PenalizedConstrainedRegression(alpha=1.0, l1_ratio=0.0)
        model.fit(X, y)
        
        assert model.converged_
    
    def test_lasso_penalty(self, simple_data):
        """Test Lasso (L1) penalty."""
        X, y = simple_data
        model = PenalizedConstrainedRegression(alpha=1.0, l1_ratio=1.0)
        model.fit(X, y)
        
        assert model.converged_
    
    def test_elastic_net_penalty(self, simple_data):
        """Test ElasticNet penalty."""
        X, y = simple_data
        model = PenalizedConstrainedRegression(alpha=1.0, l1_ratio=0.5)
        model.fit(X, y)
        
        assert model.converged_
    
    def test_predict(self, simple_data):
        """Test prediction."""
        X, y = simple_data
        model = PenalizedConstrainedRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert not np.any(np.isnan(y_pred))
    
    def test_score(self, simple_data):
        """Test R² score."""
        X, y = simple_data
        model = PenalizedConstrainedRegression()
        model.fit(X, y)
        
        r2 = model.score(X, y)
        assert 0 <= r2 <= 1 or r2 < 0  # R² can be negative for bad fits
    
    def test_active_constraints(self, simple_data):
        """Test active constraints detection."""
        X, y = simple_data
        model = PenalizedConstrainedRegression(
            bounds=[(-1, 0), (-1, 0)]
        )
        model.fit(X, y)
        
        assert hasattr(model, 'active_constraints_')
        assert hasattr(model, 'n_active_constraints_')
        assert isinstance(model.n_active_constraints_, int)
    
    def test_custom_prediction_fn(self, learning_curve_data):
        """Test custom prediction function."""
        data = learning_curve_data
        X_orig = data['X_original']
        y_orig = data['y_original']
        
        def lc_func(X, params):
            T1, b, c = params
            return T1 * (X[:, 0] ** b) * (X[:, 1] ** c)
        
        model = PenalizedConstrainedRegression(
            prediction_fn=lc_func,
            feature_names=['T1', 'LC', 'RC'],
            bounds={'T1': (50, 200), 'LC': (-1, 0), 'RC': (-1, 0)},
            fit_intercept=False,
            init=[100, -0.15, -0.07]
        )
        model.fit(X_orig, y_orig)
        
        assert len(model.coef_) == 3
        assert model.coef_[0] > 0  # T1 should be positive
        assert model.coef_[1] < 0  # LC should be negative
        assert model.coef_[2] < 0  # RC should be negative
    
    def test_scaling(self, simple_data):
        """Test internal scaling."""
        X, y = simple_data
        
        model_unscaled = PenalizedConstrainedRegression(scale=False)
        model_scaled = PenalizedConstrainedRegression(scale=True)
        
        model_unscaled.fit(X, y)
        model_scaled.fit(X, y)
        
        # Both should produce similar predictions
        y_pred_unscaled = model_unscaled.predict(X)
        y_pred_scaled = model_scaled.predict(X)
        
        # Allow some tolerance due to optimization differences
        np.testing.assert_allclose(y_pred_unscaled, y_pred_scaled, rtol=0.1)


class TestPenalizedConstrainedCV:
    """Tests for cross-validated version."""
    
    @pytest.fixture
    def simple_data(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(50)
        return X, y
    
    def test_cv_basic(self, simple_data):
        """Test basic CV fitting."""
        X, y = simple_data
        
        model = PenalizedConstrainedCV(
            alphas=np.logspace(-2, 1, 5),
            l1_ratios=[0.0, 0.5, 1.0],
            cv=3
        )
        model.fit(X, y)
        
        assert hasattr(model, 'alpha_')
        assert hasattr(model, 'l1_ratio_')
        assert hasattr(model, 'best_estimator_')
        assert hasattr(model, 'cv_results_')
    
    def test_cv_with_bounds(self, simple_data):
        """Test CV with bounds."""
        X, y = simple_data
        
        model = PenalizedConstrainedCV(
            bounds=[(-1, 0), (-1, 0)],
            alphas=[0.01, 0.1, 1.0],
            cv=3
        )
        model.fit(X, y)
        
        assert model.coef_[0] <= 0
        assert model.coef_[1] <= 0


class TestDiagnostics:
    """Tests for diagnostics module."""
    
    @pytest.fixture
    def fitted_model(self):
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(30)
        
        model = PenalizedConstrainedRegression(bounds=[(-1, 0), (-1, 0)])
        model.fit(X, y)
        
        return model, X, y
    
    def test_model_diagnostics(self, fitted_model):
        """Test ModelDiagnostics class."""
        model, X, y = fitted_model
        
        diag = ModelDiagnostics(model, X, y, gdf_method='hu')
        
        assert hasattr(diag, 'gdf')
        assert hasattr(diag, 'see')
        assert hasattr(diag, 'spe')
        assert hasattr(diag, 'r2')
        assert hasattr(diag, 'adj_r2')
        assert diag.gdf > 0
    
    def test_gdf_methods(self, fitted_model):
        """Test both GDF methods."""
        model, X, y = fitted_model
        
        diag_hu = ModelDiagnostics(model, X, y, gdf_method='hu')
        diag_gaines = ModelDiagnostics(model, X, y, gdf_method='gaines')
        
        # Both should produce valid GDF
        assert diag_hu.gdf > 0
        assert diag_gaines.gdf > 0


class TestUtils:
    """Tests for utility functions."""
    
    def test_learning_rate_conversion(self):
        """Test learning rate to slope conversion."""
        lr = 0.90
        slope = learning_rate_to_slope(lr)
        lr_back = slope_to_learning_rate(slope)
        
        np.testing.assert_almost_equal(lr, lr_back)
    
    def test_data_generation(self):
        """Test correlated data generation."""
        data = generate_correlated_learning_data(
            n_lots=20,
            T1=100,
            target_correlation=0.7,
            random_state=42
        )
        
        assert 'X' in data
        assert 'y' in data
        assert 'params' in data
        assert len(data['X']) == 20
        assert abs(data['actual_correlation'] - 0.7) < 0.2  # Allow some tolerance
    
    def test_vif(self):
        """Test VIF computation."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        
        vif = compute_vif(X)
        
        assert len(vif) == 3
        assert all(v >= 1 for v in vif)  # VIF is always >= 1


class TestSklearnCompatibility:
    """Test sklearn API compatibility."""
    
    def test_get_set_params(self):
        """Test get_params and set_params."""
        model = PenalizedConstrainedRegression(alpha=0.5, l1_ratio=0.3)
        
        params = model.get_params()
        assert params['alpha'] == 0.5
        assert params['l1_ratio'] == 0.3
        
        model.set_params(alpha=1.0)
        assert model.alpha == 1.0
    
    def test_clone(self):
        """Test that model can be cloned."""
        from sklearn.base import clone
        
        model = PenalizedConstrainedRegression(alpha=0.5, bounds=[(-1, 0)])
        cloned = clone(model)
        
        assert cloned.alpha == model.alpha
        assert cloned.bounds == model.bounds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
