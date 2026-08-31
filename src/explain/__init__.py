"""SHAP Explainability modules for GlassBox."""

from .explainer import FraudExplainer, describe_feature_impact

__all__ = ["FraudExplainer", "describe_feature_impact"]
