"""Ponto de entrada para construir as features pelo terminal."""

from prf_accidents_feature_importance.features.build_features import (
    main as construir_features,
)


def main() -> None:
    """Executa a construção do conjunto de features."""
    construir_features()


if __name__ == "__main__":
    main()
