"""Ponto de entrada para preparar os dados pelo terminal."""

from prf_accidents_feature_importance.pipelines.prepare_data import executar


def main() -> None:
    """Executa o fluxo completo de preparação dos dados."""
    executar()


if __name__ == "__main__":
    main()
