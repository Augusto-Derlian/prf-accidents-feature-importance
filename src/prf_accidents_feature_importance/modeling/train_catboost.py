"""Treina e avalia o classificador CatBoost de severidade dos acidentes."""

import json
import logging
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from prf_accidents_feature_importance.config import (
    DIRETORIO_DADOS_PROCESSADOS,
    DIRETORIO_MODELOS,
)
from prf_accidents_feature_importance.features.build_features import (
    COLUNA_ALVO,
    COLUNAS_CATEGORICAS,
    COLUNAS_FEATURES,
)

logger = logging.getLogger(__name__)

ARQUIVO_ENTRADA = DIRETORIO_DADOS_PROCESSADOS / "acidentes_features.parquet"
DIRETORIO_SAIDA = DIRETORIO_MODELOS / "catboost"
SEMENTE = 42


def carregar_dados(caminho: Path = ARQUIVO_ENTRADA) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega a matriz de features e valida o contrato esperado pelo modelo."""
    dados = pd.read_parquet(caminho)
    colunas_ausentes = {*COLUNAS_FEATURES, COLUNA_ALVO} - set(dados.columns)
    if colunas_ausentes:
        raise ValueError(
            "Colunas obrigatórias ausentes: " + ", ".join(sorted(colunas_ausentes))
        )

    dados = dados.dropna(subset=[COLUNA_ALVO]).copy()
    features = dados[COLUNAS_FEATURES].copy()

    # O CatBoost exige strings (e não valores ausentes) nas colunas categóricas.
    for coluna in COLUNAS_CATEGORICAS:
        features[coluna] = features[coluna].fillna("desconhecido").astype(str)

    alvo = dados[COLUNA_ALVO].astype(str)
    if alvo.nunique() < 2:
        raise ValueError("O alvo precisa conter pelo menos duas classes.")
    return features, alvo


def dividir_dados(
    features: pd.DataFrame,
    alvo: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Cria partições estratificadas de treino (80%), validação (10%) e teste (10%)."""
    x_treino, x_temporario, y_treino, y_temporario = train_test_split(
        features,
        alvo,
        test_size=0.20,
        random_state=SEMENTE,
        stratify=alvo,
    )
    x_validacao, x_teste, y_validacao, y_teste = train_test_split(
        x_temporario,
        y_temporario,
        test_size=0.50,
        random_state=SEMENTE,
        stratify=y_temporario,
    )
    return x_treino, x_validacao, x_teste, y_treino, y_validacao, y_teste


def calcular_metricas(y_real: pd.Series, y_predito: object) -> dict[str, object]:
    """Calcula métricas adequadas ao desbalanceamento do problema multiclasse."""
    classes = sorted(y_real.unique().tolist())
    return {
        "accuracy": accuracy_score(y_real, y_predito),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_predito),
        "f1_macro": f1_score(y_real, y_predito, average="macro"),
        "f1_weighted": f1_score(y_real, y_predito, average="weighted"),
        "classes": classes,
        "confusion_matrix": confusion_matrix(
            y_real, y_predito, labels=classes
        ).tolist(),
        "classification_report": classification_report(
            y_real,
            y_predito,
            labels=classes,
            output_dict=True,
            zero_division=0,
        ),
    }


def treinar_modelo(
    caminho_entrada: Path = ARQUIVO_ENTRADA,
    diretorio_saida: Path = DIRETORIO_SAIDA,
    iteracoes: int = 1_000,
) -> dict[str, object]:
    """Treina o CatBoost, avalia no teste e persiste modelo e resultados."""
    features, alvo = carregar_dados(caminho_entrada)
    (
        x_treino,
        x_validacao,
        x_teste,
        y_treino,
        y_validacao,
        y_teste,
    ) = dividir_dados(features, alvo)

    diretorio_saida.mkdir(parents=True, exist_ok=True)
    treino = Pool(x_treino, y_treino, cat_features=COLUNAS_CATEGORICAS)
    validacao = Pool(x_validacao, y_validacao, cat_features=COLUNAS_CATEGORICAS)

    modelo = CatBoostClassifier(
        iterations=iteracoes,
        depth=8,
        learning_rate=0.08,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        auto_class_weights="Balanced",
        random_seed=SEMENTE,
        early_stopping_rounds=75,
        # Evita uma limitação do backend nativo com caminhos Windows acentuados.
        allow_writing_files=False,
        verbose=50,
    )
    modelo.fit(treino, eval_set=validacao, use_best_model=True)

    predicoes = modelo.predict(x_teste).reshape(-1)
    metricas = calcular_metricas(y_teste, predicoes)
    metricas["dataset"] = {
        "entrada": str(caminho_entrada),
        "registros": len(features),
        "treino": len(x_treino),
        "validacao": len(x_validacao),
        "teste": len(x_teste),
        "distribuicao_alvo": alvo.value_counts().to_dict(),
    }
    metricas["modelo"] = {
        "algoritmo": "CatBoostClassifier",
        "random_seed": SEMENTE,
        "best_iteration": modelo.get_best_iteration(),
        "auto_class_weights": "Balanced",
    }

    modelo.save_model(diretorio_saida / "modelo.cbm")
    importancia = pd.DataFrame(
        {
            "feature": modelo.feature_names_,
            "importance": modelo.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    importancia.to_csv(diretorio_saida / "feature_importance.csv", index=False)
    with (diretorio_saida / "metrics.json").open("w", encoding="utf-8") as arquivo:
        json.dump(metricas, arquivo, ensure_ascii=False, indent=2)

    logger.info("Modelo salvo em: %s", diretorio_saida / "modelo.cbm")
    logger.info("F1 macro no teste: %.4f", metricas["f1_macro"])
    logger.info("Acurácia balanceada no teste: %.4f", metricas["balanced_accuracy"])
    return metricas


def main() -> None:
    """Executa o treinamento com a configuração padrão do projeto."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=== Treinamento do CatBoost ===")
    treinar_modelo()


if __name__ == "__main__":
    main()
