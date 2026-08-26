"""Treina e avalia o classificador CatBoost de severidade dos acidentes."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split

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
METRICA_VALIDACAO = "TotalF1:average=Macro;use_weights=false"

# O ParameterGrid gera 2 × 2 × 2 = 8 combinações automaticamente.
GRADE_PARAMETROS = {
    "depth": [6, 8],
    "learning_rate": [0.03, 0.06],
    "expoente_pesos": [0.50, 0.65],
}

# Estes parâmetros permanecem iguais em todos os treinamentos.
PARAMETROS_FIXOS = {
    "l2_leaf_reg": 8,
    "random_strength": 1.0,
    "bagging_temperature": 0.0,
}


def carregar_dados(caminho: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Lê o parquet e prepara as colunas categóricas para o CatBoost."""
    dados = pd.read_parquet(caminho)
    colunas_ausentes = {*COLUNAS_FEATURES, COLUNA_ALVO} - set(dados.columns)
    if colunas_ausentes:
        raise ValueError(
            "Colunas obrigatórias ausentes: " + ", ".join(sorted(colunas_ausentes))
        )

    dados = dados.dropna(subset=[COLUNA_ALVO]).copy()
    features = dados[COLUNAS_FEATURES].copy()
    for coluna in COLUNAS_CATEGORICAS:
        features[coluna] = features[coluna].fillna("desconhecido").astype(str)

    alvo = dados[COLUNA_ALVO].astype(str)
    if alvo.nunique() < 2:
        raise ValueError("O alvo precisa conter pelo menos duas classes.")
    return features, alvo


def calcular_metricas(y_real: pd.Series, y_predito: object) -> dict[str, object]:
    """Calcula as métricas e a matriz de confusão do modelo."""
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
    iteracoes: int = 800,
    grade_parametros: dict[str, list[float | int]] = GRADE_PARAMETROS,
) -> dict[str, object]:
    """Testa as configurações, avalia a melhor no teste e salva os artefatos."""
    features, alvo = carregar_dados(caminho_entrada)

    # 80% para treino, 10% para validação e 10% para teste.
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

    contagens = y_treino.value_counts()
    maior_classe = float(contagens.max())
    resultados = []
    melhor_f1 = -1.0
    melhor_modelo = None
    melhor_configuracao = None
    melhores_pesos = None
    configuracoes = list(ParameterGrid(grade_parametros))

    for numero, configuracao_original in enumerate(configuracoes, start=1):
        configuracao = configuracao_original.copy()
        expoente = float(configuracao.pop("expoente_pesos"))
        pesos = {
            classe: (maior_classe / float(quantidade)) ** expoente
            for classe, quantidade in contagens.items()
        }

        modelo = CatBoostClassifier(
            **PARAMETROS_FIXOS,
            **configuracao,
            iterations=iteracoes,
            loss_function="MultiClass",
            eval_metric=METRICA_VALIDACAO,
            class_weights=pesos,
            early_stopping_rounds=60,
            random_seed=SEMENTE,
            allow_writing_files=False,
            verbose=False,
        )
        modelo.fit(
            x_treino,
            y_treino,
            cat_features=COLUNAS_CATEGORICAS,
            eval_set=(x_validacao, y_validacao),
            use_best_model=True,
        )

        predicoes_validacao = modelo.predict(x_validacao).reshape(-1)
        f1_validacao = f1_score(
            y_validacao,
            predicoes_validacao,
            average="macro",
        )
        resultados.append(
            {
                "tentativa": numero,
                **configuracao,
                "expoente_pesos": expoente,
                "best_iteration": modelo.get_best_iteration(),
                "f1_macro_validacao": f1_validacao,
                "balanced_accuracy_validacao": balanced_accuracy_score(
                    y_validacao, predicoes_validacao
                ),
                "accuracy_validacao": accuracy_score(y_validacao, predicoes_validacao),
            }
        )
        logger.info(
            "Configuração %d/%d | F1 macro: %.4f",
            numero,
            len(configuracoes),
            f1_validacao,
        )

        if f1_validacao > melhor_f1:
            melhor_f1 = f1_validacao
            melhor_modelo = modelo
            melhor_configuracao = {
                **PARAMETROS_FIXOS,
                **configuracao,
                "expoente_pesos": expoente,
            }
            melhores_pesos = pesos

    if melhor_modelo is None or melhor_configuracao is None:
        raise ValueError("Informe pelo menos uma configuração para o treinamento.")

    # O conjunto de teste só é consultado depois de escolher o melhor modelo.
    predicoes_teste = melhor_modelo.predict(x_teste).reshape(-1)
    metricas = calcular_metricas(y_teste, predicoes_teste)
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
        "iterations": melhor_modelo.tree_count_,
        "validation_metric": METRICA_VALIDACAO,
        "class_weights": melhores_pesos,
        "best_validation_f1_macro": melhor_f1,
        "best_hyperparameters": melhor_configuracao,
        "search_trials": len(configuracoes),
    }

    diretorio_saida.mkdir(parents=True, exist_ok=True)
    melhor_modelo.save_model(diretorio_saida / "modelo.cbm")
    importancias = pd.DataFrame(
        {
            "feature": melhor_modelo.feature_names_,
            "importance": melhor_modelo.get_feature_importance(), # importâncias nativas do CatBoost, não valores SHAP
        }
    ).sort_values("importance", ascending=False)
    importancias.to_csv(
        diretorio_saida / "feature_importance.csv",
        index=False,
    )
    figura, eixo = plt.subplots(
        figsize=(10, max(5, len(importancias) * 0.45)),
    )
    grafico = importancias.sort_values("importance")
    eixo.barh(grafico["feature"], grafico["importance"], color="#2f6690")
    eixo.set(
        xlabel="Importância",
        ylabel="Feature",
        title="Importância das features",
    )
    eixo.grid(axis="x", alpha=0.25)
    figura.tight_layout()
    figura.savefig(diretorio_saida / "feature_importance.png", dpi=150)
    plt.close(figura)
    pd.DataFrame(resultados).sort_values(
        "f1_macro_validacao",
        ascending=False,
    ).to_csv(diretorio_saida / "tuning_results.csv", index=False)
    with (diretorio_saida / "metrics.json").open("w", encoding="utf-8") as arquivo:
        json.dump(metricas, arquivo, ensure_ascii=False, indent=2)

    figura, eixo = plt.subplots(figsize=(8, 6))
    matriz = metricas["confusion_matrix"]
    imagem = eixo.imshow(matriz, cmap="Blues")
    figura.colorbar(imagem, ax=eixo)
    eixo.set(
        xticks=range(len(metricas["classes"])),
        yticks=range(len(metricas["classes"])),
        xticklabels=metricas["classes"],
        yticklabels=metricas["classes"],
        xlabel="Classe predita",
        ylabel="Classe real",
        title="Matriz de confusão",
    )
    plt.setp(eixo.get_xticklabels(), rotation=30, ha="right")
    limite = max(max(linha) for linha in matriz)
    for linha in range(len(matriz)):
        for coluna in range(len(matriz[linha])):
            valor = matriz[linha][coluna]
            eixo.text(
                coluna,
                linha,
                valor,
                ha="center",
                va="center",
                color="white" if valor > limite / 2 else "black",
            )
    figura.tight_layout()
    figura.savefig(diretorio_saida / "confusion_matrix.png", dpi=150)
    plt.close(figura)

    logger.info("Modelo salvo em: %s", diretorio_saida / "modelo.cbm")
    logger.info("F1 macro no teste: %.4f", metricas["f1_macro"])
    return metricas


def main() -> None:
    """Executa o treinamento com as configurações definidas acima."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=== Treinamento do CatBoost ===")
    treinar_modelo()


if __name__ == "__main__":
    main()
