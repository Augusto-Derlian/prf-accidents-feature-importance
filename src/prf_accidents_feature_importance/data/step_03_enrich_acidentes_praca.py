"""Adiciona aos acidentes o volume da praça de pedágio mais próxima."""

import logging

import pandas as pd

from prf_accidents_feature_importance.config import DIRETORIO_DADOS_INTERMEDIARIOS

logger = logging.getLogger(__name__)

ARQUIVO_ACIDENTES = (
    DIRETORIO_DADOS_INTERMEDIARIOS / "01_acidentes_ocorrencia_concat.parquet"
)
ARQUIVO_PEDAGIOS = DIRETORIO_DADOS_INTERMEDIARIOS / "02_pedagio_volume_enriched.parquet"
ARQUIVO_SAIDA = DIRETORIO_DADOS_INTERMEDIARIOS / "03_acidentes_praca_enriched.parquet"

COLUNAS_LOCALIZACAO = ["uf", "br", "km"]
COLUNAS_PRACA = ["uf", "br", "km_praca_mais_proxima"]


def criar_volume_mensal(pedagios: pd.DataFrame) -> pd.DataFrame:
    """Soma o volume de cada ponto de pedágio em cada mês."""
    pedagios_com_localizacao = pedagios.dropna(subset=COLUNAS_LOCALIZACAO)

    volume_mensal = (
        pedagios_com_localizacao.groupby(
            ["mes_ano", *COLUNAS_LOCALIZACAO],
            as_index=False,
        )
        .agg(volume_pedagio=("volume_total", "sum"))
        .rename(
            columns={
                "mes_ano": "mes_volume",
                "km": "km_praca_mais_proxima",
            }
        )
    )

    return volume_mensal


def encontrar_pracas_mais_proximas(
    acidentes: pd.DataFrame,
    pedagios: pd.DataFrame,
) -> pd.DataFrame:
    """Encontra o km de pedágio mais próximo para cada local de acidente."""
    locais_acidentes = acidentes[COLUNAS_LOCALIZACAO].dropna().drop_duplicates()

    locais_pracas = (
        pedagios[COLUNAS_LOCALIZACAO]
        .dropna()
        .drop_duplicates()
        .rename(columns={"km": "km_praca_mais_proxima"})
    )

    # O merge cria as combinações possíveis apenas na mesma UF e BR.
    candidatos = locais_acidentes.merge(
        locais_pracas,
        on=["uf", "br"],
        how="left",
    )
    candidatos["distancia_praca_km"] = (
        candidatos["km"] - candidatos["km_praca_mais_proxima"]
    ).abs()

    # Após ordenar pela distância, a primeira linha de cada local é a
    # praça mais próxima. Se não houver praça na UF e BR, o km fica vazio.
    pracas_mais_proximas = candidatos.sort_values(
        ["distancia_praca_km", "km_praca_mais_proxima"],
        na_position="last",
    ).drop_duplicates(COLUNAS_LOCALIZACAO)[
        COLUNAS_LOCALIZACAO + ["km_praca_mais_proxima"]
    ]

    return pracas_mais_proximas


def encontrar_volumes_mais_proximos(
    acidentes: pd.DataFrame,
    volume_mensal: pd.DataFrame,
) -> pd.DataFrame:
    """Busca o volume disponível no mês mais próximo de cada acidente."""
    colunas_consulta = ["mes_acidente", *COLUNAS_PRACA]
    consultas = acidentes[colunas_consulta].dropna().drop_duplicates()

    # Cada consulta é combinada somente com os meses da praça escolhida.
    candidatos = consultas.merge(
        volume_mensal,
        on=COLUNAS_PRACA,
        how="left",
    )
    candidatos["distancia_meses"] = (
        (candidatos["mes_acidente"].dt.year - candidatos["mes_volume"].dt.year) * 12
        + candidatos["mes_acidente"].dt.month
        - candidatos["mes_volume"].dt.month
    ).abs()

    # Em caso de empate, ordenar também pelo mês faz o mês anterior ser usado.
    volumes_mais_proximos = candidatos.sort_values(
        ["distancia_meses", "mes_volume"],
    ).drop_duplicates(colunas_consulta)[
        colunas_consulta + ["volume_pedagio"]
    ]

    return volumes_mais_proximos


def enriquecer_acidentes(
    acidentes: pd.DataFrame,
    pedagios: pd.DataFrame,
) -> pd.DataFrame:
    """Adiciona o volume do mês disponível mais próximo em cada praça."""
    colunas_originais = acidentes.columns.tolist()
    pracas_mais_proximas = encontrar_pracas_mais_proximas(
        acidentes,
        pedagios,
    )
    volume_mensal = criar_volume_mensal(pedagios)

    resultado = acidentes.merge(
        pracas_mais_proximas,
        on=COLUNAS_LOCALIZACAO,
        how="left",
        validate="many_to_one",
    )
    resultado["mes_acidente"] = (
        pd.to_datetime(resultado["data_inversa"]).dt.to_period("M").dt.to_timestamp()
    )
    volumes_mais_proximos = encontrar_volumes_mais_proximos(
        resultado,
        volume_mensal,
    )
    resultado = resultado.merge(
        volumes_mais_proximos,
        on=["mes_acidente", *COLUNAS_PRACA],
        how="left",
        validate="many_to_one",
    )

    return resultado[colunas_originais + ["volume_pedagio"]]


def main() -> None:
    """Executa o enriquecimento e salva o arquivo resultante."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    acidentes = pd.read_parquet(ARQUIVO_ACIDENTES)
    pedagios = pd.read_parquet(ARQUIVO_PEDAGIOS)

    resultado = enriquecer_acidentes(acidentes, pedagios)

    if len(resultado) != len(acidentes):
        raise RuntimeError("O enriquecimento alterou a quantidade de acidentes.")

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_parquet(ARQUIVO_SAIDA, index=False)

    quantidade_com_volume = int(resultado["volume_pedagio"].notna().sum())
    percentual_com_volume = (
        quantidade_com_volume / len(resultado) * 100 if len(resultado) else 0
    )

    logger.info("\n=== Enriquecimento dos dados de acidente com volume ===")
    logger.info("Acidentes processados: %s", len(resultado))
    logger.info(
        "Acidentes com volume de pedágio: %s (%.2f%%)",
        quantidade_com_volume,
        percentual_com_volume,
    )
    logger.info("Arquivo salvo em: %s", ARQUIVO_SAIDA)


if __name__ == "__main__":
    main()
