"""Padroniza e concatena os arquivos anuais de volume de pedágio da ANTT."""
"""Download os CSVs da ANTT listados em references/SOURCES.md para data/01_raw/antt/pedagio_volume e renomeie como pedagio_volume_YYYY.csv."""

from pathlib import Path

import logging

import pandas as pd
from tqdm import tqdm

from prf_accidents_feature_importance.config import (
    DIRETORIO_DADOS_BRUTOS,
    DIRETORIO_DADOS_INTERMEDIARIOS,
)

logger = logging.getLogger(__name__)

ANO_INICIAL = 2023
ANO_FINAL = 2026

DIRETORIO_ENTRADA = DIRETORIO_DADOS_BRUTOS / "antt" / "pedagio_volume"
ARQUIVO_SAIDA = DIRETORIO_DADOS_INTERMEDIARIOS / "01_pedagio_volume_concat.parquet"

ALIAS_COLUNAS = {
    "categoria": "categoria_eixo",
}

COLUNAS_ESPERADAS = [
    "concessionaria",
    "mes_ano",
    "sentido",
    "praca",
    "tipo_de_cobranca",
    "categoria_eixo",
    "tipo_de_veiculo",
    "volume_total",
    "multiplicador_de_tarifa",
    "volume_veiculo_equivalente",
]

COLUNAS_TEXTO = [
    "concessionaria",
    "sentido",
    "praca",
    "tipo_de_cobranca",
    "categoria_eixo",
    "tipo_de_veiculo",
]

COLUNAS_NUMERICAS = [
    "volume_total",
    "multiplicador_de_tarifa",
    "volume_veiculo_equivalente",
]


def localizar_arquivos_pedagio() -> list[Path]:
    if not DIRETORIO_ENTRADA.exists():
        raise FileNotFoundError(
            f"Diretório de entrada ausente: {DIRETORIO_ENTRADA}"
        )

    arquivos = sorted(
        arquivo
        for arquivo in DIRETORIO_ENTRADA.rglob("pedagio_volume_*")
        if arquivo.is_file()
        and arquivo.stem.removeprefix("pedagio_volume_").isdigit()
        and ANO_INICIAL
        <= int(arquivo.stem.removeprefix("pedagio_volume_"))
        <= ANO_FINAL
    )

    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo pedagio_volume_* encontrado em {DIRETORIO_ENTRADA}."
        )

    return arquivos


def normalizar_colunas(dados: pd.DataFrame) -> pd.DataFrame:
    """Padroniza os nomes e seleciona as colunas usadas no projeto."""
    dados = dados.copy()
    dados.columns = (
        dados.columns.astype("string")
        .str.strip()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    dados = dados.rename(columns=ALIAS_COLUNAS)

    colunas_ausentes = set(COLUNAS_ESPERADAS) - set(dados.columns)
    if colunas_ausentes:
        raise ValueError(
            f"Colunas ausentes no volume de pedágio: {sorted(colunas_ausentes)}"
        )

    return dados[COLUNAS_ESPERADAS]


def normalizar_dados(dados: pd.DataFrame) -> pd.DataFrame:
    """Converte textos, datas e números para formatos consistentes."""
    dados = normalizar_colunas(dados)

    for coluna in COLUNAS_TEXTO:
        dados[coluna] = (
            dados[coluna]
            .astype("string")
            .str.strip()
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
            .str.lower()
            .replace({"": pd.NA, "n/i": "nao informado"})
            .astype("string")
        )

    dados["mes_ano"] = (
        pd.to_datetime(
            dados["mes_ano"].astype("string").str.strip(),
            format="mixed",
            dayfirst=True,
            errors="raise",
        )
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    dados[COLUNAS_NUMERICAS] = (
        dados[COLUNAS_NUMERICAS]
        .apply(
            lambda coluna: pd.to_numeric(
                coluna.astype("string").str.strip().str.replace(",", ".", regex=False),
                errors="raise",
            )
        )
        .astype("Float64")
    )

    return dados


def main() -> None:
    """Executa a concatenação do volume de pedágio."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=== Concatenação do volume de pedágio (ANTT) ===")

    arquivos = localizar_arquivos_pedagio()

    tabelas_normalizadas = []
    for arquivo in tqdm(
        arquivos,
        desc="Processando arquivos de pedágio",
        unit="arquivo",
    ):
        dados_arquivo = pd.read_csv(
            arquivo,
            sep=";",
            encoding="cp1252",
            dtype="string",
        )
        tabelas_normalizadas.append(normalizar_dados(dados_arquivo))

    dados_concatenados = pd.concat(
        tabelas_normalizadas,
        ignore_index=True,
        sort=False,
    )

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    dados_concatenados.to_parquet(ARQUIVO_SAIDA, index=False)

    logger.info(f"Arquivos processados: {len(arquivos):,}")
    logger.info(f"Registros gerados: {len(dados_concatenados):,}")
    logger.info(
        "Período encontrado: "
        f"{dados_concatenados['mes_ano'].min():%Y-%m} a "
        f"{dados_concatenados['mes_ano'].max():%Y-%m}"
    )
    logger.info(f"Arquivo de saída: {ARQUIVO_SAIDA}")
    logger.info("")


if __name__ == "__main__":
    main()
