"""Padroniza e concatena os arquivos anuais de acidentes da PRF."""

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
DATA_CORTE = pd.Timestamp("2023-11-01")

DIRETORIO_ENTRADA = DIRETORIO_DADOS_BRUTOS / "prf" / "acidentes_ocorrencia"
ARQUIVO_SAIDA = (
    DIRETORIO_DADOS_INTERMEDIARIOS
    / "01_acidentes_ocorrencia_concat.parquet"
)

ALIAS_COLUNAS = {
    "condicao_metereologica": "condicao_meteorologica",
}

COLUNAS_ESPERADAS = [
    "id",
    "data_inversa",
    "dia_semana",
    "horario",
    "uf",
    "br",
    "km",
    "municipio",
    "causa_acidente",
    "tipo_acidente",
    "classificacao_acidente",
    "fase_dia",
    "sentido_via",
    "condicao_meteorologica",
    "tipo_pista",
    "tracado_via",
    "uso_solo",
    "pessoas",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "feridos",
    "ilesos",
    "ignorados",
    "veiculos",
    "latitude",
    "longitude",
    "regional",
    "delegacia",
    "uop",
]

COLUNAS_INTEIRAS = [
    "id",
    "br",
    "pessoas",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "feridos",
    "ilesos",
    "ignorados",
    "veiculos",
]

COLUNAS_DECIMAIS = ["km", "latitude", "longitude"]


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
        raise ValueError(f"Colunas ausentes nos acidentes: {sorted(colunas_ausentes)}")

    return dados[COLUNAS_ESPERADAS]


def normalizar_dados(dados: pd.DataFrame) -> pd.DataFrame:
    """Converte textos, datas e números para formatos consistentes."""
    dados = normalizar_colunas(dados)
    colunas_texto = dados.columns.difference(
        COLUNAS_INTEIRAS + COLUNAS_DECIMAIS + ["data_inversa", "horario"]
    )

    for coluna in colunas_texto:
        valores = (
            dados[coluna].astype("string").str.strip().replace({"": pd.NA, "NA": pd.NA})
        )
        dados[coluna] = (
            valores.str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
            .str.lower()
            .astype("string")
        )

    dados["uf"] = dados["uf"].str.upper()
    dados["data_inversa"] = pd.to_datetime(
        dados["data_inversa"],
        format="mixed",
        dayfirst=True,
        errors="raise",
    )
    dados["horario"] = pd.to_datetime(
        dados["horario"],
        format="%H:%M:%S",
        errors="raise",
    ).dt.time

    dados[COLUNAS_INTEIRAS] = (
        dados[COLUNAS_INTEIRAS].apply(pd.to_numeric, errors="raise").astype("Int64")
    )
    dados[COLUNAS_DECIMAIS] = dados[COLUNAS_DECIMAIS].apply(
        lambda coluna: pd.to_numeric(
            coluna.astype("string").str.strip().str.replace(",", ".", regex=False),
            errors="raise",
        )
    )

    return dados


def main() -> None:
    """Executa a concatenação dos acidentes."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=== Concatenação dos acidentes (PRF) ===")

    arquivos = []
    for arquivo in DIRETORIO_ENTRADA.rglob("acidentes_ocorrencia_*"):
        ano_texto = arquivo.stem.removeprefix("acidentes_ocorrencia_")
        if arquivo.is_file() and ano_texto.isdigit():
            ano = int(ano_texto)
            if ANO_INICIAL <= ano <= ANO_FINAL:
                arquivos.append(arquivo)
    arquivos.sort()

    tabelas_normalizadas = []
    for arquivo in tqdm(
        arquivos,
        desc="Processando arquivos de acidentes",
        unit="arquivo",
    ):
        dados_arquivo = pd.read_csv(
            arquivo,
            sep=";",
            encoding="cp1252",
            dtype="string",
        )
        tabelas_normalizadas.append(normalizar_dados(dados_arquivo))

    todos_acidentes = pd.concat(
        tabelas_normalizadas,
        ignore_index=True,
        sort=False,
    )
    acidentes_selecionados = todos_acidentes[
        todos_acidentes["data_inversa"] > DATA_CORTE
    ].copy()

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    acidentes_selecionados.to_parquet(ARQUIVO_SAIDA, index=False)

    logger.info(f"Arquivos processados: {len(arquivos):,}")
    logger.info(f"Registros lidos: {len(todos_acidentes):,}")
    logger.info(f"Registros após o corte temporal: {len(acidentes_selecionados):,}")
    logger.info(
        "Registros anteriores ou iguais à data de corte: "
        f"{len(todos_acidentes) - len(acidentes_selecionados):,}"
    )
    logger.info(
        "Período encontrado: "
        f"{acidentes_selecionados['data_inversa'].min():%Y-%m-%d} a "
        f"{acidentes_selecionados['data_inversa'].max():%Y-%m-%d}"
    )
    logger.info(f"Arquivo de saída: {ARQUIVO_SAIDA}")
    logger.info("")


if __name__ == "__main__":
    main()
