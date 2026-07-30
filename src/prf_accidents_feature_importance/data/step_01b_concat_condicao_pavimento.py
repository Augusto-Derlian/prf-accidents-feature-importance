"""Padroniza e concatena os levantamentos de condição do pavimento do DNIT."""

import logging

import pandas as pd
from tqdm import tqdm

from prf_accidents_feature_importance.config import (
    DIRETORIO_DADOS_BRUTOS,
    DIRETORIO_DADOS_INTERMEDIARIOS,
)

logger = logging.getLogger(__name__)

DIRETORIO_ENTRADA = DIRETORIO_DADOS_BRUTOS / "dnit" / "condicao_pavimento"
ARQUIVO_SAIDA = (
    DIRETORIO_DADOS_INTERMEDIARIOS
    / "01_condicao_pavimento_concat.parquet"
)

ALIAS_COLUNAS = {
    "data_aval": "data_inversa",
    "data": "data_inversa",
    "rodovia": "br",
    "sentido": "sentido_via",
}

PLANILHAS_EXCEL = {
    "condicao_pavimento_2023_11.xlsx": "ICM - Levantamentos PAV",
    "condicao_pavimento_2023_12.xlsx": "ICM - PAV",
    "condicao_pavimento_2024_01.xlsx": "ICM - PAV",
    "condicao_pavimento_2024_02.xlsx": "ICM - PAV",
    "condicao_pavimento_2024_03.xlsx": "ICM - Levantamentos PAV",
    "condicao_pavimento_2024_04.xlsx": "ICM - PAV",
    "condicao_pavimento_2024_05.xlsx": "ICM - PAV",
    "condicao_pavimento_2024_06.xlsx": "ICM 06_2024_PAV",
    "condicao_pavimento_2024_07.xlsx": "ICM - Levantamentos PAV",
    "condicao_pavimento_2024_08.xlsx": "ICM-PAV",
    "condicao_pavimento_2024_09.xlsx": "ICM - Levantamentos PAV",
    "condicao_pavimento_2024_10.xlsx": "ICM - Levantamentos PAV",
    "condicao_pavimento_2024_11.xlsx": "ICM - Levantamentos PAV",
    "condicao_pavimento_2024_12.xlsx": "ICM - Levantamentos PAV",
    "condicao_pavimento_2025_01.xlsx": "ICM - Levantamentos PAV",
    "condicao_pavimento_2025_02.xlsx": "ICM - Levantamentos",
    "condicao_pavimento_2025_03.xlsx": "ICM - Levantamentos",
    "condicao_pavimento_2025_04.xlsx": "ICM - Levantamentos",
    "condicao_pavimento_2025_05.xlsx": "ICM - Levantamentos",
    "condicao_pavimento_2025_06.xlsx": "ICM - Levantamentos",
    "condicao_pavimento_2025_07.xlsx": "ICM - Levantamentos",
    "condicao_pavimento_2025_11.xlsx": "Pavimentada",
}

COLUNAS_NUMERICAS = ["km_inicial", "km_final", "icm"]
COLUNAS_FINAIS = [
    "uf",
    "br",
    "sentido_via",
    "km_inicial",
    "km_final",
    "data_inversa",
    "icm",
]


def normalizar_cabecalho_simples(dados: pd.DataFrame) -> pd.DataFrame:
    """Normaliza arquivos que já possuem uma única linha de cabeçalho."""
    dados = dados.copy()
    dados.columns = (
        dados.columns.astype("string")
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    dados = dados.rename(columns=ALIAS_COLUNAS)

    colunas_obrigatorias = {
        "uf",
        "br",
        "km_inicial",
        "km_final",
        "data_inversa",
        "icm",
    }
    colunas_ausentes = colunas_obrigatorias - set(dados.columns)
    if colunas_ausentes:
        raise ValueError(
            f"Colunas ausentes na condição do pavimento: {sorted(colunas_ausentes)}"
        )

    dados["uf"] = (
        dados["uf"]
        .astype("string")
        .str.strip()
        .str.upper()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.extract(r"^([A-Z]{2})$", expand=False)
    )
    dados["br"] = pd.to_numeric(
        dados["br"].astype("string").str.strip().str.extract(r"(\d+)", expand=False),
        errors="coerce",
    ).astype("Int64")
    dados["data_inversa"] = pd.to_datetime(
        dados["data_inversa"],
        errors="coerce",
    )
    dados[COLUNAS_NUMERICAS] = dados[COLUNAS_NUMERICAS].apply(
        lambda coluna: pd.to_numeric(
            coluna.astype("string").str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )
    )

    if "sentido_via" not in dados.columns:
        dados["sentido_via"] = pd.NA

    dados["sentido_via"] = (
        dados["sentido_via"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .replace({"c": "crescente", "d": "decrescente"})
    )

    return dados[COLUNAS_FINAIS]


def normalizar_cabecalho_mesclado(dados: pd.DataFrame) -> pd.DataFrame:
    """Monta o cabeçalho de planilhas que usam as três primeiras linhas."""
    dados = dados.dropna(axis=1, how="all").copy()
    cabecalho = dados.iloc[:3].ffill(axis=0).ffill(axis=1)

    dados.columns = cabecalho.apply(
        lambda coluna: "_".join(pd.unique(coluna.dropna().astype(str))),
        axis=0,
    )
    dados = dados.iloc[3:].reset_index(drop=True)

    return normalizar_cabecalho_simples(dados)


def main() -> None:
    """Executa a concatenação da condição do pavimento."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=== Concatenação da condição do pavimento (DNIT) ===")

    tabelas_normalizadas: dict[str, pd.DataFrame] = {}

    diretorio_2026 = DIRETORIO_ENTRADA / "2026"
    diretorio_2025 = DIRETORIO_ENTRADA / "2025"
    diretorio_2024 = DIRETORIO_ENTRADA / "2024"
    diretorio_2023 = DIRETORIO_ENTRADA / "2023"
    meses_cabecalho_mesclado_2025 = [7, 6, 5, 4, 3, 2, 1]
    meses_excel_2023 = [12, 11]

    total_arquivos = (
        len(list(diretorio_2026.glob("*.csv")))
        + len(list(diretorio_2025.glob("*.csv")))
        + 1
        + len(meses_cabecalho_mesclado_2025)
        + len(list(diretorio_2024.glob("*.xlsx")))
        + len(meses_excel_2023)
    )
    progresso = tqdm(
        total=total_arquivos,
        desc="Processando arquivos de pavimento",
        unit="arquivo",
    )

    for arquivo in sorted(diretorio_2026.glob("*.csv")):
        progresso.set_postfix_str(arquivo.name)
        dados = pd.read_csv(arquivo, sep=";")
        tabelas_normalizadas[arquivo.name] = normalizar_cabecalho_simples(dados)
        progresso.update()

    for arquivo in sorted(diretorio_2025.glob("*.csv")):
        progresso.set_postfix_str(arquivo.name)
        dados = pd.read_csv(arquivo, sep=";")
        tabelas_normalizadas[arquivo.name] = normalizar_cabecalho_simples(dados)
        progresso.update()

    arquivo_2025_11 = diretorio_2025 / "condicao_pavimento_2025_11.xlsx"
    progresso.set_postfix_str(arquivo_2025_11.name)
    dados_2025_11 = pd.read_excel(
        arquivo_2025_11,
        sheet_name=PLANILHAS_EXCEL[arquivo_2025_11.name],
    )
    tabelas_normalizadas[arquivo_2025_11.name] = normalizar_cabecalho_simples(
        dados_2025_11
    )
    progresso.update()

    for mes in meses_cabecalho_mesclado_2025:
        arquivo = diretorio_2025 / f"condicao_pavimento_2025_{mes:02}.xlsx"
        progresso.set_postfix_str(arquivo.name)
        dados = pd.read_excel(
            arquivo,
            header=None,
            sheet_name=PLANILHAS_EXCEL[arquivo.name],
        )
        tabelas_normalizadas[arquivo.name] = normalizar_cabecalho_mesclado(dados)
        progresso.update()

    for arquivo in sorted(diretorio_2024.glob("*.xlsx")):
        progresso.set_postfix_str(arquivo.name)
        dados = pd.read_excel(
            arquivo,
            header=None,
            sheet_name=PLANILHAS_EXCEL[arquivo.name],
            usecols="A:AE"
            if arquivo.name == "condicao_pavimento_2024_09.xlsx"
            else None,
            skiprows=range(5)
            if arquivo.name == "condicao_pavimento_2024_08.xlsx"
            else None,
        )
        tabelas_normalizadas[arquivo.name] = normalizar_cabecalho_mesclado(dados)
        progresso.update()

    for mes in meses_excel_2023:
        arquivo = diretorio_2023 / f"condicao_pavimento_2023_{mes:02}.xlsx"
        progresso.set_postfix_str(arquivo.name)
        dados = pd.read_excel(
            arquivo,
            header=None,
            sheet_name=PLANILHAS_EXCEL[arquivo.name],
        )
        tabelas_normalizadas[arquivo.name] = normalizar_cabecalho_mesclado(dados)
        progresso.update()

    progresso.close()

    # A ordem dos arquivos define a ordem dos registros no arquivo final.
    tabelas_normalizadas = dict(sorted(tabelas_normalizadas.items()))
    dados_concatenados = pd.concat(
        tabelas_normalizadas.values(),
        ignore_index=True,
    )

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    dados_concatenados.to_parquet(ARQUIVO_SAIDA, index=False)

    sem_localizacao = (
        dados_concatenados[["uf", "br", "km_inicial", "km_final"]]
        .isna()
        .any(axis=1)
        .sum()
    )

    logger.info(f"Arquivos processados: {len(tabelas_normalizadas):,}")
    logger.info(f"Registros gerados: {len(dados_concatenados):,}")
    logger.info(f"Registros sem localização completa: {sem_localizacao:,}")
    logger.info(
        "Registros sem data de avaliação: "
        f"{dados_concatenados['data_inversa'].isna().sum():,}"
    )
    logger.info(f"Registros sem ICM: {dados_concatenados['icm'].isna().sum():,}")
    logger.info(f"Arquivo de saída: {ARQUIVO_SAIDA}")
    logger.info("")


if __name__ == "__main__":
    main()
