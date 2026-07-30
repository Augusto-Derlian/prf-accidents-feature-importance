"""Adiciona BR, km e UF ao volume das praças de pedágio."""

import logging

import pandas as pd

from prf_accidents_feature_importance.config import (
    DIRETORIO_DADOS_BRUTOS,
    DIRETORIO_DADOS_INTERMEDIARIOS,
)

logger = logging.getLogger(__name__)

ARQUIVO_VOLUME = (
    DIRETORIO_DADOS_INTERMEDIARIOS
    / "01_pedagio_volume_concat.parquet"
)
ARQUIVO_PRACAS = (
    DIRETORIO_DADOS_BRUTOS
    / "antt"
    / "praca_pedagio"
    / "2026"
    / "praca_pedagio_2026_05.csv"
)
ARQUIVO_SAIDA = (
    DIRETORIO_DADOS_INTERMEDIARIOS
    / "02_pedagio_volume_enriched.parquet"
)

ALIAS_CONCESSIONARIAS = {
    "autopista fernao dias": "motiva minas sp",
    "eco101": "ecovias capixaba",
    "ecoponte": "ecovias ponte",
    "ecoriominas": "ecovias rio minas",
    "msvia": "pantanal",
}

ALIAS_PRACAS = {
    "conselheiro lafaiete": "lafaiete",
    "br 116 mg km 433 600": "engenheiro caldas",
}


def main() -> None:
    """Relaciona o volume ao cadastro de praças e salva o resultado."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    volume = pd.read_parquet(ARQUIVO_VOLUME)
    pracas = pd.read_csv(ARQUIVO_PRACAS, sep=";", low_memory=False)

    colunas_originais = volume.columns.tolist()
    quantidade_original = len(volume)
    colunas_chave = ["concessionaria_chave", "praca_chave"]

    for dados, coluna_praca in (
        (volume, "praca"),
        (pracas, "praca_de_pedagio"),
    ):
        dados["concessionaria_chave"] = (
            dados["concessionaria"]
            .astype("string")
            .str.strip()
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
            .str.lower()
            .str.replace(r"[^a-z0-9]+", " ", regex=True)
            .str.strip()
            .replace(ALIAS_CONCESSIONARIAS)
        )
        dados["praca_chave"] = (
            dados[coluna_praca]
            .astype("string")
            .str.strip()
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
            .str.lower()
            .str.replace(r"[^a-z0-9]+", " ", regex=True)
            .str.strip()
            .str.replace(
                r"^(?:praca\s+)?p?0*\d+\s+(?=.*[a-z])",
                "",
                regex=True,
            )
            .str.replace(r"^praca\s+(?!\d+$)", "", regex=True)
            .str.replace(r"^eng\s+", "engenheiro ", regex=True)
            .str.replace(r"^prof\s+", "professor ", regex=True)
            .str.replace(r"^pres\s+", "presidente ", regex=True)
            .replace(ALIAS_PRACAS)
        )

    pracas = pracas.rename(columns={"km_m": "km"})
    pracas["br"] = pd.to_numeric(
        pracas["rodovia"].str.extract(r"(\d+)", expand=False),
        errors="coerce",
    ).astype("Int64")
    pracas["km"] = pd.to_numeric(pracas["km"], errors="coerce").astype("Float64")
    pracas["uf"] = pracas["uf"].astype("string").str.strip().str.upper()

    localizacoes = (
        pracas[colunas_chave + ["br", "km", "uf"]]
        .dropna(subset=colunas_chave)
        .drop_duplicates()
    )

    # Primeira tentativa: concessionária e praça.
    por_concessionaria_e_praca = localizacoes.drop_duplicates(
        colunas_chave,
        keep=False,
    )
    volume_enriquecido = volume.merge(
        por_concessionaria_e_praca,
        on=colunas_chave,
        how="left",
        validate="many_to_one",
    )

    # Segunda tentativa: apenas praças que possuem uma única localização.
    por_praca = (
        localizacoes[["praca_chave", "br", "km", "uf"]]
        .drop_duplicates()
        .drop_duplicates("praca_chave", keep=False)
        .rename(
            columns={
                "br": "br_por_praca",
                "km": "km_por_praca",
                "uf": "uf_por_praca",
            }
        )
    )
    volume_enriquecido = volume_enriquecido.merge(
        por_praca,
        on="praca_chave",
        how="left",
        validate="many_to_one",
    )

    for coluna in ["br", "km", "uf"]:
        volume_enriquecido[coluna] = volume_enriquecido[coluna].fillna(
            volume_enriquecido[f"{coluna}_por_praca"]
        )

    resultado = volume_enriquecido[colunas_originais + ["br", "km", "uf"]]

    if len(resultado) != quantidade_original:
        raise RuntimeError("A correspondência alterou a quantidade de registros.")

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_parquet(ARQUIVO_SAIDA, index=False)

    correspondencias = int(resultado["br"].notna().sum())
    percentual = correspondencias / len(resultado) * 100 if len(resultado) else 0
    
    logger.info("\n=== Enriquecimento dos dados de volume com praça ===")
    logger.info("Registros de entrada: %s", quantidade_original)
    logger.info("Registros de saída: %s", len(resultado))
    logger.info(
        "Correspondências encontradas: %s (%.2f%%)", correspondencias, percentual
    )
    logger.info("Arquivo salvo em: %s", ARQUIVO_SAIDA)


if __name__ == "__main__":
    main()
