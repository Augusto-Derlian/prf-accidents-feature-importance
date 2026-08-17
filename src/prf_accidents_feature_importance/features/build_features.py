"""Constrói o conjunto de features usado no treinamento dos modelos."""

import logging

import numpy as np
import pandas as pd

from prf_accidents_feature_importance.config import (
    DIRETORIO_DADOS_INTERMEDIARIOS,
    DIRETORIO_DADOS_PROCESSADOS,
)

logger = logging.getLogger(__name__)

ARQUIVO_ENTRADA = (
    DIRETORIO_DADOS_INTERMEDIARIOS / "04_acidentes_pavimento_enriched.parquet"
)
ARQUIVO_SAIDA = DIRETORIO_DADOS_PROCESSADOS / "acidentes_features.parquet"

COLUNA_ALVO = "classificacao_acidente"
TAMANHO_TRECHO_KM = 10

COLUNAS_CATEGORICAS = [
    "dia_semana",
    "br",
    "causa_acidente",
    "tipo_acidente",
    "fase_dia",
    "condicao_meteorologica",
    "tipo_pista",
    "inclinacao",
    "reta",
]

COLUNAS_NUMERICAS = [
    "km",
    "volume_pedagio",
    "icm",
    "fator_humano",
]

COLUNAS_FEATURES = [*COLUNAS_CATEGORICAS, *COLUNAS_NUMERICAS]
COLUNAS_SAIDA = [*COLUNAS_FEATURES, COLUNA_ALVO]

COLUNAS_PARA_REMOVER = [
    # Identificadores
    "id",
    # Temporalidade
    "data_inversa",
    "horario",
    # Localização
    "uf",
    "municipio",
    "sentido_via",
    "latitude",
    "longitude",
    "regional",
    "delegacia",
    "uop",
    "uso_solo",
    # Vazamento do alvo
    "pessoas",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "feridos",
    "ilesos",
    "ignorados",
    "veiculos",
]

CAUSAS_DE_FATOR_HUMANO = {
    "ausencia de reacao do condutor",
    "reacao tardia ou ineficiente do condutor",
    "acessar a via sem observar a presenca dos outros veiculos",
    "condutor deixou de manter distancia do veiculo da frente",
    "velocidade incompativel",
    "manobra de mudanca de faixa",
    "ingestao de alcool pelo condutor",
    "transitar na contramao",
    "condutor dormindo",
    "ultrapassagem indevida",
    "trafegar com motocicleta (ou similar) entre as faixas",
    "desrespeitar a preferencia no cruzamento",
    "conversao proibida",
    "entrada inopinada do pedestre",
    "pedestre andava na pista",
    "pedestre cruzava a pista fora da faixa",
    "mal subito do condutor",
    "transitar no acostamento",
    "retorno proibido",
    "frear bruscamente",
    "condutor desrespeitou a iluminacao vermelha do semaforo",
    "carga excessiva e/ou mal acondicionada",
    "estacionar ou parar em local proibido",
    "suicidio (presumido)",
    "condutor usando celular",
    "pedestre - ingestao de alcool/ substancias psicoativas",
    "ingestao de substancias psicoativas pelo condutor",
    "transtornos mentais (exceto suicidio)",
    "participar de racha",
    "deixar de acionar o farol da motocicleta (ou similar)",
    "modificacao proibida",
    "transitar na calcada",
    "ingestao de alcool e/ou substancias psicoativas pelo pedestre",
    "ingestao de alcool ou de substancias psicoativas pelo pedestre",
    "obstrucao via tentativa assalto",
}

DIA_DA_SEMANA_PARA_NUMERO = {
    "segunda-feira": 0,
    "terca-feira": 1,
    "quarta-feira": 2,
    "quinta-feira": 3,
    "sexta-feira": 4,
    "sabado": 5,
    "domingo": 6,
}


def remover_colunas_nao_preditivas(acidentes: pd.DataFrame) -> pd.DataFrame:
    """Remove identificadores, localização direta e vazamentos do alvo."""
    return acidentes.drop(columns=COLUNAS_PARA_REMOVER)


def criar_trecho_br(acidentes: pd.DataFrame) -> pd.DataFrame:
    """Junta a BR, o estado e o trecho de 10 km em uma categoria."""
    resultado = acidentes.copy()
    inicio_trecho = ((resultado["km"] // TAMANHO_TRECHO_KM) * TAMANHO_TRECHO_KM).astype(
        "Int64"
    )
    fim_trecho = inicio_trecho + TAMANHO_TRECHO_KM

    resultado["br"] = (
        "BR-"
        + resultado["br"].astype("string")
        + " / "
        + resultado["uf"].astype("string")
        + " (km "
        + inicio_trecho.astype("string")
        + " a "
        + fim_trecho.astype("string")
        + ")"
    )
    return resultado


def criar_fator_humano(acidentes: pd.DataFrame) -> pd.DataFrame:
    """Indica se a causa registrada está associada a um fator humano."""
    resultado = acidentes.copy()
    resultado["fator_humano"] = (
        resultado["causa_acidente"].isin(CAUSAS_DE_FATOR_HUMANO).astype("int8")
    )
    return resultado


def codificar_dia_semana(acidentes: pd.DataFrame) -> pd.DataFrame:
    """Representa o dia da semana por meio de coordenadas cíclicas."""
    resultado = acidentes.copy()
    numero_dia = resultado.pop("dia_semana").map(DIA_DA_SEMANA_PARA_NUMERO)

    resultado["dia_semana_seno"] = np.sin(2 * np.pi * numero_dia / 7)
    resultado["dia_semana_cosseno"] = np.cos(2 * np.pi * numero_dia / 7)
    return resultado


def separar_tracado_via(acidentes: pd.DataFrame) -> pd.DataFrame:
    """Separa o traçado da via em inclinação e geometria horizontal."""
    resultado = acidentes.copy()
    tracado_via = resultado.pop("tracado_via").astype("string")

    resultado["inclinacao"] = "nivelado"
    resultado.loc[
        tracado_via.str.contains("aclive", regex=False, na=False),
        "inclinacao",
    ] = "aclive"
    resultado.loc[
        tracado_via.str.contains("declive", regex=False, na=False),
        "inclinacao",
    ] = "declive"

    resultado["reta"] = "reta"
    resultado.loc[
        tracado_via.str.contains("curva", regex=False, na=False),
        "reta",
    ] = "curva"
    return resultado


def construir_features(acidentes: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas as transformações de engenharia de features."""
    resultado = criar_trecho_br(acidentes)
    resultado = remover_colunas_nao_preditivas(resultado)
    resultado = criar_fator_humano(resultado)
    # resultado = codificar_dia_semana(resultado)
    resultado = separar_tracado_via(resultado)
    resultado = resultado.dropna(subset=[COLUNA_ALVO]).reset_index(drop=True)

    return resultado[COLUNAS_SAIDA]


def main() -> None:
    """Lê os dados intermediários e salva a matriz de features processada."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=== Construção das features ===")

    acidentes = pd.read_parquet(ARQUIVO_ENTRADA)
    features = construir_features(acidentes)

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(ARQUIVO_SAIDA, index=False)

    logger.info("Registros de entrada: %s", f"{len(acidentes):,}")
    logger.info("Registros gerados: %s", f"{len(features):,}")
    logger.info("Features geradas: %s", len(COLUNAS_FEATURES))
    logger.info("Arquivo de saída: %s", ARQUIVO_SAIDA)


if __name__ == "__main__":
    main()
