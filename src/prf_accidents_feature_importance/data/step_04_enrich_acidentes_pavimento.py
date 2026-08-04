"""Adiciona aos acidentes o ICM do trecho de pavimento correspondente."""

import logging

import numpy as np
import pandas as pd

from prf_accidents_feature_importance.config import DIRETORIO_DADOS_INTERMEDIARIOS

logger = logging.getLogger(__name__)

ARQUIVO_ACIDENTES = (
    DIRETORIO_DADOS_INTERMEDIARIOS / "03_acidentes_praca_enriched.parquet"
)
ARQUIVO_PAVIMENTO = (
    DIRETORIO_DADOS_INTERMEDIARIOS / "01_condicao_pavimento_concat.parquet"
)
ARQUIVO_SAIDA = (
    DIRETORIO_DADOS_INTERMEDIARIOS / "04_acidentes_pavimento_enriched.parquet"
)

COLUNAS_LOCALIZACAO = ["uf", "br"]
SENTIDOS_CONHECIDOS = ["crescente", "decrescente"]
DISTANCIA_MAXIMA_TRECHO_KM = 10


def preparar_trechos_pavimento(pavimento: pd.DataFrame) -> pd.DataFrame:
    """Prepara os trechos para a busca por UF, BR, sentido e km."""
    colunas_necessarias = [
        "uf",
        "br",
        "sentido_via",
        "km_inicial",
        "km_final",
        "data_inversa",
        "icm",
    ]
    trechos = pavimento[colunas_necessarias].dropna(
        subset=[
            "uf",
            "br",
            "km_inicial",
            "km_final",
            "data_inversa",
            "icm",
        ]
    )
    trechos = trechos.copy()

    # No sentido decrescente, o km inicial costuma ser maior que o km final.
    # Estes dois campos facilitam a comparação para qualquer sentido da via.
    trechos["km_menor"] = trechos[["km_inicial", "km_final"]].min(axis=1)
    trechos["km_maior"] = trechos[["km_inicial", "km_final"]].max(axis=1)

    # Parte dos arquivos antigos não informa o sentido. Nesses casos, a ordem
    # dos limites do trecho permite identificar o sentido do levantamento.
    sentido_pela_ordem_dos_kms = pd.Series(
        np.where(
            trechos["km_final"] >= trechos["km_inicial"],
            "crescente",
            "decrescente",
        ),
        index=trechos.index,
        dtype="string",
    )
    trechos["sentido_pavimento"] = trechos["sentido_via"].fillna(
        sentido_pela_ordem_dos_kms
    )
    trechos = trechos.rename(columns={"data_inversa": "data_avaliacao_pavimento"})

    # Pode haver mais de uma medição para o mesmo trecho na mesma data.
    # A média produz um único ICM para cada combinação.
    colunas_do_trecho = [
        *COLUNAS_LOCALIZACAO,
        "sentido_pavimento",
        "km_menor",
        "km_maior",
        "data_avaliacao_pavimento",
    ]
    trechos = trechos.groupby(
        colunas_do_trecho,
        as_index=False,
        dropna=False,
        sort=False,
    ).agg(icm=("icm", "mean"))

    # O km inteiro funciona como uma chave auxiliar para evitar comparar cada
    # acidente com todos os milhões de trechos disponíveis.
    primeiro_km_inteiro = np.floor(trechos["km_menor"]).astype("int64")
    ultimo_km_inteiro = np.ceil(trechos["km_maior"]).astype("int64") - 1
    ultimo_km_inteiro = np.maximum(
        primeiro_km_inteiro,
        ultimo_km_inteiro,
    )

    trechos["km_chave"] = [
        range(inicio, fim + 1)
        for inicio, fim in zip(
            primeiro_km_inteiro,
            ultimo_km_inteiro,
            strict=True,
        )
    ]
    trechos = trechos.explode("km_chave", ignore_index=True)
    trechos["km_chave"] = trechos["km_chave"].astype("int64")

    return trechos


def criar_consultas_de_acidentes(acidentes: pd.DataFrame) -> pd.DataFrame:
    """Seleciona os dados necessários para localizar o ICM de cada acidente."""
    consultas = acidentes[["uf", "br", "km", "sentido_via", "data_inversa"]].copy()
    consultas.insert(0, "_indice_acidente", np.arange(len(consultas)))
    consultas = consultas.dropna(subset=["uf", "br", "km", "data_inversa"])
    consultas = consultas.rename(
        columns={
            "sentido_via": "sentido_acidente",
            "data_inversa": "data_acidente",
        }
    )

    # Em um trecho crescente, o km exato 1 pertence ao trecho que começa em 1.
    # Em um trecho decrescente, ele pertence ao trecho que termina em 1.
    consultas["km_chave"] = np.floor(consultas["km"]).astype("int64")
    sentido_decrescente = consultas["sentido_acidente"].eq("decrescente")
    consultas.loc[sentido_decrescente, "km_chave"] = (
        np.ceil(consultas.loc[sentido_decrescente, "km"]).astype("int64") - 1
    )

    return consultas


def encontrar_icm_exato(
    acidentes: pd.DataFrame,
    trechos: pd.DataFrame,
) -> pd.DataFrame:
    """Encontra o ICM espacial e temporalmente mais próximo de cada acidente."""
    consultas = criar_consultas_de_acidentes(acidentes)

    tem_sentido = consultas["sentido_acidente"].isin(SENTIDOS_CONHECIDOS)
    consultas_com_sentido = consultas[tem_sentido]
    consultas_sem_sentido = consultas[~tem_sentido]

    # Quando o sentido é conhecido, ele também faz parte do cruzamento.
    candidatos_com_sentido = consultas_com_sentido.merge(
        trechos,
        left_on=[
            *COLUNAS_LOCALIZACAO,
            "sentido_acidente",
            "km_chave",
        ],
        right_on=[
            *COLUNAS_LOCALIZACAO,
            "sentido_pavimento",
            "km_chave",
        ],
        how="inner",
    )

    # Para acidentes sem sentido informado, são considerados os dois sentidos.
    candidatos_sem_sentido = consultas_sem_sentido.merge(
        trechos,
        on=[*COLUNAS_LOCALIZACAO, "km_chave"],
        how="inner",
    )

    candidatos = pd.concat(
        [candidatos_com_sentido, candidatos_sem_sentido],
        ignore_index=True,
    )
    if candidatos.empty:
        return pd.DataFrame(columns=["_indice_acidente", "icm"])

    # Confirma que o km do acidente realmente está dentro do trecho. A chave
    # inteira usada no merge apenas reduz o número de comparações necessárias.
    trecho_pontual = candidatos["km_menor"].eq(candidatos["km_maior"]) & candidatos[
        "km"
    ].eq(candidatos["km_menor"])
    dentro_do_trecho_crescente = (
        candidatos["sentido_acidente"].eq("crescente")
        & candidatos["km"].ge(candidatos["km_menor"])
        & candidatos["km"].lt(candidatos["km_maior"])
    )
    dentro_do_trecho_decrescente = (
        candidatos["sentido_acidente"].eq("decrescente")
        & candidatos["km"].gt(candidatos["km_menor"])
        & candidatos["km"].le(candidatos["km_maior"])
    )
    sem_sentido_informado = ~candidatos["sentido_acidente"].isin(SENTIDOS_CONHECIDOS)
    dentro_do_trecho_sem_sentido = sem_sentido_informado & candidatos["km"].between(
        candidatos["km_menor"],
        candidatos["km_maior"],
    )

    candidatos = candidatos[
        trecho_pontual
        | dentro_do_trecho_crescente
        | dentro_do_trecho_decrescente
        | dentro_do_trecho_sem_sentido
    ].copy()
    if candidatos.empty:
        return pd.DataFrame(columns=["_indice_acidente", "icm"])

    candidatos["distancia_dias"] = (
        (candidatos["data_acidente"] - candidatos["data_avaliacao_pavimento"])
        .abs()
        .dt.days
    )
    candidatos["centro_trecho"] = (candidatos["km_menor"] + candidatos["km_maior"]) / 2
    candidatos["distancia_centro_trecho"] = (
        candidatos["km"] - candidatos["centro_trecho"]
    ).abs()

    # A avaliação mais próxima da data do acidente é escolhida. Se houver
    # empate, vence o trecho com centro mais próximo e depois a data anterior.
    melhores_candidatos = candidatos.sort_values(
        [
            "_indice_acidente",
            "distancia_dias",
            "distancia_centro_trecho",
            "data_avaliacao_pavimento",
        ],
        kind="stable",
    ).drop_duplicates("_indice_acidente")

    return melhores_candidatos[["_indice_acidente", "icm"]]


def encontrar_trechos_mais_proximos(
    consultas: pd.DataFrame,
    trechos: pd.DataFrame,
    considerar_sentido: bool,
) -> pd.DataFrame:
    """Encontra o trecho anterior ou posterior mais próximo de cada km."""
    colunas_do_trecho = [
        *COLUNAS_LOCALIZACAO,
        "sentido_pavimento",
        "km_menor",
        "km_maior",
    ]
    locais_dos_trechos = trechos[colunas_do_trecho].drop_duplicates()
    consultas = consultas.copy()

    # Os arquivos podem representar as chaves com tipos internos diferentes.
    # Usar os mesmos tipos evita incompatibilidades durante o cruzamento.
    consultas["uf"] = consultas["uf"].astype("object")
    locais_dos_trechos["uf"] = locais_dos_trechos["uf"].astype("object")
    consultas["br"] = consultas["br"].astype("int64")
    locais_dos_trechos["br"] = locais_dos_trechos["br"].astype("int64")
    consultas["sentido_acidente"] = consultas["sentido_acidente"].astype("object")
    locais_dos_trechos["sentido_pavimento"] = locais_dos_trechos[
        "sentido_pavimento"
    ].astype("object")

    if considerar_sentido:
        consultas = consultas[consultas["sentido_acidente"].isin(SENTIDOS_CONHECIDOS)]
        chaves_consulta = [
            *COLUNAS_LOCALIZACAO,
            "sentido_acidente",
        ]
        chaves_trecho = [
            *COLUNAS_LOCALIZACAO,
            "sentido_pavimento",
        ]
    else:
        chaves_consulta = COLUNAS_LOCALIZACAO
        chaves_trecho = COLUNAS_LOCALIZACAO

    if consultas.empty or locais_dos_trechos.empty:
        return pd.DataFrame()

    # Para cada acidente, procura-se um trecho antes e outro depois de seu km.
    # Depois, a distância até os limites desses dois trechos decide o mais perto.
    consultas_ordenadas = consultas.sort_values("km")

    trechos_anteriores = pd.merge_asof(
        consultas_ordenadas,
        locais_dos_trechos.sort_values("km_maior"),
        left_on="km",
        right_on="km_maior",
        left_by=chaves_consulta,
        right_by=chaves_trecho,
        direction="backward",
    )
    trechos_posteriores = pd.merge_asof(
        consultas_ordenadas,
        locais_dos_trechos.sort_values("km_menor"),
        left_on="km",
        right_on="km_menor",
        left_by=chaves_consulta,
        right_by=chaves_trecho,
        direction="forward",
    )

    candidatos = pd.concat(
        [trechos_anteriores, trechos_posteriores],
        ignore_index=True,
    ).dropna(subset=["km_menor", "km_maior"])
    if candidatos.empty:
        return candidatos

    distancia_antes_do_trecho = (candidatos["km_menor"] - candidatos["km"]).clip(
        lower=0
    )
    distancia_depois_do_trecho = (candidatos["km"] - candidatos["km_maior"]).clip(
        lower=0
    )
    candidatos["distancia_trecho_km"] = (
        distancia_antes_do_trecho + distancia_depois_do_trecho
    )
    candidatos = candidatos[
        candidatos["distancia_trecho_km"] <= DISTANCIA_MAXIMA_TRECHO_KM
    ]

    return candidatos.sort_values(
        [
            "_indice_acidente",
            "distancia_trecho_km",
            "km_menor",
            "km_maior",
        ],
        kind="stable",
    ).drop_duplicates("_indice_acidente")


def buscar_icm_nos_trechos(
    acidentes_com_trecho: pd.DataFrame,
    trechos: pd.DataFrame,
) -> pd.DataFrame:
    """Busca a avaliação temporalmente mais próxima nos trechos escolhidos."""
    if acidentes_com_trecho.empty:
        return pd.DataFrame(columns=["_indice_acidente", "icm"])

    colunas_chave = [
        *COLUNAS_LOCALIZACAO,
        "sentido_pavimento",
        "km_menor",
        "km_maior",
    ]
    medicoes = trechos[
        [
            *colunas_chave,
            "data_avaliacao_pavimento",
            "icm",
        ]
    ].drop_duplicates()
    candidatos = acidentes_com_trecho.merge(
        medicoes,
        on=colunas_chave,
        how="inner",
    )
    candidatos["distancia_dias"] = (
        (candidatos["data_acidente"] - candidatos["data_avaliacao_pavimento"])
        .abs()
        .dt.days
    )

    melhores_candidatos = candidatos.sort_values(
        [
            "_indice_acidente",
            "distancia_dias",
            "data_avaliacao_pavimento",
        ],
        kind="stable",
    ).drop_duplicates("_indice_acidente")

    return melhores_candidatos[["_indice_acidente", "icm"]]


def encontrar_icm_dos_acidentes(
    acidentes: pd.DataFrame,
    trechos: pd.DataFrame,
) -> pd.DataFrame:
    """Encontra o ICM exato ou, como alternativa, o trecho mais próximo."""
    resultados_exatos = encontrar_icm_exato(acidentes, trechos)
    consultas = criar_consultas_de_acidentes(acidentes)

    # Primeiro fallback: mantém UF, BR e sentido, alterando apenas o km.
    consultas_pendentes = consultas[
        ~consultas["_indice_acidente"].isin(resultados_exatos["_indice_acidente"])
    ]
    trechos_mesmo_sentido = encontrar_trechos_mais_proximos(
        consultas_pendentes,
        trechos,
        considerar_sentido=True,
    )
    resultados_mesmo_sentido = buscar_icm_nos_trechos(
        trechos_mesmo_sentido,
        trechos,
    )

    # Segundo fallback: usado quando não há trecho aceitável no mesmo sentido.
    indices_encontrados = pd.concat(
        [
            resultados_exatos["_indice_acidente"],
            resultados_mesmo_sentido["_indice_acidente"],
        ],
        ignore_index=True,
    )
    consultas_pendentes = consultas[
        ~consultas["_indice_acidente"].isin(indices_encontrados)
    ]
    trechos_qualquer_sentido = encontrar_trechos_mais_proximos(
        consultas_pendentes,
        trechos,
        considerar_sentido=False,
    )
    resultados_qualquer_sentido = buscar_icm_nos_trechos(
        trechos_qualquer_sentido,
        trechos,
    )

    return pd.concat(
        [
            resultados_exatos,
            resultados_mesmo_sentido,
            resultados_qualquer_sentido,
        ],
        ignore_index=True,
    )


def enriquecer_acidentes(
    acidentes: pd.DataFrame,
    pavimento: pd.DataFrame,
) -> pd.DataFrame:
    """Adiciona uma coluna de ICM sem alterar a quantidade de acidentes."""
    if "icm" in acidentes.columns:
        raise ValueError("Os dados de acidentes já possuem a coluna 'icm'.")

    trechos = preparar_trechos_pavimento(pavimento)
    icm_dos_acidentes = encontrar_icm_dos_acidentes(acidentes, trechos)

    resultado = acidentes.copy()
    resultado.insert(0, "_indice_acidente", np.arange(len(resultado)))
    resultado = resultado.merge(
        icm_dos_acidentes,
        on="_indice_acidente",
        how="left",
        validate="one_to_one",
    )
    resultado = (
        resultado.sort_values("_indice_acidente", kind="stable")
        .drop(columns="_indice_acidente")
        .reset_index(drop=True)
    )

    return resultado


def main() -> None:
    """Executa o enriquecimento e salva o arquivo resultante."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    acidentes = pd.read_parquet(ARQUIVO_ACIDENTES)
    pavimento = pd.read_parquet(ARQUIVO_PAVIMENTO)

    resultado = enriquecer_acidentes(acidentes, pavimento)

    if len(resultado) != len(acidentes):
        raise RuntimeError("O enriquecimento alterou a quantidade de acidentes.")

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_parquet(ARQUIVO_SAIDA, index=False)

    quantidade_com_icm = int(resultado["icm"].notna().sum())
    percentual_com_icm = (
        quantidade_com_icm / len(resultado) * 100 if len(resultado) else 0
    )

    logger.info("\n=== Enriquecimento dos acidentes com pavimento ===")
    logger.info("Acidentes processados: %s", len(resultado))
    logger.info(
        "Acidentes com ICM: %s (%.2f%%)",
        quantidade_com_icm,
        percentual_com_icm,
    )
    logger.info(
        "Limite do fallback por distância: %s km",
        DISTANCIA_MAXIMA_TRECHO_KM,
    )
    logger.info("Arquivo salvo em: %s", ARQUIVO_SAIDA)


if __name__ == "__main__":
    main()
