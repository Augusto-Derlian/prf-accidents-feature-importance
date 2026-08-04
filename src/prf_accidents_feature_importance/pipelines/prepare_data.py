"""Define a ordem de execução da preparação dos dados."""

from prf_accidents_feature_importance.data import (
    step_01a_concat_pedagio_volume,
    step_01b_concat_condicao_pavimento,
    step_01c_concat_acidentes_ocorrencia,
    step_02_enrich_pedagio_volume,
    step_03_enrich_acidentes_praca,
    step_04_enrich_acidentes_pavimento,
)

ETAPAS = {
    "01a_concatenar_pedagio": step_01a_concat_pedagio_volume.main,
    "01b_concatenar_pavimento": step_01b_concat_condicao_pavimento.main,
    "01c_concatenar_acidentes": step_01c_concat_acidentes_ocorrencia.main,
    "02_enriquecer_pedagio": step_02_enrich_pedagio_volume.main,
    "03_enriquecer_acidentes": step_03_enrich_acidentes_praca.main,
    "04_enriquecer_acidentes": step_04_enrich_acidentes_pavimento.main,
}


def executar() -> None:
    """Executa todas as etapas do pipeline na ordem definida."""
    for executar_etapa in ETAPAS.values():
        executar_etapa()
