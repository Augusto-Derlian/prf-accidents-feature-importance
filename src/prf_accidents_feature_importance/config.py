"""Caminhos compartilhados pelos módulos do projeto."""

from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
DIRETORIO_DADOS = RAIZ_PROJETO / "data"
DIRETORIO_DADOS_BRUTOS = DIRETORIO_DADOS / "01_raw"
DIRETORIO_DADOS_INTERMEDIARIOS = DIRETORIO_DADOS / "02_intermediate"
DIRETORIO_DADOS_PROCESSADOS = DIRETORIO_DADOS / "03_processed"
DIRETORIO_MODELOS = RAIZ_PROJETO / "models"
