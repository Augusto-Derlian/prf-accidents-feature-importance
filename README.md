# 🛣️ Predição de Severidade em Acidentes Rodoviários (CatBoost + SHAP)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-Optimized-yellow.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-green.svg)
![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-lightgrey.svg)

## 📌 Visão Geral
Este repositório contém o framework de fusão de dados e modelagem preditiva desenvolvido para a disciplina de Ciência de Dados (ESE410031 - PPGESE/UFSC). O objetivo é classificar a severidade de acidentes nas rodovias federais da região Sul (2024-2026), relacionando o sinistro às condições da via (ICM) e ao volume de tráfego.

A arquitetura resolve o problema da "caixa-preta" em IA utilizando **CatBoost** para lidar com os dados tabulares e categóricos, e **SHAP (Shapley Additive exPlanations)** para extrair a inferência causal e o impacto de cada feature.

## 🏗️ Arquitetura do Projeto

O desenvolvimento é dividido em dois pipelines independentes:

1. **Pipeline de ETL (Engenharia de Dados):**
   - Extração e padronização das bases abertas (PRF, DNIT, ANTT).
   - *Linear Referencing:* Cruzamento geoespacial via `KM` da rodovia para vincular o acidente ao respectivo trecho de tráfego e conservação.
   - Os dados brutos não são versionados no Git. Utilizamos **DVC** para versionamento de dados pesados.

2. **Pipeline de Modelagem (Ciência de Dados):**
   - Treinamento do classificador multiclasse (CatBoost).
   - Avaliação de métricas.
   - Geração das explicações globais e locais (SHAP).

## � Estrutura do Repositório

- [data/processed/](data/processed/) — dados processados e notebooks de geração do dataset sintético.
- [src/models/catboost/model/](src/models/catboost/model/) — notebook de treino do CatBoost, resumo do modelo e artefatos gerados.
- [src/models/catboost/visualizacao/](src/models/catboost/visualizacao/) — notebook de visualização e arquivos de explicabilidade gerados.
- [src/visualization/](src/visualization/) — scripts auxiliares para visualização e explicabilidade.

## �� Contrato de Dados (Matriz de Treinamento)

A tabela final consolidada (`model_input.csv`), livre de identificadores espaciais diretos para evitar vazamento de dados, respeita a seguinte estrutura:

| Feature | Tipo | Descrição / Domínio |
| :--- | :--- | :--- |
| `dia_semana` | Categórica | Segunda, Terça, etc. |
| `horario` | Categórica | Manhã, Tarde, Noite, Madrugada |
| `condicao_metereologica`| Categórica | Céu claro, Chuva, Neblina, etc. |
| `tipo_pista` | Categórica | Simples, Dupla, Múltipla |
| `inclinacao` | Categórica | Nível, Aclive, Declive |
| `reta` | Categórica | Reta, Curva |
| `volume_pedagio` | Numérica | Volume Diário Médio (VDM) do trecho |
| `icm_via` | Numérica | Índice de Condição de Manutenção (DNIT) |
| `classificacao_acidente` | Categórica (Alvo) | Sem vítimas, Leves, Graves/Fatais |

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos:
Adicione um environment Python 3.12.*:
> py -3.12 -m venv .venv

Ative-o:
- Windows:
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
A linha acima é necessária para permitir a execução de scripts PowerShell no Windows. Em seguida, ative o ambiente virtual:
> .venv\Scripts\activate.ps1

Atualize o pip e instale as dependências:
> python -m pip install --upgrade pip
> pip install -e

Registre o environment no Jupyter:
> python -m ipykernel install --user --name=.venv

### 2. Como Rodar com dados dummy:
1. rode o dummy_input.ipynb para gerar o dataset sintético
2. rode o catboost_training.ipynb para treinar o modelo
3. rode o catboost_visualizacao.ipynb para gerar as explicações SHAP

### 3. Como rodar com dados reais:
1. Consulte os comentários no topo dos arquivos em `src/prf_accidents_feature_importance/data/step_*.py` para saber onde baixar, onde salvar e como renomear cada arquivo.
2. Coloque os dados brutos em `data/01_raw/` usando as pastas esperadas:
   - `antt/pedagio_volume/`
   - `antt/praca_pedagio/2026/`
   - `dnit/condicao_pavimento/`
   - `prf/acidentes_ocorrencia/`
3. Execute o pipeline de ETL:
   - `prepare-data`
4. Construa as features:
   - `build-features`
5. Treine o modelo:
   - `train-model`
6. Execute os notebooks de visualização, se precisar de explicações SHAP.

### Treinamento reproduzível do CatBoost

Depois de gerar `data/03_processed/acidentes_features.parquet`, execute:

```powershell
train-model
```

Também é possível executar diretamente pelo módulo Python:

```powershell
python -m prf_accidents_feature_importance.cli.train_model
```

O pipeline usa divisões estratificadas de treino, validação e teste, balanceia as
classes e aplica parada antecipada. Os artefatos são gravados em
`models/catboost/`: `modelo.cbm`, `metrics.json` e `feature_importance.csv`.
