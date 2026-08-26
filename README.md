# Predição de Severidade em Acidentes Rodoviários (CatBoost + SHAP)

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-Optimized-yellow.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-green.svg)
![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-lightgrey.svg)

## Visão Geral
Este repositório contém o pipeline de fusão de dados e modelagem preditiva desenvolvido para a disciplina de Ciência de Dados (ESE410031 - PPGESE/UFSC). O objetivo é classificar a severidade de acidentes registrados em rodovias federais no Brasil, usando dados da PRF, do DNIT e da ANTT.

O recorte atualmente configurado no pipeline cobre os anos de 2023 a 2026. O escopo efetivo também depende dos trechos e períodos disponíveis para o enriquecimento por volume de tráfego e condição do pavimento.

O modelo usa **CatBoost** para dados tabulares com variáveis categóricas e **SHAP (Shapley Additive exPlanations)** para analisar a contribuição das features. As explicações SHAP descrevem o comportamento do modelo e não devem ser interpretadas como inferência causal.

## Arquitetura do Projeto

O desenvolvimento é dividido em dois pipelines independentes:

1. **Pipeline de ETL:**
   - Extração e padronização das bases abertas (PRF, DNIT, ANTT).
   - Associação por UF, BR, sentido e km para vincular acidentes aos trechos de tráfego e pavimento.
   - Os dados brutos não são versionados no Git. As fontes e os links estão em [`references/SOURCES.md`](references/SOURCES.md).

2. **Pipeline de modelagem:**
   - Treinamento do classificador multiclasse (CatBoost).
   - Avaliação no conjunto de teste, incluindo métricas, matriz de confusão e importância nativa das features.
   - Geração de explicações globais e por classe com SHAP.

## Estrutura do Repositório

- [`data/01_raw/`](data/01_raw/) — arquivos brutos da PRF, DNIT e ANTT.
- [`data/02_intermediate/`](data/02_intermediate/) — arquivos Parquet gerados pelas etapas de ETL.
- [`data/03_processed/`](data/03_processed/) — conjunto final de features usado na modelagem.
- [`notebooks/`](notebooks/) — notebooks de EDA e análise das relações com a classificação.
- [`src/prf_accidents_feature_importance/`](src/prf_accidents_feature_importance/) — código do pipeline, features, treinamento e explicabilidade.
- [`models/catboost/`](models/catboost/) — modelo treinado, métricas, importâncias e gráficos.
- [`references/SOURCES.md`](references/SOURCES.md) — catálogo das fontes originais.

## Contrato de Dados

A tabela final [`data/03_processed/acidentes_features.parquet`](data/03_processed/acidentes_features.parquet) contém as features selecionadas e a variável alvo. Identificadores, localização direta e variáveis que representam consequências observadas após o acidente são removidos para reduzir vazamento do alvo.

| Feature | Tipo | Descrição / Domínio |
| :--- | :--- | :--- |
| `dia_semana` | Categórica | Dia da semana |
| `br` | Categórica | BR, UF e trecho de 10 km |
| `causa_acidente` | Categórica | Causa registrada pela PRF |
| `tipo_acidente` | Categórica | Tipo do acidente |
| `fase_dia` | Categórica | Fase do dia |
| `condicao_meteorologica` | Categórica | Condição meteorológica |
| `tipo_pista` | Categórica | Simples, Dupla ou Múltipla |
| `inclinacao` | Categórica | Nivelado, Aclive ou Declive |
| `reta` | Categórica | Reta ou Curva |
| `km` | Numérica | Quilômetro do acidente |
| `volume_pedagio` | Numérica | Volume Diário Médio (VDM) do trecho |
| `icm` | Numérica | Índice de Condição de Manutenção do trecho, segundo o DNIT |
| `fator_humano` | Numérica | Indicador binário derivado de `causa_acidente` |
| `classificacao_acidente` | Categórica (Alvo) | Sem vítimas, Leves, Graves/Fatais |

### Legenda do ICM

O ICM é interpretado conforme as seguintes faixas de condição da via:

| Faixa do ICM | Condição |
| :--- | :--- |
| ICM < 30 | Bom |
| 30 < ICM < 50 | Regular |
| 50 < ICM < 70 | Ruim |
| ICM > 70 | Péssimo |

Essa legenda é baseada no dicionário de dados do ICM do DNIT, cuja referência está em [`references/SOURCES.md`](references/SOURCES.md).

## Como Executar o Projeto

### Pré-requisitos

O projeto exige Python 3.12.x. Na raiz do repositório, crie e ative um ambiente virtual:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Se a política do PowerShell impedir a ativação, execute antes, apenas na sessão atual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Instale o projeto e, opcionalmente, registre o kernel do Jupyter:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
python -m ipykernel install --user --name=prf-accidents-feature-importance
```

### Dados brutos

Baixe os arquivos listados em [`references/SOURCES.md`](references/SOURCES.md), renomeie-os conforme os comentários nos módulos `step_*.py` e coloque-os nas pastas esperadas:

- `data/01_raw/antt/pedagio_volume/`
- `data/01_raw/antt/praca_pedagio/`
- `data/01_raw/dnit/condicao_pavimento/`
- `data/01_raw/prf/acidentes_ocorrencia/`

O pipeline atualmente procura os arquivos de acidentes, volume e pavimento referentes a 2023–2026.

### Sequência completa

Execute os comandos a seguir na raiz do projeto, depois de colocar os dados brutos:

```powershell
prepare-data
build-features
train-model
python -m prf_accidents_feature_importance.modeling.explain_shap
```

As três primeiras etapas também podem ser chamadas diretamente pelos módulos:

```powershell
python -m prf_accidents_feature_importance.cli.prepare_data
python -m prf_accidents_feature_importance.cli.build_features
python -m prf_accidents_feature_importance.cli.train_model
```

O comando `prepare-data` executa as seis etapas de concatenação e enriquecimento na ordem definida em `pipelines/prepare_data.py`. Em seguida, `build-features` gera `data/03_processed/acidentes_features.parquet`.

### Treinamento reproduzível do CatBoost

Depois de gerar `data/03_processed/acidentes_features.parquet`, `train-model` divide os dados em treino, validação e teste, balanceia as classes, seleciona a configuração pelo F1 macro de validação e salva o modelo e as métricas.

```powershell
train-model
```

Também é possível executar diretamente pelo módulo Python:

```powershell
python -m prf_accidents_feature_importance.cli.train_model
```

Os artefatos são gravados em `models/catboost/`:

- `modelo.cbm`: modelo CatBoost treinado;
- `metrics.json`: métricas, classes, matriz de confusão e informações do conjunto de dados;
- `confusion_matrix.png`: matriz de confusão em imagem;
- `feature_importance.csv` e `feature_importance.png`: importância nativa do CatBoost;
- `tuning_results.csv`: resultados das configurações avaliadas.

### Explicabilidade SHAP

Com o modelo treinado, execute:

```powershell
python -m prf_accidents_feature_importance.modeling.explain_shap
```

O módulo reproduz a divisão do conjunto de teste e salva em `models/catboost/shap/`:

- `shap_feature_importance.csv`: importância média absoluta por classe;
- `shap_values.parquet`: valores SHAP por observação e classe;
- `shap_summary_global.png`: summary plot global;
- `shap_summary_<classe>.png`: summary plot de cada classe;
- `classes.csv`: ordem das classes usada nos resultados.

Para os gráficos, são excluídas somente as observações com `icm` ou `volume_pedagio` ausentes. Essa filtragem não é aplicada ao modelo nem aos arquivos completos de valores SHAP.

### Notebooks

Os notebooks atuais são auxiliares à análise:

- [`01_eda_classificacao_acidente.ipynb`](notebooks/01_eda_classificacao_acidente.ipynb): EDA e preparação das features;
- [`02_analise_relacoes_classificacao_acidente.ipynb`](notebooks/02_analise_relacoes_classificacao_acidente.ipynb): análise das relações entre variáveis e classificação.

O treinamento e a explicabilidade principais devem ser executados pelos comandos acima.
