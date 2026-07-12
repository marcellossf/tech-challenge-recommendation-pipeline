# Tech Challenge Fase 02 - Sistema de Recomendacao

Pipeline completo de recomendacao com MovieLens 100K, baseline Scikit-Learn, Neural Collaborative Filtering em PyTorch, dados e pipeline versionados com DVC, experimentos rastreados no MLflow e execucao containerizada.

## Problema

Uma empresa precisa recomendar itens a partir do comportamento de navegacao. Avaliacoes com nota maior ou igual a 4 sao tratadas como feedback positivo implicito. O objetivo do modelo e ordenar itens ainda nao vistos por cada usuario.

## Principais recursos

| Area | Implementacao |
|---|---|
| Clean code | `src/`, funcoes pequenas, type hints, docstrings, Factory e Strategy |
| Reprodutibilidade | Poetry, `poetry.lock`, seeds, `.env.example`, Pydantic Settings |
| Docker | Dockerfile multi-stage, usuario nao-root e Compose com MLflow + trainer |
| DVC | dataset versionado e pipeline com quatro stages |
| PyTorch | embeddings user/item, MLP, negative sampling e early stopping |
| Scikit-Learn | baseline NMF com a mesma divisao temporal |
| MLflow | tres runs minimas, artefatos, Registry e promocao para Production |
| Qualidade | Ruff, pre-commit, testes unitarios e teste de integracao |

## Arquitetura

```text
MovieLens -> preprocess -> feature_eng -> train -> evaluate
                              |             |         |
                         split temporal  MLflow   metricas top-K
                                        Registry  Model Card
```

O pattern `ModelFactory` desacopla a criacao do NMF e do modelo neural. A Strategy `RandomNegativeSampling` encapsula a geracao de exemplos negativos.

## Estrutura

```text
configs/             configuracao base
data/                dados brutos, intermediarios e processados
docs/                arquitetura e Model Card
models/              checkpoints gerados
reports/             metricas e comparacoes
scripts/             validacao do ambiente e promocao
src/recommender/     codigo da aplicacao
tests/               testes unitarios e integracao
dvc.yaml             pipeline reproduzivel de quatro stages
docker-compose.yml   servicos MLflow e treino
```

## Instalacao local

Requisitos: Python 3.11 e Poetry 2.x.

```bash
poetry install
poetry run pre-commit install
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/macOS
poetry run validate-env
```

## Dados e DVC

Baixe o MovieLens da fonte oficial e adicione o arquivo ao DVC:

```bash
poetry run download-data
poetry run dvc add data/raw/u.data
poetry run dvc repro
```

O remote padrao e local (`../dvc-storage`):

```bash
poetry run dvc push
poetry run dvc pull
```

## Pipeline

```text
preprocess -> feature_eng -> train -> evaluate
```

- `preprocess`: filtra avaliacoes positivas e codifica IDs.
- `feature_eng`: aplica leave-two-out temporal por usuario.
- `train`: executa NMF e duas configuracoes NCF, totalizando tres runs.
- `evaluate`: compara os modelos com cinco metricas top-K.

Execute tudo com:

```bash
poetry run dvc repro
poetry run dvc metrics show
poetry run dvc dag
```

## MLflow e Model Registry

Por padrao, o pipeline usa SQLite local (`mlflow.db`) e artefatos em `mlruns/`.

```bash
poetry run mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Depois do treino e avaliacao:

```bash
poetry run promote-model
```

O script promove o modelo de `Staging` para `Production` e tambem cria o alias `champion`.

## Docker

```bash
docker compose build
docker compose up -d mlflow
docker compose run --rm trainer
```

A interface do MLflow fica em `http://localhost:5000`.

## Qualidade e testes

```bash
poetry run ruff check src tests scripts
poetry run ruff format --check src tests scripts
poetry run pytest
poetry run pre-commit run --all-files
```

## Resultados

O pipeline gera:

- `reports/metrics.json`: metricas finais legiveis pelo DVC;
- `reports/model_comparison.md`: tabela comparativa dos modelos;
- `reports/training_summary.json`: run vencedora e validation loss;
- `models/ncf_best.pt`: checkpoint neural selecionado.

As metricas utilizadas sao Precision@10, Recall@10, HitRate@10, NDCG@10 e MAP@10.

Resultados da execucao reproduzida em 12/07/2026:

| Modelo | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 | MAP@10 |
|---|---:|---:|---:|---:|---:|
| NMF (Scikit-Learn) | 0.0120 | 0.1205 | 0.1205 | 0.0619 | 0.0445 |
| NCF (PyTorch) | 0.0078 | 0.0778 | 0.0778 | 0.0401 | 0.0288 |

O NMF venceu neste recorte. O baseline linear generalizou melhor que as duas configuracoes neurais curtas. A NCF demonstra embeddings, MLP, negative sampling e early stopping. O resultado evidencia o trade-off entre complexidade e desempenho offline.

Resumo: 55.361 interacoes positivas, 938 usuarios e 1.447 itens. Cada reproducao cria tres runs MLflow, duas delas para configuracoes NCF. A melhor NCF e promovida para `Production` com o alias `champion`.

## Documentacao

- [Arquitetura](docs/architecture.md)
- [Model Card](docs/model_card.md)

## Limitacoes

- MovieLens representa consumo de filmes, servindo como proxy para interacoes user-item de e-commerce.
- Feedback implicito nao captura o motivo de uma preferencia.
- Usuarios e itens novos sofrem com cold start.
- A avaliacao offline nao garante impacto de negocio em producao.
- O treinamento curto em CPU favorece reprodutibilidade, nao tuning exaustivo.
