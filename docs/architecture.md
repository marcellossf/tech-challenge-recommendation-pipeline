# Arquitetura

## Visao geral

O repositorio adota arquitetura modular orientada ao pipeline. Cada stage possui entradas e saidas declaradas no DVC e pode ser reproduzido sem notebooks.

```text
data/raw/u.data
    |
    v
preprocess ----> data/interim/interactions.csv
    |
    v
feature_eng ---> train.csv + validation.csv + test.csv
    |
    v
train ----------> NMF + dois NCF + runs MLflow + Registry
    |
    v
evaluate -------> metrics.json + model_comparison.md
```

## Decisoes

- Split temporal por usuario evita usar o futuro para prever o passado.
- Feedback positivo e definido como rating maior ou igual a 4.
- NMF e NCF recebem a mesma informacao de treino para comparacao justa.
- A rede usa embeddings de usuario e item, concatenacao e MLP.
- A avaliacao remove itens ja vistos e mede ranking top-K.
- CPU e seed fixa priorizam portabilidade e reproducibilidade.

## Design patterns

### Factory

`ModelFactory` centraliza a construcao de modelos. A orquestracao nao depende das classes concretas e pode trocar `nmf` por `neural_cf` via configuracao.

### Strategy

`SamplingStrategy` define o contrato de amostragem. `RandomNegativeSampling` implementa a estrategia usada no treino sem acoplar essa regra ao Dataset ou Trainer.

## Componentes de infraestrutura

- Poetry: ambiente e lock de dependencias.
- DVC: lineage dos dados, cache e pipeline.
- MLflow: parametros, metricas, artefatos e Registry.
- Docker Compose: servidor MLflow e container efemero de treino.

