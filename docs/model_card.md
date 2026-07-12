# Model Card - MovieLens NCF

## Resumo

- Nome: `movielens-ncf`
- Framework: PyTorch
- Tipo: recomendador neural de feedback implicito
- Dataset: MovieLens 100K
- Arquitetura: embeddings de usuario/item + MLP
- Execucao: CPU, seed 42

## Uso pretendido

Demonstrar ranking personalizado de itens com base no historico positivo de cada usuario. A solucao e educacional e foi criada para avaliacao offline reproduzivel.

## Dados

Avaliacoes com nota maior ou igual a 4 sao tratadas como interacoes positivas. Usuarios com menos de cinco interacoes positivas sao removidos. Para cada usuario, a interacao mais recente e teste e a penultima e validacao.

## Metricas

O relatorio final usa Precision@10, Recall@10, HitRate@10, NDCG@10 e MAP@10.

| Modelo | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 | MAP@10 |
|---|---:|---:|---:|---:|---:|
| NMF | 0.0120 | 0.1205 | 0.1205 | 0.0619 | 0.0445 |
| NCF | 0.0078 | 0.0778 | 0.0778 | 0.0401 | 0.0288 |

A melhor NCF obteve validation loss de 0.623872.

## Comparacao

O modelo neural e comparado com NMF do Scikit-Learn usando os mesmos splits e catalogo candidato. O NMF venceu em todas as metricas. Maior complexidade nao garantiu melhor generalizacao com o budget curto de treino. A NCF permanece registrada como modelo neural central e o resultado real e reportado sem cherry-picking.

## Limitacoes

- Proxy de filmes para um problema de produtos de e-commerce.
- Cold start para usuarios e itens sem historico.
- Negative sampling altera a distribuicao observada no treino.
- Uma interacao positiva nao representa necessariamente intencao de compra.
- Metricas offline nao medem diversidade, novidade ou impacto comercial real.
- O tuning foi limitado a duas configuracoes neurais para execucao rapida em CPU.

## Vieses e riscos

- Itens populares recebem mais observacoes e podem dominar recomendacoes.
- Usuarios muito ativos influenciam mais o treinamento.
- O recorte temporal do dataset e antigo e nao representa catalogos atuais.
- O uso fora do contexto educacional exige avaliacao de privacidade e impacto.

## Reprodutibilidade

Versoes ficam em `poetry.lock`, parametros em `params.yaml`, lineage em `dvc.lock` e experimentos no MLflow. Seeds sao fixadas em Python, NumPy e PyTorch.
