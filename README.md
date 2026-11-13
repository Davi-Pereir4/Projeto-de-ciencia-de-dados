# Predição da qualidade de vinhos

Comparação entre regressão linear e Random Forest para prever a nota sensorial de vinhos tintos e brancos. O conjunto é o Wine Quality da [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality).

Os arquivos `data/raw/winequality-red.csv` e `data/raw/winequality-white.csv` vieram do pacote oficial dessa ficha. O artigo usa somente contagens e métricas produzidas por `main.py`.

## Como executar

O ambiente do projeto é o diretório `.venv`, com Python 3.12. Na raiz do projeto:

```text
.venv\Scripts\python.exe main.py
.venv\Scripts\python.exe -m pytest
```

O notebook `notebooks/wine_quality_analysis.ipynb` chama os módulos de `src` e repete a análise.

Para recriar o ambiente com [uv](https://docs.astral.sh/uv/):

```text
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## O que o pipeline faz

1. Lê os dois CSVs, marca branco como 0 e tinto como 1 e empilha as linhas.
2. Confere ausentes e duplicatas. Remove só as duplicatas exatas.
3. Mede outliers pelo intervalo interquartil e os mantém.
4. Divide a base tratada em 80% de treino e 20% de teste, com `random_state=42`.
5. Ajusta `LinearRegression` e `RandomForestRegressor` (100 árvores, `random_state=42`) em pipelines com `StandardScaler`.
6. Calcula MAE, MSE, RMSE e R² no teste, e MAE e R² na validação cruzada de cinco partições do treino.
7. Grava tabelas, figuras, previsões e modelos em `results/` e `models/`.

## Resultados desta execução

Base tratada: 5.320 registros (3.961 brancos e 1.359 tintos), a partir de 6.497 linhas e 1.177 duplicatas. Treino: 4.256. Teste: 1.064.

| Métrica | Regressão linear | Random Forest |
| --- | ---: | ---: |
| MAE | 0,5557 | 0,5323 |
| MSE | 0,5180 | 0,4851 |
| RMSE | 0,7197 | 0,6965 |
| R² | 0,3458 | 0,3874 |

Os valores completos estão em `results/metrics/test_metrics.json`.

## Auditoria

Uma implementação anterior foi citada na especificação do trabalho, e os arquivos dela não estavam no workspace. As contagens foram recalculadas. Duplicatas (1.177), base limpa (5.320), brancos (3.961), tintos (1.359), treino (4.256) e teste (1.064) coincidiram com os valores descritos naquela especificação.

As métricas não foram ajustadas para se aproximar da tabela antiga. Aquela tabela, não reproduzida aqui, informava MAE de 0,5602 e 0,5097 e R² de 0,3090 e 0,4142. A diferença é compatível com a ausência do código e dos hiperparâmetros originais. Esta entrega documenta o scikit-learn 1.9.1 e a linha de base acima.

## Texto

- Artigo, no formato pedido pela disciplina: `reports/article/artigo.md`
- Roteiro de 15 minutos: `reports/presentation/roteiro.md`

O PDF do artigo está em `reports/article/artigo.pdf`, gerado a partir de `reports/article/artigo.md` com `reports/article/build_pdf.py`.
