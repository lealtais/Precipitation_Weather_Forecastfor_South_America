# Previsão climática de precipitação sobre a América do Sul

Solução para o desafio de dados do WORCAP (Workshop de Computação Aplicada),
promovido pelo Programa de Pós-Graduação do INPE, no Kaggle:
[previsao-climatica-de-precipitacao-sobre-a-america-do-sul](https://www.kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul).

## Tarefa

Prever a precipitação média do mês seguinte (mm/dia), por ponto de grade
(0.25°), sobre a América do Sul, a partir do histórico de reanálise ERA5
(1940–2022). Avaliação por RMSE contra o ERA5, comparando com a climatologia
histórica como régua mínima a bater.

## Dados

10 variáveis mensais em grade (301 x 261 pontos, lat -60..15, lon -90..-25):
`t2`, `cloud_cover`, `shum_850`, `surface_pressure`, `u_850`, `v_850`,
`temperature_850`, `rel_hum_850`, `geopotential_850`, `tp` (precipitação do
mês) e `tp_alvo` (precipitação do mês seguinte = alvo de treino).

O teste (`teste_features.nc`) cobre 24 meses (jan/2023–dez/2024): cada linha
já traz as variáveis atmosféricas reais do "mês de origem" (`time_origem`),
então a tarefa é sempre prever 1 mês à frente a partir de dado observado —
não há necessidade de previsão recursiva multi-passo.

## Estratégia

O modelo aprende o **resíduo em relação à climatologia** (média histórica por
mês-calendário e ponto de grade), não o valor bruto — isso remove a enorme
variância geográfica (Amazônia vs. Andes vs. Patagônia) e deixa o modelo focado
em aprender o desvio real (sinal de ENSO, persistência, etc.).

Features:
- Anomalias das 9 variáveis atmosféricas no mês atual (em relação à
  climatologia daquele mês-calendário)
- Persistência: anomalia de `tp` no mês atual, mês-1 e mês-2
- Suavização espacial (média dos vizinhos, kernel 3x3) da anomalia de `tp`,
  capturando padrões de chuva em escala regional — a feature isolada mais
  importante do modelo
- **Índice ONI (Oceanic Niño Index, NOAA)**, com lag 0/1/2 meses — proxy
  direto do estado do ENSO (El Niño/La Niña); a resposta da chuva ao ENSO é
  defasada, então os 3 lags juntos somam ~26% da importância do modelo
- Ensemble de 3 seeds (média das previsões), pequeno ganho extra e menos ruído
- Latitude, longitude, seno/cosseno do mês-alvo
- Climatologia do mês-alvo (usada tanto como feature quanto como baseline
  somado de volta à previsão do resíduo)

Modelo: LightGBM (`objective="regression"`), aprendendo o resíduo.

### Validação: por que "últimos N meses" não bastava

O período de teste real (jan/2023–dez/2024) coincide com um dos El Niño mais
fortes já registrados. Validar em "últimos 5 anos" genéricos deu RMSE local
de ~1.80, mas o placar real do Kaggle veio em **1.9557** — um gap grande
demais pra ser só ruído. Trocamos a validação para os picos de El Niño forte
do passado (DJF de 1982-83, 1997-98, 2009-10, 2015-16): o RMSE de validação
foi pra ~1.90, muito mais perto do real, confirmando que esses episódios são
estruturalmente mais difíceis (climatologia erra muito mais neles — e é
justamente aí que o índice ONI ajuda o modelo a compensar).

## Log de resultados

| Versão | Validação | Histórico | Features extras | RMSE modelo | RMSE climatologia | Ganho | Leaderboard (público) |
|---|---|---|---|---|---|---|---|
| v1 | últimos 5 anos | 1979+ | — (baseline) | 1.8039 | 1.8266 | 1.2% | — |
| v2 | últimos 5 anos | 1995+ | lags 0/1/2 | 1.7690 | 1.7871 | 1.0% | — |
| v3 | últimos 5 anos | 1979+ | lags 0/1/2 | 1.7981 | 1.8266 | 1.6% | — |
| v4 | últimos 5 anos | 1965+ | lags + suavização 3x3 | 1.7980 | 1.8352 | 2.0% | **1.95569** |
| v5 | El Niño (48 meses) | 1965+ | igual a v4 | 1.9083 | 2.0599 | 7.4% | — |
| v6 | El Niño (48 meses) | 1965+ | v5 + índice ONI | 1.8905 | 2.0599 | 8.2% | — |
| v7 | El Niño (48 meses) | 1965+ | v6 + ONI com lag (0/1/2) + ensemble 3 seeds | 1.8861 | 2.0599 | 8.4% | **1.96904** (pior que v4!) |
| — | últimos 5 anos | 1995+ | igual a v4 (menos anos) | 1.9311 | 2.0469 | 5.7% | — |

**⚠️ Alerta importante: a validação em anos de El Niño (v5-v7) não previu o
resultado real.** v7 parecia melhor que v4 na validação local (8.4% vs. 2.0%
de ganho), mas no leaderboard real v7 saiu **pior** (1.96904 vs. 1.95569 do
v4). A suspeita é que, ao validar só nos mesmos 4 eventos históricos de El
Niño usados no treino (pool de 48 meses vindos de só 4 episódios), o modelo
com ONI pode ter aprendido a reconhecer a assinatura específica desses 4
eventos em vez do efeito genérico do ENSO -- o que não necessariamente
generaliza pro evento real de 2023-2024. **Até isso ser resolvido, v4 (mais
simples, sem ONI) é o nosso melhor resultado real confirmado.**

Tentamos uma validação "leave-one-event-out" (`validate_leave_one_out.py`),
mas abortamos por travar com pouca RAM livre. Baseado nos notebooks de
referência do Rob Mulla (ver Referências), trocamos pra uma validação
**walk-forward** de verdade (`validate_walkforward.py`, estilo
`TimeSeriesSplit` do sklearn): vários folds cobrindo janelas de tempo
diferentes ao longo de todo o histórico (não só os 4 eventos de El Niño),
cada um treinando só com o passado e validando num pedaço nunca visto do
"futuro" -- é o jeito estatisticamente correto de saber se o ONI generaliza,
em vez de validar numa seleção enviesada por características do próprio
evento que queremos prever. Resultado: *(rodando)*.

Tentativas descartadas:
- `objective="tweedie"` sobre o valor absoluto (em vez do resíduo): piorou (-2.9%)
- Suavização espacial 5x5 (além do 3x3) + modelo maior: sem ganho (2.0% → 2.0%)
- Índices TNA/TSA (SST Atlântico Norte/Sul) + gradiente, testado por hipótese de
  ligação com a ZCAS: piorou o RMSE (1.8861 → 1.9010) e deixou o Centro do
  Brasil pior que a climatologia (-0.2%) — provavelmente redundante com o ONI
  e virou ruído com só 48 meses de validação
- Modelo separado por região (Sul/Central/Norte): RMSE 1.8876 vs. 1.8861 do
  modelo global — empate técnico, sem ganho real
- Menos anos de histórico (1995+ em vez de 1965+): piorou (5.7% vs. 8.4% de
  ganho na validação de El Niño) — mais histórico ajuda mais que recência
- CatBoost em vez de LightGBM: abortado, estimativa de 15+ horas de treino
  pra esse volume de dados (LightGBM treina o mesmo em minutos)

## Diagnóstico: onde o modelo ainda erra mais

RMSE por faixa de latitude (validação em anos de El Niño forte, v7):

| Região | RMSE modelo | RMSE climatologia | Ganho |
|---|---|---|---|
| Sul (Patagônia/Sul BR-AR) | 1.2969 | 1.3151 | 1.4% |
| Central (Brasil central/Bolívia/Paraguai) | 1.6622 | 1.6666 | 0.3% |
| Norte (Amazônia) | 2.5429 | 2.9088 | **12.6%** |

Quase todo o ganho sobre a climatologia vem da Amazônia (onde o El Niño tem
efeito forte e conhecido). No Centro do Brasil (região de influência da ZCAS)
o modelo mal supera a climatologia — e a correlação ONI x chuva medida
diretamente nessa faixa é quase zero, então não é surpresa. Um [paper recente
sobre o mesmo tipo de problema](https://arxiv.org/abs/2512.13910) reporta o
mesmo padrão: ZCAS/ZCIT é a região mais difícil pra modelos baseados em árvore.

## Arquivos

- `kaggle_notebook.py` — versão para rodar direto num notebook do Kaggle
  (dataset já montado em `/kaggle/input/...`)
- `local_run.py` — versão equivalente para rodar localmente, processando um
  arquivo NetCDF por vez para caber em pouca RAM livre; gera `submission.csv`
  pronto para envio direto pelo Kaggle (Submit Predictions)
- `inspect_data.py` / `inspect_data2.py` — scripts usados para entender a
  estrutura dos arquivos NetCDF e a relação entre `tp` / `tp_alvo` / teste
- `oni.ascii.txt` — índice ONI histórico (NOAA), usado como feature de ENSO
- `tna.data` / `tsa.data` — índices SST do Atlântico (NOAA), testados e descartados
- `validate_leave_one_out.py` — validação "deixa 1 evento de El Niño de fora" (abortada por RAM)
- `validate_walkforward.py` — validação walk-forward (`TimeSeriesSplit`), a correta
- `reference_notebooks/` — notebooks do Rob Mulla (Kaggle Grandmaster) usados como
  referência de técnica (cross-validation, feature engineering, tuning)

## Como reproduzir

```bash
python -m kagglehub competition_download previsao-climatica-de-precipitacao-sobre-a-america-do-sul
curl -o oni.ascii.txt https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
python local_run.py
```

Ajuste `DATA_DIR`, `YEAR_START` e `SMOOTH_SIZES` no topo de `local_run.py`
conforme a RAM disponível.

## Referências

- Índice ONI (El Niño/La Niña): [NOAA CPC](https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt)
- Índices TNA/TSA (SST Atlântico tropical, testados e descartados): [NOAA PSL](https://psl.noaa.gov/data/timeseries/month/DS/TNA/) / [TSA](https://psl.noaa.gov/data/timeseries/month/DS/TSA/)
- [Exploring Machine Learning, Deep Learning, and Explainable AI Methods for Seasonal Precipitation Prediction in South America](https://arxiv.org/abs/2512.13910) — confirmou de forma independente dois achados nossos: multicolinearidade entre `surface_pressure`/`geopotential_850` e entre `t2`/`temperature_850`, e que a região de ZCAS/ZCIT é a mais difícil pra modelos baseados em árvore (Random Forest/XGBoost)
- [Relação entre SACZ e SST durante eventos extremos de precipitação no Centro-Leste do Brasil](https://www.sciencedirect.com/science/article/abs/pii/S0377026523000738) — motivou (sem sucesso, ver "tentativas descartadas") testar SST do Atlântico como feature
- Notebooks do [Rob Mulla](https://www.kaggle.com/robikscube) (Kaggle Grandmaster), baixados via API do Kaggle e usados como referência de técnica:
  - [Cross Validation Visualized](https://www.kaggle.com/code/robikscube/cross-validation-visualized-youtube-tutorial) — a lição central: "o score médio out-of-fold é uma estimativa muito melhor de como o modelo vai performar em dado nunca visto" do que uma validação de conjunto único/enviesado
  - [Tutorial: Time Series Forecasting with XGBoost (Parte 1 e 2)](https://www.kaggle.com/code/robikscube/tutorial-time-series-forecasting-with-xgboost) — `TimeSeriesSplit`, lag features, walk-forward validation
  - [Ion Switching - 5KFold LGBM & Tracking](https://www.kaggle.com/code/robikscube/ion-switching-5kfold-lgbm-tracking) — padrão de out-of-fold (OOF) pooling e tracking sistemático de experimentos
  - [Fast Tuning with XGBoost + Optuna](https://www.kaggle.com/code/robikscube/fast-tuning-with-xgboost-3-0-optuna-gpus) — "hyperparameter tuning é inútil sem uma validação adequada" (por isso resolvemos a validação antes de ajustar hiperparâmetro)
