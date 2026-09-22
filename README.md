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
| **v12** | **walk-forward (5 folds, 1970-2019)** | 1970+ | ONI + mapa de sensibilidade ao ENSO por ponto de grade, fold-safe | 1.7915 | 1.8600* | 3.68%* | **1.86374** |

\* RMSE/ganho da validação v12 são a média entre os 5 folds (climatologias
diferentes por período); não comparável linha a linha com v1-v7, que usam uma
única janela de validação. O número que importa é o leaderboard real.

**⚠️ Alerta importante: a validação em anos de El Niño (v5-v7) não previu o
resultado real.** v7 parecia melhor que v4 na validação local (8.4% vs. 2.0%
de ganho), mas no leaderboard real v7 saiu **pior** (1.96904 vs. 1.95569 do
v4). A suspeita é que, ao validar só nos mesmos 4 eventos históricos de El
Niño usados no treino (pool de 48 meses vindos de só 4 episódios), o modelo
com ONI pode ter aprendido a reconhecer a assinatura específica desses 4
eventos em vez do efeito genérico do ENSO -- o que não necessariamente
generaliza pro evento real de 2023-2024.

Tentamos uma validação "leave-one-event-out" (`validate_leave_one_out.py`),
mas abortamos por travar com pouca RAM livre. Baseado nos notebooks de
referência do Rob Mulla (ver Referências), trocamos pra uma validação
**walk-forward** de verdade (`kaggle_notebook_v12_walkforward.py`, estilo
`TimeSeriesSplit` do sklearn): 5 folds cobrindo janelas de tempo diferentes
ao longo de todo o histórico (1970-2019, não só os 4 eventos de El Niño),
cada um treinando só com o passado, recalculando climatologia e o mapa de
sensibilidade ao ENSO só com dado do próprio fold (sem vazamento), e
validando num pedaço nunca visto do "futuro".

| Config | RMSE médio (5 folds) | Ganho médio sobre climatologia |
|---|---|---|
| baseline_sem_oni | 1.7997 | 3.24% |
| oni_simples | 1.7964 | 3.42% |
| **oni_sensmap** | **1.7915** | **3.68%** |
| oni_sensmap_fisica_anom | 1.7914 | 3.69% |
| oni_sensmap_fisica_raw | 1.7906 | 3.72% |

As features físicas (fluxo de umidade, instabilidade térmica) não trouxeram
ganho fora do ruído entre folds — ficaram de fora. `oni_sensmap` (ONI + mapa
de sensibilidade ao ENSO por ponto de grade, recalculado por fold) venceu com
a mesma performance das versões com física, só que mais simples.

**Confirmado no leaderboard real com `kaggle_notebook_v12_final_submission.py`: RMSE 1.86374** —
melhora real sobre o v4 (1.95569), e dessa vez a validação local (walk-forward,
~3.7% de ganho) bateu com o resultado real, ao contrário do que aconteceu com
v5-v7. **v12 (`oni_sensmap`) é agora o nosso melhor resultado real confirmado.**

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

## Notebooks do Kaggle (histórico de versões)

A máquina local não tinha RAM suficiente pra rodar validação walk-forward
completa, então a partir daqui os experimentos passaram a rodar direto num
notebook do Kaggle (mais memória). Cada arquivo `kaggle_notebook_*.py` é uma
versão independente — cole o conteúdo num notebook novo (Internet: ON) e use
"Save Version → Save & Run All (Commit)" pra não perder o progresso se a
sessão cair.

| Arquivo | O que testa | Status |
|---|---|---|
| `kaggle_notebook_v2.py` | Decide se o ONI ajuda, com validação walk-forward + OOF pooling | ✅ rodou, ver resultado abaixo |
| `kaggle_notebook_v3_trend.py` | v2 + feature de tendência temporal (ano) | testado dentro do v6 |
| `kaggle_notebook_v4_modelcompare.py` | Compara LightGBM/LightGBM-RF/XGBoost + tuning com Optuna | superado pelo v5 |
| `kaggle_notebook_v5_full.py` | v4 + HistGradientBoosting + Random Forest (sklearn) + GridSearchCV | ⚠️ travava no HistGB/RF (sklearn não aguenta o volume) |
| **`kaggle_notebook_v5.2.py`** | **Igual ao v5, mas só com LightGBM/LightGBM-RF/XGBoost** (tirado o que travava) | ✅ versão atual pra comparação de modelo |
| `kaggle_notebook_v6.py` | Testa as 4 combinações (nada/ONI/tendência/ONI+tendência) num pass só | ✅ rodando |
| **`kaggle_notebook_v6.2.py`** | **v6 + testa blend com regressão Ridge** por cima da combinação vencedora | ✅ versão atual, mais completa |
| **`kaggle_notebook_v8_final_xgboost.py`** | **Modelo FINAL pra submissão**: treina direto com XGBoost + hiperparâmetros do GridSearchCV, sem comparar modelos de novo | ✅ pronto pra rodar e submeter |

**Recomendação de uso agora:** `v6.2` (decide ONI/tendência/Ridge, ainda
rodando) e depois `v8_final_xgboost` pra gerar a submissão de verdade — os
outros ficam só de histórico.

### Resultados parciais (fold 1 de 5, validação walk-forward 1998-2002)

| Config | RMSE |
|---|---|
| base (sem ONI, sem tendência) | 1.7243 |
| **oni** | **1.7136** (melhor) |
| trend (só tendência de ano) | 1.7339 (pior que base) |
| oni_trend (os dois) | 1.7150 |
| lightgbm_rf | 1.7227 |
| xgboost | 1.7100 (melhor entre os tipos de modelo) |

Primeira leitura (ainda parcial, só 1 de 5 folds): **ONI ajuda, tendência de
ano sozinha piora**, e XGBoost está ligeiramente à frente do LightGBM. Precisa
dos outros 4 folds pra confirmar.

### Comparação de modelos + modelo espacial (rodados localmente, grid reduzido)

Enquanto o Kaggle processava o `v6.2`, rodamos localmente duas verificações
com o grid espacial reduzido (1/3 ou 1/2 das células, pra caber na RAM da
máquina local) — resultado direcional, não é o número final:

| Modelo | RMSE (OOF, 3 folds) |
|---|---|
| **xgboost** | **1.7527** (melhor) |
| lightgbm | 1.7539 |
| random_forest | 1.7568 |
| GridSearchCV (xgboost) | 1.7519 com `learning_rate=0.05, max_depth=6, min_child_weight=20` |

Bate com o resultado parcial do `v5.2` no Kaggle (XGBoost levemente à frente).

**Modelo espacial (CNN)** — implementado em `local_cnn_spatial.py` (ver
detalhes na seção "Ideias exploradas" abaixo). Resultado: RMSE 1.8018 contra
1.8336 da climatologia (ganho de só 1.7%) — **pior que LightGBM/XGBoost**
(~1.75), confirmando a suspeita de que ~700 meses de histórico não são
dados suficientes para uma CNN generalizar bem. Não vale a pena para
submissão nesse momento.

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
- `validate_walkforward.py` — validação walk-forward (`TimeSeriesSplit`) local, janela deslizante fixa (pouca RAM)
- `kaggle_notebook_v2.py` a `kaggle_notebook_v6.2.py` — ver tabela de notebooks acima
- `reference_notebooks/` — notebooks do Rob Mulla (Kaggle Grandmaster) usados como
  referência de técnica (cross-validation, feature engineering, tuning)
- `local_model_compare.py` — comparação local de LightGBM/XGBoost/RandomForest
  + GridSearchCV, com grid espacial reduzido pra caber na RAM local
- `local_cnn_spatial.py` — modelo espacial (CNN) local, ver resultado acima
- `kaggle_solution_breakthrough.py` — primeira versão com Z-score por ponto de
  grade, mapa de sensibilidade ao ENSO e features físicas de transporte de
  umidade; **não tinha validação alguma antes de submeter** — serviu de ponto
  de partida pro v12, mas não deve ser usado diretamente
- `kaggle_notebook_v12_walkforward.py` — validação walk-forward fold-safe
  (5 folds, recalcula climatologia e mapa de sensibilidade ao ENSO só com o
  passado de cada fold) comparando 5 configs; ver tabela de resultado acima
- `kaggle_notebook_v12_final_submission.py` — modelo final pra submissão,
  config vencedora (`oni_sensmap`) treinada em 100% do histórico; gera
  `submission.csv`. **Só roda no Kaggle** (grade cheia + ~52 anos de histórico
  não cabe na RAM de uma máquina local de 16GB)

## Como reproduzir

```bash
python -m kagglehub competition_download previsao-climatica-de-precipitacao-sobre-a-america-do-sul
curl -o oni.ascii.txt https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
python local_run.py
```

Ajuste `DATA_DIR`, `YEAR_START` e `SMOOTH_SIZES` no topo de `local_run.py`
conforme a RAM disponível.

## Ideias exploradas (pesquisa adicional)

**Modelo espacial (CNN)** — um colega de competição sugeriu treinar uma rede
convolucional que enxerga a grade 2D inteira (como uma imagem multi-canal)
em vez de tratar cada ponto de grade como uma linha independente. A
literatura confirma que CNN/LSTM captura melhor a ZCAS/ZCIT do que modelos
baseados em árvore -- mas tem uma armadilha: pra um CNN, cada **mês inteiro**
é 1 amostra de treino, e só temos ~700 meses de histórico (contra ~40 milhões
de linhas no formato tabular atual). Um estudo específico aponta que CNN com
mais de 4 camadas convolucionais sofre overfitting sério com datasets desse
tamanho.

**Implementado e testado** (`local_cnn_spatial.py`): CNN rasa (3 camadas
convolucionais, dropout 0.3, weight decay), grid reduzido pela metade (151x131),
10 canais de entrada (anomalias com lag, suavização, ONI, mês, climatologia),
holdout cronológico dos últimos 60 meses. Parou por early stopping na epoch 32.
Resultado: RMSE 1.8018 vs. 1.8336 da climatologia (ganho de só 1.7%) --
bem pior que LightGBM/XGBoost (~1.75 no mesmo tipo de recorte). Confirma a
suspeita: **~700 imagens mensais não são dados suficientes para a CNN
generalizar melhor que gradient boosting nesse problema.** Não recomendado
para submissão.

**Regressão estatística (Ridge/Lasso/GAM/quantílica)** — pesquisa mostrou que
regressão regularizada (Ridge/Lasso) lida bem com a multicolinearidade que já
tínhamos identificado entre nossas variáveis atmosféricas, e é usada na
literatura de downscaling climático com índices de teleconexão (ENSO etc.).
Não costuma superar gradient boosting em RMSE puro, mas um blend
(LightGBM + Ridge) é barato de testar -- implementado no `kaggle_notebook_v6.2.py`.

## Próximos passos sugeridos pela pesquisa (2026-09-18, ainda não implementados)

Pesquisa em artigos/Kaggle/arXiv sobre o que mais pode ajudar. As duas
primeiras ideias são as mais promissoras e mais baratas de testar (não
exigem dados novos nem reescrever o pipeline inteiro):

1. **Trocar a função de perda do treino pra `objective="tweedie"`** (LightGBM
   e XGBoost já suportam nativamente). RMSE puro assume erro "gaussiano",
   mas chuva é zero-inflada e assimétrica (muitos meses perto de zero, poucos
   bem altos) -- a distribuição Tweedie foi feita exatamente pra esse tipo de
   dado. Mesmo continuando a AVALIAR por RMSE (é a métrica da competição),
   treinar com Tweedie pode gerar previsões melhores. É só trocar um
   parâmetro, fácil de testar.
2. **Adicionar mais índices de teleconexão além do ONI**: PDO (Pacific
   Decadal Oscillation), SAM/Antarctic Oscillation e MJO (Madden-Julian
   Oscillation) -- a literatura mostra que todos afetam a chuva na América
   do Sul, e a força da própria teleconexão do ENSO muda dependendo da fase
   do PDO. Todos têm índice histórico baixável do NOAA, igual o ONI.
3. **Pós-processamento com quantile mapping** -- corrige viés sistemático
   comparando a distribuição das previsões com a distribuição real, aplicado
   DEPOIS do modelo já ter previsto (não mexe no treino).
4. Achamos papers específicos sobre esse EXACT problema (gradient boosting
   pra precipitação sazonal na América do Sul) -- vale ler antes de decidir
   o que priorizar. Ver [REFERENCIAS.md](REFERENCIAS.md) pra lista completa
   com todos os links dessa rodada de pesquisa.

## Referências

Dados/índices usados de fato no modelo:
- Índice ONI (El Niño/La Niña): https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
- Índices TNA/TSA (SST Atlântico tropical, testados e descartados): https://psl.noaa.gov/data/timeseries/month/DS/TNA/ / https://psl.noaa.gov/data/timeseries/month/DS/TSA/

Notebooks do [Rob Mulla](https://www.kaggle.com/robikscube) (Kaggle Grandmaster), usados como referência de técnica de validação:
- https://www.kaggle.com/code/robikscube/cross-validation-visualized-youtube-tutorial
- https://www.kaggle.com/code/robikscube/tutorial-time-series-forecasting-with-xgboost
- https://www.kaggle.com/code/robikscube/pt2-time-series-forecasting-with-xgboost
- https://www.kaggle.com/code/robikscube/time-series-forecasting-with-machine-learning-yt
- https://www.kaggle.com/code/robikscube/ion-switching-5kfold-lgbm-tracking
- https://www.kaggle.com/code/robikscube/fast-tuning-with-xgboost-3-0-optuna-gpus

Lista completa (~100 links) da pesquisa sobre modelo espacial (CNN/ConvLSTM/
U-Net/GNN) e regressão estatística em clima: ver [REFERENCIAS.md](REFERENCIAS.md).
