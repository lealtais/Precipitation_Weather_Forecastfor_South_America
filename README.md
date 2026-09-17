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

**Recomendação de uso agora:** `v5.2` (decide o melhor tipo de modelo) e
`v6.2` (decide ONI/tendência/Ridge) — os outros ficam só de histórico.

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
tamanho. Conclusão: vale tentar, mas só com uma CNN rasa e bem regularizada
-- ainda não implementado.

**Regressão estatística (Ridge/Lasso/GAM/quantílica)** — pesquisa mostrou que
regressão regularizada (Ridge/Lasso) lida bem com a multicolinearidade que já
tínhamos identificado entre nossas variáveis atmosféricas, e é usada na
literatura de downscaling climático com índices de teleconexão (ENSO etc.).
Não costuma superar gradient boosting em RMSE puro, mas um blend
(LightGBM + Ridge) é barato de testar -- implementado no `kaggle_notebook_v6.2.py`.

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

Todos os links levantados na pesquisa (paper principal já citado no corpo do README + toda a busca sobre modelo espacial/CNN e regressão estatística):

**CNN / modelos espaciais para precipitação:**
- https://arxiv.org/abs/2512.13910 (paper principal, ZCAS/ZCIT e multicolinearidade)
- https://medium.com/@sujanbhattarai.jr/convolutional-neural-networks-in-weather-forecasting-a-technical-assessment-9b0329fcd736
- https://mdpi.com/2072-4292/12/17/2731/htm
- https://www.sciencedirect.com/science/article/abs/pii/S0022169421013512
- https://www.researchgate.net/publication/368381015_Improving_Precipitation_Forecasts_with_Convolutional_Neural_Networks
- https://arxiv.org/pdf/2503.19943
- https://arxiv.org/pdf/2401.03746
- https://www.sciencedirect.com/science/article/pii/S2590197425000783
- https://journals.ametsoc.org/view/journals/wefo/38/2/WAF-D-22-0002.1.xml
- https://www.sciencedirect.com/science/article/abs/pii/S0022169423008089
- https://arxiv.org/pdf/2409.09607
- https://www.sciencedirect.com/science/article/abs/pii/S0377026523000738 (SACZ x SST, motivou o teste do TNA/TSA)

**ConvLSTM / nowcasting:**
- https://arxiv.org/abs/1506.04214
- https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-33/issue-4/043053/Precipitation-nowcasting-based-on-ConvLSTM-UNet-deep-spatiotemporal-network/10.1117/1.JEI.33.4.043053.short
- https://arxiv.org/pdf/2406.04867
- http://papers.neurips.cc/paper/7145-deep-learning-for-precipitation-nowcasting-a-benchmark-and-a-new-model.pdf
- https://arxiv.org/pdf/2511.11197
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10346408/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7065808/
- https://dl.acm.org/doi/10.5555/2969239.2969329

**U-Net:**
- https://arxiv.org/pdf/2407.02125
- https://arxiv.org/pdf/2607.04862
- https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2024.1439906/full
- https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2023.1116672/xml
- https://www.sciencedirect.com/science/article/pii/S1877050925016588
- https://www.sciencedirect.com/science/article/pii/S1364682625002986
- https://arxiv.org/pdf/2008.09090
- https://arxiv.org/pdf/2501.02814

**Kaggle/PyTorch + ERA5:**
- https://www.kaggle.com/code/hanjoonchoe/cnn-time-series-forecasting-with-pytorch
- https://arxiv.org/pdf/2112.06571
- https://docs.digitalearthafrica.org/en/latest/sandbox/notebooks/Datasets/Climate_Data_ERA5_AWS.html
- https://mldata.pangeo.io/preprocessed_datasets.html
- https://arxiv.org/pdf/2501.02905
- https://arxiv.org/pdf/2302.04102
- https://arxiv.org/pdf/2411.16098
- https://github.com/indrakalita/RainfallForecasting

**Graph Neural Networks (clima):**
- https://arxiv.org/html/2512.00546
- https://www.sciencedirect.com/org/science/article/pii/S1546221825006307
- https://arxiv.org/pdf/2402.14861
- https://ramkrishna2910.medium.com/graphcast-a-breakthrough-in-weather-forecasting-d70fae9ac365
- https://arxiv.org/pdf/2306.00012
- https://arxiv.org/html/2410.12938v1
- https://arxiv.org/pdf/2606.27202
- https://arxiv.org/pdf/2409.11605
- https://arxiv.org/pdf/2411.03223
- https://arxiv.org/pdf/2403.17384

**Deep learning pra precipitação sazonal na América do Sul:**
- https://doi.org/10.3390/rs13132468
- https://arxiv.org/pdf/2602.04757
- https://sol.sbc.org.br/index.php/wcama/article/download/36093/35880/

**WeatherBench (benchmark ERA5):**
- https://www.emergentmind.com/topics/weatherbench
- https://mediatum.ub.tum.de/doc/1597509/1597509.pdf
- https://arxiv.org/pdf/2306.05420
- https://arxiv.org/pdf/2205.00865
- https://www.cs.toronto.edu/~soukayna/weatherbench.html
- https://arxiv.org/pdf/2002.00469
- https://github.com/pangeo-data/WeatherBench
- https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS004019

**Overfitting de CNN em dataset climático pequeno (achado-chave):**
- https://www.nature.com/articles/s43247-021-00225-4
- https://www.researchgate.net/publication/328846810_Monthly_Rainfall_Forecasting_Using_One-Dimensional_Deep_Convolutional_Neural_Network
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11928313/
- https://arxiv.org/pdf/2408.10550
- https://www.tandfonline.com/doi/full/10.1080/22797254.2025.2540106
- https://arxiv.org/pdf/1811.04817
- https://arxiv.org/pdf/2504.20442

**Gradient boosting vs. deep learning em dado tabular:**
- https://arxiv.org/pdf/2307.14338
- https://arxiv.org/pdf/2106.03253
- https://arxiv.org/pdf/2209.12309
- https://arxiv.org/pdf/2502.02672
- https://www.nature.com/articles/s41598-025-04634-9
- https://arxiv.org/pdf/2301.01252
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12944068/
- https://www.sciencedirect.com/science/article/abs/pii/S1566253521002360
- https://blog.dailydoseofds.com/p/top-gradient-boosting-methods
- https://arxiv.org/pdf/2302.11777

**Regressão estatística em clima (geral):**
- https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2024.1378598/full
- https://www.nature.com/articles/s41598-025-13567-2
- https://www.sciencedirect.com/science/article/pii/S2214581825008171
- https://www.sciencedirect.com/science/article/abs/pii/S0957417424001258
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9734427/
- https://arxiv.org/pdf/2501.16900
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10256754/
- https://arxiv.org/pdf/2602.08865
- https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2024.1445967/full

**Regressão quantílica (eventos extremos):**
- https://iopscience.iop.org/article/10.1088/1742-6596/1918/4/042031
- https://arxiv.org/abs/1806.05429
- https://link.springer.com/article/10.1007/s10687-019-00355-1
- https://arxiv.org/pdf/2103.00808

**GAM (Generalized Additive Models):**
- https://link.springer.com/article/10.1007/s00704-023-04725-5
- https://arxiv.org/pdf/1907.13095
- https://www.sciencedirect.com/science/article/abs/pii/S1125786520300825
- https://arxiv.org/pdf/2604.15067
- https://pubmed.ncbi.nlm.nih.gov/31398655/
- https://arxiv.org/pdf/2603.14984
- https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003753

**Downscaling estatístico e teleconexões:**
- https://www.sciencedirect.com/science/article/pii/S2212094723000907
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8549900/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10393634/
- https://rmets.onlinelibrary.wiley.com/doi/10.1002/joc.7290
- https://arxiv.org/pdf/1702.04018
- https://arxiv.org/pdf/2403.17847
- https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025GL119307
- https://climate.copernicus.eu/sites/default/files/2021-01/infosheet8.pdf

**Ridge/Lasso/PCR + ENSO:**
- https://link.springer.com/article/10.1007/s12517-024-12111-2
- https://arxiv.org/pdf/1901.05397
- https://www.tandfonline.com/doi/full/10.1080/02626667.2019.1570210
- https://link.springer.com/article/10.1007/s00704-026-06243-6 (precipitação extrema na América do Sul com índices climáticos)
- https://journals.ametsoc.org/view/journals/mwre/148/10/mwrD190302.xml
- https://arxiv.org/pdf/2207.04794
- https://arxiv.org/pdf/1602.06872
- https://arxiv.org/pdf/2307.01872
