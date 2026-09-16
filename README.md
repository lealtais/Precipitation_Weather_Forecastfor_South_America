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
- **Índice ONI (Oceanic Niño Index, NOAA)** do mês atual — proxy direto do
  estado do ENSO (El Niño/La Niña), 2ª feature mais importante
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
| v4 | últimos 5 anos | 1965+ | lags + suavização 3x3 | 1.7980 | 1.8352 | 2.0% | **1.9557** (#20) |
| v5 | El Niño (48 meses) | 1965+ | igual a v4 | 1.9083 | 2.0599 | 7.4% | — |
| v6 | El Niño (48 meses) | 1965+ | v5 + índice ONI | **1.8905** | 2.0599 | **8.2%** | *(a submeter)* |

Tentativas descartadas:
- `objective="tweedie"` sobre o valor absoluto (em vez do resíduo): piorou (-2.9%)
- Suavização espacial 5x5 (além do 3x3) + modelo maior: sem ganho (2.0% → 2.0%)

## Arquivos

- `kaggle_notebook.py` — versão para rodar direto num notebook do Kaggle
  (dataset já montado em `/kaggle/input/...`)
- `local_run.py` — versão equivalente para rodar localmente, processando um
  arquivo NetCDF por vez para caber em pouca RAM livre; gera `submission.csv`
  pronto para envio direto pelo Kaggle (Submit Predictions)
- `inspect_data.py` / `inspect_data2.py` — scripts usados para entender a
  estrutura dos arquivos NetCDF e a relação entre `tp` / `tp_alvo` / teste
- `oni.ascii.txt` — índice ONI histórico (NOAA), usado como feature de ENSO

## Como reproduzir

```bash
python -m kagglehub competition_download previsao-climatica-de-precipitacao-sobre-a-america-do-sul
curl -o oni.ascii.txt https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
python local_run.py
```

Ajuste `DATA_DIR`, `YEAR_START` e `SMOOTH_SIZES` no topo de `local_run.py`
conforme a RAM disponível.
