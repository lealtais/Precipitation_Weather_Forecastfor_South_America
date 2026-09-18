# Referências completas da pesquisa

Lista completa de todos os links levantados durante a pesquisa sobre modelos
espaciais (CNN/ConvLSTM/U-Net/GNN) e regressão estatística em clima, feita
depois que o colega de competição sugeriu testar um "modelo espacial". Ver o
[README](README.md) pra um resumo curto do que foi usado de fato e das
conclusões práticas.

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

## Pesquisa adicional (2026-09-18): loss function, teleconexões extras, papers do mesmo problema

**Tweedie loss (a ideia mais promissora encontrada nessa rodada)** — RMSE
puro assume erro "gaussiano", mas precipitação é zero-inflada e assimétrica
(muitos meses perto de zero, poucos com valores bem altos). A distribuição
Tweedie foi feita exatamente pra esse tipo de dado, e tanto LightGBM
(`objective="tweedie"`) quanto XGBoost já suportam nativamente:
- https://arxiv.org/pdf/2509.08369 ("Stop using root-mean-square error as a precipitation target!")
- https://arxiv.org/pdf/2604.19340 (Tweedie + feature selection em pós-processamento de previsão do tempo)
- https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/155334 (Tweedie no LightGBM, discussão prática)
- https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
- https://github.com/catboost/tutorials/blob/master/regression/tweedie.ipynb
- https://sathesant.medium.com/tweedie-loss-function-395d96883f0b

**Papers específicos sobre o MESMO problema (precipitação sazonal na América
do Sul com gradient boosting)** — vale ler antes de qualquer coisa, é o mais
próximo do nosso caso exato:
- https://arxiv.org/abs/2512.13910 (compara ML/DL/XAI pra previsão sazonal de precipitação na América do Sul)
- https://www.researchgate.net/publication/358256945_South_America_Seasonal_Precipitation_Prediction_by_Gradient-Boosting_Machine-Learning_Approach (usa Optuna pra tuning, mesma técnica que já testamos)
- https://link.springer.com/article/10.1007/s40314-025-03438-x (LightGBM com quantificação de incerteza pra precipitação na América do Sul)

**Teleconexões além do ONI** — a literatura aponta outros índices climáticos
que também influenciam a chuva na América do Sul e que ainda não testamos:
- PDO (Pacific Decadal Oscillation) — https://www.nature.com/articles/s41612-024-00852-6
- SAM / Antarctic Oscillation — https://journals.ametsoc.org/view/journals/clim/22/22/2009jcli3036.1.xml
- MJO (Madden-Julian Oscillation) — https://arxiv.org/pdf/2507.20289 e https://psl.noaa.gov/mjo/MJOprimer/
- Visão geral de todas as teleconexões relevantes pra América do Sul — https://nyaspubs.onlinelibrary.wiley.com/doi/10.1111/nyas.14592

**Pós-processamento (bias correction) — aplicar depois da previsão, não no treino:**
- https://medium.com/@juanmi.gutierrez/quantile-mapping-bias-correction-63ed01d5a618
- https://doi.org/10.3390/rs15071743
- https://link.springer.com/article/10.1007/s12145-021-00577-7 (correção de viés com Random Forest, comparado com quantile mapping)

**Técnicas gerais de Kaggle Grandmaster (tabular):**
- https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data
- https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/
- https://app.daily.dev/posts/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data-l9wf9m7oo
