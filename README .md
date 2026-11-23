# Detecção de Anomalias Não Supervisionada em Séries Temporais

Atividade de detecção de falhas em um sistema monitorado por **sensor de temperatura**, com algoritmos de **aprendizado de máquina não supervisionado**. Os dados vêm do *Numenta Anomaly Benchmark* (NAB).

## Objetivo

Aplicar, comparar e combinar diferentes algoritmos não supervisionados para identificar comportamentos anômalos (indícios de falha) em uma série temporal de temperatura **sem usar rótulos no treinamento**.

## Dataset

- **Fonte:** [Numenta Anomaly Benchmark (NAB) no Kaggle](https://www.kaggle.com/datasets/boltzmannbrain/nab)
- **Arquivo usado:** `realKnownCause/ambient_temperature_system_failure.csv`
- **Conteúdo:** temperatura ambiente de um escritório medida de hora em hora, de julho de 2013 a maio de 2014 (7.267 registros), com falhas registradas no sistema de climatização.

O notebook baixa o arquivo automaticamente do [repositório oficial do NAB](https://github.com/numenta/NAB).

## Metodologia

1. **Pré-processamento:** conversão de Fahrenheit para Celsius e criação de atributos temporais (hora, dia/noite, dia da semana, dia útil/fim de semana).
2. **Análise exploratória:** a série foi dividida em **4 regimes operacionais** (dia útil ou fim de semana × dia ou noite), porque a temperatura "normal" muda conforme o contexto.
3. **Modelos aplicados**, todos assumindo 1% de anomalias (`outliers_fraction = 0.01`):

| # | Algoritmo | Ideia principal |
|---|---|---|
| 1 | **K-Means + PCA** | Pontos mais distantes do centro do seu grupo são anômalos |
| 2 | **Elliptic Envelope** | Distribuição Gaussiana ajustada separadamente para cada regime |
| 3 | **Isolation Forest** | Anomalias são isoladas com poucos cortes aleatórios |
| 4 | **One-Class SVM** | Aprende uma fronteira em volta dos dados normais |
| 5 | **LSTM** | Rede neural prevê a próxima temperatura; erros grandes indicam anomalia |

4. **Consenso entre modelos:** um ponto é considerado anomalia forte quando **3 ou mais modelos** concordam.
5. **Validação:** os alertas foram comparados com as janelas de falha que o próprio NAB registra para esta série. Esses rótulos **não foram usados no treinamento**:
   - 15/12/2013 a 30/12/2013 (pico de temperatura)
   - 29/03/2014 a 20/04/2014 (quedas bruscas de temperatura)

## Resultados

| Modelo | Nº de alertas | % dentro das janelas de falha | Falhas detectadas |
|---|---|---|---|
| K-Means + PCA | 72 | 59,7% | 2/2 |
| Elliptic Envelope | 75 | 54,7% | 2/2 |
| Isolation Forest | 73 | 34,2% | 2/2 |
| One-Class SVM | 74 | 71,6% | 2/2 |
| LSTM | 72 | 8,3% | 1/2 |
| **Consenso (≥ 3 modelos)** | **36** | **83,3%** | **2/2** |

> Como referência, um detector aleatório acertaria cerca de **10%** (a fração da série coberta pelas janelas de falha).
> Os resultados da LSTM podem variar um pouco a cada execução.

### Principais conclusões

- Os modelos baseados em **valor e contexto** (K-Means, Elliptic Envelope, Isolation Forest e One-Class SVM) detectaram as **duas falhas reais**.
- A **LSTM** ficou próxima do acaso: a maioria dos seus alertas caiu no trecho final da série, que não foi usado no treino. Ela reagiu mais a dados novos do que a falhas reais.
- O **consenso entre modelos** foi a estratégia mais precisa: 83,3% dos alertas caíram dentro das janelas de falha.
- O contexto temporal (hora e dia da semana) é essencial: uma mesma temperatura pode ser normal à noite e anômala durante o expediente.

## Como executar

**Opção 1: Google Colab (recomendado)**
1. Abra o notebook no [Google Colab](https://colab.research.google.com/) (Arquivo → Fazer upload de notebook).
2. Clique em **Ambiente de execução → Executar tudo**.

**Opção 2: localmente**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
jupyter notebook deteccao_anomalias_nab.ipynb
```

> O comando `!wget` da primeira célula funciona no Colab e no Linux. No Windows, baixe o arquivo CSV manualmente do Kaggle e coloque-o na mesma pasta do notebook.

## Estrutura do repositório

```
├── deteccao_anomalias_nab.ipynb   # notebook com todo o desenvolvimento
└── README.md                      # este arquivo
```

## Tecnologias

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · TensorFlow/Keras

## Referências

- Lavin, A.; Ahmad, S. *Evaluating Real-time Anomaly Detection Algorithms – the Numenta Anomaly Benchmark*. 14th International Conference on Machine Learning and Applications (IEEE ICMLA), 2015.
- [Numenta Anomaly Benchmark (NAB) no GitHub](https://github.com/numenta/NAB)
- [Unsupervised Anomaly Detection, Victor Ambonati (Kaggle)](https://www.kaggle.com/code/victorambonati/unsupervised-anomaly-detection)
