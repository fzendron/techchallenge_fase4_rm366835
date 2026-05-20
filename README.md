# FIAP Pos Tech MLET - Tech Challenge Fase 4

Projeto de previsão do preço de fechamento de ações usando uma rede neural LSTM, FastAPI, Docker e monitoramento compatível com Prometheus.

O experimento padrão prevê o valor de `Close` do próximo pregão para `MELI` usando dados do Yahoo Finance a partir de `2018-01-01`. O ticker, o intervalo de datas, os ajustes do modelo e os caminhos dos artefatos podem ser configurados em [configs/default.yaml](configs/default.yaml).

Este projeto é destinado apenas a uso acadêmico e educacional. Não é aconselhamento financeiro.

## Escopo do Desafio

Este repositório implementa os principais requisitos do desafio FIAP Pos Tech MLET Fase 4:

- Coletar dados históricos OHLCV de ações com `yfinance`.
- Construir um conjunto de dados supervisionado de série temporal para previsão do `Close` do próximo dia.
- Treinar um modelo LSTM com Keras/TensorFlow.
- Avaliar o modelo com métricas de regressão: MAE, RMSE e MAPE.
- Comparar o LSTM com uma linha de base ingênua em que o fechamento de amanhã é igual ao de hoje.
- Salvar o modelo treinado, o scaler, os metadados, as métricas e os gráficos.
- Expor previsões por meio de uma API REST com FastAPI.
- Oferecer execução com Docker e monitoramento compatível com Prometheus.
- Incluir testes automatizados para geração de features, validação de inferência e rotas da API.

## Estrutura do Repositório

```text
.
+-- README.md
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
+-- pyproject.toml
+-- .gitignore
+-- .dockerignore
+-- app/
|   +-- main.py          # Aplicação FastAPI e rotas
|   +-- schemas.py       # Modelos Pydantic de request/response
|   +-- inference.py     # Carregamento de artefatos e lógica de predição
|   +-- monitoring.py    # Métricas Prometheus e middleware de latência
|   +-- config.py        # Configurações em tempo de execução
+-- src/
|   +-- data.py          # download com yfinance e validação de dados
|   +-- features.py      # seleção de features, scaling, splits, windows
|   +-- model.py         # Arquitetura LSTM
|   +-- train.py         # CLI de treinamento
|   +-- evaluate.py      # CLI de avaliação
|   +-- utils.py         # helpers compartilhados
+-- configs/
|   +-- default.yaml
+-- models/
|   +-- .gitkeep         # artefatos gerados são ignorados pelo git
+-- reports/
|   +-- .gitkeep         # relatórios gerados são ignorados pelo git
+-- tests/
    +-- test_features.py
    +-- test_inference.py
    +-- test_api.py
```

## Abordagem de Modelagem

O projeto trata a previsão de ações como um problema supervisionado de regressão:

- Features de entrada: `Open`, `High`, `Low`, `Close`, `Volume`
- Target: `Close` do próximo pregão
- Ticker padrão: `AAPL`
- Data inicial padrão: `2018-01-01`
- Tamanho padrão da janela: `60` pregões
- Horizonte padrão de previsão: `1` pregão
- Estratégia de split: cronológica, com `70%` para treino, `15%` para validação e `15%` para teste

A avaliação final evita intencionalmente splits aleatórios de treino e teste, porque eles vazam informação futura do mercado para o conjunto de treinamento. O scaler é ajustado apenas no período de treino e depois reutilizado para validação, teste e inferência da API.

## Arquitetura LSTM

Arquitetura padrão:

```text
Input(shape=(window_size, n_features))
LSTM(64, return_sequences=True)
Dropout(0.2)
LSTM(32)
Dropout(0.2)
Dense(16, activation="relu")
Dense(1)
```

O treinamento usa Adam e MSE por padrão. O loop de treinamento também usa:

- `EarlyStopping`
- `ModelCheckpoint`
- `ReduceLROnPlateau`

Todos os padrões podem ser alterados em [configs/default.yaml](configs/default.yaml).

## Pré-requisitos

- Python 3.11 ou mais recente
- `pip`
- Docker e Docker Compose, se a execução for via containers
- Acesso à internet para instalação de dependências e download de dados do Yahoo Finance

O suporte do TensorFlow varia conforme a versão do Python e o sistema operacional. O `Dockerfile` fornecido usa Python 3.11 para manter o runtime reproduzível.

## Configuração Local

Crie e ative um ambiente virtual:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Se sua máquina não tiver `python3.11`, use outra versão suportada do Python 3:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuração

A configuração padrão está em [configs/default.yaml](configs/default.yaml). Campos importantes:

```yaml
data:
  ticker: MELI
  start_date: "2008-01-01"
  end_date: null

features:
  window_size: 60
  horizon: 1
  train_ratio: 0.70
  validation_ratio: 0.15
  test_ratio: 0.15

artifacts:
  model_path: models/lstm_stock_model.keras
  scaler_path: models/feature_scaler.joblib
  metadata_path: models/metadata.json
```

Defina `data.end_date` com uma data fixa se quiser resultados acadêmicos reproduzíveis. Deixe como `null` para usar os dados mais recentes disponíveis no Yahoo Finance.

## Treinamento

Execute:

```bash
python -m src.train --config configs/default.yaml
```

O treinamento executa o pipeline completo:

1. Baixa dados históricos com `yfinance`.
2. Valida as colunas OHLCV obrigatórias.
3. Ordena as linhas por data e remove linhas inválidas ou ausentes.
4. Cria splits cronológicos de treino, validação e teste.
5. Ajusta o scaler apenas no período de treino.
6. Gera janelas deslizantes com shape `(samples, timesteps, features)`.
7. Treina o modelo LSTM.
8. Calcula métricas do LSTM e da linha de base ingênua.
9. Salva artefatos e gráficos.

Arquivos gerados:

- `models/lstm_stock_model.keras`
- `models/feature_scaler.joblib`
- `models/metadata.json`
- `reports/metrics.json`
- `reports/training_history.png`
- `reports/predictions.png`

Os arquivos gerados do modelo são ignorados pelo git de propósito, porque podem ser grandes e específicos do ambiente.

## Avaliação

Execute:

```bash
python -m src.evaluate --config configs/default.yaml
```

A avaliação recarrega o modelo salvo, o scaler e os metadados, e então recalcula as métricas de teste usando o mesmo split cronológico. As métricas são gravadas em:

```text
reports/metrics.json
```

Definições das métricas:

- `MAE`: erro absoluto médio da predição em unidades de preço.
- `RMSE`: raiz da média do erro quadrático, mais sensível a erros grandes.
- `MAPE`: erro percentual médio, útil para comparação sensível à escala.

As previsões são retransformadas para valores de preço antes de as métricas serem calculadas.

## Execução Rápida de Smoke Test

O treinamento completo pode levar tempo dependendo do hardware. Para um smoke run rápido, reduza temporariamente os valores em `configs/default.yaml`:

```yaml
data:
  start_date: "2023-01-01"

features:
  window_size: 20

model:
  epochs: 2
  batch_size: 16
```

Para o relatório acadêmico final, use um período histórico mais longo e mantenha a estratégia de avaliação cronológica.

## Executar a API Localmente

Treine o modelo primeiro para que a API consiga carregar os artefatos necessários:

```bash
python -m src.train --config configs/default.yaml
```

Inicie o FastAPI:

```bash
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```

Abra:

```text
http://localhost:8000/docs
```

Endpoints disponíveis:

| Método | Caminho | Finalidade |
| --- | --- | --- |
| `GET` | `/health` | Saúde da API e status de modelo carregado |
| `GET` | `/model-info` | Metadados do modelo carregado e status dos artefatos |
| `POST` | `/predict` | Prediz o próximo preço de fechamento |
| `GET` | `/metrics` | Métricas compatíveis com Prometheus |

Se os artefatos estiverem ausentes, `/health` ainda responde, mas `model_loaded` será `false` e `/predict` retornará `503`.

## Solicitação de Previsão

A API espera pelo menos `window_size` linhas históricas. O padrão é 60 linhas. Cada linha deve incluir:

- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

O campo opcional `date` é recomendado. Se as datas forem fornecidas, a API ordena as linhas cronologicamente antes de selecionar a janela mais recente para predição.

Exemplo de solicitação com 60 linhas sintéticas:

```bash
python - <<'PY' > /tmp/predict.json
import json
from datetime import date, timedelta

rows = []
start = date(2024, 1, 2)
for i in range(60):
    rows.append({
        "date": str(start + timedelta(days=i)),
        "Open": 180 + i * 0.1,
        "High": 181 + i * 0.1,
        "Low": 179 + i * 0.1,
        "Close": 180.5 + i * 0.1,
        "Volume": 80000000 + i * 10000
    })

print(json.dumps({"request_id": "demo-1", "historical_prices": rows}))
PY

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  --data @/tmp/predict.json
```

Exemplo de resposta:

```json
{
  "ticker": "AAPL",
  "horizon": 1,
  "predicted_close": 189.42,
  "currency": null,
  "model_version": "2026-05-18T21:00:00+00:00",
  "input_rows": 60,
  "feature_order": ["Open", "High", "Low", "Close", "Volume"],
  "request_id": "demo-1"
}
```

O valor numérico acima é apenas ilustrativo. A saída real depende do modelo treinado e dos dados de entrada.

## Docker

Construa e execute:

```bash
docker compose up --build
```

A API é exposta em:

```text
http://localhost:8000
```

O arquivo compose monta `./models` no container como somente leitura. Treine localmente primeiro ou copie artefatos válidos para `models/` antes de iniciar o container da API.

Comandos úteis do Docker:

```bash
docker compose ps
docker compose logs -f stock-api
docker compose down
```

## Monitoramento

`GET /metrics` expõe métricas em texto no formato Prometheus, incluindo:

- `stock_api_requests_total`
- `stock_api_request_latency_seconds`
- `stock_api_errors_total`
- `stock_api_predictions_total`
- `stock_api_model_loaded`
- `stock_api_process_memory_bytes`

Exemplo:

```bash
curl http://localhost:8000/metrics
```

O app também emite logs JSON estruturados com eventos de predição e erros, o que facilita monitorar o serviço em Docker, logs de cloud ou ferramentas de observabilidade.

## Testes

Execute:

```bash
pytest
```

Os testes cobrem:

- Formato das janelas deslizantes.
- Premissas de split cronológico.
- Ajuste do scaler apenas com os dados de treino.
- Rejeição de inferência quando poucas linhas são fornecidas.
- `/health`, `/model-info`, `/predict` e `/metrics`.

Os testes de API e inferência usam objetos fake leves de modelo, então não exigem um artifact TensorFlow treinado.

## Limitações Atuais

- O modelo usa apenas dados OHLCV; ele não inclui notícias, variáveis macroeconômicas, eventos de earnings ou sentiment.
- Um horizonte de um dia é mais simples de avaliar, mas não garante decisões de investimento úteis.
- Desempenho histórico não implica desempenho futuro.
- Modelos LSTM podem overfit em séries financeiras ruidosas, então a comparação com linha de base é obrigatória.
- MAPE pode ser instável para alvos próximos de zero, embora os preços de fechamento de ações normalmente fiquem longe de zero.
