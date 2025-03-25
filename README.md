import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

# Carregar os dados
url = 'https://raw.githubusercontent.com/rosimotasp/https-br.investing.com-indices-bovespa-historical-data/main/Dados%20Invest.xlsx'
df = pd.read_excel(url)

# Converter a coluna 'Data' para datetime e ordenar
df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
df = df.sort_values(by='Data')

# Manter apenas a coluna 'Último' (preço de fechamento)
df = df[['Data', 'Último']].dropna()
df.columns = ['ds', 'y']  # Renomear para uso no Prophet

# Definir data limite para treino (80% dos dados)
data_limite = df['ds'].quantile(0.8)
df_treino = df[df['ds'] <= data_limite].copy()
df_teste = df[df['ds'] > data_limite].copy()

# Criar e ajustar o modelo Prophet
modelo = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.1)
modelo.add_seasonality(name='monthly', period=30.5, fourier_order=5)
modelo.fit(df_treino)

# Criar datas futuras apenas para o período de teste
future = df_teste[['ds']]
forecast = modelo.predict(future)

# Calcular a acurácia (MAPE) nos dados de teste
mape = mean_absolute_percentage_error(df_teste['y'], forecast['yhat']) * 100
print(f"\nAcurácia do modelo (MAPE) nos dados de teste: {100 - mape:.2f}%")

# Comparar dados reais com previsões
df_comparacao = df_teste[['ds', 'y']].copy()
df_comparacao['Previsto'] = forecast['yhat'].values
print("\nAmostra dos dados reais e previstos:")
print(df_comparacao.head(10))

# Plotar os resultados
plt.figure(figsize=(12, 6))
plt.plot(df_treino['ds'], df_treino['y'], label='Treino', color='blue')
plt.plot(df_teste['ds'], df_teste['y'], label='Teste (real)', color='green')
plt.plot(forecast['ds'], forecast['yhat'], label='Previsão', linestyle='dashed', color='red')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2)
plt.axvline(x=df_teste['ds'].iloc[0], color='black', linestyle='dotted', label='Início da previsão')
plt.title('Previsão do IBOVESPA com Prophet')
plt.xlabel('Data')
plt.ylabel('Valor')
plt.legend()
plt.grid()
plt.show()
