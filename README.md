import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Carregando os dados do Excel
df = pd.read_excel("ibov.xlsx")

# Convertendo a coluna 'Último' para numérico
df['Último'] = pd.to_numeric(df['Último'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
df = df.dropna(subset=['Último'])

# Criando uma série temporal
df['Data'] = pd.to_datetime(df['Data'])
df = df.set_index('Data')
data = df['Último'].values.reshape(-1, 1)

# Normalizando os dados
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Dividindo os dados em treino e teste
train_size = int(len(data_scaled) * 0.7)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# Definição do look_back
look_back = 7  # Ajustável conforme necessidade

# Função para criar os conjuntos de dados para a LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# Redimensionando os dados para o formato 3D
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Criando o modelo LSTM
model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=100, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')

# Treinando o modelo
model.fit(X_train, Y_train, epochs=300, batch_size=16, verbose=1)

# Fazendo previsões
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invertendo a escala para obter os valores originais
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# Avaliação do modelo
train_rmse = np.sqrt(mean_squared_error(Y_train[0], train_predict[:, 0]))
test_rmse = np.sqrt(mean_squared_error(Y_test[0], test_predict[:, 0]))
r2 = r2_score(Y_test[0], test_predict[:, 0])

print(f'Train RMSE: {train_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')
print(f'R² Score: {r2:.2f}')

# Cálculo da acurácia em percentual
correct_predictions = np.sum((np.abs(Y_test[0] - test_predict[:, 0]) / Y_test[0]) < 0.05)  # Tolerância de 5%
accuracy = (correct_predictions / len(Y_test[0])) * 100
print(f'Accuracy: {accuracy:.2f}%')  # Imprimindo a acurácia em percentual

# Preparando para análise por data
test_dates = df.index[train_size + look_back:len(df)]
df_results = pd.DataFrame({'Real': Y_test[0], 'Previsão': test_predict[:, 0]}, index=test_dates)
print(df_results)

# Plot dos resultados
plt.figure(figsize=(12, 6))
plt.plot(df_results['Real'], label='Real')
plt.plot(df_results['Previsão'], label='Previsão')
plt.legend()
plt.show()
