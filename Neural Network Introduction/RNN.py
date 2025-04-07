import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import random

# ============================
# Helpers: Ativações e Dropout
# ============================
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class Dropout:
    def __init__(self, dropout_rate, seed=42):
        self.rate = dropout_rate
        self.seed = seed
        
    def forward(self, X, training=True):
        if training:
            np.random.seed(self.seed)  # Garante reprodutibilidade para a máscara de dropout
            self.mask = (np.random.rand(*X.shape) >= self.rate).astype(float) / (1 - self.rate)
            return X * self.mask
        else:
            return X

    def backward(self, dY):
        return dY * self.mask

# ============================
# Camada Dense (Totalmente Conectada)
# ============================
class DenseLayer:
    def __init__(self, input_dim, output_dim, seed=42):
        np.random.seed(seed)
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.b = np.zeros((1, output_dim))
    
    def forward(self, X):
        self.X = X  # Armazena para o backprop
        return np.dot(X, self.W) + self.b

    def backward(self, d_out, learning_rate):
        dW = np.dot(self.X.T, d_out)
        db = np.sum(d_out, axis=0, keepdims=True)
        dX = np.dot(d_out, self.W.T)
        # Atualização dos parâmetros
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return dX

# ============================
# Camada RNN Simples
# ============================
class RNNLayer:
    def __init__(self, input_dim, hidden_dim, return_sequences=True, seed=42):
        np.random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        # Pesos de entrada e recorrentes
        self.Wx = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2. / hidden_dim)
        self.b = np.zeros(hidden_dim)
    
    def forward(self, X, training=True):
        # X: shape (batch, seq_length, input_dim)
        batch_size, seq_length, _ = X.shape
        self.x = X
        self.h = np.zeros((batch_size, seq_length, self.hidden_dim))
        self.z = np.zeros((batch_size, seq_length, self.hidden_dim))
        # Estado oculto inicial: zeros
        h_prev = np.zeros((batch_size, self.hidden_dim))
        for t in range(seq_length):
            self.z[:, t, :] = np.dot(X[:, t, :], self.Wx) + np.dot(h_prev, self.Wh) + self.b
            self.h[:, t, :] = relu(self.z[:, t, :])
            h_prev = self.h[:, t, :].copy()
        return self.h if self.return_sequences else self.h[:, -1, :]
    
    def backward(self, d_out, learning_rate):
        # d_out: gradiente vindo da camada seguinte.
        batch_size, seq_length = self.x.shape[0], self.x.shape[1]
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(self.x)
        dh_next = np.zeros((batch_size, self.hidden_dim))
        
        # Organiza o gradiente para todos os timesteps
        if not self.return_sequences:
            d_h = np.zeros((batch_size, seq_length, self.hidden_dim))
            d_h[:, -1, :] = d_out
        else:
            d_h = d_out
        
        # Backpropagation through time (BPTT)
        for t in reversed(range(seq_length)):
            dh = d_h[:, t, :] + dh_next  # soma do gradiente com o do próximo timestep
            dz = dh * relu_derivative(self.z[:, t, :])
            db += np.sum(dz, axis=0)
            dWx += np.dot(self.x[:, t, :].T, dz)
            h_prev = self.h[:, t-1, :] if t > 0 else np.zeros((batch_size, self.hidden_dim))
            dWh += np.dot(h_prev.T, dz)
            dX[:, t, :] = np.dot(dz, self.Wx.T)
            dh_next = np.dot(dz, self.Wh.T)
        
        # Atualização dos parâmetros
        self.Wx -= learning_rate * dWx
        self.Wh -= learning_rate * dWh
        self.b  -= learning_rate * db
        
        return dX

# ============================
# Modelo RNN Completo
# ============================
class RNNModel:
    def __init__(self, seed=42):
        self.rnn1 = RNNLayer(input_dim=1, hidden_dim=64, return_sequences=True, seed=seed)
        self.dropout1 = Dropout(dropout_rate=0.3, seed=seed)
        self.rnn2 = RNNLayer(input_dim=64, hidden_dim=32, return_sequences=False, seed=seed)
        self.dropout2 = Dropout(dropout_rate=0.3, seed=seed)
        self.dense = DenseLayer(input_dim=32, output_dim=1, seed=seed)
    
    def forward(self, X, training=True):
        out1 = self.rnn1.forward(X, training)
        out1 = self.dropout1.forward(out1, training)
        out2 = self.rnn2.forward(out1, training)
        out2 = self.dropout2.forward(out2, training)
        out = self.dense.forward(out2)
        return out
    
    def backward(self, d_out, learning_rate):
        d_dense = self.dense.backward(d_out, learning_rate)
        d_dropout2 = self.dropout2.backward(d_dense)
        d_rnn2 = self.rnn2.backward(d_dropout2, learning_rate)
        d_dropout1 = self.dropout1.backward(d_rnn2)
        self.rnn1.backward(d_dropout1, learning_rate)

# ============================
# Função para criar sequências
# ============================
def create_sequences(data, seq_length):
    """Cria sequências e rótulos para o treinamento."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# ============================
# Função de denormalização
# ============================
def denormalize(normalized_data, data_min, data_max):
    return normalized_data * (data_max - data_min) + data_min

# ============================
# Main: Carregamento, Normalização, Divisão, Treinamento e Avaliação
# ============================
if __name__ == "__main__":
    # 1. Configurar semente global para reprodutibilidade
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    
    # 2. Carregar a planilha com polars
    filepath = r"C:\Users\Leo\Desktop\Portfolio\EconDynamicsHub\Neural Network Introduction\AEP_hourly1.xlsx"
    df = pl.read_excel(filepath)
    # Converter a coluna "Datetime" para datetime se necessário (não usada no treinamento)
    if df["Datetime"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("Datetime").str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S"))
    
    # 3. Extrair a série de consumo ("AEP_MW") e normalizar para [0,1]
    data_series = df["AEP_MW"].to_numpy()
    data_min = np.min(data_series)
    data_max = np.max(data_series)
    data_normalized = (data_series - data_min) / (data_max - data_min)
    
    # 4. Criar sequências (exemplo: janela de 24 horas)
    SEQ_LENGTH = 24  # ajuste conforme necessário
    X, y = create_sequences(data_normalized, SEQ_LENGTH)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # formato: (samples, timesteps, features)
    
    # 5. Dividir os dados sequencialmente: 70% treino, 20% validação, 10% teste
    n_samples = X.shape[0]
    train_end = int(n_samples * 0.7)
    val_end = train_end + int(n_samples * 0.2)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print("Shapes dos dados:")
    print("Treino:", X_train.shape, y_train.shape)
    print("Validação:", X_val.shape, y_val.shape)
    print("Teste:", X_test.shape, y_test.shape)
    
    # ============================
    # Treinamento do Modelo com EarlyStopping
    # ============================
    model = RNNModel(seed=SEED)
    epochs = 50
    batch_size = 32
    learning_rate = 0.001
    patience = 5

    best_val_loss = np.inf
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Embaralhar os dados de treino para cada época
        perm = np.random.permutation(X_train.shape[0])
        X_train = X_train[perm]
        y_train = y_train[perm]
        
        train_loss = 0
        num_batches = int(np.ceil(X_train.shape[0] / batch_size))
        
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, X_train.shape[0])
            X_batch = X_train[start:end]
            y_batch = y_train[start:end].reshape(-1, 1)
            
            # Forward pass
            predictions = model.forward(X_batch, training=True)
            loss = np.mean((predictions - y_batch) ** 2)
            train_loss += loss * (end - start)
            
            # Backward pass: derivada do MSE
            d_loss = (2 * (predictions - y_batch)) / y_batch.shape[0]
            model.backward(d_loss, learning_rate)
        
        train_loss /= X_train.shape[0]
        history["train_loss"].append(train_loss)
        
        # Cálculo da loss no conjunto de validação
        val_predictions = model.forward(X_val, training=False)
        val_loss = np.mean((val_predictions - y_val.reshape(-1, 1)) ** 2)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch+1:02d} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        # EarlyStopping: interrompe se não houver melhora por 'patience' épocas
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # ============================
    # 11. Gerar previsões e denormalizar para validação e teste
    # ============================
    # Previsões no conjunto de validação
    pred_val_norm = model.forward(X_val, training=False)
    pred_val_denorm = denormalize(pred_val_norm, data_min, data_max)
    y_val_denorm = denormalize(y_val.reshape(-1, 1), data_min, data_max)
    
    # Previsões no conjunto de teste
    pred_test_norm = model.forward(X_test, training=False)
    pred_test_denorm = denormalize(pred_test_norm, data_min, data_max)
    y_test_denorm = denormalize(y_test.reshape(-1, 1), data_min, data_max)
    
    # ============================
    # 12. Cálculo das métricas: MSE e MAPE para validação e teste
    # ============================
    def compute_metrics(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mse, mape

    val_mse, val_mape = compute_metrics(y_val_denorm, pred_val_denorm)
    test_mse, test_mape = compute_metrics(y_test_denorm, pred_test_denorm)
    
    print("\nMétricas no conjunto de Validação:")
    print(f"MSE: {val_mse:.3f} | MAPE: {val_mape:.3f}%")
    print("\nMétricas no conjunto de Teste:")
    print(f"MSE: {test_mse:.3f} | MAPE: {test_mape:.3f}%")
    
    # ============================
    # 13. Plot do histórico de treinamento (loss)
    # ============================
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Épocas")
    plt.ylabel("MSE")
    plt.title("Histórico de Treinamento")
    plt.legend()
    plt.show()
    
    # ============================
    # 14. Plot Predito vs. Real para o conjunto de Teste
    # ============================
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_denorm, label="Real", alpha=0.7)
    plt.plot(pred_test_denorm, label="Predito", alpha=0.7)
    plt.xlabel("Índice da amostra")
    plt.ylabel("Consumo (MW)")
    plt.title("Previsão vs. Valor Real (Teste)")
    plt.legend()
    plt.show()
