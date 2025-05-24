import csv
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import randn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.header = []
        self.data = []

        try:
            self._load_data()
        except FileNotFoundError:
            print(f"soubor nenalezen: {file_path}")
        except IsADirectoryError:
            print(f"Očekáván soubor, ale '{file_path}' je složka.")

    def _load_data(self):
        """Načtení CSV souboru"""
        with open(self.file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)

            try:
                self.header = next(reader)
            except StopIteration:
                raise ValueError("Soubor je prázdný!")

            self.data = [row for row in reader]

    def shape(self):
        """Vrací tvar dat (počet řádků, počet sloupců)"""
        rows = len(self.data)
        cols = len(self.header)
        return (rows, cols)

    def columns(self):
        """Vrací názvy sloupců"""
        return self.header

    def head(self, n=5):
        """Vrací prvních n řádků"""
        return self.data[:n]


class DataFrameBuilder:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def to_dataframe(self):
        """Převede načtená data na pandas DataFrame"""
        return pd.DataFrame(self.data_loader.data, columns=self.data_loader.header)

    def to_clean_dataframe(self):
        """Vrací DataFrame bez řádků obsahujících None nebo NaN"""
        df = self.to_dataframe()
        return df.dropna()


# Cesta k datum
file_path = "/Users/anetastrnadova/Desktop/archive/train.csv"

# Objekt
loader = DataLoader(file_path)
df_builder = DataFrameBuilder(loader)
df_raw = df_builder.to_dataframe()
df_cleaned = df_builder.to_clean_dataframe()

temp_age = df_cleaned.loc[501:575, 'Age'].copy()
temp_gender = df_cleaned.loc[501:575, 'Gender'].copy()

df_cleaned.loc[501:575, 'Age'] = temp_gender
df_cleaned.loc[501:575, 'Gender'] = temp_age

temp_age = df_cleaned.loc[1505:1579, 'Age'].copy()
temp_gender = df_cleaned.loc[1505:1579, 'Gender'].copy()

df_cleaned.loc[1505:1579, 'Age'] = temp_gender
df_cleaned.loc[1505:1579, 'Gender'] = temp_age

# Výstupy
print("Shape:", loader.shape())
print("Názvy sloupců:", loader.columns())
print("Prvních 5 řádků:")
for row in loader.head():
    print(row)

df = df_cleaned.copy()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_neuron(X, y, mu=0.01, epochs=1000, test_size=0.2, val_size=0.1, random_state=42, patience=20, plot=True):
    # --- Rozdělení dat ---
    # Nejprve na train + temp (test + val)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=random_state)
    # Poté z temp udělej val a test
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - val_ratio, random_state=random_state)

    # --- Škálování ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Inicializace ---
    nw = X_train_scaled.shape[1]
    w = randn(nw) / nw
    b = 0.0

    Q_train = np.zeros(epochs)
    Q_val = np.zeros(epochs)
    best_val_loss = np.inf
    best_w, best_b = None, None
    wait = 0

    # --- Učení ---
    for epoch in range(epochs):
        z_train = np.dot(X_train_scaled, w) + b
        sigma_train = sigmoid(z_train)
        e_train = y_train - sigma_train
        Q_train[epoch] = np.mean(e_train ** 2)

        # Gradienty
        dw = -mu * 2 * np.dot(X_train_scaled.T, e_train) / X_train_scaled.shape[0]
        db = -mu * 2 * np.mean(e_train)
        w -= dw
        b -= db

        # Validační ztráta
        z_val = np.dot(X_val_scaled, w) + b
        sigma_val = sigmoid(z_val)
        e_val = y_val - sigma_val
        Q_val[epoch] = np.mean(e_val ** 2)

        # Early stopping
        if Q_val[epoch] < best_val_loss:
            best_val_loss = Q_val[epoch]
            best_w = w.copy()
            best_b = b
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    # --- Test ---
    z_test = np.dot(X_test_scaled, best_w) + best_b
    y_pred_prob = sigmoid(z_test)
    y_pred = np.where(y_pred_prob >= 0.5, 1, 0)

    # --- Metriky ---
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))

        axes[0].plot(y_test.values, 'k', label="y true")
        axes[0].plot(y_pred, 'g', label="y pred")
        axes[0].legend()
        axes[0].set_title("Predikce vs. Skutečnost")

        axes[1].semilogy(Q_train[:epoch], label="Train Loss")
        axes[1].semilogy(Q_val[:epoch], label="Val Loss")
        axes[1].set_title("Chybová funkce")
        axes[1].set_xlabel("Epochy")
        axes[1].legend()
        axes[1].grid()

        plt.tight_layout()
        plt.show()

    return {
        'weights': best_w,
        'bias': best_b,
        'accuracy': acc,
        'report': report,
        'loss_train': Q_train,
        'loss_val': Q_val,
        'scaler': scaler
    }


y = df['Dominant_Emotion'].apply(lambda x: 1 if x == 'Happiness' else 0)
X = df.drop(columns=['Dominant_Emotion', 'User_ID'])
X = pd.get_dummies(X, columns=['Gender', 'Platform'])  # převede stringové kategorie na čísla
X = X.astype(float)

result = train_neuron(X, y)

print("Test Accuracy:", result['accuracy'])
print(result['report'])

weights = pd.Series(result['weights'], index=X.columns)
print("Váhy s názvy vstupních proměnných:")
print(weights)

# Bias zvlášť
print("\nBias (b):", result['bias'])

import requests
import pandas as pd

API_KEY = "de882d30"

# Inline schéma v URL
url = (
    f"https://api.mockaroo.com/api/generate.json?key={API_KEY}&count=100"
    "&fields=["
    '{"name":"User_ID","type":"Row Number"},'
    '{"name":"Age","type":"Number","min":18,"max":50},'
    '{"name":"Gender","type":"Gender (Binary)"},'
    '{"name":"Platform","type":"Custom List","values":["Facebook","Instagram","LinkedIn","Snapchat","Twitter","Whatsapp"]},'
    '{"name":"Daily_Usage_Time (minutes)","type":"Number","min":10,"max":200},'
    '{"name":"Posts_Per_Day","type":"Number","min":1,"max":20},'
    '{"name":"Likes_Received_Per_Day","type":"Number","min":1,"max":70},'
    '{"name":"Comments_Received_Per_Day","type":"Number","min":1,"max":70},'
    '{"name":"Messages_Sent_Per_Day","type":"Number","min":1,"max":50}'
    "]"
)

try:
    response = requests.get(url)
    response.raise_for_status()

    # Načtení JSON dat do DataFrame
    df_new = pd.DataFrame(response.json())
    print(df_new.head())

    # Uložení do CSV
    df_new.to_csv("mockaroo_data.json", index=False)
    print("Data byla uložena do souboru 'mockaroo_data.json'")

except requests.exceptions.RequestException as e:
    print("Chyba při stahování dat:", e)

X_new = df_new.drop(columns=['User_ID'])
X_new = pd.get_dummies(X_new, columns=['Gender', 'Platform'])
X_new = X_new.astype(float)

scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

weights_filtered = weights.loc[weights.index.intersection(X_new.columns)]

perceptron = np.dot(X_new_scaled, weights_filtered) + result['bias']
y_activation = sigmoid(perceptron)
y_pred_new = np.where(y_activation >= 0.5, 1, 0)

y = pd.Series(y)
y_pred_new = pd.Series(y_pred_new)

# Spočítáme četnosti
counts_y = y.value_counts().sort_index()
counts_y.index = ['Negativní emoce', 'Pozitivní emoce']

counts_pred = y_pred_new.value_counts().sort_index()
counts_pred.index = ['Negativní emoce', 'Pozitivní emoce']

# Vykreslíme vedle sebe
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# První koláč – skutečné hodnoty
counts_y.plot.pie(ax=axs[0], autopct='%1.1f%%', ylabel='', startangle=90)
axs[0].set_title('Skutečné hodnoty')

# Druhý koláč – predikce
counts_pred.plot.pie(ax=axs[1], autopct='%1.1f%%', ylabel='', startangle=90)
axs[1].set_title('Predikované hodnoty')

plt.tight_layout()
plt.show()