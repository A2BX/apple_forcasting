import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

# Chargement et prétraitement des données
file_path = 'World-Stock-Prices-Dataset.csv'
data = pd.read_csv(file_path)
apple_data = data[data['Brand_Name'] == 'apple']['Close']
apple_data.index = pd.to_datetime(data[data['Brand_Name'] == 'apple']['Date'])
apple_data = apple_data.asfreq('B').dropna()  # Assurez une fréquence quotidienne

# Transformation logarithmique et différenciation
apple_log = np.log(apple_data)
apple_diff = apple_log.diff().dropna()

# ACF et PACF pour déterminer l'ordre des modèles AR et MA
fig, axes = plt.subplots(1, 2, figsize=(16, 3))
plot_acf(apple_diff, ax=axes[0])
plot_pacf(apple_diff, ax=axes[1])
plt.show()

# Ajustement des modèles AR, MA, et ARMA
# Note: Les ordres doivent être déterminés en fonction des plots ACF/PACF
order_ma = (0, 0, 1)  # Par exemple
order_ar = (1, 0, 0)  # Par exemple
order_arma = (1, 0, 1)  # Par exemple

model_ma = ARIMA(apple_diff, order=order_ma).fit()
model_ar = ARIMA(apple_diff, order=order_ar).fit()
model_arma = ARIMA(apple_diff, order=order_arma).fit()

# Affichage des résultats
print(model_ma.summary())
print(model_ar.summary())
print(model_arma.summary())

# Résidus du modèle ARMA
residus_arma = model_arma.resid

# Ajustement du modèle GARCH
garch_model = arch_model(residus_arma, vol='Garch', p=1, q=1)
garch_results = garch_model.fit()
print(garch_results.summary())

# Votre code pour tester la nullité des paramètres et ajuster éventuellement un modèle ARCH
