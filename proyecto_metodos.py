import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, norm, t

# Funciones MCF integradas
def obtener_datos(stocks):
    df = yf.download(stocks, start="2010-01-01")['Close']
    return df

def calcular_rendimientos(df):
    return df.pct_change().dropna()

def calcular_var_parametrico(returns, alpha, distrib='normal'):
    mean = np.mean(returns)
    stdev = np.std(returns)
    
    if distrib == 'normal':
        VaR = norm.ppf(1-alpha, mean, stdev)
    elif distrib == 't':
        df_t = 10  # Grados de libertad
        VaR = t.ppf(1-alpha, df_t, mean, stdev)
    return VaR

def calcular_var_historico(returns, alpha):
    hVaR = returns.quantile(1-alpha)
    return hVaR

def calcular_var_montecarlo(returns, alpha, n_sims=100000):
    mean = np.mean(returns)
    stdev = np.std(returns)
    sim_returns = np.random.normal(mean, stdev, n_sims)
    return np.percentile(sim_returns, (1-alpha)*100)

def calcular_cvar(returns, hVaR):
    return returns[returns <= hVaR].mean()

# Inicia la aplicación Streamlit
st.title('Análisis de Riesgo de Activos')

# Definir activos y obtener datos
M7 = ['NVDA']
df_precios = obtener_datos(M7)
df_rendimientos = calcular_rendimientos(df_precios)

# Mostrar los primeros datos
st.subheader('Datos de Precios y Rendimientos')
st.write(df_precios.head())
st.write(df_rendimientos.head())

# Promedio de rendimientos
promedio_rendi_diario = df_rendimientos['NVDA'].mean()
st.write(f"Promedio de rendimiento diario: {promedio_rendi_diario:.4f}")

# Estadísticas de Kurtosis y Skewness
kurt = kurtosis(df_rendimientos['NVDA'])
skewness = skew(df_rendimientos['NVDA'])
st.write(f"Kurtois: {kurt:.4f}")
st.write(f"Sesgo: {skewness:.4f}")

# Cálculo de VaR y CVaR para diferentes niveles de confianza
alpha_vals = [0.95, 0.975, 0.99]
resultados_df = pd.DataFrame(columns=['VaR (Normal)', 'VaR (t-Student)', 'VaR (Histórico)', 'VaR (Monte Carlo)', 
                                      'CVaR (Normal)', 'CVaR (t-Student)', 'CVaR (Histórico)', 'CVaR (Monte Carlo)'])

for alpha in alpha_vals:
    var_normal = calcular_var_parametrico(df_rendimientos['NVDA'], alpha, distrib='normal')
    var_t = calcular_var_parametrico(df_rendimientos['NVDA'], alpha, distrib='t')
    var_historico = calcular_var_historico(df_rendimientos['NVDA'], alpha)
    var_montecarlo = calcular_var_montecarlo(df_rendimientos['NVDA'], alpha)
    
    cvar_normal = calcular_cvar(df_rendimientos['NVDA'], var_normal)
    cvar_t = calcular_cvar(df_rendimientos['NVDA'], var_t)
    cvar_historico = calcular_cvar(df_rendimientos['NVDA'], var_historico)
    cvar_montecarlo = calcular_cvar(df_rendimientos['NVDA'], var_montecarlo)
    
    resultados_df.loc[f'{int(alpha*100)}% Confidence'] = [
        var_normal * 100, var_t * 100, var_historico * 100, var_montecarlo * 100,
        cvar_normal * 100, cvar_t * 100, cvar_historico * 100, cvar_montecarlo * 100
    ]

# Mostrar el DataFrame de resultados
st.subheader('Resultados de VaR y CVaR por intervalo de confianza')
st.write(resultados_df)

# Graficar el histograma de rendimientos
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(df_rendimientos['NVDA'], bins=50, color='dodgerblue', alpha=0.7, label='Rendimientos')

# Colorear los bins que corresponden al VaR histórico
for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
    if bin_left < var_historico:
        patch.set_facecolor('salmon')

# Marcar los diferentes VaR y CVaR
plt.axvline(x=var_normal, color='lightseagreen', linestyle='--', label='VaR 95% (Normal)')
plt.axvline(x=var_montecarlo, color='darkgrey', linestyle='--', label='VaR 95% (Monte Carlo)')
plt.axvline(x=var_historico, color='forestgreen', linestyle='--', label='VaR 95% (Histórico)')
plt.axvline(x=cvar_normal, color='purple', linestyle='-.', label='CVaR 95% (Normal)')

plt.title('Histograma de Rendimientos con VaR y CVaR', fontsize=14)
plt.xlabel('Rendimientos', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.legend()

# Mostrar gráfico
st.pyplot(plt)

# Calcular VaR y ES rolling
window_size = 252
rolling_mean = df_rendimientos['NVDA'].rolling(window=window_size).mean()
rolling_std = df_rendimientos['NVDA'].rolling(window=window_size).std()

VaR_rolling = {
    95: norm.ppf(1 - 0.95, rolling_mean, rolling_std),
    99: norm.ppf(1 - 0.99, rolling_mean, rolling_std)
}

VaR_hist_rolling = {
    95: df_rendimientos['NVDA'].rolling(window=window_size).quantile(0.05),
    99: df_rendimientos['NVDA'].rolling(window=window_size).quantile(0.01)
}

# Calcular ES para 95% y 99% (paramétrico y histórico)
ES_rolling = {alpha: [calcular_cvar(df_rendimientos['NVDA'][i-window_size:i], VaR_rolling[alpha][i]) for i in range(window_size, len(df_rendimientos))]
              for alpha in [95, 99]}

ES_hist_rolling = {alpha: [calcular_cvar(df_rendimientos['NVDA'][i-window_size:i], VaR_hist_rolling[alpha][i]) for i in range(window_size, len(df_rendimientos))]
                   for alpha in [95, 99]}

# Crear DataFrame con resultados
rolling_results_df = pd.DataFrame({
    'Date': df_rendimientos.index[window_size:],
    **{f'VaR_{alpha}_Rolling': VaR_rolling[alpha][window_size:] for alpha in [95, 99]},
    **{f'VaR_{alpha}_Rolling_Hist': VaR_hist_rolling[alpha][window_size:] for alpha in [95, 99]},
    **{f'ES_{alpha}_Rolling': ES_rolling[alpha] for alpha in [95, 99]},
    **{f'ES_{alpha}_Rolling_Hist': ES_hist_rolling[alpha] for alpha in [95, 99]}
})

rolling_results_df.set_index('Date', inplace=True)

# Graficar los resultados de VaR rolling
plt.figure(figsize=(14, 7))

plt.plot(df_rendimientos.index, df_rendimientos['NVDA'] * 100, label='Rendimientos Diarios (%)', color='blue', alpha=0.5)

for alpha in [95, 99]:
    plt.plot(rolling_results_df.index, rolling_results_df[f'VaR_{alpha}_Rolling'] * 100, label=f'VaR {alpha}% Rolling Paramétrico')
    plt.plot(rolling_results_df.index, rolling_results_df[f'VaR_{alpha}_Rolling_Hist'] * 100, label=f'VaR {alpha}% Rolling Histórico')
    plt.plot(rolling_results_df.index, rolling_results_df[f'ES_{alpha}_Rolling'] * 100, label=f'ES {alpha}% Rolling Paramétrico')
    plt.plot(rolling_results_df.index, rolling_results_df[f'ES_{alpha}_Rolling_Hist'] * 100, label=f'ES {alpha}% Rolling Histórico')

plt.title('Rendimientos Diarios y VaR/ES Rolling Window (252 días)')
plt.xlabel('Fecha')
plt.ylabel('Valor (%)')
plt.legend()

# Mostrar gráfico
st.pyplot(plt)

# Función para contar violaciones
def contar_violaciones(returns, risk_measure):
    violations = returns < risk_measure
    num_violations = violations.sum()
    violation_percentage = (num_violations / len(returns)) * 100
    return num_violations, violation_percentage

# Preparar los datos para el análisis de violaciones
returns_for_test = df_rendimientos['NVDA'].iloc[window_size:]

# Diccionario con medidas de riesgo y sus nombres
risk_measures = {
    'VaR 95% Paramétrico': rolling_results_df['VaR_95_Rolling'],
    'VaR 99% Paramétrico': rolling_results_df['VaR_99_Rolling'],
    'VaR 95% Histórico': rolling_results_df['VaR_95_Rolling_Hist'],
    'VaR 99% Histórico': rolling_results_df['VaR_99_Rolling_Hist'],
    'ES 95% Paramétrico': rolling_results_df['ES_95_Rolling'],
    'ES 99% Paramétrico': rolling_results_df['ES_99_Rolling'],
    'ES 95% Histórico': rolling_results_df['ES_95_Rolling_Hist'],
    'ES 99% Histórico': rolling_results_df['ES_99_Rolling_Hist']
}

# Calcular violaciones para cada medida de riesgo
violation_results = {
    'Número de violaciones': [],
    'Porcentaje de violaciones': []
}

for measure_name, measure_values in risk_measures.items():
    violations, violation_percent = contar_violaciones(returns_for_test, measure_values)
    violation_results['Número de violaciones'].append(violations)
    violation_results['Porcentaje de violaciones'].append(violation_percent)

# Crear DataFrame para los resultados de violaciones
violation_results_df = pd.DataFrame(violation_results, index=risk_measures.keys())

# Mostrar los resultados de violaciones
st.subheader('Resultados de violaciones de VaR y ES')
st.write(violation_results_df)

# Graficar VaR con volatilidad móvil
window_size = 252
q_95 = norm.ppf(0.05)
q_99 = norm.ppf(0.01)

rolling_vol = df_rendimientos['NVDA'].rolling(window=window_size).std()
rolling_var_95 = q_95 * rolling_vol
rolling_var_99 = q_99 * rolling_vol

plt.figure(figsize=(12, 6))
plt.plot(df_rendimientos.index, df_rendimientos['NVDA'], label='Retornos', alpha=0.5)
plt.plot(df_rendimientos.index, rolling_var_95, label='VaR 95% (Volatilidad Móvil)', linestyle='dashed')
plt.plot(df_rendimientos.index, rolling_var_99, label='VaR 99% (Volatilidad Móvil)', linestyle='dashed')
plt.legend()
plt.title('VaR con Volatilidad Móvil')
plt.xlabel('Fecha')
plt.ylabel('VaR')

# Mostrar gráfico
st.pyplot(plt)

# Evaluación de eficiencia
violaciones = {'VaR_95': 0, 'VaR_99': 0}

total_pruebas = len(df_rendimientos) - window_size
for i in range(window_size, len(df_rendimientos)):
    r_t = df_rendimientos['NVDA'].iloc[i]
    if r_t < rolling_var_95.iloc[i]:
        violaciones['VaR_95'] += 1
    if r_t < rolling_var_99.iloc[i]:
        violaciones['VaR_99'] += 1

# Calcular porcentaje de violaciones
violaciones_pct = {key: (value / total_pruebas) * 100 for key, value in violaciones.items()}

# Mostrar resultados de violaciones
violaciones_df = pd.DataFrame([violaciones, violaciones_pct], index=['Violaciones', 'Porcentaje (%)']).T
st.subheader('Número y porcentaje de violaciones')
st.write(violaciones_df)
