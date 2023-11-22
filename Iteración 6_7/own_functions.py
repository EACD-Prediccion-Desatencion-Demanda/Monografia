#Librerias necesarias
#Math
import math
import numpy as np
#Plots
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def multi_plot(df, cols, num_cols, num_rows, tipo, targetVar, figsize=(16,8)):
    
    plt.rcParams['figure.figsize'] = figsize
    
    #num_plots = len(cols)
    #num_cols = math.ceil(np.sqrt(num_plots))
    #num_rows = math.ceil(num_plots/num_cols)
        
    fig, axs = plt.subplots(num_rows, num_cols)
    plt.style.use("seaborn-v0_8-muted")
    for ind, col in enumerate(cols):
        i = math.floor(ind/num_cols)
        j = ind - i*num_cols
        
        if num_rows == 1:
            if num_cols == 1:
                if tipo == 'c': 
                    sns.countplot(x=df[col], ax=axs, dodge = False)
                if tipo == 'b':
                    sns.boxplot(x=df[col], y=df[targetVar], ax=axs)
                if tipo == 's':
                    sns.scatterplot(x=df[col], y=df[targetVar], ax=axs)
            else:
                if tipo == 'c':
                    sns.countplot(x=df[col], ax=axs[j], dodge = False)
                if tipo == 'b':
                    sns.boxplot(x=df[col], y=df[targetVar], ax=axs[j])
                if tipo == 's':
                    sns.scatterplot(x=df[col], y=df[targetVar], ax=axs[j])
        else:
            if num_cols == 1:
                if tipo == 'c':
                    sns.countplot(x=df[col], ax=axs[i], dodge = False)
                if tipo == 'b':
                    sns.boxplot(x=df[col], y=df[targetVar], ax=axs[i])
                if tipo == 's':
                    sns.scatterplot(x=df[col], y=df[targetVar], ax=axs[i])
            else:
                if tipo == 'c':
                    sns.countplot(x=df[col], ax=axs[i, j], dodge = False)
                if tipo == 'b':
                    sns.boxplot(x=df[col], y=df[targetVar], ax=axs[i, j])
                if tipo == 's':
                    sns.scatterplot(x=df[col], y=df[targetVar], ax=axs[i, j])

    if len(cols)<(num_rows*num_cols):
        if num_rows==1 or num_cols ==1:
            axs[-1].set_axis_off()
        else:
            axs[-1, -1].set_axis_off()

# Función para graficar los residuos de los modelos

def residuos(y_train, prediccion_train, residuos_train):
  fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
  axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4)
  axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                  'k--', color = 'black', lw=2)
  axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
  axes[0, 0].set_xlabel('Real')
  axes[0, 0].set_ylabel('Predicción')
  axes[0, 0].tick_params(labelsize = 7)

  axes[0, 1].scatter(list(range(len(y_train))), residuos_train, edgecolors=(0, 0, 0), alpha = 0.4)
  axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
  axes[0, 1].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
  axes[0, 1].set_xlabel('id')
  axes[0, 1].set_ylabel('Residuo')
  axes[0, 1].tick_params(labelsize = 7)

  sns.histplot(
      data    = residuos_train,
      stat    = "density",
      kde     = True,
      line_kws= {'linewidth': 1},
      color   = "firebrick",
      alpha   = 0.3,
      ax      = axes[1, 0]
  )

  axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10,
                      fontweight = "bold")
  axes[1, 0].set_xlabel("Residuo")
  axes[1, 0].tick_params(labelsize = 7)


  sm.qqplot(
      residuos_train,
      fit   = True,
      line  = 'q',
      ax    = axes[1, 1], 
      color = 'firebrick',
      alpha = 0.4,
      lw    = 2
  )
  axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
  axes[1, 1].tick_params(labelsize = 7)


  axes[2, 0].scatter(prediccion_train, residuos_train,
                    edgecolors=(0, 0, 0), alpha = 0.4)
  axes[2, 0].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
  axes[2, 0].set_title('Residuos del modelo vs predicción', fontsize = 10, fontweight = "bold")
  axes[2, 0].set_xlabel('Predicción')
  axes[2, 0].set_ylabel('Residuo')
  axes[2, 0].tick_params(labelsize = 7)

  # Se eliminan los axes vacíos
  fig.delaxes(axes[2,1])
  fig.tight_layout()
  plt.subplots_adjust(top=0.9)
  fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold");

# Función para graficar la variable de salida original vs la predicción del modelo
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'

def comparisson(Y_Test, Y_pred, name, fig_size=(20,5)):
  plt.style.use("seaborn-v0_8-muted")
  fig = plt.figure(figsize=fig_size)
  line_1 = plt.plot(np.arange(len(Y_Test)), Y_Test, 
                    color = '#4e79a7', label='Real') 
  line_2 = plt.plot(np.arange(len(Y_Test)), Y_pred, 
                    color = 'indianred', linestyle = 'dashed', label='Predicted') 
  plt.axhline(0, color="black") # Elegir color de la linea horizontal de referencia
  plt.legend()
  plt.title(f'{name}') # Titulo de la gráfica
  plt.xlabel('Samples') # Etiqueta del eje x
  plt.ylabel('3h Load Shortfall') # Etiqueta del eje y
  plt.show() # Mostrar gráfica

#Clasificación de la BD por estación
import pandas as pd

def apply_season(month, day):
    if (month == 3 and day >= 20) or (4 <= month <= 5) or (month == 6 and day <= 20):
        return 'Primavera'
    elif (month == 6 and day >= 21) or (7 <= month <= 8) or (month == 9 and day <= 22):
        return 'Verano'
    elif (month == 9 and day >= 23) or (10 <= month <= 11) or (month == 12 and day <= 20):
        return 'Otoño'
    else:
        return 'Invierno' 

#Generador de nombres
def secuencia_PCA(inicio, fin):
    lista = []
    for x in range(inicio, fin + 1):
        cadena = f"C{x}"
        lista.append(cadena)
    return lista

#Evaluación del modelo
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

def scorer(y_true  = None, y_pred  = None, model_name=None):
    metrics = {'MSE' : mean_squared_error (y_true  = y_true, y_pred  = y_pred, squared = True),
    'RMSE'  : mean_squared_error (y_true  = y_true, y_pred  = y_pred, squared = False),
    'MAE'   : mean_absolute_error (y_true  = y_true, y_pred  = y_pred),
    'MedAE' : median_absolute_error (y_true  = y_true, y_pred  = y_pred),
    'MAPE'  : mean_absolute_percentage_error (y_true  = y_true, y_pred  = y_pred),
    'R2'    : r2_score (y_true  = y_true, y_pred  = y_pred)
    }

    #for key,value in metrics.items():
    #    print(f'El error {key} de test para el {model_name} es: {value}')

    return metrics

def ConvertirWeatherID(value):
    if value<300:
        return 'Thunderstorm'
    elif value>=300 and value<400:
        return 'Drizzle'
    elif value>=400 and value<500:
        return 'Other'
    elif value>=500 and value<600:
        return 'Rain'
    elif value>=600 and value<700:
        return 'Rain'
    elif value>=700 and value<800:
        return 'Atmosphere'
    elif value==800:
        return 'Clear'
    else:
        return 'Clouds'