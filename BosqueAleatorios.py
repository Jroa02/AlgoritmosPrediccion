# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 10:25:22 2021

@author: Dari
"""

###### Librerias a utilizar ###########
#Se importan las librerias a utilizar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import datetime

#Realice la limpieza y adecuación de la data (DataGames)

#importamos la data
df=pd.read_csv('DataGames.csv',sep=";")

#Informacion de la data antes del Pre-procesamiento
print('Información antes del Pre-procesamiento de la data')
print('# de filas:',len(df.index))
print('Numero de Columnas:', df.shape[1]) 
#Verificar si hay datos nulos en la data registro a registro
DataNull=df.isnull()

#columnas de la data
cols = list(df)  

#eliminamos la columna Hours Played
#eliminar columna vacias
#DEFINIMOS EL PARAMETRO AXIS=1 PARA INDICAR QUE VAMOS A OPERAR CON COLUMNAS 
df = df.drop(['Hours Played'], axis=1)
df = df.drop('Unnamed: 8', axis=1)

#2. condicionar el campo
CatGame = df.Game.unique() 
df.Game = pd.factorize(df.Game, sort= True)[0]

#conficionar el campo incremento
CatIncremento = df.Incremento.unique() 
df.Incremento = pd.factorize(df.Incremento, sort= True)[0]

#convertir el campo Month en tipo de dato Fecha
df.Month = pd.to_datetime(df.Month,format="%m-%y")

#formatear la fecha a mes corto y año corto 
df.Month= df.Month.dt.strftime("%m-%Y")

# remplazar el dato "-" por "0" ya que es el primer dia que se registro en la base de datos y este depende
#de los meses anteriores
df.Gain = df['Gain'].replace("-","0")
df["% Gain"] = df['% Gain'].replace("-","0")

#convertir str a numeros de las columnas Avg. Players,Gain,% Gain

df.Gain=pd.to_numeric(df['Gain'],downcast='float')
df['Avg. Players']=pd.to_numeric(df['Avg. Players'],downcast='float')
df['Peak Players']=pd.to_numeric(df['Peak Players'],downcast='float')
df['% Gain']=pd.to_numeric(df['% Gain'],downcast='float')
#Tipo de dato en cada columna
TypeColumn = df.dtypes

#Informacion de la data despues del Pre-procesamiento
print('-----------------------------------------------------')
print('Información despues del Pre-procesamiento de la data')
print('Numero de filas:',len(df.index))
print('Numero de Columnas:', df.shape[1]) 
print('Tipo de Dato Columnas \n',TypeColumn)
print('-----------------------------------------------------------')
#media de los jugadores promedio
print('Media de los Jugadores Promedio')
for n in range(0,6):
   GameMean=df['Avg. Players'][df['Game']==n]
   print('Media del Juego #',n,GameMean.mean())

#valor max de Peak Players por juego
print('-----------------------------------------------------------')
for n in range(0,6):
    PeakMax= df['Peak Players'][df['Game']==n]
    print('Valor Maximo de Peak Players juego',n,PeakMax.max())
    
    
df1 = df.loc[df['Game']==2]

df1 = df1.drop(['Month'], axis= 1)

df1 = df1.to_numpy(dtype=int)

#Seleccionamos solamente la columna 6 del dataset
X_bar = df1[:, np.newaxis, 0]

#Defino los datos correspondientes a las etiquetas
Y_bar = df1[:, 3]

#Graficamos los datos correspondientes
plt.scatter(X_bar, Y_bar)
plt.show()


########## IMPLEMENTACIÓN DE BOSQUES ALEATORIOS REGRESIÓN ##########
from sklearn.model_selection import train_test_split
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_bar, Y_bar, test_size=0.2)
from sklearn.ensemble import RandomForestRegressor
#Defino el algoritmo a utilizar

bar = RandomForestRegressor(n_estimators = 300, max_depth = 15)

#Entreno el modelo
bar.fit(X_train, y_train)

#Realizo una predicción
Y_pred = bar.predict(X_test)

#Graficamos los datos de prueba junto con la predicción
X_grid = np.arange(min(X_test), max(X_test), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test)
plt.plot(X_grid, bar.predict(X_grid), color='red', linewidth=3)
plt.show()
#
print('DATOS DEL MODELO BOSQUES ALEATORIOS REGRESION')
print()
print('Precisión del modelo:')
print(bar.score(X_train, y_train))
