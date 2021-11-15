# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 19:50:42 2021

@author: Juan,Felipe,Dariel
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
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


##  aplicamos el algoritmo de regresion Polinomial
for n in range(0,6):
    print('-----------------------------------------------------------')
    DataPre = df[df['Game']==n]


    x = DataPre.iloc[:,1:2]
    y = DataPre.iloc[:,4]


    from sklearn.model_selection import train_test_split 

    X_train, X_test, y_train, y_test = train_test_split(x,y)

    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(degree = 2)

    X_train_p = poly.fit_transform(X_train)
    X_test_p = poly.fit_transform(X_test)

    pr = linear_model.LinearRegression()
    
    pr.fit(X_train_p, y_train)

    y_predit = pr.predict(X_test_p)

    # Visualización de los resultados de la regresión polinomial
    plt.scatter(X_test, y_test) 
    plt.title('Game#'+str(n)+' '+CatGame[n]) 
    plt.xlabel("Avg. Player") 
    plt.ylabel("Peak Players")  
    plt.plot(X_test, y_predit, color = 'red', linewidth = 3) 
    plt.show()
    
    print('Precisión del Modelo Juego#',n,':',pr.score(X_train_p, y_train))



