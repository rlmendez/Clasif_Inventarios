#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime, date, timedelta
import time
import io
import csv
import sys
import seaborn as sns
 
#matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
#plt.rcParams['figure.figsize'] = (16, 9)
#plt.style.use('ggplot')

url = "/Users/rlmendez/Desktop/Terpel/Base_Inventario.csv"
print("Carga de Informacion")
Categoria, Denominacion, Alm, CMv, E, Docmat, Pos, Ndoc, Fecontab, Registrado, Hora, Cantidad, UMB, ImporteML = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
data = csv.reader(open('/Users/rlmendez/Desktop/Terpel/Base_Inventario.csv','rb'), delimiter=",")
for index, row in enumerate(data):
	Categoria.append(int(row[0]))
	Denominacion.append(row[1])
	Alm.append(row[2])
	CMv.append(row[3])
	E.append(row[4])
	Docmat.append(str(row[5]))
	Pos.append(row[6])
	Ndoc.append(int(row[7]))
	Fecontab.append(row[8])
	Registrado.append(row[9])
	Hora.append(row[10])
	Cantidad.append(row[11])
	UMB.append(row[12])
	ImporteML.append(row[13])

for x in range(0,len(Cantidad)):
	Cantidad[x] = Cantidad[x].replace(".","")
	ImporteML[x] = ImporteML[x].replace(".","")
	try:
		Cantidad[x] = float(Cantidad[x])/1000
		ImporteML[x] = float(ImporteML[x])/10
	except ValueError,e:
		print "Error",e,"on line",x

df = pd.DataFrame({'Categoria':Categoria, 'Denominacion':Denominacion, 'Alm':Alm, 'CMv':CMv, 'E':E , 'Docmat':Docmat, 'Pos':Pos, 'Ndoc':Ndoc, 'Fecontab':Fecontab, 'Registrado':Registrado, 'Hora':Hora, 'Cantidad':Cantidad, 'UMB':UMB, 'ImporteML':ImporteML})
print("Inicio Prueba")
#print(df.head())
#print("Estadistica:")
#print(df.describe())

#Definicio de Entrada
X = np.array(df[["Cantidad","Ndoc","ImporteML"]])
y = np.array(df['Categoria'])
X.shape

#Obtener el valor K
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Numero de Clusters')
plt.ylabel('Puntuacion')
plt.title('Analisis con la curva K para la seleccion del numero de Cluster')
plt.show()

#Ejecutamos K-Means (Por criterio del minero, escoge el numero de cluster)
K = 5
kmeans = KMeans(n_clusters=K).fit(X)
centroids = kmeans.cluster_centers_
#print("Impresion de Centroides: ",centroids)

# Prediccion de los clusters
labels = kmeans.predict(X)
# Se obtiene los centros de los centroides
C = kmeans.cluster_centers_
colores=['red','green','blue','cyan','yellow']
asignar=[]
for row in labels:
    asignar.append(colores[row])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar,s=60)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
plt.show()
print("Finalizacion Prueba")

#print("Dispersion de los valores")
#time.sleep(5)
#df.drop(['Categoria'],1).hist()
#plt.show()

#print("Relacion de Cantidad ImporteML y Ndoc")
#sb.pairplot(df.dropna(), hue='Categoria',size=3,vars=["Cantidad","ImporteML"],kind='scatter')
#plt.show()

#Impresion grafica para representar la Denominacion
#fig = plt.figure()
#ax = Axes3D(fig)
#colores=['blue','red','green','blue','cyan','yellow','orange','black','pink','brown','purple']
#asignar=[]
#for row in y:
#    asignar.append(colores[row])
#ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar,s=60)
#plt.show()

#Clasificacion
#Hipotesis: A partir del movimientom valor y numero_doc se puede identificar algun movimiento de fraude
#X_new = np.array([[50,4900000212,100400]])
#new_labels = kmeans.predict(X_new)
#print(new_labels)

#Evaluacion
X = df.drop(['Cantidad','Ndoc','ImporteML'], axis=1).values  
y = df.loc[:, 'Categoria'].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)

print('Precision: %.2f' % accuracy_score(X_train, y_train))


