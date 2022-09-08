#!/usr/bin/env python
# coding: utf-8




# importamos las librerias que necesitaremos
import numpy as np
import matplotlib.pyplot as plt
import math



                        ####funciones para importar registros de tiempos neuronales
######################################################
#### Apertura y procesamiento inicial de datos 
def abrir(archivo):
    with open(archivo, mode='r') as datos: # abrimos el archivo
        contenido= datos.readlines() #contenido aqui es una lista de strings 
    return contenido
def procesado(contenido):
    ensayos=[] # declaro una lista vacía para guardar arreglos donde cada arreglo contendra los registros correspondientes a un ensayo 
    for string in contenido:# hay n str en contenido1 n-k de esos str son '\n' y los demás son los datos de k ensayos
        if string!='\n': # solo 60 str de contenido cumplen esta condición y esos son ensayos
            lista_n=[] # entonces declararé en total k listas vacías una en cada ciclo
            string=string.replace(','," ") # reemplazo las comas por espacios para poder usar split después
            string=string.split() #split hace cada número dentro del str de ensayos un str individual, y pone estos str dentro de una lista # ahora la variable string es una lista donde cada elemento es un str de un número flotante
            for numero in string: # itero cada elemento de la lista string
                numero=float(numero) # cada str de un número lo convierto a float 
                lista_n.append(numero)# cada float agrego con append a lista_n
            ensayo_n=np.array(lista_n)# al final de el segundo ciclo for agrego la lista_n correspondiente a un array que llamo ensayo_n
            ensayos.append(ensayo_n)# luego agrego mi arreglo ensayo_n a mi lista de arrays "ensayos"
    return ensayos

#############################################################
def importar(archivo): # función para importar datos
    content=abrir(archivo)
    Datos=procesado(content)
    return Datos

###########################################################Funciones para importar datos de tiempos neuronales con formato irregular

####################################################################

def procesado2(contenido):
    ensayos=[] # declaro una lista vacía para guardar arreglos donde cada arreglo contendra los registros correspondientes a un ensayo 
    for string in contenido:# hay n str en contenido1 n-k de esos str son '\n' y los demás son los datos de los k ensayos
        s=string.strip().split(",")  # al hacer esto identifico los separadores aunque no sean \n
        if  s.count('')!=len(s): # solo los ensayos cumplen esto mientras que los separadores de bloque no
            lista_n=[] # entonces declararé en total k listas vacías una en cada ciclo
            string=string.replace(','," ") # reemplazo las comas por espacios para poder usar split después
            string=string.split() #split hace cada número dentro del str de ensayos un str individual, y pone estos str dentro de una lista # ahora la variable string es una lista donde cada elemento es un str de un número flotante
            for numero in string: # itero cada elemento de la lista string
                numero=float(numero) # cada str de un número lo convierto a float 
                lista_n.append(numero)# cada float agrego con append a lista_n
            ensayo_n=np.array(lista_n)# al final de el segundo ciclo for agrego la lista_n correspondiente a un array que llamo ensayo_n
            ensayos.append(ensayo_n)# luego agrego mi arreglo ensayo_n a mi lista de arrays "ensayos"
    return ensayos



#############################################################

def importar2(archivo): # función para importar datos
    content=abrir(archivo)
    Datos=procesado2(content)
    return Datos


                                                        ###Funciones para calcular tasas de disparo
##############################################################
#bordes para ventanas deterministicas

def bordesup(start,stop,paso,ventana,nv=False): # función para obtener los bordes superiores de mi ventana dadas las especificaciones de cada caso
    """
    Esta función cálcula los bordes superiores (para ventanas deterministicas) necesarios para calcular la tasa de disparo.
    alternativamente devuelve también el número de ventanas determinadas por estos bordes.
    La función recibe en donde empieza mi rango de tiempo y donde termina (en segundos), el tamaño del paso
    tamaño de la ventana. y cálcula los bordes superiores.alternativamente también cálcula 
    el número de ventanas determinadas por los bordes
    <parametros>
    start  es en donde empieza mi rango # en segundos
    stop es donde donde termina mi rango #segundos
    paso es el tamaño del paso #en segundos
    ventana es el tamaño de la ventana # en segundos
    nv es un parametro opcional que indica si regresa también el número de ventanas para el valor default es false
    <devuelve>
    np.array que contiene los valores de los bordes superiores (para  ventanas deterministicas)
    int que corresponde al número de ventanas determinadas por esos bordes
    """
    ventana,paso,stop,start=ventana*1000,paso*1000,stop*1000,start*1000
    sup=np.arange(start+ventana,stop+1,paso)/1000 #obtengamos los bordes superiores de mi ventana dadas las especificaciones
    if nv==True:
        nvent=(len(sup))
        return sup,nvent
    else:
        return sup


# In[13]:

#################################################################################
#### Cálculo de la tasa de disparo
def tasa_prom_clase(ensayos,nventanas,paso,ventana,start=0):
    """
    Esta función calcula la tasa de disparo con un algoritmo optimizado para reducir el uso de ciclos 
    for al mínimo.Para esto toma ventaja de la regularidad y el ordenamiento de los datos y hace uso de la 
    división entera y la vectorización.La función calcula la tasa de disparo dado un slice de una lista de
    una lista de numpy arrays donde cada array es un ensayo.
    <parámetros>
    LOS PARÁMETROS OBLIGATORIOS SON:
    ensayos debe ser una lista de arrays o un slice de una lista de arrays donde cada array debe ser un ensayo.
    nventanas es el número de ventanas para el que estamos calculando
    paso es el tamaño de paso que vamos usar para calcular la tasa 
    ventana es el tamaño del la ventana que vamos a usar
    Parámetro opcional
    el parametro opcional
    start indica donde comienza el registro debido a que si empieza por debajo del tiempo cero para que funcione el algoritmo de
    división entera es necesario hacer una correción. si los datos comienzan por debajo de del tiempo cero y no se le especifica
    a la función en el parámetro start en que tiempo se empieza los datos calculados serán erroneos
    
    <devuelve>
    un numpy array donde cada elemento corresponde a la tasa de disparo para cada ventana determinística(a los pasos establecidos).
    """
    if (start>0): # el parametro opcional por default vale cero
        raise NameError('El parametro start debe ser un valor negativo o cero')
    else:
        frecuencia=np.zeros(nventanas)# array vacío con un cero por cada ventana 
        ultima=ventana/paso # este número me dice en cuantas ventanas cae una espiga
        for ensayo in ensayos:#itero un array de ensayos a la vez 
            vent=((ensayo-start)//paso)# hay que sumar un -start si empezamos el resgistro de datos en  un punto distinto de cero para hacer el ajuste del algoritmo al caso #los elementos de la matriz ((ensayo-start)//paso) son la última ventana en que aparece un dato
            for nv in range(int(ultima)): # nv cuenta 0, 1,2,3,4 que es lo que necesito restar para ver si una espiga aparce en ventanas anteriores a la última en que aparece (((ensayo-start)//paso))
                ven=vent-nv #los elementos de ((ensayo-start)//paso)-nv en la primera iteración son la última ventana en la que aparece un dato y en las siguientes iteraciones son las ventanas anteriores en las que puede aparecer
                ven=(ven[ven>=0])# la condicion [ven>=0] me evita que al restar nv aparezcan negativos cuando un valor no aparece en una ventana
                ven=(ven[ven<nventanas]).astype(int) # esto evita que si por ejemplo al tomar una espiga que cayo en los últimos pasos al hacer la división entera me de indices superiores a mi número de ventanas
                indices,cuentas=np.unique(ven,return_counts=True)#en este caso los valores unicos corresponden a los indices # y las cuentas a la cantidad de veces que aparece un índice
                frecuencia[indices]+=cuentas # matriz con las frecuencias por ventana
                tasa=(frecuencia/ventana)/len(ensayos) # calculamos la tasa de disparo
    return tasa
####################################################################
# tasa de disparo para un solo ensayo
def tasa_original(ensayo,nventanas,paso,ventana,start=0):
    """
    Este algoritmo cálcula la tasa para un solo ensayo
    Esta función calcula la tasa de disparo con un algoritmo optimizado para reducir el uso de ciclos 
    for al mínimo.Para esto toma ventaja de la regularidad y el ordenamiento de los datos y hace uso de la 
    división entera y la vectorización.La función calcula la tasa de disparo dado numpy array que es un ensayo.
    <parámetros>
    LOS PARÁMETROS OBLIGATORIOS SON:
    ensayos debe ser una lista de arrays o un slice de una lista de arrays donde cada array debe ser un ensayo.
    nventanas es el número de ventanas para el que estamos calculando
    paso es el tamaño de paso que vamos usar para calcular la tasa 
    ventana es el tamaño del la ventana que vamos a usar
    Parámetro opcional
    el parametro opcional
    start indica donde comienza el registro debido a que si empieza por debajo del tiempo cero para que funcione el algoritmo de
    división entera es necesario hacer una correción. si los datos comienzan por debajo de del tiempo cero y no se le especifica
    a la función en el parámetro start en que tiempo se empieza los datos calculados serán erroneos
    
    <devuelve>
    un numpy array donde cada elemento corresponde a la tasa de disparo para cada ventana determinística(a los pasos establecidos).
    """
    if (start>0): # el parametro opcional por default vale cero
        raise NameError('El parametro start debe ser un valor negativo o cero')
    else:
        frecuencia=np.zeros(nventanas)# array vacío con un cero por cada ventana 
        ultima=ventana/paso # este número me dice en cuantas ventanas cae una espiga
        vent=((ensayo-start)//paso)# hay que sumar un -start si empezamos el resgistro de datos en  un punto distinto de cero para hacer el ajuste del algoritmo al caso #los elementos de la matriz ((ensayo-start)//paso) son la última ventana en que aparece un dato
        for nv in range(int(ultima)): # nv cuenta 0, 1,2,3,4... que es lo que necesito restar para ver si una espiga aparce en ventanas anteriores a la última en que aparece (((ensayo-start)//paso))
            ven=vent-nv #los elementos de ((ensayo-start)//paso)-nv en la primera iteración son la última ventana en la que aparece un dato y en las siguientes iteraciones son las ventanas anteriores en las que puede aparecer
            ven=(ven[ven>=0])# la condicion [ven>=0] me evita que al restar nv aparezcan negativos cuando un valor no aparece en una ventana
            ven=(ven[ven<nventanas]).astype(int) # esto evita que si por ejemplo al tomar una espiga que cayo en los últimos pasos al hacer la división entera me de indices superiores a mi número de ventanas
            indices,cuentas=np.unique(ven,return_counts=True)#en este caso los valores unicos corresponden a los indices # y las cuentas a la cantidad de veces que aparece un índice
            frecuencia[indices]+=cuentas # matriz con las frecuencias por ventana
    tasa=(frecuencia/ventana) # calculamos la tasa de disparo
    return tasa
#########################################################################
####################################################################
# tasa de disparo para un solo ensayo
def tasa(ensayo,nventanas,paso,ventana,start=0):
    """
    Este algoritmo cálcula la tasa para un solo ensayo
    Esta función calcula la tasa de disparo con un algoritmo optimizado para reducir el uso de ciclos 
    for al mínimo.Para esto toma ventaja de la regularidad y el ordenamiento de los datos y hace uso de la 
    división entera y la vectorización.La función calcula la tasa de disparo dado numpy array que es un ensayo.
    <parámetros>
    LOS PARÁMETROS OBLIGATORIOS SON:
    ensayos debe ser una lista de arrays o un slice de una lista de arrays donde cada array debe ser un ensayo.
    nventanas es el número de ventanas para el que estamos calculando
    paso es el tamaño de paso que vamos usar para calcular la tasa 
    ventana es el tamaño del la ventana que vamos a usar
    Parámetro opcional
    el parametro opcional
    start indica donde comienza el registro debido a que si empieza por debajo del tiempo cero para que funcione el algoritmo de
    división entera es necesario hacer una correción. si los datos comienzan por debajo de del tiempo cero y no se le especifica
    a la función en el parámetro start en que tiempo se empieza los datos calculados serán erroneos
    
    <devuelve>
    un numpy array donde cada elemento corresponde a la tasa de disparo para cada ventana determinística(a los pasos establecidos).
    """
    frecuencia=np.zeros(nventanas)# array vacío con un cero por cada ventana 
    ultima=ventana/paso # este número me dice en cuantas ventanas cae una espiga
    vent=((ensayo-start)//paso)# hay que sumar un -start si empezamos el resgistro de datos en  un punto distinto de cero para hacer el ajuste del algoritmo al caso #los elementos de la matriz ((ensayo-start)//paso) son la última ventana en que aparece un dato
    for nv in range(int(ultima)): # nv cuenta 0, 1,2,3,4... que es lo que necesito restar para ver si una espiga aparce en ventanas anteriores a la última en que aparece (((ensayo-start)//paso))
        ven=vent-nv #los elementos de ((ensayo-start)//paso)-nv en la primera iteración son la última ventana en la que aparece un dato y en las siguientes iteraciones son las ventanas anteriores en las que puede aparecer
        ven=(ven[ven>=0])# la condicion [ven>=0] me evita que al restar nv aparezcan negativos cuando un valor no aparece en una ventana
        ven=(ven[ven<nventanas]).astype(int) # esto evita que si por ejemplo al tomar una espiga que cayo en los últimos pasos al hacer la división entera me de indices superiores a mi número de ventanas
        indices,cuentas=np.unique(ven,return_counts=True)#en este caso los valores unicos corresponden a los indices # y las cuentas a la cantidad de veces que aparece un índice
        frecuencia[indices]+=cuentas # matriz con las frecuencias por ventana
    tasa=(frecuencia/ventana) # calculamos la tasa de disparo
    return tasa

###########################################################################

#################################################################


#####################################################
def col_mu_sg(arreglo):
    """
    Dado un array de ensayos donde donde cada row es la tasa o z-score 
    un ensayo y las columnas las ventanas temporales. 
    Esta función calcula promedio por columna (ventana a ventana) y la std por
    columna (ventana a ventana).Esta función devuelve una lista con 
    dos arrays renglón. un array es el promedio por columna(ventana a ventana) 
    y el otro array es la std por columna(ventana a ventana).
    """  
    mu=np.mean(arreglo,axis=0) # calculamos el z-score promedio 
    sg=np.std(arreglo,axis=0) # calculamos las std 
    return mu,sg #devuelve una lista con dos ndarrays
###########################################################

def clases(Neuron,ebloques):
    clases=[]
    nbloques=len(ebloques) #número de bloques (clases)
    superior=0 # indice superior
    for index in range(nbloques):# tantos ciclos como número de clases,es decir un ciclo por clase.
        inferior=superior# actualizamos el valor de del indice inferior 
        superior=inferior+ebloques[index]#actualizamos el valor del indice superior
        clase=Neuron[inferior:superior] # este slice corresponde a los elementos de una clase
        clases.append(clase)
    return clases


                                        #funciones para plotear
###########################################################################
def c4plot(ejex,zm,zsg,titulo,ejey='tasa'):
    """
    Esta función unicamente para listas con 4 clases.
    Dado un ejex comúm y un lista o tupla que contiene arrays con los zscore de cada clase y otra lista o tupla que contiene
    los std de cada clase,así como un título para la figura. Esta función crea una figura con las graficas del zscore promedio 
    de cada clase. 
    """
    c1,c2,c3,c4=zm# desempaquetamos 
    sg1,sg2,sg3,sg4=zsg #desempaquetamos los std de cada clase
    # una figura
    fig,(ax1,ax2,ax3,ax4)=plt.subplots(nrows=4,ncols=1,sharex=True)
    # titulo
    fontdict_title = {'family': 'serif','color':  'darkred','weight': 'normal','size': 16,} # formato para el título
    ax1.set_title(titulo, fontdict_title) #título de la gráfic.plot(xs1f,c1_zNeuS1,label='clase1')# creamos la gráfica del z-score
    #######
    ax1.plot(ejex,c1,label='clase1',color='red')# creamos la gráfica 
    ax1.fill_between(ejex,c1-sg1, c1+sg1)# +- 1std
    ax2.plot(ejex,c2,label='clase2',color='purple')# creamos la gráfica 
    ax2.fill_between(ejex,c2-sg2, c2+sg2)# +- 1std
    ax3.plot(ejex,c3,label='clase3',color='darkblue')# creamos la gráfica 
    ax3.fill_between(ejex,c3-sg3, c3+sg3)# +- 1std
    ax4.plot(ejex,c4,label='clase4',color='black')# creamos la gráfica 
    ax4.fill_between(ejex,c4-sg4, c4+sg4)# +- 1std
    fig.legend()
    # labels para los ejes
    ax4.set_xlabel(" Time [s] ") # Configuramos la etiqueta del eje X
    ax1.set_ylabel(ejey) # Configuramos la etiqueta del eje Y
    ax2.set_ylabel(ejey) # Configuramos la etiqueta del eje Y
    ax3.set_ylabel(ejey) # Configuramos la etiqueta del eje Y
    ax4.set_ylabel(ejey) # Configuramos la etiqueta del eje Y
    #show
    
    return plt.tight_layout()

##################################
def fourgr(ejex,cl,titulo,ejey):
    """
    Esta función unicamente para listas con 4 clases.
    Dado un ejex comúm y un lista o tupla que contiene arrays de cada clase,así como un título para la figura. Esta función crea una figura con las graficas  promedio 
    de cada clase. 
    """
    c1,c2,c3,c4=cl# desempaquetamos las clases
    
    # una figura
    fig,(ax1,ax2,ax3,ax4)=plt.subplots(nrows=4,ncols=1,sharex=True)
    # titulo
    fontdict_title = {'family': 'serif','color':  'darkred','weight': 'normal','size': 16,} # formato para el título
    ax1.set_title(titulo, fontdict_title) #título de la gráfic.plot(xs1f,c1_zNeuS1,label='clase1')# creamos la gráfica del z-score
    #######
    ax1.plot(ejex,c1,label='clase1',color='red')# creamos la gráfica del z-score
    ax2.plot(ejex,c2,label='clase2',color='purple')# creamos la gráfica del z-score
    ax3.plot(ejex,c3,label='clase3',color='darkblue')# creamos la gráfica del z-score
    ax4.plot(ejex,c4,label='clase4',color='black')# creamos la gráfica del z-score
    fig.legend()
    # labels para los ejes
    ax4.set_xlabel(" Time [s] ") # Configuramos la etiqueta del eje X
    ax1.set_ylabel(ejey) # Configuramos la etiqueta del eje Y
    ax2.set_ylabel(ejey) # Configuramos la etiqueta del eje Y
    ax3.set_ylabel(ejey) # Configuramos la etiqueta del eje Y
    ax4.set_ylabel(ejey) # Configuramos la etiqueta del eje Y
    #show
    
######################################################



                                         ##PRÁCTICA 3##
###########################################PRÁCTICA 3#################################################
def gaussian(x,mu,sigma): 
    """Esta función sive para calcular una distribución gaussiana y por tanto
    sirve para calcular un kernel gaussiano dados los paramatros  """  
    f1=(sigma*(np.sqrt(2*np.pi)))**(-1)#factor 1 # por comodidad calculo los factores por separado
    f2=np.power(math.e,(-1/2)*(((x-mu)/sigma)**2))#factor 2
    gauss=f1*f2 # factor1 x factor 2 es la formula de la distribución Gausiana
    return gauss

#####################################
def gaussian_cut(x,mu,sigma): 
    """esta función cálcula una gaussiana truncada a la derecha de la media es decir sin la parte de la
    izquierda de la media. la cuál dados los parámetros sirve para calcular una kernel gaussiano determinístico"""
    trunc=x[x>=mu]
    cortados=x[x<mu]
    cortada=np.zeros(len(cortados))
    f1=(sigma*(np.sqrt(2*np.pi)))**(-1)#factor 1 # por comodidad calculo los factores por separado
    f2=np.power(math.e,(-1/2)*(((trunc-mu)/sigma)**2))#factor 2
    gausscut=f1*f2 # factor1 x factor 2 es la formula de la distribución Gausiana
    gc=np.insert(cortada,int(len(cortados)),gausscut,axis=0)
    return gc

#####################################
def Kgc(Neu,ti,tf,vnt,sigma):
    """
    Esta función calcula a partir de una lista de ensayos(tiempos) un array de donde cada fila es 
    la tasa de disparo de un ensayo calculada con un kernel gaussiano centrado.
    Entrada:
    Neu-lista de arrays(de ensayos) donde cada array contiene los tiempos de un ensayo de una neurona
    ti-tiempo inicial en segundos
    tf-tiempo final en segundos
    vnt-ventana o intervalo que tiene nuestro array de tiempos, en segundos
    sigma- sigma de la distribución gaussiana en segundos
    Salida:
    Devuelve un array de tiempos y el array de tasa de disparo.
    """
    Tiempos=np.arange(ti,tf,vnt) #array con los tiempos con ventanas de 10 ms 
    frecuencias=[]# Frecuencias 
    for ensayo in Neu:# itero cada ensayo de la neurona
        gaussens=[]#lista para todas las gaussianitas centradas en cada dato del ensayo
        for dato in ensayo:#itero cada dato en el ensayo
            mu=dato# cada dato en el ensayo va a tener su propia gaussianita
            x=Tiempos# los tiempos
            g=gaussian(x,mu,sigma) # calculo la gaussinita para un dato # cada gaussianita es un array
            gaussens.append(g)# aquí voy guardando cada gaussiannita a final del ciclo contiene todas las gaussianitas dea cada dato del ensayo # es una lista de arrays
        gaussens=tuple(gaussens)#convierto la lista de arrays en una tupla de arrays
        fens=np.vstack(gaussens)#array donde cada fila es la gaussianita de un dato
        uessay=np.sum(fens,axis=0) #array para la tasa de disparo de un ensayo #calculo la tasa de disparo promedio con kernel gaussiano de todo un ensayo
        frecuencias.append(uessay) #lista de arrays, donde cada array es la tasa de disparo de un ensayo
    fr=np.vstack(tuple(frecuencias))
    return Tiempos,fr

################################################

def Kg_det(Neu,ti,tf,vnt,sigma):
    """
    Esta función calcula a partir de una lista de ensayos(tiempos) un array de donde cada fila es 
    la tasa de disparo de un ensayo calculada con un kernel gaussiano deterministico.
    Entrada:
    Neu-lista de arrays(de ensayos) donde cada array contiene los tiempos de un ensayo de una neurona
    ti-tiempo inicial en segundos
    tf-tiempo final en segundos
    vnt-ventana o intervalo que tiene nuestro array de tiempos, en segundos
    sigma- sigma de la distribución gaussiana en segundos
    Devuelve un array de tiempos y el array de tasa de disparo.
    """
    Tiempos=np.arange(ti,tf,vnt) #array con los tiempos con ventanas de 10 ms 
    frecuencias=[]# Frecuencias 
    for ensayo in Neu:# itero cada ensayo de la neurona
        gaussens=[]#lista para todas las gaussianitas centradas en cada dato del ensayo
        for dato in ensayo:#itero cada dato en el ensayo
            mu=dato# cada dato en el ensayo va a tener su propia gaussianita
            x=Tiempos# los tiempos
            g=gaussian_cut(x,mu,sigma) # calculo la gaussinita para un dato # cada gaussianita es un array
            gaussens.append(g)# aquí voy guardando cada gaussiannita a final del ciclo contiene todas las gaussianitas dea cada dato del ensayo # es una lista de arrays
        gaussens=tuple(gaussens)#convierto la lista de arrays en una tupla de arrays
        fens=np.vstack(gaussens)#array donde cada fila es la gaussianita de un dato
        uessay=np.sum(fens,axis=0) #array para la tasa de disparo de un ensayo #calculo la tasa de disparo promedio con kernel gaussiano de todo un ensayo
        frecuencias.append(uessay) #lista de arrays, donde cada array es la tasa de disparo de un ensayo
    fr=np.vstack(tuple(frecuencias))
    return Tiempos,fr


                                    #funciones para importar datos psicométricos
#########################################################
def cargar3(archivo):
    """carga un archivo de datos psicométricos y lo convierte en un array, donde cada elemento es una lista
    de dos elementos, la amplitud del estímulo y si acerto (donde 1 corresponde a acierto y 0 a error)"""
    datos=np.loadtxt(archivo, skiprows=1, delimiter=",")
    return datos
##############################################
def filtro3(datos,amplitudes):
    """ entradas : amplitudes es una lista o tupla con las amplitudes, datos es un array de datos psicometricos
    esta función filtro el array por amplitudes y devuelve una lista de arrays dónde cada array son los datos
    psicometricos de una sola frecuencia, los arrays están en el mismo orden en el que las amplitudes de
    la lista de amplitudes"""
    amp=[]# lista vacía para guardar arrays de datos pscometricos(acierto o erro, 1 o 0)
    for amplitud in amplitudes:
        dat=datos[datos[:,0]==amplitud] #filtro por amplitud
        amp.append(dat[:,1]) #agrego un array 
    return amp #amp es una lista de arrays
#############################################
def importar3(archivo,amplitudes):
    """archivo es un archivo de datos psicométricos, amplitudes una lista o tupla de amplitudes"""
    datos=cargar3(archivo)# array de datos sin filtra
    amp=filtro3(datos,amplitudes)# lista de arrays filtrados por amplitud
    return amp #regresa lista de arrays
################################################    

                                       ####Cálculos psicometricos
    
def proba_psi(setspsico,amps):
    """Dado un set psicométrico calcula la probabilidad de decir que si, 
    recibe el set de datos psicométricos 0's y 1's """
    ps=[]
    for i in range(len(amps)):
        if amps[i]==0:
            prob=(np.sum(setspsico[i]==0))/len(setspsico[i])#hay que notar que para los casos en que la amplitud cero los 1 es decir los aciertos corresponden decir que no. y los 0, es decir los errores corresponden a decir que si por esto para amplitud cero para saber la probabilidad de decir que sí contamos los errores en vez de los aciertos
            ps.append(prob)
        else:
            prob=(np.sum(setspsico[i]==1))/len(setspsico[i])#En cambio en los casos normales los 1 es decir los aciertos corresponde a decir que si  por eso contamos los aciertos en vez de los errores para conocer la probabilidad de decir que si
            ps.append(prob)
    return ps

def curvepsi(amps,probdsi,nombre):
    """está función Gráfica una curva psicométrica dado una lista de amplitudes para el eje x,
    una lista de probabilidades, y el nombre de la neurona a la que pertenece la curva"""
    fontdict_title = {'family': 'serif','color':  'darkblue','weight': 'bold','size': 16,} # formato para el título
    plt.title(nombre, fontdict_title) #título de la gráfica
    plt.xlabel("Amplitud del estímulo",color='blue') # Configuramos la etiqueta del eje X
    plt.ylabel("Probabilidad de decir que si",color='red') # Configuramos la etiqueta del eje Y
    plt.scatter(amps,probdsi)
    plt.plot(amps,probdsi)
    plt.show() # mostramos la gráfica
    
    
    ##################################Cargar Datos determinar bloques
# ahora en forma de función
def bloques(contenido):
    """
    Esta función calcula el numero de clases por bloque. si el separador entre bloques es un tipo de espacio vacio.
    Contenido deben ser el contenido de mis datos.
    """
    contador=0# declaro un contador vacío
    clases=[] #lista vacía para guardar variables
    for string in contenido:# itero los elementos
        s=string.strip().split(",")  # al hacer esto lograre identificar los separadores aunque no sean \n
        if s.count('')!=len(s):# en la mayoría de los casos donde no es un separador
            contador+=1#sumo 1 al contador
        else: # si es un separador de bloques
            clases.append(contador) #agrego el acumulado en contandor a la lista clases
            contador=0 # reinicio el contador
            
    if contador!=0:
        clases.append(contador)# agrego el acumulado despues del último contador
        bloques=np.array(clases)# lo convierto en un array por si necesito operar con los datos
    else:
        bloques=np.array(clases)# lo convierto en un array por si necesito operar con los datos
    return bloques # devuelve un array que contien el número de elemento por bloque en el orden de aparición de los bloques 


###################################
def clases(Neuron,ebloques):
    clases=[]
    nbloques=len(ebloques) #número de bloques (clases)
    superior=0 # indice superior
    for index in range(nbloques):# tantos ciclos como número de clases,es decir un ciclo por clase.
        inferior=superior# actualizamos el valor de del indice inferior 
        superior=inferior+ebloques[index]#actualizamos el valor del indice superior
        clase=Neuron[inferior:superior] # este slice corresponde a los elementos de una clase
        clases.append(clase)
    return clases


##################################DICCIONARIOS Y SHANON

def dicneuronrate(dictneu,keyneuron,start=-2,stop=3.5,paso=0.01,ventana=0.2):
    """
    Esta función crea un diccionario donde las claves son los nombres de las neuronas y los valores
    son las tasas de disparo de cada neurona. 
    keyneurons:son las claves/llaves de cada neurona. Es una lista. o tupla.
    dictneu:es un diccionario dónde las llaves son keyneurons y los valores son los 
    datos crudos de todos los ensayos de una neurona
    salida es un diccionario."""
    Tns={clave:0 for clave in keyneuron}
    bsup,nven=bordesup(start,stop,paso,ventana,nv=True) 
    func=lambda ensayo : tasa(ensayo,nventanas=nven,paso=paso,ventana=ventana,start=start)# esta función calcula la tasa de disparo de un ensayo nventanas, paso=0.01,ventana=0.05,start=-2 
    for llave in keyneuron:
        fr=map(func,dictneu[llave]) # array con las tasas de disparo para cada ventana
        Tns[llave]=np.array(list(fr))
    return Tns



def txclases(tasas,keyneuron,keyclases,bqs):
    """tasas"""
    #tasas de disparo separadas por amplitud del estímulo en un diccionario de diccionarios
    Txclases={a:0 for a in keyneuron}
    for llave in keyneuron:
        bloques=clases(tasas[llave],bqs[llave])
        clas={keyclases[i]:bloques[i] for i in range(len(keyclases))}
        Txclases[llave]=clas
    return Txclases

def mMv2(Tasaxc,keys):
    """solo tiene dos parametros de entrada la tasa en un diccionario donde 
    las claves son correponden a neuronas y las claves """
    minymax={e:0 for e in keys}
    for llave in keys:
        dat=Tasaxc[llave].values()
        dat=tuple([e for e in dat])
        datos=np.concatenate(dat)
        m,M=(np.min(datos),np.max(datos))
        minymax[llave]=(m,M)
    return minymax

def mM(Tasa,keys):
    """solo tiene dos parametros de entrada la tasa en un diccionario donde 
    las claves son correponden a neuronas y las claves """
    minymax={e:0 for e in keys}
    for llave in keys:
        m,M=(np.min(Tasa[llave]),np.max(Tasa[llave]))
        minymax[llave]=(m,M)
    return minymax

def Probabilidades(Txclases,Tns,bines,keyclases,keyneuron):
    """
    Esta función sirve para calcular distribuciones de probabilidad de las tasas de disparo y las
    disribuciones de probabilidad de las tasas de disparo condicionadas al estímulo en toda la tarea.
    Txclases: es un diccionario de diccionarios, donde las claves de cada neuronas tienen asignado un
    diccionario con las las claves para cada clase y cada clave de clase tiene asignado las tasas de disparo 
    de esa neurona que corresponden a esa clase
    Tns: es un diccionario donde las claves de cada neurona tienen asignado un array donde cada fila son las tasas
    disparo de un ensayo a cada ventan
    bines: el número de bines que tendrán nuestras distribuciones de probabilidad
    keyclases: son las llaves/claves de cada clase
    keyneuron: son las llaves/claves de cada neurona"""
    llaves=keyneuron # las llaves de que corresponden a cada neurona
    minymax=mM(Tns,llaves)# los valores minimos y máximos que puede tomar la tasa de cada neurona sirven para limitar el bineo en el rango entre el min y el máx
    keycmt=list(keyclases)+['dtotal'] # las llaves de las clases más la llave 'dtotal'
    ProbArc={a:0 for a in keyneuron}# creo diccionario que tiene las llaves de clase + la llave dtotal
    for neuron in llaves: # itero para cada neurona
        AG={a:0 for a in keycmt}
        total=[]
        for clase in keyclases:
            counts, bins=np.histogram(Txclases[neuron][clase],bins=bines,range=minymax[neuron]) #count guarda un array con la frecuencia y bins un array con los bordes de bins
            counts=counts/counts.sum() #convertimos la frecuencia en probabilidad (frecuencia relativa)
            AG[clase]=[bins,counts]
    #calculamos bines y probabilidades de la ddistribución total:
        total=np.concatenate(tuple(Tns[neuron]))
        counts, bins=np.histogram(total,bins=bines,range=minymax[neuron]) #count guarda un array con la frecuencia y bins un array con los bordes de bins
        counts=counts/counts.sum() #convertimos la frecuencia en probabilidad
        AG['dtotal']=[bins,counts]
        ProbArc[neuron]=AG# diccionario de diccionarios 
    return ProbArc
 
def probaventana(Txclases,Tns,bines,keyclases,keyneuron):
    """
    Esta función sirve para calcular distribuciones de probabilidad de las tasas de disparo a cada tiempo/ventana 
    y las disribuciones de probabilidad de las tasas de disparo condicionadas al estímulo a cada tiempo/ventana
    Los parametros de entrada son:
    Txclases: es un diccionario de diccionarios, donde las claves de cada neuronas tienen asignado un
    diccionario con las las claves para cada clase y cada clave de clase tiene asignado las tasas de disparo 
    de esa neurona que corresponden a esa clase
    Tns: es un diccionario donde las claves de cada neurona tienen asignado un array donde cada fila son las tasas
    disparo de un ensayo a cada ventan
    bines: el número de bines que tendrán nuestras distribuciones de probabilidad
    keyclases: son las llaves/claves de cada clase
    keyneuron: son las llaves/claves de cada neurona"""
    llaves=keyneuron
    minymax=mM(Tns,llaves)
    keycmt=list(keyclases)+['dtotal']
    ProbArc={a:0 for a in keyneuron}
    nv=len(Txclases[keyneuron[0]][keyclases[0]][0])
    for neuron in llaves:
        AG={a:0 for a in keycmt}
        total=[]
        for clase in keyclases:
            pvamp=[]
            for v in range(nv):
                counts, bins=np.histogram(Txclases[neuron][clase][:,v],bins=bines,range=minymax[neuron]) #count guarda un array con la frecuencia y bins un array con los bordes de bins
                counts=counts/counts.sum() #convertimos la frecuencia en probabilidad (frecuencia relativa)
                pvamp.append(counts)
            AG[clase]=pvamp
    #calculamos bines y probabilidades de la ddistribución total:
        pvtotales=[]
        for v in range(nv):
            total=np.stack(tuple(Tns[neuron]))[:,v]
            counts, bins=np.histogram(total,bins=bines,range=minymax[neuron]) #count guarda un array con la frecuencia y bins un array con los bordes de bins
            counts=counts/counts.sum() #convertimos la frecuencia en probabilidad
            pvtotales.append(counts)
        
        AG['dtotal']=pvtotales
        ProbArc[neuron]=AG# diccionario de diccionarios 
    return ProbArc


def I_shanon(data,keys,ps):
    """data es los datos correspondientes a un archivo, keys=son las llaves de los estímulos
    ps=probabilidad de aparición del estímulo
    ptasa=probabilidad de la tasa
    pcondicionada= probabilidad de la tasa condicionada al estímulo"""
    llaves=keys[0:len(keys)-1]
    ptasa=data[keys[-1]]
    Info_stim=[]
    for llave in llaves:
        pcondicionada=data[llave]
        Is=np.nansum(pcondicionada*np.log2(pcondicionada/ptasa))
        Info_stim.append(Is)
    Ish=ps*np.sum(np.array(Info_stim))
    return Ish


def I_shanonxv(data,ctotal,clase,ps,v):
    """data es los datos correspondientes a una neurona, 
    total= es la llave que uso para las probabilidades de tasa totales de una neurona, es decir,las no condicionadas a ningún estímulo 
    ps=probabilidad de aparición del estímulo
    ptasa=probabilidad de la tasa
    pcondicionada= probabilidad de la tasa condicionada al estímulo
    v: ventana"""
    ptasa=data[ctotal][v] # es igual a : data['dtotal']
    pcondicionada=data[clase][v]
    Info_stim=np.nansum(pcondicionada*np.log2(pcondicionada/ptasa))
    Ish=ps*Info_stim
    return Ish


def I_mutuaxvent(dicprobas,keyclases,keyneuron,ps,nv,ctotal='dtotal'):
    """"Párametros de entrada:dicprobas: es un diccionario de diccionarios donde las claves de neurona tienen de value diccionarios donde las claves de amplitud tiene las probabilidades de las distribución de probabilidad de cada ventana 
    keyclases: son las claves de cada clases
    keyneuron: son las claves de cada neurona
    ps: son las probabilidades que tiene cada tipo de estímulo de ocurrir(dado el diseño experimental) es un diccionario donde las claves son las clases
    nv:es el número de ventanas que a las que calculamos las tasa de disparo y la distribución de probabilidad"""
    #keycmt=list(keyclases)+[ctotal] # es una lista con las claves de clase extendidas con un 'dtotal'
    #Info_mutua={n:0 for n in keyneuron} #es un diccionario de información mutua con claves por neurona
    Info_mutua=dict() #Crea diccionario vacío. # es un diccionario con donde las claves son las clases más dtotal y los values son cero
    for neuron in keyneuron:# iteramos un ciclo por neurona :5 neuronas
        Itotal_v=[]#es una lista vacía para almacenar arrays con la info de cada clase por ventana para después calcular el total por neurona por ventana al sumar las info de cada clase ventana a ventana
        I_clases=dict() # es un diccionario vacio que se convertira en un diccionario con claves por amplitud y 'dtotal' para almacenar los arrays con los valores de info por ventana, los values son ceros para poder asignar el array de infos de clase ventana a ventana
        for clase in keyclases:# itero un ciclo por cada clase :6 clases
            shanoncv=np.zeros(nv)# este array de zeros servirá para almacenar la info por ventana de una clase tendrá 531 elementos uno por ventana
            for v in range(nv):#nv un ciclo por número de ventana :531 ventanas
                info=I_shanonxv(dicprobas[neuron],ctotal,clase,ps[clase],v)# cálcula la info de shanon de una ventana particular 
                shanoncv[v]=info# vamos almacenando la info de cada ventana hasta junta la info # al final del último ciclo for de ventanas shanoncv tiene la info de todas las ventanas de una clase (de una neurona)
            Itotal_v.append(shanoncv)#al final de cada ciclo de clase agrego la lista de la info de las ventanas de esa clase a Itotal_v
            I_clases.update({clase:shanoncv})#agrego clave y valor a diccionario las claves son las clases y los valores son arrays con la info por ventana de un clase 
        I_v=np.sum(np.stack(Itotal_v),axis=0) # cálculo de la info total por ventanas de un ensayo
        I_clases.update({'dtotal':I_v}) # asignamos a d
        Info_mutua.update({neuron: I_clases}) #solo está pegando el array que corresponde a la última neurona en todas las neuronas #ERROR
    return Info_mutua


def mM_one(datos):
    """solo tiene dos parametros de entrada la tasa en un diccionario donde 
    las claves son correponden a neuronas y las claves """
    m,M=(np.min(datos),np.max(datos))
    return m,M


def P_ventana_Tn(Txclases,Tns,bines,keyclases):
    """
    Esta función sirve para calcular distribuciones de probabilidad de las tasas de disparo a cada tiempo/ventana 
    y las disribuciones de probabilidad de las tasas de disparo condicionadas al estímulo a cada tiempo/ventana
    Los parametros de entrada son:
    Txclases: es un diccionario de diccionarios, donde las claves de cada neuronas tienen asignado un
    diccionario con las las claves para cada clase y cada clave de clase tiene asignado las tasas de disparo 
    de esa neurona que corresponden a esa clase
    Tns: es un diccionario donde las claves de cada neurona tienen asignado un array donde cada fila son las tasas
    disparo de un ensayo a cada ventan
    bines: el número de bines que tendrán nuestras distribuciones de probabilidad
    keyclases: son las llaves/claves de cada clase"""
    minymax=mM_one(Tns)
    keycmt=list(keyclases)+['dtotal']
    nv=len(Txclases[keyclases[0]][0])
    AG=dict()
    total=[]
    for clase in keyclases:
        pvamp=[]
        for v in range(nv):
            counts, bins=np.histogram(Txclases[clase][:,v],bins=bines,range=minymax) #count guarda un array con la frecuencia y bins un array con los bordes de bins
            counts=counts/counts.sum() #convertimos la frecuencia en probabilidad (frecuencia relativa)
            pvamp.append(counts)
        AG.update({clase:pvamp})
    #calculamos bines y probabilidades de la ddistribución total:
    pvtotales=[]
    for v in range(nv):
        total=np.stack(tuple(Tns))[:,v]
        counts, bins=np.histogram(total,bins=bines,range=minymax) #count guarda un array con la frecuencia y bins un array con los bordes de bins
        counts=counts/counts.sum() #convertimos la frecuencia en probabilidad
        pvtotales.append(counts)
        
    AG.update({'dtotal':pvtotales})
    return AG

def I_one(dicprobas,keyclases,ps,nv,ctotal='dtotal'):
    """"Párametros de entrada:dicprobas: es un diccionario de diccionarios donde las claves de neurona tienen de value diccionarios donde las claves de amplitud tiene las probabilidades de las distribución de probabilidad de cada ventana 
    keyclases: son las claves de cada clases
    ps: son las probabilidades que tiene cada tipo de estímulo de ocurrir(dado el diseño experimental) es un diccionario donde las claves son las clases
    nv:es el número de ventanas que a las que calculamos las tasa de disparo y la distribución de probabilidad"""
    keycmt=list(keyclases)+[ctotal] # es una lista con las claves de clase extendidas con un 'dtotal'
    cl={clave:0 for clave in keycmt} # es un diccionario con donde las claves son las clases más dtotal y los values son cero
    Itotal_v=[]#es una lista vacía para almacenar arrays con la info de cada clase por ventana para después calcular el total por neurona por ventana al sumar las info de cada clase ventana a ventana
    I_clases=dict() # es un diccionario vacio que se convertira en un diccionario con claves por amplitud y 'dtotal' para almacenar los arrays con los valores de info por ventana, los values son ceros para poder asignar el array de infos de clase ventana a ventana
    for clase in keyclases:# itero un ciclo por cada clase :6 clases
        shanoncv=np.zeros(nv)# este array de zeros servirá para almacenar la info por ventana de una clase tendrá 531 elementos uno por ventana
        for v in range(nv):#nv un ciclo por número de ventana :531 ventanas
            info=I_shanonxv(dicprobas,ctotal,clase,ps[clase],v)# cálcula la info de shanon de una ventana particular 
            shanoncv[v]=info# vamos almacenando la info de cada ventana hasta junta la info # al final del último ciclo for de ventanas shanoncv tiene la info de todas las ventanas de una clase (de una neurona)
        Itotal_v.append(shanoncv)#al final del ciclo de clase agrego la lista de la info de las ventanas de esa clase a Itotal_v
        I_clases.update({clase:shanoncv})#agrego clave y valor a diccionario las claves son las clases y los valores son arrays con la info por ventana de un clase 
    I_v=np.sum(np.stack(Itotal_v),axis=0) # cálculo de la info total por ventanas de un ensayo
    I_clases.update({'dtotal':I_v}) # asignamos a d
    return I_clases

def zscoreTotal(tasa):
    media=np.mean(tasa)
    sigma=np.std(tasa)
    print(media,sigma)
    z=(tasa-media)/sigma
    return z

##############################################separar por primer o segundo estímulo
def zej11(tasa,musg):
    """calucula el z-score con sigma y mu total de un array, tasa es el array de tasas, musg es una lista 
    de dos elemetos con mu en el inidice 0 y sg en el indice 1"""
    z=(tasa-musg[0])/musg[1]
    return z
###########################################################################
def B_1er_est4(Neuron,ebloques):
    """Neuron es un array con las tasas de disparo por ensayo 
    ebloques una lista con el número de elementos por clase"""
    clases=[]
    nbloques=len(ebloques) #número de bloques (clases)
    superior=0 # indice superior
    for index in range(nbloques):# tantos ciclos como número de clases,es decir un ciclo por clase.
        inferior=superior# actualizamos el valor de del indice inferior 
        superior=inferior+ebloques[index]#actualizamos el valor del indice superior
        clase=Neuron[inferior:superior] # este slice corresponde a los elementos de una clase
        clases.append(clase)
    b1=np.vstack((clases[0],clases[1]))
    b2=np.vstack((clases[2],clases[3]))
    Primer_est=[b1,b2]
    return Primer_est
###########################################################################
def B_2do_est4(Neuron,ebloques):
    """Neuron es un array con las tasas de disparo por ensayo 
    ebloques una lista con el número de elementos por clase"""
    clases=[]
    nbloques=len(ebloques) #número de bloques (clases)
    superior=0 # indice superior
    for index in range(nbloques):# tantos ciclos como número de clases,es decir un ciclo por clase.
        inferior=superior# actualizamos el valor de del indice inferior 
        superior=inferior+ebloques[index]#actualizamos el valor del indice superior
        clase=Neuron[inferior:superior] # este slice corresponde a los elementos de una clase
        clases.append(clase)
    b1=np.vstack((clases[0],clases[2]))
    b2=np.vstack((clases[1],clases[3]))
    seg_est=[b1,b2]
    return seg_est
###########################################################################
def txprimerest(tasas,keyneuron,nkeyclases,bqs):
    """tasas"""
    #tasas de disparo separadas por amplitud del estímulo en un diccionario de diccionarios
    Txclases=dict()
    for llave in keyneuron:
        bloques=B_1er_est4(tasas[llave],bqs[llave])# (b1,b2)
        clas={nkeyclases[i]:bloques[i] for i in range(len(nkeyclases))}#aquí hay un error AQUI
        Txclases.update({llave:clas})
    return Txclases
#########################################################################
def txsegundoest(tasas,keyneuron,nkeyclases,bqs):
    """tasas"""
    #tasas de disparo separadas por amplitud del estímulo en un diccionario de diccionarios
    Txclases=dict()
    for llave in keyneuron:
        bloques=B_2do_est4(tasas[llave],bqs[llave])
        clas={nkeyclases[i]:bloques[i] for i in range(len(nkeyclases))}
        Txclases.update({llave:clas})
    return Txclases

###############################
def ztot(zxclas,keyneuron,keydivision):
    """de un diccionario de diccionarios de z-scores por neuronas y clases 
    esta función crea un sólo diccionario de clases combinando todas las neuronas pero por clases"""
    zto=dict()
    for amp in keydivision:
        z_juntadoc=[zxclas[neuron][amp] for neuron in keyneuron]
        z_jc=np.vstack(z_juntadoc)
        zto.update({amp:z_jc})
    return zto
####################################
def z_one(ztotal,clases):
    """a partir de  diccionario por clases de varias neuronas juntas crea un solo arreglo total"""
    Todoztot=[ztotal[amp] for amp in clases]
    Zeta_one=np.vstack(Todoztot)
    return Zeta_one

######################################AUROC AUROC AUROC
def Distributions_bvst(zbasal,ztot,nbins=200):
    mi=np.floor(min((zbasal.min(),ztot.min())))# el mínimo de mínimos de las dos distribuciones
    mx=np.ceil(max((zbasal.max(),ztot.max())))# el máximo de máximos de las dos distribuciones
    bins=np.linspace(mi,mx,nbins) # genero un linspace entre el minimo de minimos y el maximo de maximos, y este linspace va a tener el número de bines que yo indique en nbins 
    nv=len(ztot[0]) # cuantos datos tengo en el sample y que me va a servir para normalizar las frecuencias y convertirlas en probabilidades
    dist1=np.histogram(zbasal,bins=bins)[0]/len(zbasal) #probabilidades de los bines de la muestra 1
    dist2=[]
    for v in range(nv):
        dist=np.histogram(ztot[:,v],bins=bins)[0]/(len(ztot[:,v])) #probabilidades de los bines de la muestra 2
        dist2.append(dist)
    dist2=np.vstack(dist2)  
    return dist1,dist2,bins

####################################
def getDistributions(sample1,sample2,nbins=100):
    """esta función crea dos distribuciones de probabilidad cada distribución a partir de una respectiva muestra 
    numérica.
    los parámetros de entrada:
    sample1=
    nbins=número de bines que van a tener mis distribuciones ambas van a tener el mismo número de vines en el mismo rango """
    mi=np.floor(min((sample1.min(),sample2.min())))# el mínimo de mínimos de las dos distribuciones redondeado hacia abajo # [0] el mínimo de la distribución 1 y [1] el mínimo de las distribución 2
    mx=np.ceil(max((sample1.max(),sample2.max())))# el máximo de máximos de las dos distribuciones redondeado hacia arriba # [0] el Máximo de la distribución 1 y [1] el Máximo de la distribución 2
    bins=np.linspace(mi,mx,nbins) # genero un linspace entre el minimo de minimos y el maximo de maximos, y este linspace va a tener el número de bines que yo indique en nbins 
    ndata=len(sample1) # cuantos datos tengo en el sample y que me va a servir para normalizar las frecuencias y convertirlas en probabilidades
    dist1=np.histogram(sample1,bins=bins)[0]/ndata #probabilidades de los bines de la muestra 1
    dist2=np.histogram(sample2,bins=bins)[0]/ndata #probabilidades de los bines de la muestra 2
    return dist1,dist2,bins

###################################################
def getAUROC(dist1,dist2):
    """dadas dos distribuciones comparables es decir con la misma cantidad de bines en el mismo rango
    esta función calcula el AUROC
    los parametros de entrada:
    dist1=distribución 1
    dist2=distribución 2
    devuelve:
    el auroc que se genera con ambas distribuciones """
    cu1=np.cumsum(dist1) # suma acumulativa de la distribución 1, 
    cu2=np.cumsum(dist2) # suma acumulativa de las distribución 2, 
    auroc=np.trapz(cu1,cu2) # calculo el area debajo de la curva , que está definida por la suma acumulativa de  1 y la suma acumulativa de 2
    # trapz es una función para la integración numérica trapezoidal
    auroc=np.abs(auroc-0.5)+0.5 # el area que tanto se aleja de 0.5
    return auroc
######################################################
def GetDistributions_wnd(est1,est2,nbins=100):
    mi=np.floor(min((est1.min(),est2.min())))# el mínimo de mínimos de las dos distribuciones
    mx=np.ceil(max((est1.max(),est2.max())))# el máximo de máximos de las dos distribuciones
    bins=np.linspace(mi,mx,nbins) # genero un linspace entre el minimo de minimos y el maximo de maximos, y este linspace va a tener el número de bines que yo indique en nbins 
    nv=len(est1[0]) # cuantos datos tengo en el sample y que me va a servir para normalizar las frecuencias y convertirlas en probabilidades
    dist1=[]
    dist2=[]
    totale=len(est1[:,0])
    for v in range(nv):
        distribucion1=np.histogram(est1[:,v],bins=bins)[0]/totale #probabilidades de los bines de la muestra 2
        distribucion2=np.histogram(est2[:,v],bins=bins)[0]/totale #probabilidades de los bines de la muestra 1
        dist1.append(distribucion1)
        dist2.append(distribucion2)
    dist1=np.vstack(dist1)
    dist2=np.vstack(dist2)  
    return dist1,dist2,bins
##########################################################
#función para separar las probabilidades condicionadas
def divid_est(zs_Est,keyneuron,keyest):
    estA=[zs_Est[neuron][keyest[0]]for neuron in keyneuron]
    estG=[zs_Est[neuron][keyest[1]]for neuron in keyneuron]
    estA=np.vstack(estA)
    estG=np.vstack(estG)
    return estA,estG    
#####################################
def AUROClist(a,b,nv):
    lista=[]
    for v in range(nv):
        lista.append(getAUROC(a[v],b[v]))
    array=np.array(lista)
    return array
