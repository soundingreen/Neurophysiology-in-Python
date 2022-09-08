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
