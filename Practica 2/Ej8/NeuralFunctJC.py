#!/usr/bin/env python
# coding: utf-8




# importamos las librerias que necesitaremos
from numba import njit, jit
import numpy as np
import matplotlib.pyplot as plt




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

###########################################################
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
def clases_Z(zNeuron,ebloques):
    """
    Dado un array con los z-score por ensayo de una neurona,y un una lista o tupla con los elementos,que contiene cada clase,
    en el mismo orden en que las clases están en el array de z-scores,está función calcula el z-score promedio(en cada ventana) 
    para una clase y además calcula la desviación estandar de ese zscore promedio (en cada ventana).Esta función devuelve dos
    listas de ndarrays donde cada array contiene las zscore promedio y otro contiene las std de esa clase.
    su parametros son obligatorios.
    paramétros
    zNeuron contiene debe ser un array donde cada row es un el z-score de un ensayo zNeuron debe contener los z-score de todos
    los ensayos de todas las clases.
    ebloques debe ser una lista o tupla donde cada elemento es el número de elementos que tiene un bloque,
    """
    sigmas=[]# lista para almacenar las desviaciones estandar de cada clase
    mus=[]# lista para almacenar los z-score promedio de cada clase
    nbloques=len(ebloques) #número de bloques (clases)
    superior=0 # indice superior
    for index in range(nbloques):# tantos ciclos como número de clases, un ciclo por clase.
        inferior=superior# actualizamos el valor de del indice inferior 
        superior=inferior+ebloques[index]#actualizamos el valor del indice superior
        clase=zNeuron[inferior:superior] # este slice corresponde a los elementos de una clase 
        mu=np.mean(clase,axis=0) # calculamos el z-score promedio de una clase
        sg=np.std(clase,axis=0) # calculamos las std de los z-score de una clase
        mus.append(mu)# agregamos el z-score promedio a nuestra lista de z-scores promedio
        sigmas.append(sg)# agregamos la std de una clase a nuestra lista de std 
    return mus,sigmas # devuelve dos listas de ndarrays

###########################################################################
def graf4c(ejex,zm,zsg,titulo):
    """
    Esta función unicamente para listas con 4 clases.
    Dado un ejex comúm y un lista o tupla que contiene arrays con los zscore de cada clase y otra lista o tupla que contiene
    los std de cada clase,así como un título para la figura. Esta función crea una figura con las graficas del zscore promedio 
    de cada clase. 
    """
    c1,c2,c3,c4=zm# desempaquetamos los z score promedio
    sg1,sg2,sg3,sg4=zsg #desempaquetamos los std de cada clase
    # una figura
    fig,(ax1,ax2,ax3,ax4)=plt.subplots(nrows=4,ncols=1,sharex=True)
    # titulo
    fontdict_title = {'family': 'serif','color':  'darkred','weight': 'normal','size': 16,} # formato para el título
    ax1.set_title(titulo, fontdict_title) #título de la gráfic.plot(xs1f,c1_zNeuS1,label='clase1')# creamos la gráfica del z-score
    #######
    ax1.plot(ejex,c1,label='clase1',color='red')# creamos la gráfica del z-score
    ax1.fill_between(ejex,c1-sg1, c1+sg1)# +- 1std
    ax2.plot(ejex,c2,label='clase2',color='purple')# creamos la gráfica del z-score
    ax2.fill_between(ejex,c2-sg2, c2+sg2)# +- 1std
    ax3.plot(ejex,c3,label='clase3',color='darkblue')# creamos la gráfica del z-score
    ax3.fill_between(ejex,c3-sg3, c3+sg3)# +- 1std
    ax4.plot(ejex,c4,label='clase4',color='black')# creamos la gráfica del z-score
    ax4.fill_between(ejex,c4-sg4, c4+sg4)# +- 1std
    fig.legend()
    # labels para los ejes
    ax4.set_xlabel(" Time [s] ") # Configuramos la etiqueta del eje X
    ax1.set_ylabel(" z-score ") # Configuramos la etiqueta del eje Y
    ax2.set_ylabel(" z-score ") # Configuramos la etiqueta del eje Y
    ax3.set_ylabel(" z-score ") # Configuramos la etiqueta del eje Y
    ax4.set_ylabel(" z-score ") # Configuramos la etiqueta del eje Y
    #show
    
    return plt.tight_layout()
#################################################################
def graf6c(ejex,zm,zsg,titulo):
    """
    Esta función unicamente para listas con 6 clases.
    Dado un ejex comúm y un lista o tupla que contiene arrays con los zscore de cada clase y otra lista o tupla que contiene
    los std de cada clase,así como un título para la figura. Esta función crea una figura con las graficas del zscore promedio 
    de cada clase. 
    """
    c1,c2,c3,c4,c5,c6=zm
    sg1,sg2,sg3,sg4,sg5,sg6=zsg
    # una figura
    fig,(ax1,ax2,ax3,ax4,ax5,ax6)=plt.subplots(nrows=6,ncols=1,sharex=True)
    # titulo
    fontdict_title = {'family': 'serif','color':  'darkred','weight': 'normal','size': 16,} # formato para el título
    ax1.set_title(titulo, fontdict_title) #título de la gráfic.plot(xs1f,c1_zNeuS1,label='clase1')# creamos la gráfica del z-score
    #######
    ax1.plot(ejex,c1,label='clase1',color='red')# creamos la gráfica del z-score
    ax1.fill_between(ejex,c1-sg1, c1+sg1)# +- 1std
    ax2.plot(ejex,c2,label='clase2',color='purple')# creamos la gráfica del z-score
    ax2.fill_between(ejex,c2-sg2, c2+sg2)# +- 1std
    ax3.plot(ejex,c3,label='clase3',color='darkblue')# creamos la gráfica del z-score
    ax3.fill_between(ejex,c3-sg3, c3+sg3)# +- 1std
    ax4.plot(ejex,c4,label='clase4',color='black')# creamos la gráfica del z-score
    ax4.fill_between(ejex,c4-sg4, c4+sg4)# +- 1std
    ax5.plot(ejex,c5,label='clase5',color='pink')# creamos la gráfica del z-score
    ax5.fill_between(ejex,c5-sg5, c5+sg5)# +- 1std
    ax6.plot(ejex,c6,label='clase6',color='green')# creamos la gráfica del z-score
    ax6.fill_between(ejex,c6-sg6, c6+sg6)# +- 1std
    fig.legend()
    # labels para los ejes
    ax6.set_xlabel(" Time [s] ") # Configuramos la etiqueta del eje X
    ax1.set_ylabel(" z-score ") # Configuramos la etiqueta del eje Y
    ax2.set_ylabel(" z-score ") # Configuramos la etiqueta del eje Y
    ax3.set_ylabel(" z-score ") # Configuramos la etiqueta del eje Y
    ax4.set_ylabel(" z-score ") # Configuramos la etiqueta del eje Y
    ax5.set_ylabel(" z-score ") # Configuramos la etiqueta del eje Y
    ax6.set_ylabel(" z-score ") # Configuramos la etiqueta del eje Y
    #show
    
    return plt.tight_layout()

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
def z_score_array(tasa,media,sigma):
    """
    Dado un array de tasas, una array de medias (o un array renglón) y 
    un array de std (o un array renglón).Esta función calcula el z-score,
    y devuelve una matriz de z-score's.
    """
    if np.sum(sigma==0)==0:
        z=(tasa-media)/sigma
    else:
        sigma[sigma==0]=1
        z=(tasa-media)/sigma
    return z
#############################################################
def zCV(FNeuron):
    """
    Dado un array donde los rows son las tasas de disparo por ensayo de 
    una clase  y las columns son las ventanas de tiempo, de una neurona.
    Está función calcula el z-score de cada ensayo usando la mu y sigma de la clase 
    por ventana de tiempo particular.Esta función devuelve un array de z-score
    """
    mu,sg=col_mu_sg(FNeuron)
    zsc=z_score_array(FNeuron,mu,sg)
    return zsc
############################################################

def zCV_clases(FNeuron,ebloques):
    """
    Dado un array con las tasas de disparo por ensayo son los rows y las 
    columns son las ventanas de tiempo, de una neurona, y un una lista o tupla
    con los elementos,que contiene cada clase,en el mismo orden en que las 
    clases están en el array de tasa. Está función calcula el z-score usando la
    mu y sigma de cada clase por ventana de tiempo.
    
    ventana y clase particular para una clase y además calcula la desviación estandar de ese zscore promedio (en cada ventana).Esta función devuelve dos
    listas de ndarrays donde cada array contiene las zscore promedio y otro contiene las std de esa clase.
    su parametros son obligatorios.
    paramétros
    FNeuron contiene debe ser un array donde cada row es un el z-score de un ensayo zNeuron debe contener los z-score de todos
    los ensayos de todas las clases.
    ebloques debe ser una lista o tupla donde cada elemento es el número de elementos que tiene un bloque,
    """
    zxclase=[]# los z-score por clase
    zm=[]# z-score promedio por clase
    zsg=[]# sigma de los z-scores de cada clase 
    nbloques=len(ebloques) #número de bloques (clases)
    superior=0 # indice superior
    for index in range(nbloques):# tantos ciclos como número de clases,es decir un ciclo por clase.
        inferior=superior# actualizamos el valor de del indice inferior 
        superior=inferior+ebloques[index]#actualizamos el valor del indice superior
        clase=FNeuron[inferior:superior] # este slice corresponde a los elementos de una clase 
        zNrnxC=zCV(clase)# z-score por clase y ventana de tiempo particular.
        zxclase.append(zNrnxC)#append
        mu,sg=col_mu_sg(zNrnxC)# calculo de z-score promedio de la clase y de std de los z-score de la clase
        zm.append(mu)#append
        zsg.append(sg)#append
    
    return zxclase,zm,zsg # devuelve tres listas de ndarrays

######################################################################
def fourgr(ejex,zm,titulo):
    """
    Esta función unicamente para listas con 4 clases.
    Dado un ejex comúm y un lista o tupla que contiene arrays con los zscore,así como un título para la figura. Esta función crea una figura con las graficas del zscore promedio 
    de cada clase. 
    """
    c1,c2,c3,c4=zm# desempaquetamos los z score promedio
    
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
    ax1.set_ylabel(" z-score ") # Configuramos la etiqueta del eje Y
    ax2.set_ylabel(" z-score ") # Configuramos la etiqueta del eje Y
    ax3.set_ylabel(" z-score ") # Configuramos la etiqueta del eje Y
    ax4.set_ylabel(" z-score ") # Configuramos la etiqueta del eje Y
    #show
    
    return plt.tight_layout()
########################################################
def sixgr(ejex,zm,titulo,ejey='z-score'):
    """
    Esta función unicamente para listas con 6 clases.
    Dado un ejex comúm y un lista o tupla que contiene arrays con los zscore ,
    así como un título para la figura. Esta función crea una figura con las 
    graficas del zscore promedio de cada clase. 
    """
    c1,c2,c3,c4,c5,c6=zm
    # una figura
    fig,(ax1,ax2,ax3,ax4,ax5,ax6)=plt.subplots(nrows=6,ncols=1,sharex=True)
    # titulo
    fontdict_title = {'family': 'serif','color':  'darkred','weight': 'normal','size': 16,} # formato para el título
    ax1.set_title(titulo, fontdict_title) #título de la gráfic.plot(xs1f,c1_zNeuS1,label='clase1')# creamos la gráfica del z-score
    #######
    ax1.plot(ejex,c1,label='clase1',color='red')# creamos la gráfica del z-score
    ax2.plot(ejex,c2,label='clase2',color='purple')# creamos la gráfica del z-score
    ax3.plot(ejex,c3,label='clase3',color='darkblue')# creamos la gráfica del z-score
    ax4.plot(ejex,c4,label='clase4',color='black')# creamos la gráfica del z-score
    ax5.plot(ejex,c5,label='clase5',color='black')# creamos la gráfica del z-score
    ax6.plot(ejex,c6,label='clase6',color='black')# creamos la gráfica del z-score
    fig.legend()
    # labels para los ejes
    ax6.set_xlabel(" Time [s] ") # Configuramos la etiqueta del eje X
    ax1.set_ylabel(ejey) # Configuramos la etiqueta del eje Y
    ax2.set_ylabel(ejey) # Configuramos la etiqueta del eje Y
    ax3.set_ylabel(ejey) # Configuramos la etiqueta del eje Y
    ax4.set_ylabel(ejey) # Configuramos la etiqueta del eje Y
    ax5.set_ylabel(ejey) # Configuramos la etiqueta del eje Y
    ax6.set_ylabel(ejey) # Configuramos la etiqueta del eje Y
    #show
    
    return plt.tight_layout()
##############################################

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


#########################################################################
def procesalist(contenido):
    nums=[] # declaro una lista vacía para guardar arreglos donde cada arreglo contendra los registros correspondientes a un ensayo 
    for string in contenido:# hay n str en contenido1 n-k de esos str son '\n' y los demás son los datos de k ensayos
        if string!='\n': # solo 60 str de contenido cumplen esta condición y esos son ensayos
            string=string.replace(','," ") # reemplazo las comas por espacios para poder usar split después
            string=string.split() #split hace cada número dentro del str de ensayos un str individual, y pone estos str dentro de una lista # ahora la variable string es una lista donde cada elemento es un str de un número flotante
            for numero in string: # itero cada elemento de la lista string
                numero=float(numero) # cada str de un número lo convierto a float 
                nums.append(numero)# cada float agrego con append a lista_n
    nums_n=np.array(nums)# al final de el segundo ciclo for agrego la lista_n correspondiente a un array que llamo ensayo_n
    return nums_n

def importarlist(archivo): # función para importar datos
    content=abrir(archivo)
    Datos=procesalist(content)
    return Datos


#######################
def proc_lim(contenido,left,right):
    ensayos=[] # declaro una lista vacía para guardar arreglos donde cada arreglo contendra los registros correspondientes a un ensayo 
    for string in contenido:# hay n str en contenido1 n-k de esos str son '\n' y los demás son los datos de k ensayos
        if string!='\n': # solo 60 str de contenido cumplen esta condición y esos son ensayos
            lista_n=[] # entonces declararé en total k listas vacías una en cada ciclo
            string=string.replace(','," ") # reemplazo las comas por espacios para poder usar split después
            string=string.split() #split hace cada número dentro del str de ensayos un str individual, y pone estos str dentro de una lista # ahora la variable string es una lista donde cada elemento es un str de un número flotante
            for numero in string: # itero cada elemento de la lista string
                numero=float(numero) # cada str de un número lo convierto a float 
                if (numero>=left) and (numero<=right):
                    lista_n.append(numero)# cada float agrego con append a lista_n
            ensayo_n=np.array(lista_n)# al final de el segundo ciclo for agrego la lista_n correspondiente a un array que llamo ensayo_n
            ensayos.append(ensayo_n)# luego agrego mi arreglo ensayo_n a mi lista de arrays "ensayos"
    return ensayos
################################
def import_between(archivo,left,right): # función para importar datos
    content=abrir(archivo)
    Datos=proc_lim(content,left,right)
    return Datos
######################

###########################################
def proc_lim2(contenido,left,right):
    ensayos=[] # declaro una lista vacía para guardar arreglos donde cada arreglo contendra los registros correspondientes a un ensayo 
    for string in contenido:# hay n str en contenido1 n-k de esos str son '\n' y los demás son los datos de los k ensayos
        s=string.strip().split(",")  # al hacer esto identifico los separadores aunque no sean \n
        if  s.count('')!=len(s): # solo los ensayos cumplen esto mientras que los separadores de bloque no
            lista_n=[] # entonces declararé en total k listas vacías una en cada ciclo
            string=string.replace(','," ") # reemplazo las comas por espacios para poder usar split después
            string=string.split() #split hace cada número dentro del str de ensayos un str individual, y pone estos str dentro de una lista # ahora la variable string es una lista donde cada elemento es un str de un número flotante
            for numero in string: # itero cada elemento de la lista string
                numero=float(numero) # cada str de un número lo convierto a float 
                if (numero>=left) and (numero<=right):
                    lista_n.append(numero)# cada float agrego con append a lista_n
            ensayo_n=np.array(lista_n)# al final de el segundo ciclo for agrego la lista_n correspondiente a un array que llamo ensayo_n
            ensayos.append(ensayo_n)# luego agrego mi arreglo ensayo_n a mi lista de arrays "ensayos"
    return ensayos
#################################################
def import_between_v2(archivo,left,right): # función para importar datos
    content=abrir(archivo)
    Datos=proc_lim2(content,left,right)
    return Datos
##################################################
def raster4plot(data,titulo,clasn=['','','','']):

    c1,c2,c3,c4=data# desempaquetamos 
    n1,n2,n3,n4=clasn
    # una figura
    fig,(ax1,ax2,ax3,ax4)=plt.subplots(nrows=4,ncols=1,sharex=True)
    # titulo
    fontdict_title = {'family': 'serif','color':  'darkblue','weight': 'normal','size': 16,} # formato para el título
    ax1.set_title(titulo, fontdict_title) #Título solo en el ax1
    #######
    ax1.eventplot(c1,color='red')# creamos la gráfica 
    ax1.plot([],label='clase1'+n1,color='red')
    ax2.eventplot(c2,color='purple')# creamos la gráfica 
    ax2.plot([],label='clase2'+n2,color='purple')
    ax3.eventplot(c3,color='darkblue')# creamos la gráfica
    ax3.plot([],label='clase3'+n3,color='darkblue')
    ax4.eventplot(c4,color='green')# creamos la gráfica 
    ax4.plot([],label='clase4'+n4,color='green')
    
    legend = fig.legend(loc="center left", shadow=True, fontsize='x-large',bbox_to_anchor=(1, 0, 0.5, 1))
    # Put a nicer background color on the legend.
    
    # labels para los ejes
    ax4.set_xlabel(" Time [s] ") # Configuramos la etiqueta del eje X
    #show
    
    return plt.tight_layout()
##########################################################
def raster6plot(data,titulo):

    c1,c2,c3,c4,c5,c6=data# desempaquetamos 
    
    # una figura
    fig,(ax1,ax2,ax3,ax4,ax5,ax6)=plt.subplots(nrows=6,ncols=1,sharex=True)
    # titulo
    fontdict_title = {'family': 'serif','color':  'darkblue','weight': 'normal','size': 16,} # formato para el título
    ax1.set_title(titulo, fontdict_title) #Título solo en el ax1
    #######
    ax1.eventplot(c1,color='red')# creamos la gráfica 
    ax1.plot([],label='clase1',color='red')
    ax2.eventplot(c2,color='purple')# creamos la gráfica 
    ax2.plot([],label='clase2',color='purple')
    ax3.eventplot(c3,color='darkblue')# creamos la gráfica
    ax3.plot([],label='clase3',color='darkblue')
    ax4.eventplot(c4,color='green')# creamos la gráfica 
    ax4.plot([],label='clase4',color='green')
    ax5.eventplot(c5,color='lightblue')# creamos la gráfica 
    ax5.plot([],label='clase5',color='lightblue')
    ax6.eventplot(c4,color='black')# creamos la gráfica 
    ax6.plot([],label='clase6',color='black')
    
    legend = fig.legend(loc="center left", shadow=True, fontsize='x-large',bbox_to_anchor=(1, 0, 0.5, 1))
    # Put a nicer background color on the legend.
    
    # labels para los ejes
    ax6.set_xlabel(" Time [s] ") # Configuramos la etiqueta del eje X
    #show
    
    return plt.tight_layout()
