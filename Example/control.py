# -*- coding: utf-8 -*-
"""
Created on Thu May  5 09:46:17 2022

@author: ferna
"""

from numpy import *
from pylab import *
from time import perf_counter

#Ejercicio 1

def f(t, y):
    """Funcion que define la ecuacion diferencial"""
    return -9*y

def exacta(t):
    """Solucion exacta del problema de valor inicial"""
    return exp(-9*t)

def euler(a, b, fun, N, y0):
    """Implementacion del metodo de Euler en el intervalo [a, b]
    usando N particiones y condicion inicial y0"""
    
    h = (b-a)/N # paso de malla
  # t=linspace(a,b,N+1), t genera 11 puntos igualmente espaciados entre 0 y 11, se podría hacer de esta forma si se da el caso de paso constante
    t = zeros(N+1) # inicializacion del vector de nodos
    y = zeros(N+1) # inicializacion del vector de resultados
    t[0] = a # nodo inicial
    y[0] = y0 # valor inicial

    # Metodo de Euler
    for k in range(N): #k toma valores desde 0 hasta N-1, (el N no se alcanza)
        y[k+1] = y[k]+h*fun(t[k], y[k])
        t[k+1] = t[k]+h
    
    return (t, y)

a=0.
b=2.
y0=1
malla=[10,20,40,80,160] #Número de particiones


for N in malla:
    tini = perf_counter() 
    (t, y) = euler(a, b, f, N, y0) #llamada al metodo de Euler
    tfin=perf_counter()
    ye = exacta(t)
    plot(t, y, '-*')
    error = max(abs(y-ye)) #Dice el error en cada iteración
    print('-----')
    print('Tiempo CPU:',(tfin-tini))
    print('Error: ',(error))
    if N>malla[0]:                   #si N es mayor que el primer elemento de la lista
        cociente= errorold/error
        print('Cociente de errores:',cociente)
    errorold = error
    print('Paso de malla: ',((b-a)/N))
   
print('-----')   
plot(t, ye, 'k') # dibuja la solucion exacta
xlabel('t')
ylabel('y')
legend(['Euler, N=10','Euler, N=20','Euler, N=40','Euler, N=80','Euler, N=160', 'exacta'])
grid(True)
show()

#Como el el cociente de error se divide entre dos, el orden del método es 1.

def f1(t,y):
    dx = y[1]
    dy = 2*(y[0]-t)*(y[1]-1)
    return array ([dx,dy])

def exacta1(t):
    return tan(t)+t

print('Podemos observar que al aumentar el numero de iteraciones, la aproximación mejora, el cociente de errores tiende a 2, lo cual nos confirma que nuestro metodo es de orden 1.')
### Ejercicio 2


def rkSistemas45(a, b, fun, y0, h0, tol):
    
    hmin = 1.e-5 # paso de malla minimo
    hmax = 0.1 # paso de malla maximo

    
    # coeficientes RK
    q = 6 # orden del metodo mas uno
    A = zeros([q, q])
    A[1, 0] = 1/4
    A[2, 0] = 3/32
    A[2, 1] = 9/32
    A[3, 0] = 1932/2197
    A[3, 1] = -7200/2197
    A[3, 2] = 7296/2197
    A[4, 0] = 439/216
    A[4, 1] = -8
    A[4, 2] = 3680/513
    A[4, 3] = -845/4104
    A[5, 0] = -8/27
    A[5, 1] = 2
    A[5, 2] = -3544/2565
    A[5, 3] = 1859/4104
    A[5, 4] = -11/40
    
    B = zeros(q)
    B[0] = 25/216
    B[2] = 1408/2565
    B[3] = 2197/4104
    B[4] = -1/5
    
    BB = zeros(q)
    BB[0] = 16/135
    BB[2] = 6656/12825
    BB[3] = 28561/56430
    BB[4] = -9/50
    BB[5] = 2/55
    
    C = zeros(q)
    for i in range(q):
        C[i] = sum(A[i,:])
    
    # inicializacion de variables
    t = array([a]) # nodos
    y = y0 # soluciones
    h = array([h0]) # pasos de malla
    K = zeros([len(y0),q])
    k = 0 # contador de iteraciones
    
    
    
    while (t[k] < b):
        h[k] = min(h[k], b-t[k]) # ajuste del ultimo paso de malla
        for i in range(q):
            K[:,i] = fun(t[k]+C[i]*h[k], y[:,k]+h[k]*dot(A[i,:],transpose(K)))
        
        incrlow = dot(B,transpose(K)) # metodo de orden 4
        incrhigh = dot(BB,transpose(K)) # metodo de orden 5
            
        error = norm(h[k]*(incrhigh-incrlow),inf) # estimacion del error
        y = column_stack((y, y[:,k]+h[k]*incrlow))
        t = append(t, t[k]+h[k]); # t_(k+1)
        hnew = 0.9*h[k]*abs(tol/error)**(1./5) # h_(k+1)
        hnew = min(max(hnew,hmin),hmax) # hmin <= h_(k+1) <= hmax
        h = append(h, hnew)
        k += 1
        
    return (t, y, h)




# Datos del problema
a = 0. # extremo inferior del intervalo
b = 1.3 # extremo superior del intervalo
y0 = array([0,2]) # condicion inicial
y0 = y0.reshape(2,1)
h0 = 2*1.e-4 #paso inicial
tol = 1.e-5 #tolerancia




tini = perf_counter()
(t1, y1, h) = rkSistemas45(a, b, f1, y0, h0, tol) # llamada al metodo RK2(3)
tfin = perf_counter()


# calculo de la solucion exacta

ye = exacta1(t1)



print('-----')
print('Ejercicio 2')
print('-----')

# Calculo del error cometido
errorX= max(abs(y1[0]-exacta(t1)))
errorY= max(abs(y1[1]-exacta(t1)))

print('El error cometido es de : '+str(errorX))
print('Tiempo CPU: ' + str(tfin-tini))


hn = min(h[:-2]) # minimo valor utilizado del paso de malla
hm = max(h[:-2]) # maximo valor utilizado del paso de malla
print("Paso de malla minimo: " + str(hn))
print("Paso de malla maximo: " + str(hm))

print('El número total de pasos es : '+str(len(y1[0])-1))

# Las gráficas
figure('Ejercicio 2')

subplot(211)
plot(t1,y1[0],t1,ye)
legend(('Aprox 1', 'Exacta'))

subplot(212)
plot(t1,h)


