def construye_adyacencia(D,m): 
    D = D.copy()
    l = [] 
    for fila in D:
        l.append(fila<=fila[np.argsort(fila)[m]] ) 
    A = np.asarray(l).astype(int) 
    np.fill_diagonal(A,0) 
    return(A)

def calculaLU(A):
    m=A.shape[0]    
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    for iteracion in range(n - 1):
        for fila in range(iteracion + 1, n):
            
            divisor = Ac[fila][iteracion] / Ac[iteracion][iteracion]
            Ac[fila][iteracion] = divisor  
            
            for columna in range(iteracion + 1, n):  
                Ac[fila][columna] = Ac[fila][columna] - divisor * Ac[iteracion][columna]
            
    L = np.tril(Ac,-1) + np.eye(n)
    U = np.triu(Ac)
    
    return L, U

def calcula_matriz_C(A):
    K_inversa = np.eye(N, k=0)*(1/m)
    A_m = construye_adyacencia(D, m)
    A_transpueta = A_m.T
    
    return A_transpueta @ K_inversa 

    
def calcula_pagerank(A,alfa):
    C = calcula_matriz_C(A)
    N = A.shape[0]
    I = np.eye(N)
    M = (N/alfa)*(I-(1-a)*C)
    L, U = calculaLU(M)
    b = np.ones(N)
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

def calcula_matriz_C_continua(D): 
    N = D.shape[0]
    sumatoria_fila = 0
    F = np.zeros((N,N))
    K_inv = np.zeros((N,N))
    
    for fila in range(N):
        sumatoria_fila = 0
        for columna in range(N):
            if fila!=columna:
                F[fila][columna] = 1/D[fila][columna]
                sumatoria_fila += F[fila][columna]
        for columna in range(N):
            if fila==columna:
                K_inv[fila][columna] = 1/sumatoria_fila
    return F @ K_inv

def calcula_B(C,cantidad_de_visitas):
    B = np.eye(C.shape[0])  # C^0
    C_k = np.eye(C.shape[0])
    for k in range(cantidad_de_visitas-1):
        C_k = C_k @ C
        B += C_k
    return B

# TP 2

def vector_1(n):
  return np.zeros((n, 1)) + 1

def calcula_K(A):
  n = A.shape[0]
  K = np.zeros((n, n))
  for i in range(n):
    K[i][i] = np.sum(A[i])
  return K

def calcula_vector_k(A):
  return calcula_K(A) @ vector_1(A.shape[0]);

def cant_conexiones(A):
  return A.sum() / 2

def calcula_P(A):
  k = calcula_vector_k(A)
  P = k * k.T / (2*cant_conexiones(A))
  return P

def calcula_L(A):
  return calcula_K(A) - A

def calcula_R(A):
    return A - calcula_P(A)

def calcula_lambda(L,v):
    s = np.sign(v)
    return s.T/4 @ L @ s

def calcula_Q(R,v):
    s = np.sign(v)
    return s.T @ R @ s

def metpot1(A, tol=1e-8, maxrep=np.inf):
    v = np.random.rand(A.shape[0])
    v = v / np.linalg.norm(v, 2) #normalizo
    v1 = A @ v
    v1 = v1 / np.linalg.norm(v1, 2)
    l = (v.T @ A @ v) / (v.T @ v)
    l1 = (v1.T @ A @ v1) / (v1.T @ v1)
    nrep = 0
    while  abs(np.linalg.norm(v1 - v, 2)) > tol and nrep < maxrep:
        v = v1
        l = l1
        v1 = A @ v
        v1 = v1 / np.linalg.norm(v1, 2)
        l1 = (v1.T @ A @ v1) / (v1.T @ v1)
        nrep += 1
    l = (v1.T @ A @ v1) / (v1.T @ v1)
    return v1, l

def deflaciona(A,tol=1e-8,maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1 = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    deflA = A - l1 * np.outer(v1, v1) # Sugerencia, usar la funcion outer de numpy
    return deflA

def metpot_inv(A, tol=1e-8, maxrep=np.inf):
  L, U = elim_gaussiana(A)
  A_inv = inversa(L, U)
  vect, val = metpot1(A_inv, tol=tol, maxrep=maxrep)
  return vect, 1/val

def metpotI(A, mu, tol=1e-8, maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    shift = A + mu * np.eye(A.shape[0])
    return metpot_inv(shift, tol=tol, maxrep=maxrep)

def metpotI2(A, mu=0, tol=1e-8, maxrep=np.inf):
    L, U = elim_gaussiana(A)
    A_inv = inversa(L, U)
    A_def = deflaciona(A_inv,tol=1e-8,maxrep=np.inf)
    vect, val = metpot1(A_def, tol=tol, maxrep=maxrep)
    return vect, 1/val

def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        mu_identidad = np.eye(A.shape[0])*2
        L = calcula_L(A) + mu_identidad # Recalculamos el L

        v,l =  metpotI2(L)# Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        Ap = A[v >= 0][:, v >= 0] # Asociado al signo positivo
        Am = A[v < 0][:, v < 0] # Asociado al signo negativo
        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )
    
def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
         return([nombres_s])
    else:
        v,l = metpot1(R) # Primer autovector y autovalor de R
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return([nombres_s])
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            Rp = R[v >= 0][:, v >= 0] # Parte de R asociada a los valores positivos de v
            Rm = R[v < 0][:, v < 0] # Parte asociada a los valores negativos de v
            vp,lp = metpot1(Rp)  # autovector principal de Rp
            vm,lm = metpot1(Rm) # autovector principal de Rm

            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles
                return(modularidad_iterativo(A, Rp,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                modularidad_iterativo(A, Rm,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0]))