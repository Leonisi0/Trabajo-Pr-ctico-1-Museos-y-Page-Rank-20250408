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