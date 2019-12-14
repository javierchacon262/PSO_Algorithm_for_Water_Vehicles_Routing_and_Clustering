import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

########################################################################################################
# Clase Solucion
class solution:
    def __init__(self):
        self.rutas = []
########################################################################################################
########################################################################################################
# Clase Ruta
class route:
    def __init__(self):
        self.nv_nodes = []
        self.v_nodes = []
        self.r_time = 0
########################################################################################################
########################################################################################################
# Clase Nodo
class node:
    def __init__(self, n_type, num, dem, coord_x, coord_y):
        self.n_type = n_type
        self.num = num
        self.dem = dem
        self.coord_x = coord_x
        self.coord_y = coord_y
########################################################################################################
########################################################################################################
# Clase Vehiculo
class veh:
    def __init__(self, num, cap):
        self.num = num
        self.cap = cap
########################################################################################################
########################################################################################################
def euclidean(node1, node2):
    return np.sqrt((node1.coord_x - node2.coord_x)**2 + (node1.coord_y - node2.coord_y)**2)
########################################################################################################
########################################################################################################
# Esta funcion devuelve el nodo mas cercado al buscado
def seek_dist(cust, cust_list, vis):
    dist_vec = []
    near = None
    #Guardar todas las distancias a los nodos
    for i in cust_list:
        if cust.num != i.num:
            euc = euclidean(cust, i)
            dist_vec.append([euc, i])
    #Ordenar nodos por cercania de menor a mayor distancia
    sort_dist = sorted(dist_vec, key=lambda x: x[0])
    #print('Metodo seek_dist: ')
    #print('')
    for i in sort_dist:
        #selecciono el nodo mas cercano que no se encuentre dentro de la lista de visitados
        #print(i[1].num)
        if i[1].num not in vis:
            near = i[1]
            break
    #print('visited: ')
    #print(vis)
    return near
########################################################################################################
########################################################################################################
def evalcust(cust, cap):
    if cap > cust.dem:
        return True
    else:
        return False
########################################################################################################
########################################################################################################
def sum_clus(clus_list):
    sum = 0
    for i in clus_list:
        sum += len(i)
    return sum
########################################################################################################
########################################################################################################
#Funcion de clustering para separar el problema en otros mas pequeños
def clustering(cust_list, vehs):
    veh = 0
    cap = np.copy(vehs[veh].cap)
    perm = list(np.random.permutation(cust_list))
    init_node = perm[0]
    clus_list = []
    clus = [init_node]
    visited = [init_node.num]
    while sum_clus(clus_list) < len(cust_list):
        for i in range(len(perm)):
            next_node = seek_dist(init_node, cust_list, visited)
            if next_node is not None:
                while next_node.num in visited:
                    next_node = seek_dist(init_node, cust_list, visited)
                if evalcust(next_node, cap):
                    clus.append(next_node)
                    cap -= next_node.dem
                    visited.append(next_node.num)
                elif i == len(perm) - 1:
                    clus_list.append(clus)
                    #print('cluster:')
                    #for j in clus:
                    #    print(j.num)
                    #print('')
                    clus = []
                    veh += 1
                    if veh == len(vehs):
                        veh = 0
                    cap = np.copy(vehs[veh].cap)
                init_node = next_node
            else:
                clus_list.append(clus)
                #print('cluster:')
                #for j in clus:
                #    print(j.num)
                #print('')
    return clus_list
########################################################################################################
########################################################################################################
#Funcion de calculo del centro de masa para un cluster
def mass_center(cluster):
    sum_x = 0
    sum_y = 0
    for i in cluster:
        sum_x += i.coord_x
        sum_y += i.coord_y
    return [sum_x/len(cluster), sum_y/len(cluster)]
########################################################################################################
########################################################################################################
#Funcion de busqueda del deposito mas cercano a una respectiva coordenada
def seek_dep(cmn, deps):
    vec = []
    vec2 = []
    for i in deps:
        sum = 0
        depn = node(2, 0, 0, i[1], i[2])
        sum += 2 * euclidean(cmn, depn)
        sum += i[3]
        if i[4] == []:
            vec.append([i[0], sum])
    i1 = vec.index(min(vec, key=lambda x: x[1]))
    return vec[i1][0]
########################################################################################################
########################################################################################################
# Funcion de asignacion de clusters a depositos(hidrantes)
def cluster_assign(cluster_list, deps, vehs):
    deps2 = []
    for i in cluster_list:
        cm = mass_center(i)
        cmn = node(2, 0, 0, cm[0], cm[1])
        dep = seek_dep(cmn, deps)
        deps[dep][-1] = i
    vcon = 0
    for i in deps:
        if i[4] != []:
            i.append(vehs[vcon])
            vcon += 1
            if vcon == len(vehs):
                vcon = 0
            deps2.append(i)
    return deps2
########################################################################################################
########################################################################################################
#Poblacion inicial para cada cluster
def init_population(cluster, pop_number):
    initp = []
    for i in range(pop_number):
        initp.append(np.random.permutation(cluster))
    return initp
########################################################################################################
########################################################################################################
#Codificacion de la solucion al espacio PSO para cada cluster
def code_solution(solution, xmin, xmax):
    coded = []
    xmin_n = xmin.num
    xmax_n = xmax.num
    n = len(solution)
    x = []
    y = []
    for i in solution:
        y.append(i.num)
    #print(y)
    for i in range(n):
        xij = xmin_n + ((xmax_n - xmin_n)/(n*(y[i]-1+np.random.uniform())))
        x.append(xij)
    return x
########################################################################################################
########################################################################################################
#Decodificacion de la solucion al espacio entero para cada cluster
def decode_solution(solution, coded):
    n = len(coded)
    sol = np.zeros((n,), dtype=int)
    for i in range(n):
        idx = coded.index(min(coded))
        coded[idx]+=n
        sol[idx] = i
    #print(sol)
    newsol = solution[sol]
    return newsol
########################################################################################################
########################################################################################################
#Actualizacion de w para el PSO
def w_act(wmax, wmin, iter, k):
    return wmax - (k * (wmax - wmin) / (iter))
########################################################################################################
########################################################################################################
#Actualizacion de velocidad para el PSO
def v_act(coded, vels, w, cp, cq, pi, pg):
    v = []
    for i in range(len(coded)):
        xi = coded[i]
        vi = vels[i]
        v_i1 = (w * vi) + (cp * np.random.rand() * (pi - xi)) + (cq * np.random.rand() * (pg - xi))
        v.append(v_i1)
    return v
########################################################################################################
########################################################################################################
#Acrtualizacion de posicion para el PSO
def x_act(coded, vels):
    x = []
    for i in range(len(coded)):
        xi = coded[i]
        vi = vels[i]
        x.append(xi + vi)
    return x
########################################################################################################
########################################################################################################
#Funcion objetivo 1, para evaluar el tiempo de operacion de un cluster
def obj_func_1(clus, vcap, pres_hid, pres_veh, m_vel):
    dep = clus[0]
    sol = clus[1]
    t_viaje = pres_hid * vcap
    d_dc1 = euclidean(dep, sol[0])
    t_viaje += d_dc1 / m_vel
    for i in range(1, len(sol)):
        d_i = euclidean(sol[i-1], sol[i])
        t_viaje += d_i /m_vel
        t_viaje += sol[i].dem * pres_veh
    return t_viaje
########################################################################################################
########################################################################################################
# Funcion Objetivo 2 para evaluar el comportamiento global de los cluster
def obj_func_2(clusters, vcap, pres_hid, pres_veh, m_vel, tmax):
    t_total = 0
    mayor = 0
    obj_vec = []
    for i in clusters:
        obj = obj_func_1(i, vcap, pres_hid, pres_veh, m_vel)
        obj_vec.append(obj)
        if obj > mayor:
            mayor = obj
        t_total += obj
    if t_total > tmax:
        sigma = 1 + 0.1*(t_total - tmax)
        bin = True
    else:
        sigma = 1
        bin = False
    sol = [bin, t_total, t_total*sigma, mayor, obj_vec]
    return sol
########################################################################################################
########################################################################################################
# Algoritmo principal
########################################################################################################
# Definicion de parametros iniciales & Lectura de datos de instancia desde excel
df = pd.read_excel('./Instancias_Excel/InsTest1.xlsx', '1')
########################################################################################################
sv = 5  # Numero de vehiculos
vcap = df['v_cap'].values[0]
########################################################################################################
df['coord_x'] = df['coord_x']/1000000
df['coord_y'] = df['coord_y']/1000000
########################################################################################################
# Depositos
dep_x = df['coord_x'].values[0:20]
dep_y = df['coord_y'].values[0:20]
dep_c = df['demands'].values[0:20]
########################################################################################################
#Cientes
cust_x = df['coord_x'].values[20:]
cust_y= df['coord_y'].values[20:]
cust_d = df['demands'].values[20:]
########################################################################################################
sd = dep_x.shape[0]  # Numero de depositos (hidrantes)
sc = cust_x.shape[0]  # Numero de clientes
deps = []
cust = []
vehs = []
########################################################################################################
# Creacion de los nodos cliente
for i in range(sc):
    cust.append(node(0, i, cust_d[i], cust_x[i], cust_y[i]))
########################################################################################################
# Creacion de los nodos deposito (hidrante)
for i in range(sd):
    deps.append(node(1, i, dep_c[i], dep_x[i], dep_y[i]))
########################################################################################################
# Creacion de vehiculos (cisternas)
caps = [250, 150, 150, 100, 100]
for i in range(sv):
    vehs.append(veh(i, caps[i]))
########################################################################################################
# LLamada a la funcion clustering
clusters = clustering(cust, vehs)
########################################################################################################
deps = list(zip(list(range(sd)), dep_x, dep_y, dep_c))
deps = [list(i) for i in deps]
for i in deps:
    i.append([])
########################################################################################################
# Acomodar clusters, super importante
con = 0
cluster_list = []
for i in clusters:
    for j in i:
        con += 1
    if con < sc:
        cluster_list.append(i)
########################################################################################################
# Asignacion de clusters a depositos (hidrantes)
deps_ass = cluster_assign(cluster_list, deps, vehs)
environ = []
clus_dem = []
for i in deps_ass:
    dep_i = node(1, i[0], i[3], i[1], i[2])
    environ.append([dep_i, i[4]])
    # Recorrer clusters para sumar demandas
    demsum_j = 0
    for j in range(len(i[4])):
        demsum_j += i[4][j].dem
    clus_dem.append(demsum_j)
#Suma de demandas de cada vehiculo
vehs_dems = np.zeros((1, len(vehs)))
vcon = 0
for i in clus_dem:
    vehs_dems[0][vcon] += i
    vcon += 1
    if vcon == len(vehs):
        vcon = 0
########################################################################################################
# Asignar demandas de clusters a cisternas

########################################################################################################
# Parametros
cp = [0.4]
cq = [0.4]
wv = [1]
iter = [100]
init_p = [100]
m_vel = [1]
pres_veh = [1]
pres_hid = [1]

big_mat_sol = []
big_mat_sol_vec = []
for i in deps_ass:
    mat_sol = []
    mat_sol_vec = []
    dep = node(1, i[0], i[3], i[1], i[2])
    clu = i[4]
    veh = i[5]
    pop = init_population(clu, init_p[0])
    #veh = 0
    for j in pop:
        sol = []
        sol_vec = []
        mi = min(j, key=lambda x: x.num)
        ma = max(j, key=lambda x: x.num)
        #Initial solution coding
        coded = code_solution(j, mi, ma)
        #Inicializacion aleatoria de las velocidades - distribucion uniforme.
        vels = np.random.rand(len(coded))
        #Inicializacion del mejor local y mejor global
        pi = 9999999999.0
        pg = 9999999999.0
        for k in range(iter[0]):
            w = w_act(2, 0.1, iter[0], k)
            vels = v_act(coded, vels, w, cp[0], cq[0], pi, pg)
            coded = x_act(coded, vels)
            decoded = decode_solution(j, coded)
            #caps[veh]
            obj_k = obj_func_1([dep, decoded], veh.cap, pres_hid[0], pres_veh[0], m_vel[0])
            if obj_k < pi:
                sol.append(obj_k)
                sol_vec.append(decoded)
                pi = obj_k
            coded = code_solution(decoded, mi, ma)

        if sol[-1] < pg:
            pg = sol[-1]
            pg_vec = sol_vec[-1]

        mat_sol.append(sol)
        mat_sol_vec.append(sol_vec)
    big_mat_sol.append(mat_sol)
    big_mat_sol_vec.append(mat_sol_vec)
########################################################################################################
# Extraer los minimos de cada cluster, funcion objetivo, ruta y vehiculo
mins = []
mins_vec = []
for i in range(len(big_mat_sol)): # Recorre cada cluster
    min_per_clus = 999999999.0
    for j in range(len(big_mat_sol[i])): # Recorre cada vector de soluciones para el cluster i
        for k in range(len(big_mat_sol[i][j])): # Recorre las soluciones del vector i, j
            if big_mat_sol[i][j][k] < min_per_clus:
                min_per_clus = big_mat_sol[i][j][k]
                min_per_clus_vec = big_mat_sol_vec[i][j][k]
    mins.append(min_per_clus)
    mins_vec.append(min_per_clus_vec)
########################################################################################################
# Calculo porcentaje de eficacia


########################################################################################################
#plots

k = 0
vec = []
vecn = []
vecnx = []
vecny = []
vecx = []
vecy = []
v100 = []
for i in range(len(clusters)):
    #print('cluster: '+str(i))
    #print(len(clusters[i]))
    for j in clusters[i]:
        vec.append(j.num)
        vecx.append(j.coord_x)
        vecy.append(j.coord_y)
    #    print('node: '+str(j.num))
    #    k += 1
    #    print(k)
for i in range(sc):
    if i not in vec:
        #print(i)
        vecn.append(cust[i])
        #print(cust[i].num)
        vecnx.append(cust[i].coord_x)
        vecny.append(cust[i].coord_y)
vecnx = np.array(vecnx)
vecny = np.array(vecny)
vecx = np.array(vecx)
vecy = np.array(vecy)
########################################################################################################
# Figura de los clusters
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(111)
ax1.scatter(vecx, vecy, c='b', alpha=0.5, label='Clientes')
ax1.scatter(dep_x, dep_y, c='r', alpha=1, label='Depositos')
plt.legend(loc='upper left')
plt.show()
########################################################################################################
# Plot No.1
x_veh = []
for i in range(len(vehs)):
    x_veh.append(i)
x_veh = np.array(x_veh)
fig2 = plt.figure(figsize=(10, 10))
ax2 = fig2.add_subplot(111)
ax2.bar(x_veh, vehs_dems[0])
ax2.title = plt.title('Demanda Por Vehículo')
ax2.xlabel = plt.xlabel('Vehículos')
ax2.ylabel = plt.ylabel('Demanda Atendida')
plt.show()
########################################################################################################
# Plot No.2
x_clus = []
for i in range(len(deps_ass)):
    x_clus.append(i)
x_clus = np.array(x_clus)
fig3 = plt.figure(figsize=(10, 10))
ax3 = fig3.add_subplot(111)
ax3.bar(x_clus, clus_dem)
ax3.title = plt.title('Demanda Por Cluster')
ax3.xlabel = plt.xlabel('Cluster')
ax3.ylabel = plt.ylabel('Demanda Atendida')
plt.show()
########################################################################################################
# Plot No.3
fig4 = plt.figure(figsize=(10, 10))
ax4 = fig4.add_subplot(111)
ax4.bar(x_clus, mins)
ax4.title = plt.title('Funcion Objetivo por Cluster')
ax4.xlabel = plt.xlabel('Cluster')
ax4.ylabel = plt.ylabel('Funcion Objetivo')
plt.show()
########################################################################################################
# Plot No.4
fig5 = plt.figure(figsize=(10, 10))
ax5 = fig5.add_subplot(111)
ax5.scatter(clus_dem, mins)
ax5.title = plt.title('Funcion Objetivo por Demanda de Cluster')
ax5.xlabel = plt.xlabel('Demanda Atendida')
ax5.ylabel = plt.ylabel('Funcion Objetivo')
plt.show()
########################################################################################################