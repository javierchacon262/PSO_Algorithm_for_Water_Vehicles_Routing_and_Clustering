import numpy as np
import scipy.io as io
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
    print('Metodo seek_dist: ')
    print('')
    for i in sort_dist:
        #selecciono el nodo mas cercano que no se encuentre dentro de la lista de visitados
        #print(i[1].num)
        if i[1].num not in vis:
            near = i[1]
            break
    print('visited: ')
    print(vis)
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
def clustering(cust_list, veh):
    cap = np.copy(veh.cap)
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
                    print('cluster:')
                    for j in clus:
                        print(j.num)
                    print('')
                    clus = []
                    cap = np.copy(veh.cap)
                init_node = next_node
            else:
                clus_list.append(clus)
                print('cluster:')
                for j in clus:
                    print(j.num)
                print('')
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
def cluster_assign(cluster_list, deps):
    deps2 = []
    for i in cluster_list:
        cm = mass_center(i)
        cmn = node(2, 0, 0, cm[0], cm[1])
        dep = seek_dep(cmn, deps)
        deps[dep][-1] = i
    for i in deps:
        if i[4] != []:
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
    n = len(solution)
    x = []
    y = []
    for i in solution:
        y.append(i.num)
    print(y)
    for i in range(n):
        xij = xmin + ((xmax - xmin)/(n*(y[i]-1+np.random.uniform())))
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
    print(sol)
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
def v_act(xi, vi, w, cp, cq, pi, pg):
    v = (w * vi) + (cp * np.random.rand() * (pi - xi)) + (cq * np.random.rand() * (pg - xi))
    return v
########################################################################################################
########################################################################################################
#Acrtualizacion de posicion para el PSO
def x_act(xi, vi):
    x = xi + vi
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
    for i in clusters:
        t_total += obj_func_1(i, vcap, pres_hid, pres_veh, m_vel)
    if t_total > tmax:
        sigma = 1 + 0.1*(t_total - tmax)
    else:
        sigma = 1
    return t_total*sigma
########################################################################################################
########################################################################################################
# Algoritmo principal
########################################################################################################
# Definicion de parametros iniciales & Lectura de datos de instancia desde excel
df = pd.read_excel('./Instancias_Excel/InsTest1.xlsx', '1')
########################################################################################################
sv = 15  # Numero de vehiculos
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
for i in range(sv):
    vehs.append(veh(i, v_cap))
########################################################################################################
# LLamada a la funcion clustering
clusters = clustering(cust, vehs[0])
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
deps_ass = cluster_assign(cluster_list, deps)
environ = []
for i in deps_ass:
    dep_i = node(1, i[0], i[3], i[1], i[2])
    environ.append([dep_i, i[4]])
########################################################################################################
# Parametros
cp = []
cq = []
wv = []
iter = []
m_vel = []
pres_veh = []
pres_hid = []


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
        print(i)
        vecn.append(cust[i])
        print(cust[i].num)
        vecnx.append(cust[i].coord_x)
        vecny.append(cust[i].coord_y)
vecnx = np.array(vecnx)
vecny = np.array(vecny)
vecx = np.array(vecx)
vecy = np.array(vecy)

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(111)
ax1.scatter(vecx, vecy, c='b', alpha=0.5, label='IN')
ax1.scatter(vecnx, vecny, c='r', alpha=1, label='OUT')
ax1.scatter(dep_x, dep_y, c='y', alpha=1, label='DEPS')
plt.legend(loc='upper left');
plt.show()