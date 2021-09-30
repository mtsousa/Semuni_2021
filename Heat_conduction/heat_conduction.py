# Autor: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
# Matéria: Transporte de Calor e Massa
#
# Esse código modela e simula o comportamento da condução de calor
# em uma placa com dois obstáculos de tamanhos diferentes. 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as gs
import timeit

def plot_graph_2d(x, y, Temp, tempo, contour=False, subplot_size=(2, 2), N_x=10):

    if contour == True:
        aux1 = np.linspace(0.0, 1.0, 8)
        aux2 = np.linspace(0.0, 1.0, N_x)
        intervals = [aux1, aux2, aux2, aux2]
        
        fig, ax = plt.subplots(2, 2, figsize=(6, 6),
                               constrained_layout=False,
                               sharex=True, sharey=True)
        
        for k, ax in enumerate(ax.flat):
            cb = ax.contourf(x, y, np.transpose(Temp[k]),
                             intervals[k], cmap='viridis')
            ax.grid()
            ax.set_title(f't =  {tempo[k]}s')
        
        # Adiciona colorbar
        fig.subplots_adjust(right=0.8)
        cb_ax = fig.add_axes([0.84, 0.105, 0.03, 0.775])
        cbar = fig.colorbar(cb, cax=cb_ax)
        cbar.set_ticks(np.arange(0, 1.1, 0.1))
        
        # Nome dos eixos
        fig.suptitle('Gráficos de Contorno', fontsize=14)
        fig.supxlabel('x', fontsize=12)
        fig.supylabel('y', fontsize=12)
        plt.savefig('Heat_conduction/data/plot2d.png', format='png', dpi=1600)
        plt.show()
    
    else:
        fig, ax = plt.subplots(subplot_size[0], subplot_size[1], figsize=(7, 7),
                               subplot_kw={"projection": "3d"},
                               sharex=False, sharey=False,
                               constrained_layout=True)
        fig.suptitle('Gráficos Tridimensionais da Temperatura', fontsize=14)
    
        for k, ax in enumerate(ax.flat):
            X,Y = np.meshgrid(x,y)

            # Nomes dos eixos
            ax.set_xlabel('x', fontsize = 14)
            ax.set_ylabel('y', fontsize = 14)
            ax.set_zlabel('T', fontsize = 14)
            ax.set_title(f't =  {tempo[k]}s')

            aux_Temp = np.copy(np.transpose(Temp[k]))
            vec_Temp = np.resize(aux_Temp, (1, len(aux_Temp)*len(aux_Temp[0])))
            X_ = np.resize(X, (1, len(X)*len(X[0])))
            Y_ = np.resize(Y, (1, len(Y)*len(Y[0])))

            # https://stackoverflow.com/questions/13730468/from-nd-to-1d-arrays
            new_x = X_.ravel()
            new_y = Y_.ravel()
            vec_Temp = vec_Temp.ravel()
            
            ax.dist = 11
            ax.plot_trisurf(new_x, new_y, vec_Temp,
                            cmap='viridis', edgecolor='none')
        plt.show()

def calculate_temperature_2d(x, y, Lx, Ly, Nx, Ny, tempos, Temp,
                             min_int=(0,0), max_int=(0,0), obstacle=False, alfa=1):
    
    dx = Lx/Nx
    dy = Ly/Ny
    dt = dx**2/4 if dx < dy else dy**2/4
    t = 0.0
    
    start = timeit.default_timer()
    
    Temp_new = np.copy(Temp.round(3))
    Temp_resp = []
    Temp_resp.append(Temp.round(3))

    for m in tempos:
        while t < m-0.01*dt:
            # Calcula a temperatura dos pontos na placa
            Temp_new[1:N_x, 1:N_y] = Temp[1:N_x, 1:N_y]\
            + alfa*(dt/(dx**2.0))*(Temp[2:N_x+1, 1:N_y]\
                                   -2.0*Temp[1:N_x, 1:N_y] + Temp[0:N_x-1, 1:N_y])\
            + alfa*(dt/(dy**2.0))*(Temp[1:N_x, 2:N_y+1]\
                                   -2.0*Temp[1:N_x, 1:N_y] + Temp[1:N_x, 0:N_y-1])
            
            # Reconfigura os pontos do obstáculo na placa
            if obstacle == True:
                Temp_new[min_int[0]:max_int[0]+1, min_int[0]:max_int[0]+1] = 1.0
                Temp_new[min_int[1]:max_int[1]+1, min_int[1]:max_int[1]+1] = 1.0
    
            Temp = np.copy(Temp_new.round(3))
            t += dt
            
        Temp_resp.append(Temp)

    end = timeit.default_timer()
    print(f'O custo computacional foi de {end-start:6.4f} s.')
    
    return Temp_resp

L_x = 1
L_y = 1
N_x = 20
N_y = 20
tempos = [0.008, 0.020, 3]

x = np.linspace(0.0, L_x, N_x+1)
y = np.linspace(0.0, L_y, N_y+1)

Temp = np.zeros((N_x+1,N_y+1),float)

# Define os limites do primeio objeto
min1 = 7*2
max1 = 8*2

# Define a temperatura do primeiro objeto
Temp[min1:max1+1, min1:max1+1] = 1.0

# Define os limites do segundo objeto
min2 = 2*2
max2 = 4*2

# Define os limites do segundo objeto
Temp[min2:max2+1, min2:max2+1] = 1.0

Temp_resp = calculate_temperature_2d(x, y, L_x, L_y, N_x, N_y, tempos,
                                     Temp, min_int=(min1, min2), max_int=(max1, max2), obstacle=True)

tempos.insert(0, 0.00)
# Gráficos de contorno
plot_graph_2d(x, y, Temp_resp, tempos, contour=True)

# Gráficos 3D
plot_graph_2d(x, y, Temp_resp, tempos)