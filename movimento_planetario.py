import numpy as np
import matplotlib.pyplot as plt

GM = 4 * (np.pi ** 2)


def funcTerra(pos, t):
    x, vx, y, vy = pos
    r = (x ** 2 + y ** 2) ** (3 / 2)
    ax = -x * GM / r
    ay = -y * GM / r
    return np.array([vx, ax, vy, ay])


def funcMerc(pos, t):
    x, vx, y, vy = pos
    r = (x ** 2 + y ** 2) ** (3 / 2)
    ax = -x * GM / r
    ay = -y * GM / r
    return np.array([vx, ax, vy, ay])


def MarJup(pos, t):
    xm, vxm, ym, vym, xj, vxj, yj, vyj = pos
    rm = (xm ** 2 + ym ** 2) ** (3 / 2)
    rj = (xj ** 2 + yj ** 2) ** (3 / 2)  # distancia de júpiter ao sol ^3
    dist = ((xm - xj) ** 2 + (ym - yj) ** 2) ** 3 / 2  # distancia entre mercúrio e júpiter ^ 3
    mj = 9.5 * 10 ** (-4)
    mm = 3.3 * 10 ** (-7)

    result = np.zeros_like(pos)
    result[0] = vxm
    result[1] = -((xm * GM) / rm + (GM * mj * (xm - xj)) / dist)
    result[2] = vym
    result[3] = -((ym * GM) / rm + (GM * mj * (ym - yj)) / dist)
    result[4] = vxj
    result[5] = -((xj * GM) / rj + (GM * mm * (xj - xm)) / dist)
    result[6] = vyj
    result[7] = -((yj * GM) / rj + (GM * mm * (yj - ym)) / dist)

    return result


def verlet(r, t, dt, f):
    k1 = f(r, t)
    r_next = r + dt * k1 + 0.5 * dt ** 2 * f(r + dt * k1, t + dt)
    k2 = f(r_next, t + dt)
    r += dt * (k1 + k2) / 2
    return r


def runge_kutta4(r, t, dt, f):
    k1 = f(r, t)
    k2 = f(r + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(r + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(r + dt * k3, t + dt)
    r += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return r


def euler(r, t, dt, f):
    k1 = f(r, t)
    r += dt * k1
    return r


def midpoint(r, t, dt, f):
    k1 = f(r, t)
    k2 = f(r + 0.5 * dt * k1, t + 0.5 * dt)
    r += dt * k2
    return r


def taylor2ordem(r, t, dt, f):
    k1 = f(r, t)
    k2 = f(r + dt * k1, t + dt)
    r += dt * (k1 + 0.5 * dt * k2)
    return r


def ex1(N, p):
    m = 5.974 * (10 ** 24)
    rt = 1
    vi = np.sqrt(GM / rt)
    ti = 0
    tf = 1
    dt = tf / N

    t = np.zeros(N)
    f = np.zeros((N, 4))

    t[0] = ti
    f[0] = [1, 0, 0, vi]

    method_dict = {1: euler, 2: taylor2ordem, 3: runge_kutta4, 4: midpoint, 5: verlet}
    integration_method = method_dict.get(p)
    for i in range(1, N):
        f[i] = integration_method(f[i - 1], t[i - 1], dt, funcTerra)
        t[i] = t[i - 1] + dt

    resx = []
    resy = []
    resvx = []
    resvy = []

    for i in range(N):
        resx.append(f[i][0])
        resvx.append(f[i][1])
        resy.append(f[i][2])
        resvy.append(f[i][3])

    Et = np.zeros(N)
    for i in range(0, N):
        d = (resx[i] ** 2 + resy[i] ** 2) ** (1 / 2)
        v = (resvx[i] ** 2 + resvy[i] ** 2) ** (1 / 2)
        Et[i] = - (GM * m / d) + (m / 2) * v ** 2

    varE = abs((Et[0] - Et[-1]) * 100 / Et[0])
    print("Variação da Energia total relativa do sistema: ", varE, "%")

    plt.plot(resx, resy, color='blue')
    plt.scatter(0, 0, color='red', s=200)
    plt.scatter(resx[0], resy[0], color='blue', s=50)
    plt.show()


def ex2(N):
    a = 0.39
    e = 0.206
    vi = np.sqrt((GM * (1 - e)) / (a * (1 + e)))
    rimerc = a * (1 + e)
    ti = 0
    tf = 89 / 365
    # N = 10000
    dt = tf / N

    t = np.zeros(N)
    f = np.zeros((N, 4))

    t[0] = ti
    f[0] = [rimerc, 0, 0, vi]

    # Integração no tempo
    for i in range(1, N):
        f[i] = runge_kutta4(f[i - 1], t[i - 1], dt, funcMerc)
        t[i] = t[i - 1] + dt

    resx = []
    resy = []
    resvx = []
    resvy = []

    # Cálculo da velocidade do afélio e perifélio
    for i in range(N):
        resx.append(f[i][0])
        resvx.append(f[i][1])
        resy.append(f[i][2])
        resvy.append(f[i][3])
    r = np.sqrt(np.array(resx) ** 2 + np.array(resy) ** 2)
    aphelion_index = np.argmax(r)  # index of aphelion
    perihelion_index = np.argmin(r)  # index of perihelion
    velocity_at_aphelion = np.sqrt(resvx[aphelion_index] ** 2 + resvy[aphelion_index] ** 2)  # velocity at aphelion
    velocity_at_perihelion = np.sqrt(
        resvx[perihelion_index] ** 2 + resvy[perihelion_index] ** 2)  # velocity at perihelion

    print("Velocidade no afélio:", velocity_at_aphelion)
    print("Velocidade no perifélio:", velocity_at_perihelion)

    plt.plot(resx, resy, color='blue')
    plt.scatter(0, 0, color='blue', s=200)
    plt.scatter(resx[-1000], resy[-1000], color='brown', s=40)
    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    plt.show()


def ex3(N):
    # Condições Iniciais
    initial_radius_mars = 1.52
    initial_velocity_mars = np.sqrt(GM / initial_radius_mars)
    initial_radius_jupiter = 5.2
    initial_velocity_jupiter = np.sqrt(GM / initial_radius_jupiter)

    # Variáveis de Tempo
    start_time = 0
    end_time = 11.862  # Em anos
    time_step = end_time * (1 / N)

    # Armazenar os resultados
    time_array = np.zeros(N)
    data = np.zeros((N, 8))

    # Valores Iniciais
    time_array[0] = start_time
    data[0] = [initial_radius_mars, 0, 0, initial_velocity_mars, initial_radius_jupiter, 0, 0, initial_velocity_jupiter]

    # Integração no Tempo
    for i in range(1, N):
        data[i] = runge_kutta4(data[i - 1], time_array[i - 1], time_step, MarJup)
        time_array[i] = time_array[i - 1] + time_step

    x_mars = []
    y_mars = []
    vx_mars = []
    vy_mars = []

    x_jupiter = []
    y_jupiter = []
    vx_jupiter = []
    vy_jupiter = []

    for i in range(N):
        x_mars.append(data[i][0])
        vx_mars.append(data[i][1])
        y_mars.append(data[i][2])
        vy_mars.append(data[i][3])

        x_jupiter.append(data[i][4])
        vx_jupiter.append(data[i][5])
        y_jupiter.append(data[i][6])
        vy_jupiter.append(data[i][7])

    plt.plot(x_mars, y_mars, color='red')
    plt.scatter(0, 0, color='blue', s=200)
    plt.scatter(x_mars[0], y_mars[0], color='red', s=80)
    plt.scatter(x_jupiter[0], y_jupiter[0], color='black', s=100)
    plt.plot(x_jupiter, y_jupiter, color='black')
    plt.show()


while True:
    orbital_positions = {1: "Orbita da Terra", 2: "Orbita de Mercúrio", 3: "Orbita de Juíter e Marte"}
    edo_methods = {1: "Euler", 2: "Taylor de 2º Ordem", 3: "Runge-Kutta de 4º Ordem", 4: "Midpoint", 5: "Verlet"}

    while True:
        opçao = int(input("Que órbita quer analizar? (1-3): "))
        if opçao not in orbital_positions:
            print("Input inválido")
            continue
        print("Selecionou:", orbital_positions[opçao])
        if opçao == 1:
            p = int(input("Que método de EDO quer usar? (1-5): "))
            if p not in edo_methods:
                print("Input inválido")
                continue
            print("Selecionou:", edo_methods[p])
            ex1(10000, p)
        if opçao == 2:
            ex2(10000)
        if opçao == 3:
            ex3(10000)
        else:
            print("Input inválido")
