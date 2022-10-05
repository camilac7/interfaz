import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from PIL import Image
import os
from matplotlib.pyplot import figure

def punto_endemico():

    Fi = (l - (((e + d + m) * o * k*m) / (beta * a * (1 - w) ** 2)))/((e + d + m) * (1 + m / (beta * (1 - w))))
    Fs= (l-(e+d+m)*Fi)/m
    Fq= e*Fi/(n+d+m)
    Fr= n*Fq/m
    Fb= ((1-w)*a*Fi)/o
    R0= (((1-w)**2)*a*beta*l)/(k*m*o*(e+d+m))
    S0= l/m
    return Fs,Fi,Fq,Fr,Fb,R0,S0

def rename_parameter():
    l = born_rate # tasa de nacimientos
    m = die_rate # tasa de muerte natural
    d = colera_die_rate# tasa de muerte por colera
    beta = beta_e  # tasa de infectividad por ingerir la bacteria del medio ambiente.
    k = k_rate  # concentración de la bacteria
    e = quarentine_rate  # tasa de infectados que entran a cuarentena
    n = colera_recuperate_rate  # tasa de recuperación de individuos en cuarentena
    a = alpha  # tasa de contribución de la bacteria por un infectado al medio ambiente
    o = sigma  # tasa de muerte de la bacteria en el medio ambiente.
    w = omega  # efecto de la campaña de educación sobre la tasa de contagio por medio ambiente.
    return l,m,d,beta,k,e,n,a,o,w

def fun(A, t):
    S, I, Q, R, B = A
    return [l - (1 - w) * (beta * B * S) / (k + B) - m * S, (1 - w) * (beta * B * S) / (k + B) - (e + d + m) * I,
                e * I - (n + d + m) * Q, n * Q - m * R, (1 - w) * a * I - o * B]

def susceptibles():
    f0 = [s0, i0, q0, r0, b0]
    t = np.linspace(0, 100, 500)
    sol = odeint(fun, f0, t)
    tiempo = t
    #susceptibles = sol[:,0]

    return tiempo, sol


header = st.container()
st.markdown("""---""")
context_proyect = st.container()
st.markdown("""---""")
parameter_model = st.container()
st.markdown("""---""")
solve_model = st.container()
st.markdown("""---""")
simulation = st.container()
with header:
    st.title("Modelo epidemiológico SIQR-B  para la transmisión del cólera")


with context_proyect:
    st.write(r'''Para la transmisión del cólera, se considerará el modelo determinístico compartimental $SIQR-B$ el cual divide la población en cuatro compartimentos $S(t)$, $I(t)$, $Q(t)$, $R(t)$ y considera un compartimento adicional $B(t)$, que representa la concentración de la bacteria Vibrio cholerae en el agua contaminada.''')
    #st.markdown(
       # r'''<div style="text-align: justify">
        #Para la transmisión del cólera, se considerará el modelo determinístico compartimental <div class="latex"> SIQR-B</div> el cual divide la población en cuatro compartimentos  $<div class="latex">S(t)</div>, $I(t)$, $Q(t)$, $R(t)$ y considera un compartimento adicional $B(t)$, que representa la concentración de la bacteria Vibrio cholerae en el agua contaminada. </div>''',
        #unsafe_allow_html=True)

    col_equation, col_diagram = st.columns(2)
    with col_equation:
        st.subheader("Ecuaciones")
        image_eq = os.path.join(os.path.dirname(__file__), 'imagen3.png')
        im_eq = Image.open(image_eq)
        st.image(im_eq, width=400)
    with col_diagram:
        st.subheader("Diagrama de flujo")
        image_dir = os.path.join(os.path.dirname(__file__), 'imagen_2.png')
        print(os.path.dirname(__file__))
        im = Image.open(image_dir)
        st.image(im,width=400)



with parameter_model:
    st.header("Parámetros del modelo")

    with st.form(key='columns_in_form'):
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Parámetros relativos a la población")
            st.markdown(r'''Tasa de nacimientos $\Lambda$''')
            born_rate = st.number_input("born_rate", step=0.001, value= 9.6274e-5, format="%e", label_visibility="collapsed")
            st.markdown(r'''Tasa de muerte natural $\mu$''')
            die_rate = st.number_input("die_rate", step=0.001, format="%e", value=2.537e-5, label_visibility="collapsed")
            st.markdown(r'''Tasa de muerte por cólera $\delta$''')
            colera_die_rate = st.slider('colera_die_rate', 0.0, 1.0, 0.4, label_visibility="collapsed")

            st.markdown(r'''Tasa de recuperación por cólera $\eta$''')
            colera_recuperate_rate = st.slider('colera_recuperate_rate', 0.0, 1.0, 0.3, label_visibility="collapsed")

            st.markdown(r'''Tasa de infectados que entran a cuarentena $\varepsilon$''')
            quarentine_rate = st.slider('quarentine_rate', 0.0, 1.0, 0.3, label_visibility="collapsed")

        with c2:
            st.subheader("Parámetros relativos a la bacteria")
            st.markdown(r'''Concentración necesaria de la bacteria tal que la probabilidad de infetarse es del $50\%$ $k$''')
            k_rate = st.number_input("k_rate", step=0.001, value= 10e6,format="%e", label_visibility="collapsed")
            st.markdown(r'''Tasa de exposición de los humanos con la bacteria $\beta_e$''')
            beta_e = st.slider('beta_e', 0.0, 1.0, 0.75, label_visibility="collapsed")
            st.markdown(r'''Tasa de contribución de los infectados a la bacteria en el medio ambiente $\alpha$''')
            alpha = st.number_input("alpha", value=10, min_value = 0, format="%d", label_visibility="collapsed")
            st.markdown(r'''Tasa de muerte de la bacteria en el medio ambiente $\sigma$''')
            sigma = st.slider('sigma', 0.0, 1.0, 0.23, label_visibility="collapsed")

        st.subheader("Parámetro relativo al efecto de la educación en salud")
        st.markdown(r'''Tasa de reducción del contagio por campañas de educación $\omega$''')
        omega = st.slider('omega', 0.0, 0.9, 0.0, step=0.1, label_visibility="collapsed")

        st.subheader("Condiciones iniciales")
        col_s0, col_i0, col_q0, col_r0, col_b0 = st.columns(5)
        with col_s0:
            st.markdown(r'''$S_0$''')
            s0 = st.number_input("s0", value=10000, min_value=0, format="%d", label_visibility="collapsed")
        with col_i0:
            st.markdown(r'''$I_0$''')
            i0 = st.number_input("i0", value=0, min_value=0, format="%d", label_visibility="collapsed")
        with col_q0:
            st.markdown(r'''$Q_0$''')
            q0 = st.number_input("q0", value=0, min_value=0, format="%d", label_visibility="collapsed")
        with col_r0:
            st.markdown(r'''$R_0$''')
            r0 = st.number_input("r0", value=0, min_value=0, format="%d", label_visibility="collapsed")
        with col_b0:
            st.markdown(r'''$B_0$''')
            b0 = st.number_input("b0", value=1000, min_value=0, format="%d", label_visibility="collapsed")

        submitted = st.form_submit_button('Simular')
with solve_model:
    st.header("Número reproductivo básico")
    r0_col1, r0_col2 = st.columns(2)
    with r0_col1:
        st.latex(r'''R_0 = \dfrac{(1-\omega)^2\alpha\beta_{e}\Lambda}{k\mu\sigma(\varepsilon+\delta+\mu)}''')
    with r0_col2:
        if submitted:
            l, m, d, beta, k, e, n, a, o, w = rename_parameter()
            Fs, Fi, Fq, Fr, Fb, R0, S0 = punto_endemico()
            st.write(" ")
            st.write(" ")
            srt_text = r'''$R_0 = $'''+str(R0)
            st.write(srt_text)
with simulation:
    st.header("Simulación de las dinámicas del modelo")

    if submitted:
        l,m,d,beta,k,e,n,a,o,w = rename_parameter()
        #print(parametros)
        tiempo, estados = susceptibles()
        col1, col2 = st.columns(2)
        with col1:
            fig_susceptibles, ax_susceptibles = plt.subplots(figsize=(12, 5))
            ax_susceptibles.plot(tiempo, estados[:,0], '#F63366')
            ax_susceptibles.set_title('Individuos susceptibles')
            ax_susceptibles.set_ylabel('Susceptibles')
            ax_susceptibles.set_xlabel('Tiempo [días]')
            ax_susceptibles.legend()
            st.pyplot(fig_susceptibles)
        with col2:
            fig_infectados, ax_infectados = plt.subplots(figsize=(12, 5))
            ax_infectados.plot(tiempo, estados[:, 1], '#F63366')
            ax_infectados.set_title('Individuos infectados')
            ax_infectados.set_ylabel('Infectados')
            ax_infectados.set_xlabel('Tiempo [días]')
            ax_infectados.legend()
            st.pyplot(fig_infectados)
        col3, col4 = st.columns(2)
        with col3:
            fig_3, ax_susceptibles = plt.subplots(figsize=(12, 5))
            ax_susceptibles.plot(tiempo, estados[:, 2], '#F63366')
            ax_susceptibles.set_title('Individuos susceptibles')
            ax_susceptibles.set_ylabel('Susceptibles')
            ax_susceptibles.set_xlabel('Tiempo [días]')
            ax_susceptibles.legend()
            st.pyplot(fig_3)
        with col4:
            fig_4, ax_infectados = plt.subplots(figsize=(12, 5))
            ax_infectados.plot(tiempo, estados[:, 3], '#F63366')
            ax_infectados.set_title('Individuos infectados')
            ax_infectados.set_ylabel('Infectados')
            ax_infectados.set_xlabel('Tiempo [días]')
            ax_infectados.legend()
            st.pyplot(fig_4)


