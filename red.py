"""
🧠 PLAYGROUND DE REDES NEURONALES — Módulo interactivo estilo TensorFlow Playground
===================================================================================
Integrado al simulador. Permite construir, entrenar y visualizar una red neuronal:

  1. Arquitectura interactiva: añadir/eliminar capas y neuronas con botones.
  2. Parámetros editables: pesos y bias a mano, activación, tasa de aprendizaje y épocas.
  3. Visualización de qué aprende y qué aporta cada neurona a la salida (activación + ablación).
  4. Evolución del error (función de coste) y de la precisión durante el entrenamiento.
  5. Superficie 3D de la función de coste sobre DOS parámetros seleccionables, con la
     trayectoria del optimizador y vista cenital (contorno).
  6. Datasets integrados (AND, OR, XOR, blobs, círculos, anillo, lunas, espirales, tablero)
     + datasets personalizados (CSV, datos pegados o generador ajustable).

El motor de la red (forward/backprop en NumPy) es propio, de modo que todo parámetro
puede inspeccionarse y modificarse. Ejecutar con:

    streamlit run red.py
"""

from __future__ import annotations

import csv
import hashlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_blobs, make_circles, make_moons

# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURACIÓN DE PÁGINA, PALETA Y ESTILO
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Playground Red Neuronal", page_icon="🧠", layout="wide")

C0, C1 = "#2563eb", "#f59e0b"          # colores de las dos clases
C_PERDIDA, C_PREC, C_PESO = "#e11d48", "#059669", "#4f46e5"
CS_FRONTERA = [(0.0, "#1e3a8a"), (0.25, "#60a5fa"), (0.5, "#ffffff"),
               (0.75, "#fbbf24"), (1.0, "#92400e")]

PLANTILLA = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#334155"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=45, r=20, t=52, b=42),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    hoverlabel=dict(bgcolor="white", font_size=12),
)

st.markdown(
    """
    <style>
    .pg-title { font-size:2.05rem; font-weight:800; letter-spacing:-.5px; margin:0;
        background:linear-gradient(90deg,#4f46e5 0%,#0891b2 55%,#059669 100%);
        -webkit-background-clip:text; background-clip:text; color:transparent; }
    .pg-sub { color:#64748b; font-size:.92rem; margin:.1rem 0 .7rem 0; }
    .pg-card { background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:10px 16px;
        box-shadow:0 1px 3px rgba(15,23,42,.06); }
    .pg-hint { color:#64748b; font-size:.82rem; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid #e2e8f0; border-radius:14px;
        padding:10px 16px; box-shadow:0 1px 3px rgba(15,23,42,.06); }
    div[data-testid="stMetricLabel"] > div { color:#64748b; font-size:.70rem; font-weight:700;
        text-transform:uppercase; letter-spacing:.07em; }
    div[data-testid="stMetricValue"] { font-weight:700; color:#0f172a; }
    section[data-testid="stSidebar"] { background:linear-gradient(180deg,#f8fafc 0%,#eef2ff 100%); }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color:#312e81; }
    .stTabs [data-baseweb="tab-list"] { gap:.35rem; }
    .stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:.45rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def style(fig: go.Figure, height: int = 420, titulo: str | None = None,
          xaxis: str | None = None, yaxis: str | None = None) -> go.Figure:
    """Aplica el estilo común (limpio, moderno y consistente) a todas las figuras."""
    fig.update_layout(**PLANTILLA, height=height)
    if titulo:
        fig.update_layout(title=dict(text=titulo, x=0.01, xanchor="left",
                                     font=dict(size=15, color="#0f172a")))
    fig.update_xaxes(gridcolor="#eef2f7", zerolinecolor="#dbe3ee")
    fig.update_yaxes(gridcolor="#eef2f7", zerolinecolor="#dbe3ee")
    if xaxis:
        fig.update_xaxes(title_text=xaxis)
    if yaxis:
        fig.update_yaxes(title_text=yaxis)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 1. MOTOR DE LA RED NEURONAL (NumPy: forward + backprop propios)
# ══════════════════════════════════════════════════════════════════════════════
ACTIVACIONES = {"Tanh": "tanh", "ReLU": "relu", "Sigmoide": "sigmoid",
                "LeakyReLU": "leaky_relu", "Lineal": "linear"}
INICIALIZACIONES = {"Xavier (Glorot)": "xavier", "He (Kaiming)": "he", "Uniforme (±0.5)": "uniform"}
FUNCIONES_COSTE = {"Entropía cruzada (log-loss)": "bce", "Error cuadrático medio (MSE)": "mse"}


def act_directo(nombre: str, z: np.ndarray) -> np.ndarray:
    if nombre == "tanh":
        return np.tanh(z)
    if nombre == "relu":
        return np.maximum(0.0, z)
    if nombre == "leaky_relu":
        return np.where(z > 0, z, 0.01 * z)
    if nombre == "sigmoid":
        return 0.5 * (1.0 + np.tanh(0.5 * z))          # sigmoid estable
    return z                                            # lineal


def act_derivada(nombre: str, z: np.ndarray, a: np.ndarray) -> np.ndarray:
    if nombre == "tanh":
        return 1.0 - a ** 2
    if nombre == "relu":
        return (z > 0).astype(float)
    if nombre == "leaky_relu":
        return np.where(z > 0, 1.0, 0.01)
    if nombre == "sigmoid":
        return a * (1.0 - a)
    return np.ones_like(z)


class MLP:
    """Red neuronal densa para clasificación binaria (salida sigmoid)."""

    def __init__(self, sizes, activacion="tanh", inicializacion="xavier",
                 semilla=42, coste="bce", activaciones=None):
        self.sizes = list(sizes)
        n_ocultas = max(len(self.sizes) - 2, 0)
        if activaciones is None:
            activaciones = [activacion] * n_ocultas
        self.activaciones = [str(a) for a in list(activaciones)[:n_ocultas]]
        while len(self.activaciones) < n_ocultas:      # por si viene incompleta
            self.activaciones.append("tanh")
        self.inicializacion = inicializacion
        self.coste = coste
        self.semilla = semilla
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.velW: list[np.ndarray] = []
        self.velb: list[np.ndarray] = []
        self.rng = np.random.default_rng(semilla + 9973)
        self.reset(semilla)

    def act_de(self, l: int) -> str:
        """Activación de la capa l (la salida siempre es sigmoide)."""
        return "sigmoid" if l == len(self.W) - 1 else self.activaciones[l]


    # ---------- inicialización ----------
    def reset(self, semilla: int | None = None):
        if semilla is not None:
            self.semilla = int(semilla)
        rng = np.random.default_rng(self.semilla)
        self.rng = np.random.default_rng(self.semilla + 9973)
        self.W, self.b = [], []
        for l in range(len(self.sizes) - 1):
            nin, nout = self.sizes[l], self.sizes[l + 1]
            if self.inicializacion == "xavier":
                W = rng.normal(0.0, np.sqrt(2.0 / (nin + nout)), (nin, nout))
            elif self.inicializacion == "he":
                W = rng.normal(0.0, np.sqrt(2.0 / nin), (nin, nout))
            else:
                W = rng.uniform(-0.5, 0.5, (nin, nout))
            self.W.append(W)
            self.b.append(np.zeros(nout))
        self.velW = [np.zeros_like(w) for w in self.W]
        self.velb = [np.zeros_like(bb) for bb in self.b]

    # ---------- propagación ----------
    def forward(self, X, devolver_cache: bool = False):
        a = np.asarray(X, float)
        acts, zs = [a], []
        L = len(self.W)
        for l in range(L):
            z = a @ self.W[l] + self.b[l]
            a = act_directo(self.act_de(l), z)
            zs.append(z)
            acts.append(a)
        return (acts, zs) if devolver_cache else a

    def forward_enmascarado(self, X, capa: int, neurona: int):
        """Propagación con una neurona concreta anulada (para medir su aporte)."""
        a = np.asarray(X, float)
        L = len(self.W)
        for l in range(L):
            z = a @ self.W[l] + self.b[l]
            a = act_directo(self.act_de(l), z)
            if l == capa:
                a[:, neurona] = 0.0
        return a

    def predict_proba(self, X) -> np.ndarray:
        return self.forward(X).ravel()

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    # ---------- coste y precisión ----------
    def loss_from_p(self, p, y) -> float:
        eps = 1e-9
        p = np.clip(np.asarray(p, float), eps, 1 - eps)
        y = np.asarray(y, float)
        if self.coste == "bce":
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return float(np.mean((p - y) ** 2))

    def accuracy(self, p, y) -> float:
        return float(np.mean((np.asarray(p, float) >= 0.5) == np.asarray(y, float)))

    # ---------- retropropagación ----------
    def _gradientes(self, acts, zs, y):
        n = len(y)
        p = acts[-1]
        ycol = np.asarray(y, float).reshape(-1, 1)
        if self.coste == "bce":
            delta = (p - ycol) / n                       # sigmoid + entropía cruzada
        else:
            delta = 2.0 * (p - ycol) * p * (1 - p) / n   # sigmoid + MSE
        L = len(self.W)
        gW: list = [None] * L
        gb: list = [None] * L
        for l in range(L - 1, -1, -1):
            gW[l] = acts[l].T @ delta
            gb[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ self.W[l].T) * act_derivada(self.activaciones[l - 1],
                                                             zs[l - 1], acts[l])
        return gW, gb

    def _aplicar(self, gW, gb, lr: float, mu: float = 0.0):
        """Descenso del gradiente con momento: v ← μ·v − η·∂L/∂θ ; θ ← θ + v."""
        for l in range(len(self.W)):
            if mu:
                self.velW[l] = mu * self.velW[l] - lr * gW[l]
                self.velb[l] = mu * self.velb[l] - lr * gb[l]
                self.W[l] += self.velW[l]
                self.b[l] += self.velb[l]
            else:
                self.W[l] -= lr * gW[l]
                self.b[l] -= lr * gb[l]

    def train_epoch(self, X, y, lr: float, lote: int | None = None, mu: float = 0.0):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n = len(X)
        if not lote or lote >= n:                        # batch completo
            acts, zs = self.forward(X, True)
            gW, gb = self._gradientes(acts, zs, y)
            self._aplicar(gW, gb, lr, mu)
        else:                                            # mini-batch / estocástico
            idx_all = self.rng.permutation(n)
            for start in range(0, n, lote):
                bs = idx_all[start:start + lote]
                acts, zs = self.forward(X[bs], True)
                gW, gb = self._gradientes(acts, zs, y[bs])
                self._aplicar(gW, gb, lr, mu)

    # ---------- acceso a parámetros (editar / superficie 3D) ----------
    @property
    def n_params(self) -> int:
        return int(sum(w.size for w in self.W) + sum(bb.size for bb in self.b))

    def flatten(self) -> np.ndarray:
        return np.concatenate([w.ravel() for w in self.W] + [bb.ravel() for bb in self.b])

    def set_flat(self, v: np.ndarray):
        v = np.asarray(v, float)
        k = 0
        for l in range(len(self.W)):
            s = self.W[l].size
            self.W[l] = v[k:k + s].reshape(self.W[l].shape).copy()
            k += s
        for l in range(len(self.b)):
            s = self.b[l].size
            self.b[l] = v[k:k + s].copy()
            k += s

    # ---------- nombres legibles ----------
    def layer_name(self, l: int) -> str:
        return "Capa de salida" if l == len(self.W) - 1 else f"Capa oculta {l + 1}"

    def in_names(self, l: int) -> list[str]:
        if l == 0:
            return [f"x{i + 1}" for i in range(self.sizes[0])]
        return [f"H{l}·n{i + 1}" for i in range(self.sizes[l])]

    def out_names(self, l: int) -> list[str]:
        if l == len(self.W) - 1:
            return ["ŷ"]
        return [f"H{l + 1}·n{j + 1}" for j in range(self.sizes[l + 1])]

    def neuron_name(self, l: int, j: int) -> str:
        return "ŷ" if l == len(self.W) - 1 else f"H{l + 1}·n{j + 1}"

    def param_labels(self) -> list[str]:
        """Etiquetas en el MISMO orden que flatten()."""
        labels = []
        for l in range(len(self.W)):
            ins, outs = self.in_names(l), self.out_names(l)
            for i in range(len(ins)):
                for j in range(len(outs)):
                    labels.append(f"{self.layer_name(l)} · {ins[i]} → {outs[j]}")
            for j in range(len(outs)):
                labels.append(f"{self.layer_name(l)} · bias → {outs[j]}")
        return labels


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASETS (integrados y personalizados)
# ══════════════════════════════════════════════════════════════════════════════
DATASETS = [
    "Compuerta AND", "Compuerta OR", "Compuerta XOR", "XOR con ruido",
    "Blobs lineales (2 clases)", "Dos gaussianas separadas",
    "Círculos concéntricos", "Círculo interior vs. anillo exterior",
    "Medias lunas", "Espirales", "Tablero de cuadrantes",
]
PERSONALIZADOS = ["📁 Subir CSV / TXT", "✏️ Escribir / pegar datos", "🎛 Nube de puntos personalizada"]


@st.cache_data(show_spinner=False)
def dataset_integrado(nombre: str, n: int, ruido: float, semilla: int = 42):
    """Genera uno de los datasets integrados. Devuelve (X, y) en float."""
    r = np.random.default_rng(semilla)

    def esquinas(tabla):
        C = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        yv = np.array(tabla, float)
        if ruido <= 0:
            return C.copy(), yv.copy()
        idx = r.integers(0, 4, max(n, 4))
        return C[idx] + r.normal(0, ruido * 0.4, (len(idx), 2)), yv[idx]

    if nombre == "Compuerta AND":
        X, y = esquinas([0, 0, 0, 1])
    elif nombre == "Compuerta OR":
        X, y = esquinas([0, 1, 1, 1])
    elif nombre == "Compuerta XOR":
        X, y = esquinas([0, 1, 1, 0])
    elif nombre == "XOR con ruido":
        X, y = esquinas([0, 1, 1, 0])                       # usa el ruido como mínimo 0.35
    elif nombre == "Blobs lineales (2 clases)":
        X, y = make_blobs(n_samples=n, centers=[[-1.5, -1.5], [1.5, 1.5]],
                          cluster_std=0.35 + 1.6 * ruido, random_state=semilla)
    elif nombre == "Dos gaussianas separadas":
        X, y = make_blobs(n_samples=n, centers=[[-2.2, 0.0], [2.2, 0.0]],
                          cluster_std=0.55 + 0.8 * ruido, random_state=semilla)
    elif nombre == "Círculos concéntricos":
        X, y = make_circles(n_samples=n, noise=ruido * 0.5, factor=0.5, random_state=semilla)
    elif nombre == "Medias lunas":
        X, y = make_moons(n_samples=n, noise=ruido * 0.6, random_state=semilla)
    elif nombre == "Círculo interior vs. anillo exterior":
        mitad = n // 2
        t1, t2 = r.uniform(0, 2 * np.pi, mitad), r.uniform(0, 2 * np.pi, n - mitad)
        rad1 = np.sqrt(r.uniform(0.0, 0.36, mitad))          # disco interior
        rad2 = r.uniform(0.72, 1.05, n - mitad)              # anillo exterior
        X = np.vstack([np.c_[rad1 * np.cos(t1), rad1 * np.sin(t1)],
                       np.c_[rad2 * np.cos(t2), rad2 * np.sin(t2)]])
        X += r.normal(0, max(ruido, 0.02) * 0.12, X.shape)
        y = np.r_[np.zeros(mitad), np.ones(n - mitad)]
        idx = r.permutation(len(y))
        X, y = X[idx], y[idx]
    elif nombre == "Espirales":
        mitad = n // 2
        t = np.linspace(0.2, 3.4 * np.pi, mitad)
        A = np.c_[t * np.cos(t), t * np.sin(t)]
        B = np.c_[t * np.cos(t + np.pi), t * np.sin(t + np.pi)]
        X = np.vstack([A, B])
        X /= np.abs(X).max()
        X += r.normal(0, max(ruido, 0.05) * 0.22, X.shape)
        y = np.r_[np.zeros(mitad), np.ones(mitad)]
    elif nombre == "Tablero de cuadrantes":
        celdas = 3
        X = r.uniform(-1, 1, (n, 2))
        y = (np.floor((X[:, 0] + 1) * celdas / 2) + np.floor((X[:, 1] + 1) * celdas / 2)) % 2
        X += r.normal(0, ruido * 0.05, X.shape)
    else:
        raise ValueError(f"Dataset desconocido: {nombre}")

    # XOR con ruido obliga a una separación mínima ruidosa
    if nombre == "XOR con ruido" and ruido < 0.35:
        idx = r.integers(0, 4, max(n, 4))
        C = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        yv = np.array([0, 1, 1, 0], float)
        X, y = C[idx] + r.normal(0, 0.35 * 0.4, (len(idx), 2)), yv[idx]

    return np.asarray(X, float), np.asarray(y, float).ravel()


def dataset_nube(n, centro_a, centro_b, std, semilla=7):
    r = np.random.default_rng(semilla)
    mitad = n // 2
    A = r.normal(0, std, (mitad, 2)) + np.asarray(centro_a, float)
    B = r.normal(0, std, (n - mitad, 2)) + np.asarray(centro_b, float)
    X = np.vstack([A, B])
    y = np.r_[np.zeros(mitad), np.ones(n - mitad)]
    idx = r.permutation(n)
    return X[idx], y[idx]


def leer_tabla(df: pd.DataFrame):
    """Interpreta una tabla x1, x2, clase y devuelve (X, y, nota)."""
    if df.shape[1] < 3:
        raise ValueError("Se necesitan 3 columnas: **x1, x2 y clase** (la última es la etiqueta).")
    extra = df.shape[1] - 3
    feats = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce")
    yraw = df.iloc[:, -1]
    ok = feats.notna().all(axis=1) & yraw.notna()
    feats, yraw = feats[ok], yraw[ok]
    if len(feats) < 4:
        raise ValueError("Faltan datos válidos: hacen falta al menos 4 filas numéricas.")
    clases = list(pd.unique(yraw))
    if len(clases) != 2:
        raise ValueError(f"El problema debe tener 2 clases; se encontraron {len(clases)}: {clases}.")
    mapeo = {c: i for i, c in enumerate(clases)}
    X = feats.to_numpy(float)
    y = yraw.map(mapeo).to_numpy(float)
    nota = f"{len(X)} muestras · {clases[0]} → clase 0, {clases[1]} → clase 1."
    if extra:
        nota += f" Se ignoraron {extra} columna(s) intermedia(s)."
    return X, y, nota


def _es_numero(v) -> bool:
    try:
        float(str(v).strip())
        return True
    except (TypeError, ValueError):
        return False


def _detectar_separador(texto: str) -> str:
    """Detecta coma, punto y coma, tabulador o espacios sin confundir filas con encabezados."""
    muestra = "\n".join(texto.splitlines()[:25])
    try:
        return csv.Sniffer().sniff(muestra, delimiters=",;\t ").delimiter
    except Exception:
        for sep in [",", ";", "\t"]:
            if sep in muestra:
                return sep
        return ","


def dataframe_desde_texto(texto: str) -> pd.DataFrame:
    """Interpreta un CSV pegado a mano, con o sin fila de encabezado y con cualquier separador."""
    texto = (texto or "").strip()
    if not texto:
        raise ValueError("No hay datos: escribe o pega filas con formato `x1, x2, clase`.")
    sep = _detectar_separador(texto)

    def _leer(s):
        return pd.read_csv(io.StringIO(texto), sep=s, engine="python",
                           header=None, skip_blank_lines=True, skipinitialspace=True)

    try:
        df = _leer(sep)
        # Si el separador elegido no parte la tabla, probamos los demás.
        if df.shape[1] < 2:
            for alt in [",", ";", "\t", r"\s+"]:
                if alt == sep:
                    continue
                candidato = _leer(alt)
                if candidato.shape[1] > df.shape[1]:
                    df = candidato
                    sep = alt
    except Exception as e:
        raise ValueError(f"No se pudo interpretar el formato del texto: {e}")
    if df.empty:
        raise ValueError("La tabla está vacía.")
    # Si la primera fila es completamente no numérica, es un encabezado (p. ej. x1,x2,clase)
    primera = [v for v in df.iloc[0].tolist() if str(v).strip() != ""]
    if primera and not any(_es_numero(v) for v in primera):
        df = df.iloc[1:].reset_index(drop=True)
    if df.empty:
        raise ValueError("No quedan filas de datos válidas (¿el texto sólo tenía encabezado "
                         "o no reconoce el separador de columnas?).")
    return df


@st.cache_data(show_spinner=False, max_entries=5)
def parse_upload(raw: bytes):
    try:
        texto = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = raw.decode("latin-1", errors="replace")
    return leer_tabla(dataframe_desde_texto(texto))


# ══════════════════════════════════════════════════════════════════════════════
# 3. ESTADO Y CALLBACKS DE LA ARQUITECTURA
# ══════════════════════════════════════════════════════════════════════════════
def _init_estado():
    s = st.session_state
    s.setdefault("arch", [4, 2])
    s.setdefault("hist", None)
    s.setdefault("modelo", None)
    s.setdefault("modelo_cfg", None)
    s.setdefault("editado", False)
    s.setdefault("superficie", None)


def _add_capa():
    st.session_state.arch.append(int(st.session_state.get("nueva_capa_sz", 4)))


def _mas_neurona(i):
    st.session_state.arch[i] = min(16, st.session_state.arch[i] + 1)


def _menos_neurona(i):
    st.session_state.arch[i] = max(1, st.session_state.arch[i] - 1)


def _borrar_capa(i):
    del st.session_state.arch[i]


_init_estado()

# ══════════════════════════════════════════════════════════════════════════════
# 4. BARRA LATERAL: datos, arquitectura y hiperparámetros
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🧠 Configuración")

# ---------- 1. Datos ----------
st.sidebar.markdown("#### 1️⃣ Datos y problema")
dataset = st.sidebar.selectbox("Problema a aprender", DATASETS + PERSONALIZADOS,
                               index=2, help="También puedes subir tus propios datos.")

X = y = None
nota_datos = None
error_datos = None

if dataset in DATASETS:
    n_muestras = st.sidebar.slider("Número de muestras", 40, 5000, 400, step=20)
    ruido = st.sidebar.slider("Ruido / dispersión", 0.0, 1.0, 0.15, step=0.05)
    X, y = dataset_integrado(dataset, n_muestras, ruido)
    nota_datos = f"{len(X)} muestras generadas · ruido = {ruido:.2f}."
elif dataset.startswith("📁"):
    subida = st.sidebar.file_uploader("Archivo CSV / TXT", type=["csv", "txt", "data"],
                                      help="Una fila por muestra: `x1, x2, clase`. "
                                           "Admite coma, punto y coma o tabulador; "
                                           "con o sin fila de encabezado.")
    if subida is not None:
        try:
            X, y, nota_datos = parse_upload(subida.getvalue())
        except Exception as e:
            error_datos = f"No se pudo leer el archivo: {e}"
    else:
        error_datos = ("Sube un archivo **CSV/TXT** con el formato `x1, x2, clase` "
                       "(por ejemplo: `0,0,0` / `0,1,0` / `1,0,0` / `1,1,1`).")
elif dataset.startswith("✏️"):
    texto_datos = st.sidebar.text_area(
        "Datos (una muestra por línea)", value="0,0,0\n0,1,0\n1,0,0\n1,1,1", height=170,
        help="Formato `x1, x2, clase`. La última columna define la clase; "
             "acepta números o etiquetas (positivo/negativo).")
    try:
        X, y, nota_datos = leer_tabla(dataframe_desde_texto(texto_datos))
    except Exception as e:
        error_datos = str(e)
else:  # nube personalizada
    n_nube = st.sidebar.slider("Número de puntos", 50, 5000, 400, step=50)
    sep_nube = st.sidebar.slider("Separación entre clases", 0.5, 8.0, 3.0, step=0.1)
    std_nube = st.sidebar.slider("Dispersión (σ)", 0.1, 3.0, 0.9, step=0.1)
    X, y = dataset_nube(n_nube, (-sep_nube / 2, 0.0), (sep_nube / 2, 0.0), std_nube)
    nota_datos = (f"{len(X)} puntos gaussianos · separación = {sep_nube:.1f} · "
                  f"σ = {std_nube:.2f}.")

data_sig = ""
if X is not None:
    data_sig = (f"{dataset}|{len(X)}|"
                + hashlib.md5(np.ascontiguousarray(X).tobytes()
                              + np.ascontiguousarray(y).tobytes()).hexdigest()[:10])

# ---------- 2. Arquitectura ----------
st.sidebar.markdown("#### 2️⃣ Arquitectura de la red")
_arch = st.session_state.arch
resumen = "Entrada (x1,x2) → " + (" → ".join(f"H{i+1}: {n}n" for i, n in enumerate(_arch)) + " → " if _arch else "sin capas ocultas → ") + "salida (ŷ)"
st.sidebar.markdown(f"`{resumen}`")

for i in range(len(_arch)):
    fila = st.columns([3.3, 0.9, 0.9, 0.9, 0.9], vertical_alignment="center")
    fila[0].markdown(f"**Capa oculta {i + 1}**")
    fila[1].button("−", key=f"menos_n_{i}", on_click=_menos_neurona, args=(i,),
                   help="Quitar una neurona", disabled=_arch[i] <= 1)
    fila[2].markdown(
        f"<div style='text-align:center;font-weight:700;color:#0f172a'>{_arch[i]}</div>",
        unsafe_allow_html=True)
    fila[3].button("+", key=f"mas_n_{i}", on_click=_mas_neurona, args=(i,),
                   help="Añadir una neurona", disabled=_arch[i] >= 16)
    fila[4].button("🗑", key=f"borrar_capa_{i}", on_click=_borrar_capa, args=(i,),
                   help="Eliminar la capa completa")

fnew = st.columns([2.4, 1.2], vertical_alignment="center")
fnew[0].number_input("Neuronas de la nueva capa", 1, 16, 4, key="nueva_capa_sz")
fnew[1].button("➕ Añadir capa", on_click=_add_capa, width="stretch")

# ---------- 3. Activación por capa + entrenamiento ----------
st.sidebar.markdown("#### 3️⃣ Activación por capa")
ACT_OPCIONES = list(ACTIVACIONES)


def _aplicar_activacion():
    """Callback: copia la activación elegida a todas las capas ocultas."""
    val = st.session_state.get("aplicar_activacion")
    if val and val != "—":
        for i in range(len(st.session_state.arch)):
            st.session_state[f"act_{i}"] = val
    st.session_state["aplicar_activacion"] = "—"


if len(_arch) == 0:
    capa_act = []
    st.sidebar.caption("Sin capas ocultas: la red es lineal (la salida usa sigmoide).")
else:
    st.sidebar.selectbox("Aplicar a todas las capas", ["—"] + ACT_OPCIONES,
                         key="aplicar_activacion", on_change=_aplicar_activacion,
                         help="Atajo: fija la misma activación en todas las capas ocultas.")
    for i in range(len(_arch)):
        st.session_state.setdefault(f"act_{i}", "Tanh")
    capa_act = [st.sidebar.selectbox(f"Activación · capa oculta {i + 1}", ACT_OPCIONES,
                                     key=f"act_{i}") for i in range(len(_arch))]

st.sidebar.markdown("#### 4️⃣ Parámetros de entrenamiento")
LR_OPCIONES = [f"{v:g}" for v in [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01,
                                  0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]]
lr_txt = st.sidebar.select_slider("Tasa de aprendizaje (η)", options=LR_OPCIONES, value="0.03")
lr = float(lr_txt)
momento = st.sidebar.slider("Momento (inercia)", 0.0, 0.95, 0.9, step=0.05,
                            help="Acumula el gradiente de épocas anteriores y acelera la "
                                 "convergencia (es lo que usa TensorFlow Playground).")
epocas = st.sidebar.number_input("Número máximo de épocas", 10, 20000, 500, step=50)
lote_txt = st.sidebar.selectbox("Lote (modo de actualización)",
                                ["Batch completo", "Mini-batch (64)", "Mini-batch (32)",
                                 "Estocástico (1)"])
LOTE = {"Batch completo": None, "Mini-batch (64)": 64, "Mini-batch (32)": 32,
        "Estocástico (1)": 1}[lote_txt]
coste = st.sidebar.selectbox("Función de coste", list(FUNCIONES_COSTE))
inicializacion = st.sidebar.selectbox("Inicialización de pesos", list(INICIALIZACIONES))
semilla = st.sidebar.number_input("Semilla (reproducibilidad)", 0, 99999, 42)
parada_temprana = st.sidebar.checkbox("Detener si la pérdida < 0.01", value=False)

st.sidebar.caption("Los cambios de arquitectura, activación, coste, inicialización o semilla "
                   "reinician la red con pesos nuevos.")

# ══════════════════════════════════════════════════════════════════════════════
# 5. SINCRONIZACIÓN MODELO ↔ CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
activaciones_cfg = tuple(ACTIVACIONES[a] for a in capa_act)
cfg = (tuple(st.session_state.arch), activaciones_cfg,
       INICIALIZACIONES[inicializacion],
       int(semilla), FUNCIONES_COSTE[coste])
if st.session_state.modelo_cfg != cfg:
    st.session_state.modelo = MLP([2] + list(st.session_state.arch) + [1],
                                  activaciones=list(activaciones_cfg),
                                  inicializacion=INICIALIZACIONES[inicializacion],
                                  semilla=int(semilla), coste=FUNCIONES_COSTE[coste])
    st.session_state.modelo_cfg = cfg
    st.session_state.hist = None
    st.session_state.editado = False
    st.session_state.superficie = None
modelo: MLP = st.session_state.modelo
hist = st.session_state.hist

# ══════════════════════════════════════════════════════════════════════════════
# 6. ENTRENAMIENTO (con progreso y curva en vivo)
# ══════════════════════════════════════════════════════════════════════════════
def _fig_viva(perdidas) -> go.Figure:
    fig = go.Figure(go.Scatter(y=list(perdidas), mode="lines",
                               line=dict(color=C_PERDIDA, width=2.2), showlegend=False))
    cfg = dict(PLANTILLA, height=200, margin=dict(l=40, r=15, t=12, b=30),
               xaxis_title="época", yaxis_title="pérdida")
    fig.update_layout(**cfg)
    return fig


def _rango_trajectoria(vals, actual):
    if vals is None or len(vals) == 0:
        return actual - 1.5, actual + 1.5
    lo, hi = float(np.min(vals)), float(np.max(vals))
    pad = max(hi - lo, 0.4) * 0.35
    return lo - pad, hi + pad


def ejecutar_entrenamiento(modelo, X, y, *, epocas, lr, lote, mu, tol, data_sig):
    n = len(X)
    prog = st.progress(0.0, text="Preparando el entrenamiento…")
    caja_grafica = st.empty()

    perdidas, precisiones, epocas_hist = [], [], []
    rec_ep, rec_param, rec_loss = [0], [modelo.flatten()], []
    snap_eps = sorted({0, max(1, epocas // 10), epocas // 4, epocas // 2,
                       (3 * epocas) // 4, epocas})
    snapshots = []

    p = modelo.predict_proba(X)
    perdida = modelo.loss_from_p(p, y)
    rec_loss.append(perdida)
    snapshots.append((0, modelo.flatten()))

    cada_registro = max(1, epocas // 400)
    cada_grafica = max(1, epocas // 20)
    epoca_final, paro = epocas, False

    for ep in range(1, epocas + 1):
        modelo.train_epoch(X, y, lr, lote, mu)
        p = modelo.predict_proba(X)
        perdida = modelo.loss_from_p(p, y)
        prec = modelo.accuracy(p, y)
        perdidas.append(perdida)
        precisiones.append(prec)
        epocas_hist.append(ep)

        if ep % cada_registro == 0 or ep == epocas or ep in snap_eps:
            rec_ep.append(ep)
            rec_param.append(modelo.flatten())
            rec_loss.append(perdida)
        if ep in snap_eps:
            snapshots.append((ep, modelo.flatten()))

        if ep % cada_grafica == 0 or ep == epocas:
            caja_grafica.plotly_chart(_fig_viva(perdidas), theme=None)
        prog.progress(ep / epocas,
                      text=f"Época {ep}/{epocas} · pérdida {perdida:.4f} · precisión {prec:.1%}")
        epoca_final = ep
        if tol and perdida < tol:
            paro = True
            break

    prog.progress(1.0, text=("✔ Entrenamiento terminado" if not paro
                             else f"✔ Detenido temprano en la época {epoca_final}"))
    return dict(loss=perdidas, acc=precisiones, ep=epocas_hist,
                rec_ep=np.array(rec_ep), rec_param=np.array(rec_param),
                rec_loss=np.array(rec_loss), snapshots=snapshots,
                epoca_final=epoca_final, paro=paro, data_sig=data_sig)


# ══════════════════════════════════════════════════════════════════════════════
# 7. CABECERA, BARRA DE ACCIÓN Y MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="pg-title">🧠 Playground de Redes Neuronales</div>',
            unsafe_allow_html=True)
st.markdown('<div class="pg-sub">Construye la red capa a capa, entrena y observa qué aprende '
            'y qué aporta cada neurona · inspirado en TensorFlow Playground</div>',
            unsafe_allow_html=True)

if error_datos:
    st.error(error_datos)
    st.stop()

if X is not None and nota_datos:
    st.caption(f"📊 **{dataset}** — {nota_datos}")

barra = st.columns([1.15, 1.0, 3.0], gap="small", vertical_alignment="center")
with barra[0]:
    btn_entrenar = st.button("🚀 Entrenar red", type="primary", width="stretch")
with barra[1]:
    btn_reiniciar = st.button("🎲 Reiniciar pesos", width="stretch")
with barra[2]:
    estado = ("⏳ sin entrenar" if hist is None else
              f"✅ entrenada en {hist['epoca_final']} épocas"
              + (" (parada temprana)" if hist.get("paro") else ""))
    marca = " · ✏️ con edición manual" if st.session_state.editado else ""
    st.markdown(
        f"<div class='pg-card pg-hint'>η = <b>{lr:g}</b> · momento = <b>{momento:g}</b> · "
        f"épocas máx = <b>{epocas}</b> · "
        f"activación = <b>{' → '.join(capa_act) if capa_act else '—'}</b> · lote = <b>{lote_txt}</b> · "
        f"coste = <b>{coste.split(' (')[0]}</b><br>{estado}{marca}</div>",
        unsafe_allow_html=True)

if btn_reiniciar:
    modelo.reset(int(np.random.randint(0, 2 ** 31 - 1)))
    st.session_state.hist = None
    st.session_state.editado = False
    st.session_state.superficie = None
    st.rerun()

if btn_entrenar:
    st.session_state.hist = ejecutar_entrenamiento(
        modelo, X, y, epocas=int(epocas), lr=lr, lote=LOTE, mu=momento,
        tol=0.01 if parada_temprana else None, data_sig=data_sig)
    st.session_state.editado = False
    st.session_state.superficie = None      # queda obsoleta al entrenar
    st.rerun()

# ---------- métricas en vivo ----------
p_actual = modelo.predict_proba(X)
acc_viva = modelo.accuracy(p_actual, y)
loss_viva = modelo.loss_from_p(p_actual, y)
_ocultas = len(modelo.sizes) - 2

m = st.columns(5, gap="small")
m[0].metric("Precisión actual", f"{acc_viva * 100:.1f}%",
            delta=f"{(acc_viva - 0.5) * 100:+.1f} pts vs. azar")
m[1].metric("Pérdida (coste)", f"{loss_viva:.4f}")
m[2].metric("Épocas entrenadas", str(hist["epoca_final"]) if hist else "0")
m[3].metric("Parámetros totales", str(modelo.n_params))
m[4].metric("Arquitectura", "2-" + ("-".join(map(str, st.session_state.arch)) or "0") + "-1")

if hist is None:
    st.info("👆 Elige un problema, diseña la arquitectura y pulsa **🚀 Entrenar red** para "
            "ver la frontera de decisión, las neuronas y la superficie de coste.")
elif hist.get("data_sig") != data_sig:
    st.warning("⚠️ El dataset cambió después del último entrenamiento: entrena de nuevo "
               "para que todas las visualizaciones tengan sentido con los datos actuales.")

# ══════════════════════════════════════════════════════════════════════════════
# 8. PESTAÑAS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════
tab_frontera, tab_evol, tab_neuronas, tab_pesos, tab_3d, tab_datos, tab_demo = st.tabs(
    ["🗺 Frontera de decisión", "🎬 Evolución del entrenamiento", "🔬 Qué aprende cada neurona",
     "⚖️ Pesos y bias", "🏔 Superficie de coste 3D", "📦 Datos", "🎓 Demostración"])


def _malla(x0, x1, y0, y1, res=170):
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    XX, YY = np.meshgrid(xs, ys)
    return XX, YY


def fig_frontera(Xv, yv, p_map, xx, yy, titulo=None, leyenda=True, altura=430):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=p_map, x=xx[0], y=yy[:, 0], zmin=0, zmax=1, colorscale=CS_FRONTERA,
        colorbar=dict(title="P(ŷ=1)", thickness=11, outlinewidth=0, len=0.72),
        hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<br>P(ŷ=1)=%{z:.3f}<extra></extra>"))
    for cls, color, nombre in [(0, C0, "Clase 0"), (1, C1, "Clase 1")]:
        sel = yv == cls
        if not sel.any():
            continue
        fig.add_trace(go.Scatter(
            x=Xv[sel, 0], y=Xv[sel, 1], mode="markers", name=nombre, showlegend=leyenda,
            marker=dict(size=8, color=color, line=dict(width=1, color="white")),
            hovertemplate=f"{nombre}<br>x1=%{{x:.2f}}<br>x2=%{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(range=[xx[0].min(), xx[0].max()], constrain="domain")
    fig.update_yaxes(range=[yy[:, 0].min(), yy[:, 0].max()], scaleanchor="x", scaleratio=1)
    return style(fig, height=altura, titulo=titulo)


def fig_curvas(h, logy=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["ep"], y=h["loss"], mode="lines", name="Pérdida (coste)",
                             line=dict(color=C_PERDIDA, width=2.4),
                             fill="tozeroy", fillcolor="rgba(225,29,72,.08)",
                             hovertemplate="época %{x}<br>pérdida %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=h["ep"], y=h["acc"], mode="lines", name="Precisión",
                             line=dict(color=C_PREC, width=2, dash="dot"), yaxis="y2",
                             hovertemplate="época %{x}<br>precisión %{y:.1%}<extra></extra>"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[-0.02, 1.02],
                                  title="Precisión", gridcolor="rgba(0,0,0,0)"))
    if logy:
        fig.update_yaxes(type="log", title="Pérdida (log)")
    else:
        fig.update_yaxes(title="Pérdida")
    return style(fig, height=430, titulo="Evolución del error durante el entrenamiento",
                 xaxis="Época")


# ---------- datos para todas las pestañas ----------
x0, x1 = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
y0, y1 = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
xx, yy = _malla(x0, x1, y0, y1)
puntos_malla = np.c_[xx.ravel(), yy.ravel()]
p_mapa = modelo.predict_proba(puntos_malla).reshape(xx.shape)

# ────────────────────────────── 8.1 Frontera ──────────────────────────────
with tab_frontera:
    escala_log = st.checkbox("Escala logarítmica en la curva de pérdida", value=False, key="logy")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.plotly_chart(fig_frontera(X, y, p_mapa, xx, yy,
                                     titulo="Frontera de decisión aprendida"), theme=None)
    with c2:
        if hist:
            st.plotly_chart(fig_curvas(hist, escala_log), theme=None)
        else:
            st.info("Entrena la red para ver la evolución del error y la precisión.")

# ──────────────────────────── 8.2 Evolución ───────────────────────────────
with tab_evol:
    st.markdown("**Cómo evolucionó la frontera de decisión** durante el entrenamiento "
                "(instantáneos al 0 %, 10 %, 25 %, 50 %, 75 % y 100 % de las épocas).")
    if hist and hist["snapshots"]:
        pesos_originales = modelo.flatten()
        cols = st.columns(len(hist["snapshots"]))
        try:
            for k, (ep_snap, flat_snap) in enumerate(hist["snapshots"]):
                modelo.set_flat(flat_snap)
                p_snap = modelo.predict_proba(puntos_malla).reshape(xx.shape)
                with cols[k]:
                    st.plotly_chart(fig_frontera(X, y, p_snap, xx, yy,
                                                 titulo=f"Época {ep_snap}", leyenda=(k == 0),
                                                 altura=300), theme=None)
        finally:
            modelo.set_flat(pesos_originales)
        if hist.get("paro"):
            st.caption(f"⏹ Detención temprana activada: el entrenamiento se detuvo en la "
                       f"época {hist['epoca_final']} porque la pérdida bajó de 0.01.")
    else:
        st.info("Aún no hay historial de entrenamiento: pulsa **🚀 Entrenar red**.")

# ───────────────────────────── 8.3 Neuronas ───────────────────────────────
with tab_neuronas:
    Lcap = len(modelo.W)
    nombres_capas = [modelo.layer_name(l) for l in range(Lcap)]
    capa_sel = st.selectbox("Capa a inspeccionar", nombres_capas,
                            index=(Lcap - 2 if Lcap > 1 else 0))
    l = nombres_capas.index(capa_sel)
    n_salidas = modelo.sizes[l + 1]

    acts_malla, _ = modelo.forward(puntos_malla, devolver_cache=True)
    MAX_PLOTS = 8
    visibles = min(n_salidas, MAX_PLOTS)
    cols_neu = st.columns(visibles, gap="small")
    for j in range(visibles):
        A = acts_malla[l + 1][:, j].reshape(xx.shape)
        nombre = modelo.neuron_name(l, j)
        fig = go.Figure(go.Heatmap(
            z=A, x=xx[0], y=yy[:, 0], colorscale="Viridis",
            colorbar=dict(thickness=10, outlinewidth=0, len=0.55) if j == visibles - 1 else None,
            hovertemplate=f"{nombre}<br>x1=%{{x:.2f}}<br>x2=%{{y:.2f}}<br>a=%{{z:.3f}}<extra></extra>"))
        fig.update_xaxes(range=[xx[0].min(), xx[0].max()], constrain="domain")
        fig.update_yaxes(range=[yy[:, 0].min(), yy[:, 0].max()], scaleanchor="x", scaleratio=1)
        st.plotly_chart(style(fig, height=300, titulo=nombre), theme=None, key=f"neu_{l}_{j}")
    if n_salidas > MAX_PLOTS:
        st.caption(f"Mostrando {MAX_PLOTS} de {n_salidas} neuronas de esta capa "
                   f"(el resto son análogas).")

    # --- aporte de cada neurona (ablación) ---
    st.markdown("---")
    st.subheader("Qué aporta cada neurona a la salida de la red")
    st.caption("**Aporte** = cambio medio absoluto de la probabilidad de salida (|Δŷ|) al "
               "anular esa neurona. Valores grandes = la red depende mucho de esa neurona.")
    ocultas_total = len(modelo.W) - 1
    if ocultas_total > 0:
        p0 = modelo.predict_proba(X)
        acts_X, _ = modelo.forward(X, devolver_cache=True)   # una sola pasada reutilizada
        etiquetas, aportes, activaciones, salientes = [], [], [], []
        for l2 in range(ocultas_total):
            for j in range(modelo.sizes[l2 + 1]):
                pm = modelo.forward_enmascarado(X, l2, j)
                etiquetas.append(modelo.neuron_name(l2, j))
                aportes.append(float(np.mean(np.abs(p0 - pm))))
                activaciones.append(float(np.mean(np.abs(acts_X[l2 + 1][:, j]))))
                salientes.append(float(np.abs(modelo.W[l2 + 1][j, :]).sum()))
        fig_abl = go.Figure()
        fig_abl.add_trace(go.Bar(x=etiquetas, y=aportes, name="Aporte a la salida (|Δŷ|)",
                                 marker_color=C_PESO, yaxis="y",
                                 hovertemplate="%{x}<br>|Δŷ|=%{y:.4f}<extra></extra>"))
        fig_abl.add_trace(go.Bar(x=etiquetas, y=activaciones, name="Activación media (|a|)",
                                 marker_color="#06b6d4", yaxis="y2",
                                 hovertemplate="%{x}<br>|a|=%{y:.3f}<extra></extra>"))
        fig_abl.update_layout(barmode="group",
                              yaxis2=dict(overlaying="y", side="right", title="|a|",
                                          gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(style(fig_abl, height=420, titulo="Aporte y actividad de cada neurona",
                              xaxis="Neurona", yaxis="|Δŷ|"), theme=None)

        df_neu = pd.DataFrame({"neurona": etiquetas,
                               "aporte |Δŷ|": np.round(aportes, 4),
                               "activación media |a|": np.round(activaciones, 3),
                               "|pesos salientes|": np.round(salientes, 3)})
        st.dataframe(df_neu, width="stretch", hide_index=True)
    else:
        st.info("Esta red **no tiene capas ocultas**: la salida depende linealmente de la "
                "entrada. Es el caso ideal para demostrar por qué hace falta no linealidad "
                "(prueba XOR sin capas ocultas).")

# ────────────────────────────── 8.4 Pesos ─────────────────────────────────
with tab_pesos:
    nombres_capas_p = [modelo.layer_name(l) for l in range(len(modelo.W))]
    lp = nombres_capas_p.index(st.selectbox("Capa a inspeccionar/editar", nombres_capas_p))
    Wl, Bl = modelo.W[lp], modelo.b[lp]
    ins, outs = modelo.in_names(lp), modelo.out_names(lp)

    ca, cb = st.columns([2, 1], gap="large")
    with ca:
        fig_w = go.Figure(go.Heatmap(
            z=Wl.T, x=ins, y=outs, colorscale="RdBu_r", zmid=0,
            text=np.round(Wl.T, 2), texttemplate="%{text:.2f}", textfont=dict(size=11),
            colorbar=dict(title="peso", thickness=11, outlinewidth=0),
            hovertemplate="de %{x} a %{y}<br>peso=%{z:.4f}<extra></extra>"))
        fig_w.update_layout(xaxis_title="Entrada de la capa", yaxis_title="Neurona destino")
        st.plotly_chart(style(fig_w, height=40 * len(outs) + 180,
                              titulo=f"Matriz de pesos · {modelo.layer_name(lp)}"), theme=None)
    with cb:
        fig_b = go.Figure(go.Bar(x=outs, y=Bl, marker_color="#f43f5e",
                                 hovertemplate="%{x}<br>bias=%{y:.4f}<extra></extra>"))
        st.plotly_chart(style(fig_b, height=40 * len(outs) + 180,
                              titulo="Bias de cada neurona", yaxis="bias"), theme=None)

    st.markdown("---")
    st.subheader("✏️ Editor manual de pesos y bias")
    st.caption("Cambia un valor y pulsa **Aplicar**: la frontera y las métricas se "
               "actualizan al instante. Así puedes observar qué hace cada parámetro.")
    neur = st.selectbox("Neurona destino", list(range(len(outs))),
                        format_func=lambda j: outs[j])

    clave_firma = hashlib.md5(np.ascontiguousarray(Wl[:, neur]).tobytes()
                              + np.ascontiguousarray(Bl[neur:neur + 1]).tobytes()).hexdigest()
    prefijo = f"ed_{lp}_{neur}"
    if st.session_state.get("_editor_firma") != clave_firma:
        for i in range(len(ins)):
            st.session_state[f"{prefijo}_w{i}"] = float(Wl[i, neur])
        st.session_state[f"{prefijo}_b"] = float(Bl[neur])
        st.session_state["_editor_firma"] = clave_firma

    valores_w, ancho = {}, min(4, len(ins))
    for inicio in range(0, len(ins), ancho):
        fila_cols = st.columns(ancho, gap="small")
        for col, i in zip(fila_cols, range(inicio, min(inicio + ancho, len(ins)))):
            with col:
                valores_w[i] = st.number_input(f"{ins[i]} → {outs[neur]}",
                                               step=0.01, format="%.4f",
                                               key=f"{prefijo}_w{i}")
    valor_b = st.number_input(f"Bias → {outs[neur]}", step=0.01, format="%.4f",
                              key=f"{prefijo}_b")

    if st.button("💾 Aplicar valores a la red", type="primary"):
        for i, v in valores_w.items():
            modelo.W[lp][i, neur] = float(v)
        modelo.b[lp][neur] = float(valor_b)
        st.session_state.editado = True
        st.session_state.superficie = None
        st.rerun()

    st.markdown("---")
    go_op = st.columns(3, gap="small")
    if go_op[0].button("🎲 Aleatorizar todos los pesos", width="stretch"):
        modelo.reset(int(np.random.randint(0, 2 ** 31 - 1)))
        st.session_state.hist = None
        st.session_state.editado = False
        st.session_state.superficie = None
        st.rerun()
    if go_op[1].button("🧹 Poner todos los pesos a cero", width="stretch"):
        for l2 in range(len(modelo.W)):
            modelo.W[l2][:] = 0.0
            modelo.b[l2][:] = 0.0
        st.session_state.editado = True
        st.session_state.superficie = None
        st.rerun()
    if go_op[2].button("↩️ Restaurar pesos iniciales (semilla)", width="stretch"):
        modelo.reset()
        st.session_state.hist = None
        st.session_state.editado = False
        st.session_state.superficie = None
        st.rerun()
    st.caption("💡 Con todos los pesos a cero todas las neuronas aprenden igual (simetría): "
               "es un buen experimento para demostrar por qué la inicialización importa.")

    fig_hist = go.Figure()
    for l2 in range(len(modelo.W)):
        fig_hist.add_trace(go.Histogram(x=modelo.W[l2].ravel(), name=modelo.layer_name(l2),
                                        opacity=0.72, nbinsx=31))
    fig_hist.update_layout(barmode="overlay", xaxis_title="valor del peso", yaxis_title="frecuencia")
    st.plotly_chart(style(fig_hist, height=300,
                          titulo="Distribución de los pesos por capa"), theme=None)

# ─────────────────────────────── 8.5 Coste 3D ─────────────────────────────
with tab_3d:
    etiquetas_param = modelo.param_labels()
    n_param = len(etiquetas_param)
    st.markdown("Elige **dos parámetros** de la red (el resto queda congelado) y la app "
                "recalcula la función de coste sobre una malla, superponiendo el camino que "
                "siguió el optimizador durante el entrenamiento.")
    s1, s2, s3 = st.columns([2.4, 2.4, 1.2], gap="small")
    idx1 = s1.selectbox("Parámetro del eje X", list(range(n_param)),
                        format_func=lambda i: etiquetas_param[i], key="p3d_x")
    idx2 = s2.selectbox("Parámetro del eje Y", list(range(n_param)),
                        format_func=lambda i: etiquetas_param[i],
                        index=min(1, n_param - 1), key="p3d_y")
    resolucion = s3.select_slider("Resolución", options=[13, 17, 21, 27, 33], value=21)

    def firma_superficie():
        token = (data_sig or "").encode()
        return hashlib.md5(np.ascontiguousarray(modelo.flatten()).tobytes()
                           + token).hexdigest()[:12]

    sup = st.session_state.superficie
    clave_actual = (idx1, idx2, resolucion)
    lista = st.button("🧮 Calcular superficie de coste", type="primary", width="stretch")

    if idx1 == idx2:
        st.warning("Selecciona dos parámetros **distintos** para los ejes X e Y.")
    elif lista:
        with st.spinner(f"Calculando la superficie ({resolucion}×{resolucion} evaluaciones)…"):
            flat0 = modelo.flatten()
            t1 = t2 = tloss = reps = None
            if hist and hist["rec_param"] is not None and len(hist["rec_ep"]) > 1:
                t1 = hist["rec_param"][:, idx1]
                t2 = hist["rec_param"][:, idx2]
                tloss = hist["rec_loss"]
                reps = hist["rec_ep"]
            lo1, hi1 = _rango_trajectoria(t1, flat0[idx1])
            lo2, hi2 = _rango_trajectoria(t2, flat0[idx2])
            g1 = np.linspace(lo1, hi1, resolucion)
            g2 = np.linspace(lo2, hi2, resolucion)
            Z = np.empty((resolucion, resolucion))
            try:
                for i in range(resolucion):
                    for j in range(resolucion):
                        f = flat0.copy()
                        f[idx1] = g1[j]
                        f[idx2] = g2[i]
                        modelo.set_flat(f)
                        Z[i, j] = modelo.loss_from_p(modelo.predict_proba(X), y)
            finally:
                modelo.set_flat(flat0)
            st.session_state.superficie = dict(
                clave=clave_actual, firma=firma_superficie(),
                x=g1, y=g2, z=Z, i1=idx1, i2=idx2,
                tray=dict(x=t1, y=t2, z=tloss, ep=reps) if t1 is not None else None,
                actual=dict(x=float(flat0[idx1]), y=float(flat0[idx2]),
                            z=loss_viva))
        st.rerun()

    sup = st.session_state.superficie
    if sup is None:
        st.info("Pulsa **🧮 Calcular superficie de coste** para generar la visualización 3D.")
    elif sup["clave"] != clave_actual:
        st.warning("⚠️ La superficie calculada corresponde a otros parámetros. Vuelve a "
                   "pulsar **🧮 Calcular superficie de coste**.")
    elif sup["firma"] != firma_superficie():
        st.warning("⚠️ Los pesos o los datos cambiaron desde el último cálculo. Vuelve a "
                   "pulsar **🧮 Calcular superficie de coste**.")
    else:
        tr = sup["tray"]
        fig3d = go.Figure()
        fig3d.add_trace(go.Surface(
            x=sup["x"], y=sup["y"], z=sup["z"], colorscale="Viridis", opacity=0.88,
            colorbar=dict(title="coste", thickness=12, outlinewidth=0, len=0.6),
            contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen"),
            hovertemplate="X=%{x:.3f}<br>Y=%{y:.3f}<br>coste=%{z:.4f}<extra></extra>"))
        if tr is not None:
            fig3d.add_trace(go.Scatter3d(
                x=tr["x"], y=tr["y"], z=tr["z"], mode="lines",
                line=dict(color="#94a3b8", width=4), name="Camino del optimizador",
                hovertemplate="época %{customdata}<extra>camino</extra>",
                customdata=tr["ep"]))
            paso = max(1, len(tr["x"]) // 60)
            fig3d.add_trace(go.Scatter3d(
                x=tr["x"][::paso], y=tr["y"][::paso], z=tr["z"][::paso], mode="markers",
                marker=dict(size=4, color=tr["ep"][::paso], colorscale="Plasma",
                            showscale=False, opacity=0.95),
                name="Épocas (color = tiempo)",
                customdata=tr["ep"][::paso],
                hovertemplate="época %{customdata}<extra></extra>"))
            fig3d.add_trace(go.Scatter3d(
                x=[tr["x"][0]], y=[tr["y"][0]], z=[tr["z"][0]], mode="markers",
                marker=dict(size=8, color="#10b981", symbol="diamond"), name="Inicio",
                hovertemplate="inicio (pesos aleatorios)<extra></extra>"))
        fig3d.add_trace(go.Scatter3d(
            x=[sup["actual"]["x"]], y=[sup["actual"]["y"]], z=[sup["actual"]["z"]],
            mode="markers", marker=dict(size=9, color="#e11d48", symbol="x"), name="Pesos actuales",
            hovertemplate="pesos actuales<extra></extra>"))
        fig3d.update_layout(
            scene=dict(xaxis_title=etiquetas_param[idx1],
                       yaxis_title=etiquetas_param[idx2],
                       zaxis_title="Función de coste",
                       aspectmode="cube", bgcolor="rgba(0,0,0,0)"))
        c3d, c2d = st.columns([2, 1], gap="large")
        with c3d:
            st.plotly_chart(style(fig3d, height=600,
                                  titulo="Superficie de coste y trayectoria de optimización"),
                            theme=None)
        with c2d:
            fig2d = go.Figure(go.Contour(
                x=sup["x"], y=sup["y"], z=sup["z"], colorscale="Viridis",
                contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
                colorbar=dict(title="coste", thickness=11, outlinewidth=0, len=0.6),
                hovertemplate="X=%{x:.3f}<br>Y=%{y:.3f}<br>coste=%{z:.4f}<extra></extra>"))
            if tr is not None:
                fig2d.add_trace(go.Scatter(x=tr["x"], y=tr["y"], mode="lines+markers",
                                           marker=dict(size=4, color="#e11d48"),
                                           line=dict(color="white", width=2),
                                           name="Optimizador",
                                           hovertemplate="época %{customdata}<extra></extra>",
                                           customdata=tr["ep"]))
                fig2d.add_trace(go.Scatter(x=[tr["x"][0]], y=[tr["y"][0]], mode="markers",
                                           marker=dict(size=11, color="#10b981", symbol="diamond"),
                                           name="Inicio"))
            st.plotly_chart(style(fig2d, height=600, titulo="Vista cenital (contornos)",
                                  xaxis=etiquetas_param[idx1],
                                  yaxis=etiquetas_param[idx2]), theme=None)
        st.caption("🟥 Cruz roja = pesos actuales · 💚 diamante = peso inicial · "
                   "camino gris/rojo = recorrido del descenso del gradiente época a época. "
                   "Si la trayectoria oscila mucho, la tasa de aprendizaje es alta; si casi "
                   "no se mueve, es baja.")

# ────────────────────────────── 8.7 Demostración ──────────────────────────
with tab_demo:
    st.markdown("### 🎓 Guía de demostración del módulo")
    st.markdown("""
Estas son pruebas **verificadas** que demuestran cómo la arquitectura, las activaciones y
los parámetros de entrenamiento afectan a la capacidad de aprendizaje de la red:

| # | Experimento | Qué observar (medido en esta app) |
|---|-------------|-----------------------------------|
| 1 | **XOR sin capas ocultas** (borra las capas con 🗑) | La frontera sólo puede ser lineal: la precisión se queda en ~41–45 % (azar) y la pérdida ≈ 0.69. Una red lineal *no puede* resolver XOR. Con 1 capa oculta de ≥2 neuronas se llega al 100 %. |
| 2 | **Activación en XOR** con 1 capa oculta y 2 neuronas | Misma red, cambiando sólo la activación: **Tanh 100 %**, **ReLU ~76 %**, **Sigmoide ~52 %**, **Lineal ~27 %**. La no linealidad es imprescindible y tanh/relu trabajan mejor que la sigmoide apilada (gradientes que desaparecen). |
| 3 | **Tasa de aprendizaje** | Con lote completo y momento, η = 0.001 apenas avanza (~42 %) y η = 0.03–1.0 converge al 100 %. Pero con **actualización estocástica (lote = 1)**, η ≥ 0.1 desestabiliza: la pérdida deja de bajar y sube hasta 1.29 (visible en la superficie 3D, con saltos enormes). |
| 4 | **Épocas insuficientes** | En *Evolución del entrenamiento* verás la frontera a mitad de camino todavía sin separar las clases; sube a 2000 y compara. |
| 5 | **Profundidad** | En el *Tablero de cuadrantes*, `[4]` neuronas en una capa se quedan ≈ 71 %, mientras `[8, 8]` alcanzan ≈ 97 %: la profundidad construye fronteras más complejas con menos neuronas por capa. |
| 6 | **Lote (optimizador)** | En *Espirales* con **batch completo** la red se estanca ≈ 60 %; con **mini-batch (32)** y **ReLU** llega a ≈ 99.5 % en sólo 500 épocas (y ≈ 99.8 % con `[16,16]`). El mismo problema se resuelve sólo cambiando el modo de actualización. |
| 7 | **Inicialización** | Pulsa *Poner todos los pesos a cero*: por simetría todas las neuronas quedan idénticas y la red no aprende (≈ 50 %, los pesos siguen a 0). En datos fáciles (círculos) Xavier, He y Uniforme llegan al 100 % todas: la inicialización pesa sobre todo en redes profundas o con ReLU. |
| 8 | **Función de coste** | Prueba BCE y MSE: en estos problemas llegan a resultados parecidos (en un desbalanceado 90/10 ambas alcanzaron 100 %). BCE (log-loss) es la elección estándar: su pérdida es probabilística y su gradiente no se apaga con la sigmoide de salida. |
| 9 | **Superficie 3D de coste** | Elige dos pesos de la *Capa oculta 1* y observa el valle: el optimizador desciende por él época a época. Con η alta la trayectoria da saltos y oscila; con η baja avanza casi en línea recta. |
| 10 | **Tus propios datos** | Sube un CSV con `x1, x2, clase` (o pégalo / genera una nube) y entrena: el módulo acepta cualquier problema de clasificación binaria en 2D, con o sin fila de encabezado y con coma, punto y coma, tabulador o espacios. |
    """)
    st.markdown("""
**Ideas clave para la exposición**
- La **no linealidad** (activación, distinta por capa) es lo que permite separar clases que no son linealmente separables.
- La **arquitectura** (nº de capas y neuronas) define la *capacidad* de la red: demasiada poca no alcanza el problema; demasiada puede memorizar (compara entrenando con pocos puntos y muchos parámetros).
- Los **hiperparámetros de entrenamiento** (η, épocas, lote, coste, inicialización) determinan *si* y *a qué velocidad* se alcanza el mínimo.
- En la superficie 3D se ve el corazón del asunto: **entrenar = descender por el valle del coste** modificando pesos y bias.
    """)


# ─────────────────────────────── 8.6 Datos ────────────────────────────────
with tab_datos:
    cd1, cd2 = st.columns([3, 2], gap="large")
    with cd1:
        fig_datos = go.Figure()
        for cls, color, nombre in [(0, C0, "Clase 0"), (1, C1, "Clase 1")]:
            sel = y == cls
            if sel.any():
                fig_datos.add_trace(go.Scatter(
                    x=X[sel, 0], y=X[sel, 1], mode="markers", name=nombre,
                    marker=dict(size=8, color=color, line=dict(width=1, color="white")),
                    hovertemplate=f"{nombre}<br>x1=%{{x:.3f}}<br>x2=%{{y:.3f}}<extra></extra>"))
        fig_datos.update_xaxes(range=[x0, x1], constrain="domain", title="x1")
        fig_datos.update_yaxes(range=[y0, y1], scaleanchor="x", scaleratio=1, title="x2")
        st.plotly_chart(style(fig_datos, height=440, titulo=f"Dataset: {dataset}"), theme=None)
    with cd2:
        d0, d1, d2 = st.columns(3)
        d0.metric("Muestras", len(X))
        d1.metric("Clase 0", int((y == 0).sum()))
        d2.metric("Clase 1", int((y == 1).sum()))
        st.markdown("**Estadísticas por característica**")
        st.dataframe(pd.DataFrame({
            "mín": [X[:, 0].min(), X[:, 1].min()],
            "máx": [X[:, 0].max(), X[:, 1].max()],
            "media": [X[:, 0].mean(), X[:, 1].mean()],
            "desv. típ.": [X[:, 0].std(), X[:, 1].std()],
        }, index=["x1", "x2"]).round(3), width="stretch", hide_index=False)
        if nota_datos:
            st.caption(nota_datos)
    st.markdown("**Vista previa de los datos**")
    st.dataframe(pd.DataFrame({"x1": X[:, 0].round(4), "x2": X[:, 1].round(4),
                               "clase": y.astype(int)}),
                 width="stretch", height=300, hide_index=True)

st.markdown("---")
st.caption("Módulo Playground de Redes Neuronales · motor propio en NumPy "
           "(propagación directa y retropropagación) · visualización con Plotly · "
           "interfaz con Streamlit")
