"""
funciones_icfes.py
==================
Funciones de análisis estadístico para el dashboard ICFES — Saber 11.

Módulos incluidos
-----------------
  1. DIAGNÓSTICO
     - diagnostico_manual          → NAs y duplicados sin pandas nativos

  2. VARIABLES CUALITATIVAS
     - frecuencia_cualitativa_manual → tabla de frecuencias
     - validar_cualitativa           → comparación manual vs nativa
     - graficar_cualitativa          → barras horizontal con color configurable

  3. VARIABLES CUANTITATIVAS
     - estadisticas_cuantitativas_manual → media, mediana, moda, desv.std
     - validar_cuantitativa              → comparación manual vs nativa
     - graficar_cuantitativa             → histograma con líneas de tendencia central

  4. MEDIDAS DE LOCALIZACIÓN
     - medidas_localizacion_manual → Q1, Q2, Q3 por interpolación lineal
     - validar_localizacion        → comparación manual vs pandas.quantile
     - graficar_localizacion       → boxplot + gráfico de cuartiles

  5. MEDIDAS DE FORMA
     - medidas_forma_manual → asimetría y curtosis poblacional
     - validar_forma        → comparación manual vs scipy
     - graficar_forma       → histograma + KDE + panel de interpretación

Uso en el dashboard
-------------------
    from funciones_icfes import *

    var, color = "PUNT_GLOBAL", PALETA[0]
    res = estadisticas_cuantitativas_manual(df[var])
    validar_cuantitativa(df[var], res)
    graficar_cuantitativa(df[var], var, res, color)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as sp


# ── Paleta y estilo global ─────────────────────────────────────────────────────
PALETA = [
    "#2563EB",  # 0 — azul
    "#16A34A",  # 1 — verde
    "#DC2626",  # 2 — rojo
    "#D97706",  # 3 — ámbar
    "#7C3AED",  # 4 — violeta
    "#0891B2",  # 5 — cian
    "#DB2777",  # 6 — rosa
    "#65A30D",  # 7 — lima
]

plt.rcParams.update({
    "figure.figsize": (6, 4),
    "axes.grid": True,
})



#  1. DIAGNÓSTICO — NAs y duplicados


def diagnostico_manual(dataframe):
    """
    Calcula manualmente el porcentaje de NAs por columna y el número de
    filas duplicadas, sin usar isnull(), duplicated() ni sum() de pandas.

    Parámetros
    ----------
    dataframe : pd.DataFrame
        Dataset a diagnosticar (preferiblemente el original, sin limpiar).

    Retorna
    -------
    tabla : pd.DataFrame
        Columnas: Variable | NAs (manual) | % NA (manual)
        Ordenado de mayor a menor porcentaje de nulos.
    duplicados : int
        Número de filas duplicadas detectadas por hash.
    total_filas : int
        Total de filas contadas manualmente.
    """
    # ── Total de filas ─────────────────────────────────────────
    total_filas = 0
    for _ in dataframe.index:
        total_filas += 1

    # ── Conteo de NAs por columna ──────────────────────────────
    conteo_na = {}
    pct_na    = {}

    for col in dataframe.columns:
        nulos = 0
        for valor in dataframe[col]:
            try:
                if valor != valor:      # NaN float: NaN != NaN es True
                    nulos += 1
            except TypeError:
                pass
        conteo_na[col] = nulos
        pct_na[col]    = round((nulos / total_filas) * 100, 4)

    # ── Conteo de filas duplicadas por hash ───────────────────
    vistos     = {}
    duplicados = 0
    for i in dataframe.index:
        clave = hash(tuple(str(v) for v in dataframe.loc[i]))
        if clave in vistos:
            duplicados += 1
        else:
            vistos[clave] = True

    # ── Tabla resultado ────────────────────────────────────────
    tabla = pd.DataFrame({
        "Variable":      list(conteo_na.keys()),
        "NAs (manual)":  list(conteo_na.values()),
        "% NA (manual)": list(pct_na.values()),
    }).sort_values("% NA (manual)", ascending=False).reset_index(drop=True)

    return tabla, duplicados, total_filas



#  2. VARIABLES CUALITATIVAS


def frecuencia_cualitativa_manual(columna):
    """
    Calcula manualmente la tabla de frecuencias de una variable cualitativa,
    sin usar value_counts(), mode(), groupby() ni crosstab().

    Parámetros
    ----------
    columna : pd.Series
        Variable cualitativa a analizar.

    Retorna
    -------
    tabla : pd.DataFrame
        Columnas: Categoría | Frec. Absoluta | Frec. Relativa % | Frec. Acumulada
    moda : cualquier tipo
        Categoría más frecuente.
    """
    # ── Conteo absoluto ────────────────────────────────────────
    frecuencias = {}
    for valor in columna:
        if str(valor) == "nan":
            continue
        frecuencias[valor] = frecuencias.get(valor, 0) + 1

    total = sum(frecuencias.values())
    moda  = max(frecuencias, key=lambda v: frecuencias[v])

    # ── Relativa y acumulada ───────────────────────────────────
    porcentajes = {}
    frec_acum   = {}
    acumulado   = 0
    for v in frecuencias:
        porcentajes[v] = round((frecuencias[v] / total) * 100, 2)
        acumulado      += frecuencias[v]
        frec_acum[v]   = acumulado

    tabla = pd.DataFrame({
        "Categoría":        list(frecuencias.keys()),
        "Frec. Absoluta":   list(frecuencias.values()),
        "Frec. Relativa %": list(porcentajes.values()),
        "Frec. Acumulada":  list(frec_acum.values()),
    })

    return tabla, moda


def validar_cualitativa(columna, tabla_manual, moda_manual):
    """
    Compara los resultados de frecuencia_cualitativa_manual() con las
    funciones nativas de pandas (value_counts, mode).

    Parámetros
    ----------
    columna : pd.Series
        Variable cualitativa original.
    tabla_manual : pd.DataFrame
        Resultado de frecuencia_cualitativa_manual().
    moda_manual : cualquier tipo
        Moda calculada manualmente.

    Imprime
    -------
    Coincidencias o diferencias entre conteos, y comparación de modas.
    """
    col_str      = columna.astype(str).replace("nan", pd.NA).dropna()
    vc           = col_str.value_counts().reset_index()
    vc.columns   = ["Categoría", "Frec. Nativa"]
    moda_nativa  = col_str.mode()[0]

    comp             = tabla_manual.merge(vc, on="Categoría", how="left")
    comp["Coincide"] = comp["Frec. Absoluta"] == comp["Frec. Nativa"]
    errores          = comp[comp["Coincide"] == False]

    if errores.empty:
        print("Frecuencias coinciden.")
    else:
        print("Diferencias encontradas:")
        print(errores.to_string(index=False))

    moda_ok = "igual" if str(moda_manual) == str(moda_nativa) else "desigual"
    print(f"   Moda manual: {moda_manual}  |  Moda nativa: {moda_nativa}  [{moda_ok}]")


def graficar_cualitativa(tabla, nombre_variable, color=None):
    """
    Genera un gráfico de barras horizontal para una variable cualitativa.

    Parámetros
    ----------
    tabla : pd.DataFrame
        Resultado de frecuencia_cualitativa_manual().
    nombre_variable : str
        Nombre que aparecerá en el título del gráfico.
    color : str, opcional
        Color hex de las barras. Si no se indica, usa PALETA[0].
    """
    c         = color or PALETA[0]
    tabla_ord = tabla.sort_values("Frec. Absoluta")

    fig, ax = plt.subplots(figsize=(7, max(2.5, len(tabla) * 0.55)))
    bars = ax.barh(
        tabla_ord["Categoría"].astype(str),
        tabla_ord["Frec. Absoluta"],
        color=c, edgecolor="white", linewidth=0.5,
    )
    ax.bar_label(bars, padding=4, fontsize=8, color="#374151")
    ax.set_xlabel("Frecuencia absoluta")
    ax.set_title(f"Distribución — {nombre_variable}", fontweight="bold", pad=8)
    plt.tight_layout()
    plt.show()


#  3. VARIABLES CUANTITATIVAS


def estadisticas_cuantitativas_manual(columna):
    """
    Calcula manualmente las medidas de tendencia central y dispersión
    sin usar mean(), median(), mode() ni std() de pandas/numpy.

    Parámetros
    ----------
    columna : pd.Series
        Variable numérica a analizar.

    Retorna
    -------
    dict con claves: n | Media | Mediana | Moda | Desv.Std
    """
    # ── Filtrar NaN ────────────────────────────────────────────
    datos = []
    for v in columna:
        try:
            if v == v:
                datos.append(float(v))
        except (TypeError, ValueError):
            pass

    n = len(datos)

    # ── Media ──────────────────────────────────────────────────
    suma = 0.0
    for v in datos:
        suma += v
    media = suma / n

    # ── Mediana ────────────────────────────────────────────────
    ordenados = sorted(datos)
    mitad     = n // 2
    mediana   = (
        (ordenados[mitad - 1] + ordenados[mitad]) / 2
        if n % 2 == 0
        else ordenados[mitad]
    )

    # ── Moda ───────────────────────────────────────────────────
    conteo = {}
    for v in datos:
        conteo[v] = conteo.get(v, 0) + 1
    moda = max(conteo, key=lambda x: conteo[x])

    # ── Desviación estándar poblacional ───────────────────────
    suma_cuad = 0.0
    for v in datos:
        suma_cuad += (v - media) ** 2
    desv_std = (suma_cuad / n) ** 0.5

    return {
        "n":        n,
        "Media":    round(media,    4),
        "Mediana":  round(mediana,  4),
        "Moda":     round(moda,     4),
        "Desv.Std": round(desv_std, 4),
    }


def validar_cuantitativa(columna, res):
    """
    Compara los resultados de estadisticas_cuantitativas_manual() con las
    funciones nativas de pandas/numpy (ddof=0 para desv. poblacional).

    Parámetros
    ----------
    columna : pd.Series
        Variable numérica original.
    res : dict
        Resultado de estadisticas_cuantitativas_manual().

    Imprime
    -------
    Tabla comparativa manual vs nativa por cada medida.
    """
    col_num = pd.to_numeric(columna, errors="coerce").dropna()
    nativa  = {
        "Media":    round(col_num.mean(),      4),
        "Mediana":  round(col_num.median(),    4),
        "Moda":     round(col_num.mode()[0],   4),
        "Desv.Std": round(col_num.std(ddof=0), 4),
    }

    print(f"  {'Medida':<12} {'Manual':>10} {'Nativa':>10}  {'OK?':>8}")
    print(f"  {'-'*44}")
    for k in ["Media", "Mediana", "Moda", "Desv.Std"]:
        ok = "igual" if res[k] == nativa[k] else "dif."
        print(f"  {k:<12} {res[k]:>10} {nativa[k]:>10}  {ok}")


def graficar_cuantitativa(columna, nombre_variable, res, color=None):
    """
    Histograma con líneas verticales para media, mediana y moda.

    Parámetros
    ----------
    columna : pd.Series
        Variable numérica a graficar.
    nombre_variable : str
        Nombre que aparecerá en el título del gráfico.
    res : dict
        Resultado de estadisticas_cuantitativas_manual().
    color : str, opcional
        Color hex de las barras del histograma. Si no se indica, usa PALETA[0].
    """
    col_num = pd.to_numeric(columna, errors="coerce").dropna()
    c       = color or PALETA[0]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(col_num, bins=40, color=c, edgecolor="white", alpha=0.85)
    ax.axvline(res["Media"],   color="#DC2626", linestyle="--", lw=1.8,
               label=f"Media={res['Media']}")
    ax.axvline(res["Mediana"], color="#16A34A", linestyle="--", lw=1.8,
               label=f"Mediana={res['Mediana']}")
    ax.axvline(res["Moda"],    color="#D97706", linestyle="--", lw=1.8,
               label=f"Moda={res['Moda']}")
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución — {nombre_variable}", fontweight="bold", pad=8)
    ax.legend()
    plt.tight_layout()
    plt.show()


#  4. MEDIDAS DE LOCALIZACIÓN


def medidas_localizacion_manual(columna):
    """
    Calcula manualmente los cuartiles Q1, Q2, Q3 por interpolación lineal,
    sin usar quantile() de pandas.

    Parámetros
    ----------
    columna : pd.Series
        Variable numérica a analizar.

    Retorna
    -------
    dict con claves: n | Q1 | Q2 | Q3
    """
    datos = []
    for v in columna:
        try:
            if v == v:
                datos.append(float(v))
        except (TypeError, ValueError):
            pass

    datos.sort()
    n = len(datos)

    def _percentil(p):
        k = (n - 1) * p
        i = int(k)
        d = k - i
        if i + 1 < n:
            return datos[i] + d * (datos[i + 1] - datos[i])
        return datos[i]

    return {
        "n":  n,
        "Q1": round(_percentil(0.25), 4),
        "Q2": round(_percentil(0.50), 4),
        "Q3": round(_percentil(0.75), 4),
    }


def validar_localizacion(columna, res):
    """
    Compara los resultados de medidas_localizacion_manual() con
    pandas.Series.quantile().

    Parámetros
    ----------
    columna : pd.Series
        Variable numérica original.
    res : dict
        Resultado de medidas_localizacion_manual().

    Imprime
    -------
    Tabla comparativa manual vs nativa para Q1, Q2, Q3.
    """
    col_num = pd.to_numeric(columna, errors="coerce").dropna()
    nativa  = {
        "Q1": round(col_num.quantile(0.25), 4),
        "Q2": round(col_num.quantile(0.50), 4),
        "Q3": round(col_num.quantile(0.75), 4),
    }

    print(f"  {'Medida':<8} {'Manual':>10} {'Nativa':>10}  {'OK?':>8}")
    print(f"  {'-'*42}")
    for k in ["Q1", "Q2", "Q3"]:
        ok = "igual" if res[k] == nativa[k] else "dif."
        print(f"  {k:<8} {res[k]:>10} {nativa[k]:>10}  {ok}")


def graficar_localizacion(columna, nombre_variable, res, color=None):
    """
    Boxplot anotado + gráfico de barras con los valores de Q1, Q2, Q3 e IQR.

    Parámetros
    ----------
    columna : pd.Series
        Variable numérica a graficar.
    nombre_variable : str
        Nombre que aparecerá en los títulos.
    res : dict
        Resultado de medidas_localizacion_manual().
    color : str, opcional
        Color hex base. Si no se indica, usa PALETA[0].
    """
    col_num = pd.to_numeric(columna, errors="coerce").dropna()
    c       = color or PALETA[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # ── Boxplot ────────────────────────────────────────────────
    bp = axes[0].boxplot(
        col_num, vert=True, patch_artist=True,
        medianprops=dict(color="#DC2626", linewidth=2),
        flierprops=dict(marker="o", markersize=3,
                        markerfacecolor="#94A3B8", alpha=0.4),
    )
    bp["boxes"][0].set_facecolor(c + "40")
    bp["boxes"][0].set_edgecolor(c)
    axes[0].set_title(f"Boxplot — {nombre_variable}", fontweight="bold", pad=8)
    axes[0].set_ylabel("Valor")
    for label, val, col_ann in [
        ("Q1", res["Q1"], PALETA[1]),
        ("Q2", res["Q2"], "#DC2626"),
        ("Q3", res["Q3"], PALETA[2]),
    ]:
        axes[0].annotate(
            f"{label}={val}", xy=(1, val), xytext=(1.22, val),
            fontsize=7.5, color=col_ann, va="center",
        )

    # ── Barras de cuartiles ────────────────────────────────────
    labels = ["Q1", "Q2\n(Mediana)", "Q3"]
    vals   = [res["Q1"], res["Q2"], res["Q3"]]
    bars   = axes[1].bar(
        labels, vals,
        color=[PALETA[1], PALETA[0], PALETA[2]],
        edgecolor="white", width=0.5,
    )
    axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9, fontweight="bold")
    axes[1].set_title("Valores de cuartiles", fontweight="bold", pad=8)
    axes[1].set_ylabel("Valor")
    IQR = round(res["Q3"] - res["Q1"], 4)
    axes[1].set_xlabel(f"IQR = Q3 − Q1 = {IQR}", fontsize=9, color="#475569")

    plt.tight_layout()
    plt.show()



#  5. MEDIDAS DE FORMA


def medidas_forma_manual(columna):
    """
    Calcula manualmente la asimetría y la curtosis poblacional,
    sin usar skew() ni kurt() de pandas.

    Parámetros
    ----------
    columna : pd.Series
        Variable numérica a analizar.

    Retorna
    -------
    dict con claves: Asimetría | Curtosis
    """
    datos = []
    for v in columna:
        try:
            if v == v:
                datos.append(float(v))
        except (TypeError, ValueError):
            pass

    n = len(datos)

    # ── Media ────────────────────────────────────────────────
    suma = 0.0
    for v in datos:
        suma += v
    media = suma / n
    
    # ── Desviación estándar poblacional ─────────────────────
    suma_cuad = 0.0
    for v in datos:
        suma_cuad += (v - media) ** 2
    desv = (suma_cuad / n) ** 0.5

    # ── Asimetría ───────────────────────────────────────────
    suma_cub = 0.0
    for v in datos:
        suma_cub += (v - media) ** 3
    asimetria = (suma_cub / n) / (desv ** 3)

    # ── Curtosis ────────────────────────────────────────────
    suma_cuarta = 0.0
    for v in datos:
        suma_cuarta += (v - media) ** 4
    curtosis = (suma_cuarta / n) / (desv ** 4)

    return {
        "Asimetría": round(asimetria, 4),
        "Curtosis":  round(curtosis,  4),
    }


def validar_forma(columna, res):
    """
    Compara los resultados de medidas_forma_manual() con scipy.stats
    (skew y kurtosis poblacional, bias=True, fisher=False).

    Parámetros
    ----------
    columna : pd.Series
        Variable numérica original.
    res : dict
        Resultado de medidas_forma_manual().

    Imprime
    -------
    Tabla comparativa manual vs scipy para Asimetría y Curtosis.
    """
    col_num  = pd.to_numeric(columna, errors="coerce").dropna()
    asim_nat = round(sp.skew(col_num,     bias=True),              4)
    kurt_nat = round(sp.kurtosis(col_num, bias=True, fisher=False), 4)

    print(f"  {'Medida':<12} {'Manual':>10} {'Scipy':>10}  {'OK?':>8}")
    print(f"  {'-'*46}")
    for k_label, v_manual, v_nativa in [
        ("Asimetría", res["Asimetría"], asim_nat),
        ("Curtosis",  res["Curtosis"],  kurt_nat),
    ]:
        ok = "igual" if v_manual == v_nativa else "aprox."
        print(f"  {k_label:<12} {v_manual:>10} {v_nativa:>10}  {ok}")


def graficar_forma(columna, nombre_variable, res, color=None):
    """
    Histograma + KDE y panel de texto con la interpretación de
    asimetría y curtosis.

    Parámetros
    ----------
    columna : pd.Series
        Variable numérica a graficar.
    nombre_variable : str
        Nombre que aparecerá en los títulos.
    res : dict
        Resultado de medidas_forma_manual().
    color : str, opcional
        Color hex del histograma. Si no se indica, usa PALETA[0].
    """
    col_num = pd.to_numeric(columna, errors="coerce").dropna()
    c       = color or PALETA[0]

    asim = res["Asimetría"]
    kurt = res["Curtosis"]

    # ── Interpretaciones ────────────────────────────────────
    if abs(asim) < 0.5:
        interp_asim = "Simétrica"
    elif asim > 0:
        interp_asim = "Sesgo positivo (cola derecha)"
    else:
        interp_asim = "Sesgo negativo (cola izquierda)"

    if kurt < 3:
        interp_kurt = "Platicúrtica (colas ligeras)"
    elif abs(kurt - 3) < 0.1:
        interp_kurt = "Mesocúrtica (≈ normal)"
    else:
        interp_kurt = "Leptocúrtica (colas pesadas)"

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    # ── Histograma + KDE ────────────────────────────────────
    axes[0].hist(col_num, bins=40, color=c, edgecolor="white",
                 alpha=0.75, density=True)
    col_num.plot.kde(ax=axes[0], color="#1E293B", linewidth=1.8)
    axes[0].set_title(f"Distribución + KDE — {nombre_variable}",
                      fontweight="bold", pad=8)
    axes[0].set_xlabel("Valor")
    axes[0].set_ylabel("Densidad")

    # ── Panel de interpretación ─────────────────────────────
    axes[1].axis("off")
    texto = (
        f"ASIMETRÍA\n"
        f"  Valor:    {asim}\n"
        f"  Tipo:     {interp_asim}\n\n"
        f"CURTOSIS\n"
        f"  Valor:    {kurt}\n"
        f"  Tipo:     {interp_kurt}"
    )
    axes[1].text(
        0.05, 0.88, texto,
        transform=axes[1].transAxes,
        fontsize=10.5, verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#F0F9FF",
                  edgecolor="#BAE6FD", linewidth=1.5),
    )
    axes[1].set_title("Interpretación de medidas de forma",
                      fontweight="bold", pad=8)

    plt.tight_layout()
    plt.show()