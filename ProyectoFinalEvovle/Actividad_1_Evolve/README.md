# 📖 Léeme: Actividad 1, qué hicimos y cómo (datos, EDA y limpieza)

**Proyecto:** HousingPulse AI · **Máster:** Evolve Data Science e IA 2026–2027 · **Lectura: 5 minutos**

Esta actividad responde a una pregunta: **¿sirve el dataset de Miami-Dade para estimar el precio de venta de una vivienda?** Aquí se explica qué se ha hecho, cómo se ha hecho y cómo repetirlo en vuestro ordenador.

> **Resumen en una línea:** de 1.015.955 inmuebles quedan **34.793 ventas recientes y limpias** (24 columnas). Hay señal suficiente para un modelo, pero hay que **excluir `list_price`** porque es una fuga de información.

---

## 1. Qué se ha hecho

1. **Auditoría de la fuente.** Se revisó el CSV de [RealEstateDataset](https://realestatedataset.com/) (condado de Miami-Dade): estructura, tipos, nulos, duplicados y valores imposibles.
2. **Análisis exploratorio (EDA).** Distribuciones, relaciones entre variables y geografía, con 11 gráficos.
3. **Limpieza reproducible.** Un script que parte siempre del CSV original y deja un dataset limpio, con el recuento de filas que se pierden en cada paso.
4. **Prueba de viabilidad.** Una regresión lineal sencilla para comprobar que los datos contienen señal.
5. **Decisión.** Mantener HousingPulse AI con las restricciones de la sección 4.

## 2. Hallazgos principales

| Hallazgo | Consecuencia |
|---|---|
| Solo el **3,6 %** del CSV (36.980 inmuebles) tiene una venta desde 2025. | Trabajamos con una muestra reciente de 34.793 ventas tras limpiar. |
| **`list_price` coincide con `last_sold_price` en el 92 %** de las ventas recientes (97 % en `sold` y `off_market`). | Es una **fuga de información**: el modelo «vería la respuesta». Queda excluido. |
| `last_sold_price` es el **último** precio de venta, no una serie histórica; el 94 % de las ventas con fecha es anterior a 2025. | Solo usamos ventas desde 2025 para que precio y atributos correspondan al mismo momento. |
| Los atributos (`sqft`, `beds`, `baths`…) son una foto actual, no la de la fecha de venta. | Riesgo bajo con ventas recientes, pero se documenta como limitación. |
| Línea base lineal (entrenada con 2025, evaluada en 2026): error % mediano ≈ **20 %**, R² ≈ **0,79** sobre log(precio). | Hay señal suficiente; Random Forest y Gradient Boosting deberían mejorar. |

Los datos tienen además valores imposibles que hubo que filtrar: `sqft` de 0 a 722.777, `beds` hasta 96, `baths` hasta 300, precios de 0 USD a 1.700 millones y 3.841 fechas de relleno `1900-01-01`.

## 3. Cómo se ha hecho

### 3.1 Limpieza (`src/clean_data.py`)

El script aplica estos filtros en orden y guarda cuántas filas elimina cada uno en `reports/cleaning_report.json`:

| # | Paso | Filas eliminadas |
|---|---|---:|
| 1 | Solo Florida y condado 12086 | 7 |
| 2 | Solo tipos residenciales (fuera `land` y `other`) | 4.885 |
| 3 | Debe tener precio y fecha de venta | 351.488 |
| 4 | Fecha válida (sin `1900-01-01` ni posteriores a la descarga) | 341 |
| 5 | **Ventas desde 2025-01-01** (parámetro `--since`) | 622.358 |
| 6 | Precio entre 20.000 y 30.000.000 USD | 278 |
| 7 | `sqft` 200–20.000, `beds` ≤ 12, `baths` 0,5–10 | 316 |
| 8 | Coordenadas dentro de Miami-Dade | 17 |
| 9 | `year_built` ≥ 1850 y no posterior a la venta | 35 |
| 10 | Precio por pie cuadrado entre 50 y 5.000 USD | 169 |
| 11 | Duplicados (una fila por calle + unidad + código postal) | 63 |
| 12 | Variables nucleares completas (`beds`, `baths`, `sqft`, `year_built`, `zip`) | 1.205 |

**Otras decisiones de limpieza**
- Booleanos (`pool`, `garage`, `waterfront`, `central_air`, `fireplace`, `basement`): la fuente solo informa de la presencia, así que *ausente → `False`*. Es un supuesto y está en las limitaciones.
- `heating_type` ausente → `unknown`. `lot_sqft` y `hoa_fee` conservan sus nulos (se imputarán en el modelado, dentro de la validación, para no filtrar información del test).
- Variables derivadas: `sale_year`, `sale_month` y `property_age_at_sale`.
- **Columnas que no pasan al dataset final:** `list_price` (fuga), `status` y `listing_type` (estado del anuncio actual), `street`, `unit` y `parcel_apn` (privacidad: no hace falta la dirección exacta), `state`, `county_fips` y `source_count` (constantes o metadatos). `last_sold_price` se renombra `sold_price`.

**Resultado:** 34.793 filas × 24 columnas, ventas del 1-ene-2025 al 18-jun-2026.

### 3.2 Exploración y viabilidad (`notebooks/01_eda_limpieza_viabilidad.ipynb`)

El notebook lee el informe de limpieza y el CSV limpio, y genera los 11 gráficos de `reports/figures/`:

| Gráficos | Qué muestran |
|---|---|
| `01`–`04` | Valores nulos, tipo y estado de los inmuebles, fechas y precios, valores atípicos originales |
| `05` | Embudo de limpieza: cuántas filas se pierden en cada paso |
| `06`–`08` | Relaciones del precio con superficie, baños, habitaciones y parcela; correlaciones y mapa; geografía y tiempo |
| `09` | Cómo `list_price` se parece cada vez más al precio de venta (51 % en 2021, 92 % en 2025–26) |
| `10` | Atributos que no coinciden con la fecha de venta |
| `11` | Resultado del modelo lineal base |

En la muestra limpia, el precio se relaciona sobre todo con la superficie (Spearman 0,72), después con baños (0,56), habitaciones (0,54) y tamaño de parcela (0,47). La ubicación pesa mucho: el mapa muestra el gradiente de precio hacia la costa y el centro.

### 3.3 Modelo base de comparación

Regresión lineal sobre log(precio) con superficie, habitaciones, baños, antigüedad, tipo de inmueble, extras y código postal. Entrenada con las ventas de 2025 y evaluada con las de 2026:

| Modelo | MAE (USD) | Error % mediano | R² (log) |
|---|---:|---:|---:|
| Sin `list_price` (el válido) | 337.603 | 20,1 % | 0,787 |
| Con `list_price` (fuga, solo para comprobar) | 270.485 | 15,5 % | 0,827 |

Esta línea base **no es el modelo final**: sirve de referencia para los siguientes modelos.

## 4. Decisión y restricciones

**Se mantiene HousingPulse AI**, con estas reglas para todo el equipo:

1. El modelo se entrena solo con ventas recientes y se presenta como estimación del **precio de venta**, nunca como tasación.
2. **No se usa `list_price`.**
3. Se valida con separación temporal y se informa del error por zona y tipo, sin ofrecer estimaciones donde haya pocas ventas.
4. La ventana temporal solo se amplía con un ajuste explícito por fecha de venta.

**Riesgo principal:** que los modelos complejos no mejoren el error base. Si el error mediano no baja del 15 %, se valorará cambiar a GoalMetrics AI o SalaryScope AI.

## 5. Cómo reproducirlo

Necesitas Git y Python 3.11 o superior.

```bash
# 1) Clonar y entrar en la rama
git clone https://github.com/Pabloskymf7/ProyectoFinal-Evolve-2026.git
cd ProyectoFinal-Evolve-2026
git checkout feature/actividad-1-eda-cleaning

# 2) Entorno
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# 3) Copiar el CSV original (no está en Git, pesa ~162 MB) a:
#    ProyectoFinalEvovle/Actividad_1_Evolve/data/raw/realestatedataset-miami-dade-county-fl-2026-10.csv

# 4) Limpiar
cd ProyectoFinalEvovle/Actividad_1_Evolve
python src/clean_data.py                       # ventas desde 2025
python src/clean_data.py --since 2024-01-01    # ventana alternativa

# 5) Ver el análisis
jupyter notebook notebooks/01_eda_limpieza_viabilidad.ipynb
```

Resultado esperado al final del paso 4: `Dataset final: 34,793 filas x 24 columnas`.

El script genera `data/processed/housing_clean.csv` y `reports/cleaning_report.json`. **Los datos no se suben a Git** (están en `.gitignore`).

## 6. Estructura de esta actividad

```
Actividad_1_Evolve/
├── README.md                          # Este documento
├── docs/entregas/01_fuentes_eda_limpieza.md   # Documento de entrega completo
├── notebooks/01_eda_limpieza_viabilidad.ipynb # EDA y viabilidad
├── src/clean_data.py                  # Limpieza reproducible
├── reports/
│   ├── cleaning_report.json           # Embudo de limpieza
│   └── figures/                       # 11 gráficos
└── data/{raw,processed}/              # Datos locales (ignorados por Git)
```

## 7. Limitaciones

1. Muestra pequeña: 34.793 ventas (3,4 % del CSV), aunque cubre 77 códigos postales con 30 ventas o más.
2. El dataset no trae un precio de anuncio fiable: el modelo estima el **precio de cierre**, no el de anuncio.
3. Los atributos son actuales: las reformas posteriores a la venta añaden ruido.
4. Faltan variables clave (estado de conservación, planta, vistas, calidad de acabados).
5. Supuesto de booleanos: ausente = `False`.
6. Periodo corto (18 meses): no capta ciclos de mercado.
7. **Pendiente:** revisar las condiciones de uso y redistribución de la fuente antes de publicar datos o desplegar la app.

## 8. Siguientes pasos

1. Revisar las condiciones de uso de la fuente.
2. Comparar Random Forest y Gradient Boosting con validación temporal.
3. Estudiar el error por tipo de inmueble, ciudad y rango de precio.
4. Decidir la ventana temporal definitiva (2024 frente a 2025 en adelante).

Para el detalle completo de cada decisión, ver [`docs/entregas/01_fuentes_eda_limpieza.md`](docs/entregas/01_fuentes_eda_limpieza.md).
