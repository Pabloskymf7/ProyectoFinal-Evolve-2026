# Actividad 1 — Fuentes de datos, EDA y limpieza

**Proyecto:** HousingPulse AI · **Máster:** Evolve Data Science e IA 2026–2027
**Equipo:** Pablo Miguel Ferrer, Jorge Martinez Cercadillo, Julen Asua Fullaondo, Joan Poveda Baixauli
**Código:** `src/clean_data.py` · **Notebook:** `notebooks/01_eda_limpieza_viabilidad.ipynb` · **Gráficos:** `reports/figures/`

## 1. Resumen ejecutivo

- El CSV de Miami-Dade tiene 1.015.955 inmuebles, pero **solo 36.980 tienen una venta desde 2025** (3,6 %). Tras la limpieza quedan **34.793 ventas recientes** con 24 columnas.
- **`list_price` no es un precio anunciado independiente**: en las ventas recientes coincide exactamente con `last_sold_price` en el 92 % de las filas (97 % en inmuebles `sold` y `off_market`). Usarlo como variable de entrada es una **fuga de información** y queda excluido.
- **`last_sold_price` es el último precio de venta**, no una serie histórica: más del 94 % de las ventas con fecha son anteriores a 2025, y los atributos de la vivienda son una foto actual, no su estado en la fecha de venta.
- Con ventas recientes y sin fugas, una regresión lineal base entrenada con 2025 y evaluada en 2026 logra un error porcentual mediano del ~20 % y un R² de 0,79 sobre log(precio): **hay señal suficiente**.
- **Decisión: mantener HousingPulse AI** como proyecto principal, con las restricciones de la sección 8.

## 2. Fuente de datos

| Campo | Valor |
|---|---|
| Fuente | [RealEstateDataset](https://realestatedataset.com/) — [metodología](https://realestatedataset.com/methodology) |
| Archivo | `realestatedataset-miami-dade-county-fl-2026-10.csv` (descarga de octubre de 2026) |
| Contenido | Una fila por inmueble del condado de Miami-Dade (FIPS 12086): dirección, ubicación, tipo, características físicas, precio de lista, última venta y datos del anuncio |
| Tamaño | 1.015.955 filas × 30 columnas (~162 MB) |
| Uso en el proyecto | Estimar el precio de venta a partir de ubicación y características |

**Variables que necesita HousingPulse AI:** `city`, `zip`, `lat`, `lng`, `property_type`, `beds`, `baths`, `sqft`, `year_built`, `lot_sqft`, extras (`pool`, `garage`, `waterfront`, `central_air`, `hoa_fee`) y, como objetivo, `last_sold_price` con su fecha `last_sold_date`.

**Adecuación.** La fuente cubre el ámbito geográfico previsto y todas las variables que un usuario puede introducir en la app. Sus puntos débiles (secciones 4 y 5) son que el precio es el de la última venta, que el precio de lista no es utilizable y que los atributos son actuales. **Pendiente:** revisar las condiciones de uso y redistribución en la web de la fuente.

## 3. Exploración inicial (EDA)

Todos los gráficos están en `reports/figures/` y se generan en el notebook.

**Estructura.** 30 columnas; no hay filas completas duplicadas ni `id` repetidos. `parcel_apn` falta en el 26,8 % de las filas y se comparte entre unidades de un mismo edificio, por lo que no sirve como clave única. Hay 21.507 inmuebles repetidos por calle+unidad+código postal.

**Valores ausentes** (`01_nulos.png`):

| Variable | % nulos | Comentario |
|---|---|---|
| `last_sold_price` / `last_sold_date` | 34,9 % / 34,6 % | Sin venta no hay objetivo |
| `sqft` | 6,0 % | Núcleo del modelo |
| `year_built` | 10,3 % | Núcleo del modelo |
| `beds`, `baths` | 3,4 %, 2,9 % | Núcleo del modelo |
| `lot_sqft` | 50,5 % | No se exige |
| `pool`, `garage`, `waterfront`, `central_air` | 67–93 % | Solo viene el valor `true`; ausente = no consta |
| `fireplace`, `basement`, `listing_type` | > 98 % | Prácticamente vacías |

**Distribuciones** (`02` a `04`). 41,5 % unifamiliares, 32,6 % condominios, 11,2 % adosados; el resto, apartamentos, multifamiliares, solares y otros. El 81 % de los registros está `off_market`. El precio de venta tiene cola larga a la derecha (se analiza en escala logarítmica). Aparecen valores imposibles: `sqft` de 0 a 722.777, `beds` hasta 96, `baths` hasta 300, `year_built` desde 70 hasta 2028, precios de venta desde 0 USD hasta 1.700 millones y **3.841 fechas de relleno `1900-01-01`**.

**Relaciones** (`06`, `07`, `08`). En la muestra limpia el precio se relaciona sobre todo con la superficie (Spearman 0,72), después con baños (0,56), habitaciones (0,54) y tamaño de parcela (0,47). La antigüedad apenas influye por sí sola (0,08); la ubicación sí: el mapa muestra el gradiente de precio hacia la costa y el centro. Los datos cubren 79 códigos postales (77 con 30 ventas o más) y 42 ciudades; Miami concentra el 47 % de las ventas.

## 4. Decisiones de limpieza

Embudo completo en `reports/cleaning_report.json` y `05_embudo_limpieza.png`.

| # | Decisión | Filas eliminadas | Justificación |
|---|---|---:|---|
| 1 | Solo FL y condado 12086 | 7 | Hay 7 filas de otros estados (AL, TN, VA, nulo): error de la fuente |
| 2 | Solo tipos residenciales (se quitan `land` y `other`) | 4.885 | Un solar no tiene habitaciones ni baños; no entra en el producto |
| 3 | Debe tener precio y fecha de venta | 351.488 | Sin venta no hay variable objetivo |
| 4 | Fecha válida (sin `1900-01-01` ni posterior a la descarga) | 341 | Fecha de relleno o imposible |
| 5 | **Ventas desde 2025-01-01** (parámetro `--since`) | 622.358 | Ver sección 5: solo así precio y atributos corresponden al mismo momento |
| 6 | Precio entre 20.000 y 30.000.000 USD | 278 | Por debajo, traspasos simbólicos (hay ventas de 158 USD); por encima, atípicos extremos |
| 7 | `sqft` 200–20.000, `beds` ≤ 12, `baths` 0,5–10 | 316 | Valores físicamente imposibles o erróneos |
| 8 | `lat`/`lng` dentro de Miami-Dade | 17 | Coordenadas fuera del condado |
| 9 | `year_built` ≥ 1850 y no posterior a la venta | 35 | Un inmueble no se vende antes de existir |
| 10 | Precio/sqft entre 50 y 5.000 USD | 169 | Detecta importes o superficies mal registrados |
| 11 | Duplicados: una fila por calle+unidad+zip (la venta más reciente) | 63 | Evita contar dos veces un mismo inmueble |
| 12 | Variables nucleares completas (`beds`, `baths`, `sqft`, `year_built`, `zip`) | 1.205 | Se prefiere perder un 3 % a imputar los datos básicos; cualquier imputación posterior se hará dentro de la validación para no filtrar información del test |

**Tratamiento del resto de columnas.**
- Booleanos (`pool`, `garage`, `waterfront`, `central_air`, `fireplace`, `basement`): ausente → `False`. La fuente solo informa de la presencia; es un supuesto que se documenta como limitación.
- `heating_type` ausente → `unknown`. `lot_sqft` y `hoa_fee` se mantienen con nulos (no son nucleares); se imputarán en el modelado.
- Tipos: `zip`, `beds` y `year_built` a entero; `last_sold_date` a fecha; `city` normalizada.
- Variables derivadas: `sale_year`, `sale_month` y `property_age_at_sale`.
- **Columnas que no pasan al dataset final:** `list_price` (fuga), `status` y `listing_type` (estado del anuncio actual, no de la fecha de venta), `street`, `unit` y `parcel_apn` (privacidad: no hace falta la dirección exacta), `state`, `county_fips` y `source_count` (constantes o metadatos). `last_sold_price` se renombra `sold_price`.

**Resultado:** 34.793 filas × 24 columnas, ventas del 1-ene-2025 al 18-jun-2026: 17.859 unifamiliares, 10.799 condominios, 4.572 adosados, 1.041 multifamiliares, 471 apartamentos y 51 prefabricadas.

## 5. Viabilidad: ¿se puede estimar un precio con estos datos?

**a) `list_price` NO es un precio anunciado (`09_list_price_igual_a_venta.png`).** Entre 2010 y 2020 nunca coincide con `last_sold_price`; desde 2021 coincide cada vez más (51 % en 2021, 92 % en 2025–26). En las ventas de 2025 en adelante, el 97 % de los inmuebles `sold` y `off_market` tiene los dos importes idénticos. Solo los `active` tienen un precio de anuncio genuino, y casi nunca igual al de venta. Conclusión: la fuente rellena `list_price` con el precio de venta, de modo que usarlo haría que el modelo «viera la respuesta». Además, un usuario real no tiene ese dato. En la prueba con el modelo base, usar `list_price` directamente como estimación da un error del 0 % en el conjunto de test.

**b) `last_sold_price` es el último precio de venta, no una serie.** Cada inmueble tiene una sola venta (la más reciente) y la mayoría son antiguas: el 14,9 % de las ventas con fecha es de 1999 o antes y el 94,4 % es anterior a 2025. Un precio de 2004 no representa el mercado actual, así que entrenar con todo el histórico mezclaría épocas distintas sin variable de contexto temporal. De ahí la decisión 5: ventana desde 2025.

**c) Los atributos no representan el estado en la fecha de venta (`10_atributos_no_coinciden.png`).** `sqft`, `beds`, `baths`, `year_built` y los extras son una foto actual. Cuanto más antigua la venta, más probable que haya habido reformas o ampliaciones. En 9.470 inmuebles el año de construcción es **posterior** a la «última venta» (lo que se vendió era otro edificio o un solar). Con ventas de 2025–2026 la distancia entre venta y foto es de meses, por lo que el riesgo es bajo, pero no nulo (reformas posteriores a la venta).

**d) Prueba empírica.** Regresión lineal sobre log(precio) con superficie, habitaciones, baños, antigüedad, tipo, extras y código postal (media de log(precio/sqft) calculada solo con entrenamiento). Entrenada con ventas de 2025 (23.893) y evaluada con las de 2026 (10.900):

| Modelo | MAE (USD) | Error % mediano | R² (log) |
|---|---:|---:|---:|
| Sin `list_price` (válido) | 337.603 | 20,1 % | 0,787 |
| Con `list_price` (fuga) | 270.485 | 15,5 % | 0,827 |

La mejora «artificial» con `list_price` es moderada en un modelo lineal porque el 8 % de filas con precio distinto distorsiona un único coeficiente; en el 92 % restante el valor está copiado de la variable objetivo. Esta línea base no es el modelo final: Random Forest y Gradient Boosting se compararán en la fase de modelado y se espera mejorarla.

## 6. Proceso reproducible

`src/clean_data.py` parte siempre del CSV original y no depende de ediciones manuales:

```bash
python src/clean_data.py                       # ventas desde 2025
python src/clean_data.py --since 2024-01-01    # ventana alternativa
```

Genera `data/processed/housing_clean.csv` y `reports/cleaning_report.json`. Los umbrales están como constantes al inicio del fichero y cada paso informa de cuántas filas elimina. El notebook lee ese informe y el CSV limpio, así que todos los números de este documento se regeneran ejecutándolo.

## 7. Limitaciones

1. **Muestra pequeña**: 34.793 ventas, un 3,4 % del CSV. Aun así, es suficiente para un modelo base y cubre 77 códigos postales con 30 ventas o más.
2. **Sin precio anunciado**: el dataset no trae un precio de anuncio fiable, así que el modelo estima el **precio de cierre** esperado, no el de anuncio.
3. **Atributos actuales**: las reformas posteriores a la venta crean ruido que no se puede medir.
4. **Faltan variables clave**: estado de conservación, planta, vistas o calidad de acabados; limitan la precisión (el error mediano base es ~20 %).
5. **Supuesto de booleanos**: ausente = `False`.
6. **Periodo corto**: 18 meses; no capta ciclos de mercado ni estacionalidad completa.
7. **Condiciones de uso** de la fuente pendientes de revisar antes de publicar datos o desplegar la aplicación.
8. **Dirección exacta**: se elimina del dataset final por privacidad.

## 8. Decisión: ¿mantenemos HousingPulse AI?

**Sí, con restricciones.** Hay volumen reciente suficiente, el precio objetivo es identificable y la línea base ya capta la mayor parte de la señal (R² de 0,79 sobre log(precio)) sin fugas. Se mantiene si el equipo asume que:

1. el modelo se entrena solo con ventas recientes y se presenta como estimación del **precio de venta**, nunca como tasación;
2. `list_price` no se usa;
3. se valida con separación temporal y se informa del error por zona y tipo, sin ofrecer estimaciones donde haya pocas ventas;
4. la ventana temporal solo se amplía con un ajuste explícito por fecha de venta.

**Riesgo principal:** que los modelos más complejos no mejoren de forma apreciable el error base (~20 %). Si en la fase de modelado el error mediano no baja del 15 %, se valorará cambiar a GoalMetrics AI o SalaryScope AI (alternativas ya evaluadas en la Actividad 0).

## 9. Siguientes pasos

1. Revisar las condiciones de uso de la fuente.
2. Comparar Random Forest y Gradient Boosting con validación temporal.
3. Estudiar el error por tipo de inmueble, ciudad y rango de precio.
4. Decidir la ventana temporal definitiva (2024 frente a 2025 en adelante).
