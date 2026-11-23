# HousingPulse AI — Proyecto transversal Evolve Data Science e IA

> Estimación orientativa del precio de venta de viviendas en **Miami-Dade (Florida)** mediante Machine Learning, con una aplicación web que explica qué factores influyen en el resultado.

Repositorio privado del proyecto transversal del **Máster de Data Science e IA de Evolve (2026–2027)**. Reúne documentación, código, pruebas y evidencias, y crece a medida que avanzan los módulos del máster.

---

## 1. Equipo

| Integrante | GitHub |
|---|---|
| Pablo Miguel Ferrer | [@Pabloskymf7](https://github.com/Pabloskymf7) |
| Jorge Martinez Cercadillo | [@Jmcercadillo3](https://github.com/Jmcercadillo3) |
| Julen Asua Fullaondo | [@fuyacck](https://github.com/fuyacck) |
| Joan Poveda Baixauli | [@JoanP21](https://github.com/JoanP21) |

El repositorio debe ser accesible para los cuatro integrantes, el coordinador y el profesorado que revise las entregas (según las instrucciones de Classroom).

## 2. Estado del proyecto

| Actividad | Contenido | Estado |
|---|---|---|
| 0 | Equipos, ideas de producto y repositorio | Entregada |
| 1 | Python: fuentes de datos, EDA y limpieza | Ver rama `feature/pablo-actividad-1-eda-cleaning` |
| 2+ | SQL, modelado, aplicación web… | Pendiente |

**Idea preferida (provisional): HousingPulse AI.** Se eligió porque ya hay una fuente de datos descargada y permite desarrollar un modelo predictivo con una aplicación web funcional. La elección definitiva depende de la calidad y representatividad de los datos (Actividad 1), de contrastar la necesidad con usuarios y del feedback del profesorado.

## 3. De qué trata el proyecto

**Problema.** Conocer el valor aproximado de una vivienda es difícil: depende de la ubicación, la superficie, el estado y el mercado de cada zona. Los portales dan estimaciones generales, pero rara vez explican qué variables pesan ni cuáles son los límites de la predicción.

**Solución.** HousingPulse AI estima el precio de venta a partir de ciudad, código postal, superficie, habitaciones, baños, año de construcción y tipo de inmueble. Devuelve una estimación orientativa, los factores que más influyen y, cuando sea posible, viviendas comparables.

**Usuarios iniciales.** Particulares que quieren conocer el valor de su vivienda, compradores que comparan inmuebles y profesionales inmobiliarios que necesitan una herramienta complementaria.

**Alcance del MVP.**
1. Auditoría y limpieza de datos.
2. Análisis exploratorio (precio frente a superficie, habitaciones, baños, antigüedad y ubicación).
3. Preparación de variables.
4. Modelos: regresión base, Random Forest y Gradient Boosting.
5. Evaluación con MAE y RMSE, con separación temporal para evitar fugas de información.
6. Interpretabilidad (importancia de variables y efecto de cada una).
7. Aplicación web con Streamlit.
8. Informe final con datos, limpieza, modelos, resultados, limitaciones y mejoras.

**Fuera del MVP.** Tasación oficial, recomendaciones de inversión, predicción garantizada de precios futuros, cobertura de todo EE. UU. e integración en tiempo real con portales.

> La estimación es una referencia orientativa, **no una tasación oficial** ni una garantía del precio de compraventa.

**Alternativas evaluadas en la Actividad 0:** GoalMetrics AI (xG en fútbol, StatsBomb Open Data) y SalaryScope AI (salarios de desarrolladores, Stack Overflow Developer Survey). Su comparación está en [`00_ideas.md`](ProyectoFinalEvovle/Actividad_0_Evolve/docs/entregas/00_ideas.md).

## 4. Datos

| Campo | Valor |
|---|---|
| Fuente | [RealEstateDataset](https://realestatedataset.com/) — condado de Miami-Dade, FL |
| Documentación | <https://realestatedataset.com/methodology> |
| Archivo de trabajo | `realestatedataset-miami-dade-county-fl-2026-10.csv` (~162 MB, ~1 millón de filas) |
| Variables clave | `city`, `zip`, `lat`, `lng`, `property_type`, `beds`, `baths`, `sqft`, `year_built`, `list_price`, `last_sold_price`, `last_sold_date` |

**Los datos no se suben a Git** (están en `.gitignore`). Cada integrante los coloca en local en `data/raw/` de la actividad correspondiente. Antes de publicar datos o desplegar la app hay que revisar las condiciones de uso de la fuente. No se guardan credenciales ni datos personales, y se evita exponer direcciones concretas.

## 5. Estructura del repositorio

```
.
├── README.md                      # Este documento
├── requirements.txt               # Entorno Python común
├── .gitignore
└── ProyectoFinalEvovle/           # (el nombre de la carpeta se mantiene tal cual)
    ├── Actividad_0_Evolve/
    │   ├── data/README.md         # Plantilla para documentar fuentes de datos
    │   └── docs/entregas/00_ideas.md
    └── Actividad_N_Evolve/        # Una carpeta por actividad
        ├── data/{raw,processed}/  # Datos locales (ignorados por Git)
        ├── src/                   # Código reproducible
        ├── notebooks/             # Exploración
        ├── reports/figures/       # Gráficos generados
        └── docs/entregas/         # Documento de entrega
```

## 6. Cómo empezar a trabajar

Requisitos: Git y Python 3.11 o superior.

```bash
git clone https://github.com/Pabloskymf7/ProyectoFinal-Evolve-2026.git
cd ProyectoFinal-Evolve-2026

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Después, coloca el CSV de la fuente en la carpeta `data/raw/` de la actividad en la que trabajes y sigue su documento de entrega.

## 7. Cómo colaborar (Git)

- **No se commitea directamente en `main`.** Cada tarea va en una rama y se integra con Pull Request revisado por al menos otro integrante.
- Nombre de rama: `tipo/descripcion-corta`, todo en minúsculas y con guiones (`feature/`, `fix/`, `docs/`, `chore/`). Ejemplo: `feature/actividad-1-eda-cleaning`.
- Commits pequeños y con mensaje claro (`feat: …`, `fix: …`, `docs: …`).
- La documentación de una actividad va en el mismo PR que su código.
- Nunca se suben datos pesados, credenciales ni ficheros con datos personales.

## 8. Reglas del proyecto

- **No confundir** `list_price` (precio anunciado) con `last_sold_price` (último precio de venta). La Actividad 1 comprueba además si `list_price` es utilizable como variable de entrada.
- Toda variable de entrada debe estar disponible en el momento en que un usuario pide una estimación.
- La evaluación del modelo usa separación temporal siempre que las fechas lo permitan.
- Cada transformación de datos se documenta y es reproducible desde el CSV original, sin ediciones manuales.
