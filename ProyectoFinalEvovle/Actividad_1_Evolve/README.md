# Actividad 1 — Python: fuentes de datos, EDA y limpieza

Proyecto **HousingPulse AI**. Comprueba que el dataset de Miami-Dade sirve para el proyecto y deja una muestra limpia y reproducible.

- Documento de entrega: [`docs/entregas/01_fuentes_eda_limpieza.md`](docs/entregas/01_fuentes_eda_limpieza.md)
- Notebook de EDA y viabilidad: [`notebooks/01_eda_limpieza_viabilidad.ipynb`](notebooks/01_eda_limpieza_viabilidad.ipynb)
- Limpieza reproducible: [`src/clean_data.py`](src/clean_data.py)
- Gráficos: [`reports/figures/`](reports/figures/) · Embudo de limpieza: [`reports/cleaning_report.json`](reports/cleaning_report.json)

## Reproducir

```bash
# desde la raíz del repo, con el entorno instalado (pip install -r requirements.txt)
cd ProyectoFinalEvovle/Actividad_1_Evolve
# 1) copiar el CSV original a data/raw/
python src/clean_data.py
# 2) abrir el notebook y ejecutarlo de arriba abajo
jupyter notebook notebooks/01_eda_limpieza_viabilidad.ipynb
```
