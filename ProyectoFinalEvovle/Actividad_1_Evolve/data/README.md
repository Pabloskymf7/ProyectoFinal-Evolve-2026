# Datos de la Actividad 1

| Campo | Valor |
|---|---|
| Fuente | RealEstateDataset — <https://realestatedataset.com/> |
| Documentación | <https://realestatedataset.com/methodology> |
| Archivo | `realestatedataset-miami-dade-county-fl-2026-10.csv` |
| Fecha de descarga | octubre de 2026 (edición 2026-10) |
| Territorio | Condado de Miami-Dade, Florida (FIPS 12086) |
| Tamaño | 1.015.955 filas × 30 columnas, ~162 MB |
| Unidad | Una fila por inmueble (foto actual). Precios en USD, superficies en pies cuadrados (sqft) |
| Cobertura temporal | Fechas de última venta de 1900 a junio de 2026 (la mayoría, anteriores a 2025) |
| Formato | CSV, UTF-8, separador coma |
| Condiciones de uso | **Pendiente de revisar** en la web de la fuente antes de publicar datos o desplegar la app |

## Dónde van los ficheros

```
data/raw/        <- aquí se copia el CSV original (NO se sube a Git)
data/processed/  <- housing_clean.csv, generado por src/clean_data.py (NO se sube a Git)
```

## Procedimiento reproducible

1. Descargar el CSV desde la fuente (o pedírselo a un compañero) y copiarlo en `data/raw/`.
2. Desde `Actividad_1_Evolve/`: `python src/clean_data.py`.
3. Se genera `data/processed/housing_clean.csv` y `reports/cleaning_report.json`.

## Transformaciones y limitaciones

Resumidas en `docs/entregas/01_fuentes_eda_limpieza.md` (cada decisión de limpieza con su justificación). Limitación principal: los atributos del inmueble son una foto actual y `last_sold_price` es el **último** precio de venta, no una serie histórica.
