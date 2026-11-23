"""Limpieza reproducible del dataset inmobiliario de Miami-Dade (HousingPulse AI).

Uso (desde la carpeta Actividad_1_Evolve):

    python src/clean_data.py
    python src/clean_data.py --input data/raw/otro.csv --since 2024-01-01

Parte siempre del CSV original y no depende de ninguna edicion manual.
Genera:
    data/processed/housing_clean.csv
    reports/cleaning_report.json          (filas que entran y salen en cada paso)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ACTIVITY_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ACTIVITY_DIR / "data" / "raw" / "realestatedataset-miami-dade-county-fl-2026-10.csv"
DEFAULT_OUTPUT_DIR = ACTIVITY_DIR / "data" / "processed"
DEFAULT_REPORT = ACTIVITY_DIR / "reports" / "cleaning_report.json"

# ---- Parametros de limpieza (cada uno se justifica en docs/entregas/01_*.md) ----
SINCE = "2025-01-01"                      # ventana temporal: mercado reciente
MAX_DATE = "2026-10-31"                   # fecha de descarga del CSV (2026-10)
PLACEHOLDER_DATE = "1900-01-01"           # fecha de relleno de la fuente
RESIDENTIAL = ["single_family", "condo", "townhouse", "apartment", "multi_family", "manufactured"]
PRICE_RANGE = (20_000, 30_000_000)        # por debajo: traspasos simbolicos, no ventas de mercado
PPSF_RANGE = (50, 5_000)                  # USD por pie cuadrado
SQFT_RANGE = (200, 20_000)
BEDS_MAX = 12
BATHS_RANGE = (0.5, 10)
YEAR_BUILT_MIN = 1850
MIAMI_BBOX = dict(lat=(24.8, 26.1), lng=(-81.2, -80.0))
CORE_FEATURES = ["beds", "baths", "sqft", "year_built", "zip"]
BOOL_COLS = ["central_air", "pool", "garage", "waterfront", "fireplace", "basement"]
# Columnas que NO pasan al dataset final:
#  - list_price: fuga de informacion (ver notebook, apartado viabilidad)
#  - status/listing_type: estado del anuncio en la foto actual, no en la fecha de venta
#  - street/unit/parcel_apn: direccion exacta, innecesaria (privacidad)
#  - state/county_fips/source_count: constantes o metadatos de la fuente
DROP_COLS = ["list_price", "status", "listing_type", "street", "unit", "parcel_apn",
             "state", "county_fips", "source_count"]


class Funnel:
    """Registra cuantas filas quedan tras cada paso."""

    def __init__(self, n0: int):
        self.steps = [{"paso": "CSV original", "filas": n0, "eliminadas": 0}]

    def log(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        prev = self.steps[-1]["filas"]
        self.steps.append({"paso": name, "filas": len(df), "eliminadas": prev - len(df)})
        return df


def clean(raw: pd.DataFrame, since: str = SINCE) -> tuple[pd.DataFrame, list[dict]]:
    f = Funnel(len(raw))
    df = raw.copy()

    # 1. Ambito geografico: solo Florida / Miami-Dade (hay 7 filas de otros estados)
    df = f.log("Solo FL y condado 12086", df[(df["state"] == "FL") & (df["county_fips"] == 12086)])

    # 2. Solo inmuebles residenciales (fuera solar 'land' y 'other')
    df = f.log("Solo tipos residenciales", df[df["property_type"].isin(RESIDENTIAL)])

    # 3. Debe tener venta registrada (precio y fecha). Sin esto no hay variable objetivo.
    df["last_sold_date"] = pd.to_datetime(df["last_sold_date"], errors="coerce")
    df = f.log("Con precio y fecha de venta",
               df[df["last_sold_price"].notna() & df["last_sold_date"].notna()])

    # 4. Fechas imposibles: relleno 1900-01-01 o posteriores a la descarga
    ok_date = (df["last_sold_date"] != pd.Timestamp(PLACEHOLDER_DATE)) & (
        df["last_sold_date"] <= pd.Timestamp(MAX_DATE))
    df = f.log("Fecha de venta valida", df[ok_date])

    # 5. Ventana temporal reciente
    df = f.log(f"Ventas desde {since}", df[df["last_sold_date"] >= pd.Timestamp(since)])

    # 6. Precio de venta plausible
    lo, hi = PRICE_RANGE
    df = f.log(f"Precio de venta entre {lo:,} y {hi:,}", df[df["last_sold_price"].between(lo, hi)])

    # 7. Atributos fisicos imposibles (se eliminan; los NaN se tratan en el paso 10)
    s = df["sqft"]
    df = f.log("sqft en rango", df[s.isna() | s.between(*SQFT_RANGE)])
    df = f.log("beds en rango", df[df["beds"].isna() | df["beds"].between(0, BEDS_MAX)])
    df = f.log("baths en rango", df[df["baths"].isna() | df["baths"].between(*BATHS_RANGE)])
    df = f.log("lat/lng dentro de Miami-Dade",
               df[df["lat"].between(*MIAMI_BBOX["lat"]) & df["lng"].between(*MIAMI_BBOX["lng"])])

    # 8. Coherencia temporal: no puede haberse vendido antes de existir
    sale_year = df["last_sold_date"].dt.year
    yb = df["year_built"]
    df = f.log("year_built coherente con la venta",
               df[yb.isna() | (yb.between(YEAR_BUILT_MIN, sale_year))])

    # 9. Precio por pie cuadrado plausible (detecta importes o superficies erroneos)
    ppsf = df["last_sold_price"] / df["sqft"]
    df = f.log("Precio/sqft plausible", df[ppsf.isna() | ppsf.between(*PPSF_RANGE)])

    # 10. Duplicados: misma direccion+unidad+zip -> nos quedamos con la venta mas reciente
    df = df.sort_values("last_sold_date")
    df = f.log("Duplicados de id", df.drop_duplicates("id", keep="last"))
    df = f.log("Duplicados de inmueble (direccion+unidad+zip)",
               df.drop_duplicates(["street", "unit", "zip"], keep="last"))

    # 11. Variables nucleares completas (la imputacion se hace despues, dentro del CV,
    #     para no filtrar informacion del test al entrenamiento)
    df = f.log("Variables nucleares completas", df.dropna(subset=CORE_FEATURES))

    # ---- Tipos y valores ausentes en el resto de columnas ----
    df["zip"] = df["zip"].astype("int64")
    for c in ["beds", "year_built"]:
        df[c] = df[c].astype("int64")
    for c in BOOL_COLS:   # la fuente solo informa 'true'; ausente = no consta
        df[c] = df[c].astype(str).str.lower().eq("true")
    df["heating_type"] = df["heating_type"].fillna("unknown")
    df["city"] = df["city"].str.strip().str.title()

    # ---- Variables derivadas (solo con informacion disponible al consultar) ----
    df["sale_date"] = df["last_sold_date"]
    df["sale_year"] = df["sale_date"].dt.year.astype("int64")
    df["sale_month"] = df["sale_date"].dt.month.astype("int64")
    df["property_age_at_sale"] = df["sale_year"] - df["year_built"]

    df = df.drop(columns=DROP_COLS + ["last_sold_date"])
    df = df.rename(columns={"last_sold_price": "sold_price"})
    df = df.sort_values("sale_date").reset_index(drop=True)
    return df, f.steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--since", default=SINCE, help="Fecha minima de venta (YYYY-MM-DD)")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"No existe {args.input}. Copia el CSV original a data/raw/ (ver README).")

    raw = pd.read_csv(args.input, low_memory=False)
    clean_df, steps = clean(raw, since=args.since)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(args.output_dir / "housing_clean.csv", index=False)
    args.report.write_text(json.dumps(
        {"input": args.input.name, "since": args.since, "pasos": steps,
         "filas_finales": len(clean_df), "columnas_finales": list(clean_df.columns)},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'paso':55s} {'filas':>9s} {'eliminadas':>11s}")
    for s in steps:
        print(f"{s['paso']:55s} {s['filas']:>9,d} {s['eliminadas']:>11,d}")
    print(f"\nDataset final: {clean_df.shape[0]:,} filas x {clean_df.shape[1]} columnas")


if __name__ == "__main__":
    main()
