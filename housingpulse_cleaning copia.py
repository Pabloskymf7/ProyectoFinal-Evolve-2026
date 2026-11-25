#!/usr/bin/env python3
"""Reproducible cleaning pipeline for the Miami-Dade HousingPulse dataset."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


LOGGER = logging.getLogger("housingpulse")
DEFAULT_INPUT = "data/realestatedataset-miami-dade-county-fl-2026-10.csv"
DEFAULT_OUTPUT = "data/housingpulse_cleaned.csv"
CHART_STYLE = "seaborn-v0_8-whitegrid"

NUMERIC_COLUMNS = (
    "lat",
    "lng",
    "beds",
    "baths",
    "sqft",
    "lot_sqft",
    "year_built",
    "list_price",
    "last_sold_price",
    "hoa_fee",
    "source_count",
)
BOOLEAN_COLUMNS = ("central_air", "pool", "garage", "waterfront", "fireplace", "basement")
STRING_COLUMNS = (
    "id",
    "street",
    "unit",
    "city",
    "state",
    "zip",
    "property_type",
    "status",
    "heating_type",
    "listing_type",
    "parcel_apn",
    "county_fips",
)


@dataclass(frozen=True)
class CleaningConfig:
    """Documented, adjustable quality thresholds; dates are anchored to the CSV snapshot."""

    as_of_date: str = "2026-10-31"
    min_list_price: float = 1
    max_list_price: float = 50_000_000
    min_sqft: float = 300
    max_sqft: float = 20_000
    min_lot_sqft: float = 1
    max_lot_sqft: float = 80_000
    min_beds: float = 1
    max_beds: float = 12
    min_baths: float = 1
    max_baths: float = 12
    min_year_built: int = 1800
    min_price_per_sqft: float = 50
    max_price_per_sqft: float = 5_000


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the original CSV and preserve identifiers with leading zeroes."""
    LOGGER.info("Loading source CSV: %s", csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV source does not exist: {csv_path}")
    frame = pd.read_csv(
        csv_path,
        low_memory=False,
        dtype={column: "string" for column in STRING_COLUMNS},
    )
    LOGGER.info("Loaded %s rows and %s columns", *frame.shape)
    return frame


def normalize_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric, date, boolean, and identifier columns without silent defaults."""
    result = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column in result:
            converted = pd.to_numeric(result[column], errors="coerce")
            coerced = result[column].notna() & converted.isna()
            if coerced.any():
                LOGGER.warning(
                    "%s contains %s non-numeric values; converting them to null",
                    column,
                    int(coerced.sum()),
                )
            result[column] = converted
    for column in STRING_COLUMNS:
        if column in result:
            result[column] = result[column].astype("string").str.strip()
    for column in BOOLEAN_COLUMNS:
        if column in result:
            values = result[column].astype("string").str.strip().str.lower()
            mapping = {
                "true": True,
                "1": True,
                "yes": True,
                "false": False,
                "0": False,
                "no": False,
            }
            unknown = values.notna() & ~values.isin(mapping)
            if unknown.any():
                LOGGER.warning(
                    "%s contains unknown boolean values: %s",
                    column,
                    values.loc[unknown].value_counts().to_dict(),
                )
            normalized = values.map(mapping)
            result[column] = normalized.astype("boolean")
    if "last_sold_date" in result:
        original_dates = result["last_sold_date"]
        converted_dates = pd.to_datetime(original_dates, errors="coerce")
        invalid_dates = original_dates.notna() & converted_dates.isna()
        if invalid_dates.any():
            LOGGER.warning(
                "last_sold_date contains %s unparseable values; converting them to null",
                int(invalid_dates.sum()),
            )
        result["last_sold_date"] = converted_dates
    return result


def run_eda(frame: pd.DataFrame) -> dict[str, Any]:
    """Log basic dataset shape, types, categories, and numeric summaries."""
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "property_type_counts": {
            str(key): int(value)
            for key, value in frame["property_type"].value_counts(dropna=False).items()
        }
        if "property_type" in frame
        else {},
        "dtypes": {key: str(value) for key, value in frame.dtypes.items()},
    }
    LOGGER.info(
        "EDA: %s rows; property types: %s",
        summary["rows"],
        summary["property_type_counts"],
    )
    numeric = frame.select_dtypes(include="number")
    if not numeric.empty:
        LOGGER.info("Numeric summary:\n%s", numeric.describe().round(2).to_string())
    return summary


def report_nulls(frame: pd.DataFrame) -> pd.DataFrame:
    """Return column-level null counts and proportions."""
    counts = frame.isna().sum()
    return pd.DataFrame(
        {"null_count": counts, "null_rate": (counts / max(len(frame), 1)).round(6)}
    ).sort_values("null_count", ascending=False)


def detect_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag duplicate IDs, parcel identifiers, and normalized street addresses."""
    result = frame.copy()
    result["flag_duplicate_id"] = (
        result["id"].notna() & result["id"].duplicated(keep=False)
        if "id" in result
        else False
    )
    result["flag_duplicate_parcel_apn"] = (
        result["parcel_apn"].notna() & result["parcel_apn"].duplicated(keep=False)
        if "parcel_apn" in result
        else False
    )

    address_columns = ("street", "unit", "city", "zip")
    if all(column in result for column in address_columns):
        key_columns = []
        for column in address_columns:
            normalized = (
                result[column]
                .fillna("")
                .astype("string")
                .str.upper()
                .str.replace(r"[^\w\s]", "", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            key = f"_normalized_{column}"
            result[key] = normalized
            key_columns.append(key)
        has_address = result["_normalized_street"].ne("") & result["_normalized_city"].ne("")
        result["flag_duplicate_address"] = False
        result.loc[has_address, "flag_duplicate_address"] = result.loc[
            has_address, key_columns
        ].duplicated(keep=False)
        result = result.drop(columns=key_columns)
    else:
        result["flag_duplicate_address"] = False
    return result


def detect_outliers(frame: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    """Add plausibility/outlier flags; these flags do not delete or cap values."""
    result = frame.copy()
    if "list_price" in result:
        result["flag_price_nonpositive"] = (
            result["list_price"].notna() & result["list_price"].le(0)
        )
        result["flag_price_above_cap"] = (
            result["list_price"].notna() & result["list_price"].gt(config.max_list_price)
        )
    if "sqft" in result:
        result["flag_sqft_outside_range"] = result["sqft"].notna() & ~result[
            "sqft"
        ].between(config.min_sqft, config.max_sqft)
    if "lot_sqft" in result:
        result["flag_lot_sqft_outside_range"] = result["lot_sqft"].notna() & ~result[
            "lot_sqft"
        ].between(config.min_lot_sqft, config.max_lot_sqft)
    if "beds" in result:
        result["flag_beds_outside_range"] = result["beds"].notna() & ~result[
            "beds"
        ].between(config.min_beds, config.max_beds)
        result["flag_beds_noninteger"] = (
            result["beds"].notna() & result["beds"].mod(1).ne(0)
        )
    if "baths" in result:
        result["flag_baths_outside_range"] = result["baths"].notna() & ~result[
            "baths"
        ].between(config.min_baths, config.max_baths)
    if "year_built" in result:
        max_year = pd.Timestamp(config.as_of_date).year
        result["flag_year_built_outside_range"] = (
            result["year_built"].notna()
            & ~result["year_built"].between(config.min_year_built, max_year)
        )
        result["flag_year_built_noninteger"] = (
            result["year_built"].notna() & result["year_built"].mod(1).ne(0)
        )
    if {"list_price", "sqft"}.issubset(result.columns):
        price_per_sqft = result["list_price"].div(result["sqft"].where(result["sqft"].gt(0)))
        result["price_per_sqft"] = price_per_sqft
        result["flag_price_per_sqft_outlier"] = (
            price_per_sqft.notna()
            & ~price_per_sqft.between(
                config.min_price_per_sqft, config.max_price_per_sqft
            )
        )
    return result


def add_quality_flags(frame: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    """Add null, historical-sale consistency, and aggregate quality indicators."""
    result = frame.copy()
    for column in (
        "list_price",
        "sqft",
        "lot_sqft",
        "beds",
        "baths",
        "year_built",
        "last_sold_price",
        "last_sold_date",
    ):
        if column in result:
            result[f"flag_{column}_missing"] = result[column].isna()

    if {"last_sold_price", "last_sold_date"}.issubset(result.columns):
        price_missing = result["last_sold_price"].isna()
        date_missing = result["last_sold_date"].isna()
        result["flag_last_sold_pair_mismatch"] = price_missing.ne(date_missing)
        result["flag_last_sold_nonpositive"] = (
            result["last_sold_price"].notna() & result["last_sold_price"].le(0)
        )
        snapshot_date = pd.Timestamp(config.as_of_date)
        result["flag_last_sold_after_snapshot"] = result["last_sold_date"].gt(
            snapshot_date
        )

    flag_columns = [
        column for column in result if isinstance(column, str) and column.startswith("flag_")
    ]
    result["flag_any_quality_issue"] = result[flag_columns].fillna(False).any(axis=1)
    return result


def build_filtered_cohort(
    frame: pd.DataFrame, config: CleaningConfig
) -> pd.DataFrame:
    """Mark membership of the single comparable cohort and return its filtered view."""
    as_of_year = pd.Timestamp(config.as_of_date).year
    include_cohort = (
        frame["list_price"].between(config.min_list_price, config.max_list_price)
        & frame["sqft"].between(config.min_sqft, config.max_sqft)
        & frame["beds"].between(config.min_beds, config.max_beds)
        & frame["beds"].mod(1).eq(0)
        & frame["baths"].between(config.min_baths, config.max_baths)
        & frame["year_built"].between(config.min_year_built, as_of_year)
        & frame["year_built"].mod(1).eq(0)
    )
    frame["include_cohort"] = include_cohort
    cohort = frame.loc[include_cohort].copy()

    assert cohort["property_type"].eq("single_family").all()
    assert cohort["list_price"].between(config.min_list_price, config.max_list_price).all()
    assert cohort["sqft"].between(config.min_sqft, config.max_sqft).all()
    assert cohort["beds"].between(config.min_beds, config.max_beds).all()
    assert cohort["beds"].mod(1).eq(0).all()
    assert cohort["baths"].between(config.min_baths, config.max_baths).all()
    assert cohort["year_built"].between(config.min_year_built, as_of_year).all()
    assert cohort["year_built"].mod(1).eq(0).all()
    LOGGER.info("Comparable cohort contains %s records", len(cohort))
    return cohort


def show_visualizations(
    cohort: pd.DataFrame,
    nulls: pd.DataFrame,
    show: bool = True,
) -> list[Figure]:
    """Display reproducible EDA charts for the comparable cohort."""
    if cohort.empty:
        raise ValueError("Cannot create EDA charts when the filtered cohort is empty")
    plt.style.use(CHART_STYLE)
    figures: list[Figure] = []

    def finish_chart(fig: Figure) -> None:
        fig.tight_layout()
        figures.append(fig)

    top_nulls = nulls.head(20).sort_values("null_rate")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(top_nulls.index.astype(str), top_nulls["null_rate"] * 100, color="#e07a5f")
    ax.set(title="Porcentaje de valores nulos por columna", xlabel="% nulos")
    ax.set_xlim(0, 100)
    finish_chart(fig)

    price_limit = float(cohort["list_price"].quantile(0.99))
    bins: list[float] = np.linspace(0, price_limit, 45).tolist()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(cohort["list_price"], bins=bins, alpha=0.65, color="#3d5a80")
    ax.set(
        title="Distribución de precio anunciado (list_price)",
        xlabel="Precio anunciado (USD; vista hasta el percentil 99 del cohorte)",
        ylabel="Número de viviendas",
    )
    finish_chart(fig)

    scatter = cohort.loc[
        cohort["list_price"].notna() & cohort["sqft"].notna()
    ].sample(n=min(8_000, len(cohort)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 7))
    points = ax.scatter(
        scatter["sqft"],
        scatter["list_price"],
        c=scatter["beds"],
        cmap="viridis",
        alpha=0.35,
        s=14,
    )
    fig.colorbar(points, ax=ax, label="Habitaciones")
    ax.set(
        title="Precio anunciado frente a superficie",
        xlabel="Superficie habitable (ft²)",
        ylabel="Precio anunciado (USD)",
    )
    finish_chart(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, column, title, color in (
        (axes[0], "beds", "Habitaciones", "#3d5a80"),
        (axes[1], "baths", "Baños (incluye medios baños)", "#98c1d9"),
    ):
        counts = cohort[column].value_counts().sort_index()
        ax.bar(
            counts.index.astype(str),
            counts.to_numpy(dtype=np.float64),
            color=color,
        )
        ax.set(title=title, xlabel="Número", ylabel="Viviendas")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Distribución de habitaciones y baños")
    finish_chart(fig)

    decades = (cohort["year_built"] // 10 * 10).astype("Int64")
    decade_counts = decades.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        decade_counts.index.astype(str),
        decade_counts.to_numpy(dtype=np.float64),
        color="#81b29a",
    )
    ax.set(
        title="Viviendas por década de construcción",
        xlabel="Década",
        ylabel="Número de viviendas",
    )
    ax.tick_params(axis="x", rotation=45)
    finish_chart(fig)

    city_summary = (
        cohort.dropna(subset=["city", "list_price"])
        .groupby("city", observed=True)
        .agg(median_price=("list_price", "median"), listings=("list_price", "size"))
        .sort_values("listings", ascending=False)
        .head(15)
        .sort_values("median_price")
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(city_summary.index.astype(str), city_summary["median_price"], color="#e07a5f")
    ax.bar_label(
        bars,
        labels=[f"{int(value):,} anuncios" for value in city_summary["listings"]],
        padding=4,
        fontsize=8,
    )
    ax.set(
        title="Precio anunciado mediano por ciudad",
        xlabel="Mediana de list_price (USD); etiqueta = número de anuncios",
        ylabel="15 ciudades con más viviendas del cohorte",
    )
    finish_chart(fig)

    status_counts = cohort["status"].fillna("Sin dato").value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        status_counts.index.astype(str),
        status_counts.to_numpy(dtype=np.float64),
        color="#3d5a80",
    )
    ax.set(title="Viviendas por estado del anuncio", xlabel="Número de viviendas", ylabel="Estado")
    finish_chart(fig)

    if show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)
    return figures


def export_output(frame: pd.DataFrame, output_path: Path) -> None:
    """Write all flagged single-family homes and cohort indicators to one CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    LOGGER.info("Exported %s rows to %s", len(frame), output_path)


def prepare_single_family(
    frame: pd.DataFrame, config: CleaningConfig
) -> pd.DataFrame:
    """Apply type normalization and quality annotation, without dropping records."""
    result = normalize_types(frame)
    result = result.loc[result["property_type"].eq("single_family")].copy()
    result = detect_duplicates(result)
    result = detect_outliers(result, config)
    return add_quality_flags(result, config)


def run_pipeline(
    input_path: Path,
    output_path: Path,
    show_plots: bool = True,
    config: CleaningConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full input-to-one-CSV workflow, then display EDA charts."""
    cleaning_config = config or CleaningConfig()
    raw = load_data(input_path)
    run_eda(raw)
    family = prepare_single_family(raw, cleaning_config)
    nulls = report_nulls(family)
    LOGGER.info("Single-family null report:\n%s", nulls.head(15).to_string())
    cohort = build_filtered_cohort(family, cleaning_config)
    export_output(family, output_path)
    show_visualizations(cohort, nulls, show=show_plots)
    return family, cohort, nulls


def run_self_tests() -> None:
    """Run focused assertions on known synthetic input without reading the full CSV."""
    sample = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "property_type": ["single_family"] * 3,
            "street": ["1 Main St.", "1 main st", "3 Pine Rd"],
            "unit": [None, None, None],
            "city": ["Miami", "Miami", "Miami"],
            "zip": ["33101"] * 3,
            "parcel_apn": ["P1", "P1", "P3"],
            "list_price": ["300000", "0", "450000"],
            "last_sold_price": ["200000", None, "0"],
            "last_sold_date": ["2020-01-01", None, "2027-01-01"],
            "sqft": ["1500", "1500", "0"],
            "lot_sqft": ["5000", "5000", "1000"],
            "beds": ["3", "3", "0"],
            "baths": ["2", "2", "0"],
            "year_built": ["1980", "1980", "2028"],
            "source_count": ["1", "1", "1"],
            "lat": [25.7, 25.7, 25.7],
            "lng": [-80.2, -80.2, -80.2],
        }
    )
    config = CleaningConfig()
    prepared = prepare_single_family(sample, config)
    cohort = build_filtered_cohort(prepared, config)
    assert len(prepared) == 3
    assert prepared["flag_duplicate_parcel_apn"].iloc[:2].all()
    assert prepared["flag_duplicate_address"].iloc[:2].all()
    assert prepared["flag_last_sold_nonpositive"].iloc[2]
    assert prepared["flag_last_sold_after_snapshot"].iloc[2]
    assert prepared["flag_sqft_outside_range"].iloc[2]
    assert len(cohort) == 1
    assert cohort["list_price"].iloc[0] == 300000
    LOGGER.info("Self-tests passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic assertions only")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Create and close charts without opening interactive plot windows",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.self_test:
        run_self_tests()
        return

    run_pipeline(
        args.input,
        args.output,
        show_plots=not args.no_show,
    )


if __name__ == "__main__":
    main()
