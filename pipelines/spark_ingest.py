"""
delta_feature_job.py
-----------------------------------------
Transforms Delta tables from the ingestion stage into
a unified feature store for ML training.

Automatically detects any fallback tables written by spark_ingest.py
if AUTO_FALLBACK=True.

Repo: clinical-lakehouse-mlops
-----------------------------------------
"""

import os
from pyspark.sql import SparkSession, functions as F

# ============================================================
# Configuration
# ============================================================
DELTA_PATH = "delta"
FEATURES_PATH = "delta/features"
AUTO_FALLBACK = True  # match spark_ingest.py

# Core modeling tables
CORE_TABLES = [
    "patients",
    "encounters",
    "conditions",
    "observations",
    "procedures",
    "medications",
    "claims",
]


def initialize_spark() -> SparkSession:
    """Create a Spark session with Delta support."""
    spark = (
        SparkSession.builder.appName("feature_engineering")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def table_exists(table_name: str) -> bool:
    """Check if a Delta table path exists."""
    return os.path.exists(os.path.join(DELTA_PATH, table_name))


def safe_read_delta(spark: SparkSession, table: str):
    """Read a Delta table safely; return empty DF if missing."""
    try:
        path = os.path.join(DELTA_PATH, table)
        return spark.read.format("delta").load(path)
    except Exception:
        print(f"Skipping {table} (not found or unreadable)")
        return spark.createDataFrame([], schema=None)


def main():
    spark = initialize_spark()
    os.makedirs(FEATURES_PATH, exist_ok=True)

    print("Starting feature store build...")

    # Load core tables
    dfs = {}
    for t in CORE_TABLES:
        if table_exists(t):
            dfs[t] = spark.read.format("delta").load(os.path.join(DELTA_PATH, t))
            print(f"Loaded {t}")
        else:
            print(f"Missing {t}, skipping this table")

    # Use encounters as fact table
    if "encounters" not in dfs:
        print("Cannot build features: encounters table missing.")
        return

    features = dfs["encounters"]

    # Join patient demographics
    if "patients" in dfs:
        features = features.join(
            dfs["patients"],
            dfs["patients"]["Id"] == features["PATIENT"],
            "left"
        ).drop(dfs["patients"]["Id"])
        print("Joined patients")

    # Join claims (economic features)
    if "claims" in dfs and "amount" in dfs["claims"].columns:
        claims = dfs["claims"].groupby("PATIENT").agg(
            F.count("*").alias("total_claims"),
            F.sum(F.col("amount")).alias("total_claim_amount")
        )
        features = features.join(claims, "PATIENT", "left")
        print("Joined claims summary")

    # Join observations for numeric summaries
    if "observations" in dfs and "VALUE" in dfs["observations"].columns:
        obs_summary = (
            dfs["observations"]
            .filter(F.col("VALUE").cast("double").isNotNull())
            .groupby("PATIENT")
            .agg(F.avg(F.col("VALUE").cast("double")).alias("avg_lab_value"))
        )
        features = features.join(obs_summary, "PATIENT", "left")
        print("Joined observation summaries")

    # Derived features
    features = (
        features.withColumn(
            "encounter_length_days",
            (F.unix_timestamp("STOP") - F.unix_timestamp("START")) / 86400.0
        )
        .withColumn(
            "is_readmitted_30d",
            F.when(F.col("encounter_length_days") < 30, 1).otherwise(0)
        )
    )

    # Auto-fallback: include any additional Delta tables dynamically
    if AUTO_FALLBACK:
        delta_dirs = [
            d for d in os.listdir(DELTA_PATH)
            if os.path.isdir(os.path.join(DELTA_PATH, d))
        ]
        extra_tables = [
            t for t in delta_dirs if t not in CORE_TABLES and t not in ["features"]
        ]
        if extra_tables:
            print(f"Auto-joining fallback Delta tables: {extra_tables}")
            for t in extra_tables:
                df_extra = safe_read_delta(spark, t)
                common_cols = set(features.columns) & set(df_extra.columns)
                if "PATIENT" in common_cols:
                    features = features.join(df_extra, "PATIENT", "left")
                elif "Id" in common_cols:
                    features = features.join(df_extra, "Id", "left")
                else:
                    print(f"Skipping join for {t}: no patient key found")

    features.write.format("delta").mode("overwrite").save(FEATURES_PATH)
    print(f"Feature store built at {FEATURES_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
