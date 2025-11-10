from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.appName("features").getOrCreate()

enc = spark.read.format("delta").load("delta/encounters")
# toy label: readmission within 30d if same patient had another encounter soon
# (placeholder for your real logic)
features = enc.select("*")

features.write.mode("overwrite").format("delta").save("delta/features")
print("Feature engineering complete.")
