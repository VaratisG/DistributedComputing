import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import config
import findspark
findspark.init()

from pyspark.sql import SparkSession, Row

spark = SparkSession.builder.appName("AllPairs_SQL").getOrCreate()
sc = spark.sparkContext

rdd, total_elements = config.get_part_a_dataset(spark, sc)

# Wrap every integer in a Row so Spark can infer a schema (id:BIGINT) and
# register the resulting DataFrame as a temporary view. Spark SQL operates on
# views, not raw RDDs, so this is the minimum viable adapter layer.
df = rdd.map(lambda x: Row(id=x)).toDF()
df.createOrReplaceTempView("dataset")

# The join predicate `a.id < b.id` does two things at once:
#   1. Filters out identical-element pairs (a==b).
#   2. Canonicalises (x, y) vs (y, x) so each unordered pair appears ONCE.
# The /*+ BROADCAST(a) */ hint asks Catalyst to ship the 'a' side to every
# executor as an in-memory hash table. Without it Catalyst falls back to a
# SortMergeJoin which performs a full shuffle — comparable in cost to the
# naive MapReduce pipeline. The broadcast hash join eliminates that shuffle
# entirely and is the reason this approach is effectively constant-time
# across all dataset sizes we tested.
query = """
    SELECT /*+ BROADCAST(a) */ a.id AS element_1, b.id AS element_2
    FROM dataset a
    JOIN dataset b ON a.id < b.id
"""

start_time = time.time()
result_df  = spark.sql(query)
pair_count = result_df.count()
end_time   = time.time()

print(f"\nSuccess: Generated {pair_count} unique pairs from {total_elements} elements.")
print(f"Execution Time: {end_time - start_time:.4f} seconds\n")

print("--- Catalyst Optimizer Physical Plan ---")
result_df.explain()

spark.stop()
