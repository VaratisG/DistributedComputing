import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import random
import config
import findspark
findspark.init()

from pyspark.sql import SparkSession, Row

spark = SparkSession.builder.appName("TernaryJoin_SQL").getOrCreate()

N = config.PART_B_N
D = config.PART_B_D

# Same seed as ternary.py and binary.py for a fair comparison
random.seed(42)
A_data = [(i,                        random.randint(0, D - 1)) for i in range(N)]
B_data = [(random.randint(0, D - 1), random.randint(0, D - 1)) for _ in range(N)]
C_data = [(random.randint(0, D - 1), i)                        for i in range(N)]

# Build DataFrames with named columns and register as SQL views
df_A = spark.createDataFrame([Row(x=x, y=y) for x, y in A_data])
df_B = spark.createDataFrame([Row(y=y, z=z) for y, z in B_data])
df_C = spark.createDataFrame([Row(z=z, w=w) for z, w in C_data])

df_A.createOrReplaceTempView("A")
df_B.createOrReplaceTempView("B")
df_C.createOrReplaceTempView("C")

# Deliberately NO join hints here. The point of this script is to see what
# Spark's Catalyst optimiser picks on its own for a three-way chain join.
# With three similarly-sized relations it typically chooses a SortMergeJoin
# for each edge: each relation is hash-partitioned by its join attribute
# (exchange) and then merged in a single pass. This avoids full replication
# of any relation and benefits from Tungsten's code-generated binary layout,
# which is why SQL is consistently the fastest Part B approach on our setup.
query = """
    SELECT A.x, A.y, B.z, C.w
    FROM   A
    JOIN   B ON A.y = B.y
    JOIN   C ON B.z = C.z
"""

start_time   = time.time()
result_df    = spark.sql(query)
result_count = result_df.count()
end_time     = time.time()

print(f"\nSuccess: {result_count} result tuples.")
print(f"Execution Time: {end_time - start_time:.4f} seconds")

print("\nSample results (x, y, z, w):")
for row in result_df.take(5):
    print(f"  {row}")

print("\n--- Catalyst Physical Plan ---")
result_df.explain()

spark.stop()
