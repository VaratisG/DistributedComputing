import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import config
import findspark
findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AllPairs_Naive").getOrCreate()
sc = spark.sparkContext

rdd, total_elements = config.get_part_a_dataset(spark, sc)

# -----------------------------------------------------------------------------
# Naive All-Pairs MapReduce
# -----------------------------------------------------------------------------
# Strategy: dedicate one reducer to every unique unordered pair (i, j).
# Each element therefore has to reach exactly N-1 reducers — one per possible
# partner — so the mapper emits N-1 messages per element. This gives:
#
#     replication rate : N-1          (messages per element)
#     total messages   : N * (N-1)    → O(N²)  total shuffle traffic
#     reducer count    : N*(N-1)/2    → one per unordered pair
#
# The key (min(i,j), max(i,j)) canonicalises the pair so both ends land on
# the same reducer. At moderate N (~2,000) the shuffle already dominates
# wall-clock time; beyond that, socket timeouts start to appear. The whole
# point of this script is to make that failure mode visible so the
# group-based and SQL approaches have a concrete baseline to beat.
# -----------------------------------------------------------------------------
def naive_mapper(element):
    emissions = []
    for j in range(total_elements):
        if element < j:
            emissions.append(((element, j), element))
        elif element > j:
            emissions.append(((j, element), element))
    return emissions

print("Starting Naïve MapReduce Job...")
start_time = time.time()

# Pipeline stages:
#   1. flatMap    — every element fans out into N-1 (key, value) messages.
#   2. groupByKey — Spark shuffles all messages sharing a pair-key to the
#                   same reducer (this is where the O(N²) traffic happens).
#   3. mapValues  — materialise the reducer's input as a concrete list so we
#                   can inspect it later; without this the iterable is lazy.
#   4. count      — triggers the whole DAG and returns the number of pairs.
mapped_rdd  = rdd.flatMap(naive_mapper)
reduced_rdd = mapped_rdd.groupByKey().mapValues(list)
pair_count  = reduced_rdd.count()

end_time = time.time()

print(f"Success: Generated {pair_count} unique pairs.")
print(f"Execution Time: {end_time - start_time:.4f} seconds")

print("\nSample reducer inputs (key -> [elem_a, elem_b]):")
for record in reduced_rdd.take(3):
    print(record)

spark.stop()
