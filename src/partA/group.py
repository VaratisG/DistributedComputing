import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import config
import findspark
findspark.init()

from pyspark.sql import SparkSession
from itertools import combinations, product

spark = SparkSession.builder.appName("AllPairs_GroupBased").getOrCreate()
sc = spark.sparkContext

rdd, total_elements = config.get_part_a_dataset(spark, sc)

# Capture G into a plain local int BEFORE defining any mapper/reducer. Spark
# serialises closures with cloudpickle and ships them to executor-side Python
# processes. Those executors do not share the driver's sys.path, so a
# reference to `config.G` inside a closure would force them to re-import the
# config module and fail with ModuleNotFoundError. Copying the value here
# captures the integer directly in the closure and sidesteps the problem.
G = config.G

# -----------------------------------------------------------------------------
# Group-Based All-Pairs MapReduce
# -----------------------------------------------------------------------------
# Strategy: partition elements into G buckets via `element % G` and let each
# reducer handle an entire pair-of-buckets (g_a, g_b). For G=10 there are 55
# reducers (G*(G+1)/2), which is a dramatic reduction from the N*(N-1)/2
# reducers used by the naive version.
#
#     replication rate : G            (every element goes to G reducers)
#     total messages   : N * G        → O(N·G)  — linear in N for fixed G
#
# The mapper below emits one message per target bucket. For mixed-bucket
# pairs it uses a canonical (small, large) key so both halves of the pair
# reach the same reducer; for same-bucket pairs the key is just (g, g).
# -----------------------------------------------------------------------------
def group_mapper(element):
    my_group = element % G
    emissions = []
    for other_group in range(G):
        if my_group < other_group:
            emissions.append(((my_group, other_group), element))
        elif my_group > other_group:
            emissions.append(((other_group, my_group), element))
        else:  # same group
            emissions.append(((my_group, my_group), element))
    return emissions

# Each reducer receives every element belonging to either of its two groups.
# Two cases to handle:
#   * g_a == g_b  → intra-group: emit C(|group|, 2) combinations, never
#                   duplicating or self-pairing an element.
#   * g_a != g_b  → inter-group: emit the full cartesian product between
#                   the two bucket contents.
# The (min, max) canonicalisation guards against duplicate pairs in case
# the same element ends up on both sides via hash quirks.
def group_reducer(record):
    reducer_key, elements = record
    group_a_id, group_b_id = reducer_key

    group_a = [e for e in elements if e % G == group_a_id]
    group_b = [e for e in elements if e % G == group_b_id]

    if group_a_id != group_b_id:
        return [(min(a, b), max(a, b)) for a, b in product(group_a, group_b)]
    else:
        return [(min(a, b), max(a, b)) for a, b in combinations(group_a, 2)]

start_time = time.time()

mapped_rdd  = rdd.flatMap(group_mapper)
reduced_rdd = mapped_rdd.groupByKey().mapValues(list).flatMap(group_reducer)
pair_count  = reduced_rdd.count()

end_time = time.time()

print(f"\nSuccess: Generated {pair_count} unique pairs.")
print(f"Execution Time: {end_time - start_time:.4f} seconds")

spark.stop()
