import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import config
import findspark
findspark.init()

from pyspark.sql import SparkSession
from itertools import combinations

spark = SparkSession.builder.appName("AllPairs_AfratiUllman").getOrCreate()
sc = spark.sparkContext

rdd, total_elements = config.get_part_a_dataset(spark, sc)

# Capture locals so closures don't force a config import on executors.
# This is the same cloudpickle workaround used in group.py.
FANO_N       = config.FANO_N
FANO_MAPPING = config.FANO_MAPPING

# The Fano Plane construction is mathematically defined ONLY for 7 elements
# — this is the unique projective plane of order 2. If the dataset happens
# to be larger (e.g. LIMIT_N=500 from person.csv), we deliberately clamp it
# down to the first 7 records to preserve the design's optimality guarantees.
# Running the algorithm on anything other than exactly 7 elements would
# silently corrupt the pair-coverage invariant.
if total_elements > FANO_N:
    print(f"Clamping: Fano Plane requires {FANO_N} elements. Taking first {FANO_N}.")
    rdd = sc.parallelize(rdd.take(FANO_N))

# Afrati–Ullman mapper: look up the element's 3 target reducers in the
# precomputed Fano mapping and emit one (reducer_id, element) message per
# target. Because the Fano design has replication rate r=3, each element is
# sent to exactly 3 reducers; each pair of elements meets in exactly one of
# them. This is provably optimal — no design with r<3 can cover C(7,2)=21
# pairs in a single MapReduce round.
def afrati_ullman_mapper(element):
    return [(r_id, element) for r_id in FANO_MAPPING.get(element, [])]

def afrati_ullman_reducer(record):
    r_id, elements = record
    return [(f"Reducer {r_id}", min(a, b), max(a, b)) for a, b in combinations(elements, 2)]

start_time = time.time()

mapped_rdd  = rdd.flatMap(afrati_ullman_mapper)
reduced_rdd = mapped_rdd.groupByKey().mapValues(list).flatMap(afrati_ullman_reducer)

results = reduced_rdd.collect()
results.sort(key=lambda x: (x[1], x[2]))

end_time = time.time()

print(f"\nSuccess: Generated {len(results)} unique pairs.")
print(f"Execution Time: {end_time - start_time:.4f} seconds\n")

print("--- Pair distribution across reducers ---")
for res in results:
    print(res)

spark.stop()
