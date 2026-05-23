import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import random
import config
import findspark
findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("BinaryJoins").getOrCreate()
sc    = spark.sparkContext

N = config.PART_B_N
D = config.PART_B_D

# Same seed as ternary.py so result counts are directly comparable
random.seed(42)
A_data = [(i,                        random.randint(0, D - 1)) for i in range(N)]
B_data = [(random.randint(0, D - 1), random.randint(0, D - 1)) for _ in range(N)]
C_data = [(random.randint(0, D - 1), i)                        for i in range(N)]

rdd_A = sc.parallelize(A_data)
rdd_B = sc.parallelize(B_data)
rdd_C = sc.parallelize(C_data)

# -----------------------------------------------------------------------------
# Two-round hash join strategy
# -----------------------------------------------------------------------------
# Round 1 computes AB = A ⋈ B on attribute y. Round 2 joins the intermediate
# AB with C on attribute z. Each round is a textbook symmetric hash join:
# tag tuples with their source relation, key by the join attribute, group,
# cross-product at the reducer.
#
# The price of this simplicity is two shuffles. The second one is especially
# costly when selectivity is high: the intermediate AB can be MUCH larger
# than either A or B (up to |A|*|B|/D tuples for a uniformly-distributed
# join attribute of domain D), and every single one of those intermediate
# tuples has to be re-shuffled in round 2. The ternary Hypercube approach
# avoids this by doing both joins at once inside a single reducer.
# -----------------------------------------------------------------------------

# --- Round 1: A(x,y) ⋈ B(y,z) → AB(x, y, z) ---
# Key on y, tag tuples by source, cross-product at reducer.

def map_A_r1(t):
    x, y = t
    return (y, ('A', x))

def map_B_r1(t):
    y, z = t
    return (y, ('B', z))

def reduce_AB(record):
    y, values = record
    A_xs = [v[1] for v in values if v[0] == 'A']
    B_zs = [v[1] for v in values if v[0] == 'B']
    return [(x, y, z) for x in A_xs for z in B_zs]

# --- Round 2: AB(x,y,z) ⋈ C(z,w) → (x, y, z, w) ---
# Key on z, same pattern as Round 1.

def map_AB_r2(t):
    x, y, z = t
    return (z, ('AB', x, y))

def map_C_r2(t):
    z, w = t
    return (z, ('C', w))

def reduce_ABC(record):
    z, values = record
    AB_pairs = [(v[1], v[2]) for v in values if v[0] == 'AB']
    C_ws     = [v[1]         for v in values if v[0] == 'C']
    return [(x, y, z, w) for (x, y) in AB_pairs for w in C_ws]

# --- Correctness check ---
def brute_force(A, B, C):
    return {(x, yb, zb, w) for x, ya in A for yb, zb in B if ya == yb for zc, w in C if zc == zb}

random.seed(0)
A_s = [(i, random.randint(0, 4)) for i in range(10)]
B_s = [(random.randint(0, 4), random.randint(0, 4)) for _ in range(10)]
C_s = [(random.randint(0, 4), i) for i in range(10)]
ref = brute_force(A_s, B_s, C_s)

AB_small = (sc.parallelize(A_s).map(map_A_r1)
              .union(sc.parallelize(B_s).map(map_B_r1))
              .groupByKey().mapValues(list).flatMap(reduce_AB))

mr_out = (AB_small.map(map_AB_r2)
            .union(sc.parallelize(C_s).map(map_C_r2))
            .groupByKey().mapValues(list).flatMap(reduce_ABC).collect())

if set(mr_out) == ref:
    print(f"Correctness check passed — {len(ref)} tuples match reference.\n")
else:
    print(f"Correctness check failed!\n  Missing: {ref - set(mr_out)}\n  Extra: {set(mr_out) - ref}\n")

# --- Full run ---
print(f"Full run: N={N}, D={D}")

AB_rdd = (rdd_A.map(map_A_r1)
               .union(rdd_B.map(map_B_r1))
               .groupByKey().mapValues(list).flatMap(reduce_AB))

start_time   = time.time()
result_rdd   = (AB_rdd.map(map_AB_r2)
                      .union(rdd_C.map(map_C_r2))
                      .groupByKey().mapValues(list).flatMap(reduce_ABC))
result_count = result_rdd.count()
end_time     = time.time()

print(f"\nSuccess: {result_count} result tuples.")
print(f"Execution Time: {end_time - start_time:.4f} seconds")

print("\nSample results (x, y, z, w):")
for row in result_rdd.take(5):
    print(f"  {row}")

spark.stop()
