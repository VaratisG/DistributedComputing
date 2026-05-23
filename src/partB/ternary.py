import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import random
import config
import findspark
findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TernaryJoin_Hypercube").getOrCreate()
sc    = spark.sparkContext

# Capture locally to avoid config import on executor side
N = config.PART_B_N
D = config.PART_B_D
K = config.PART_B_K

# --- Data generation ---
# Three equal-size relations with overlapping join-attribute domains (0..D-1).
# Same seed across all Part B scripts so results are comparable.
random.seed(42)
A_data = [(i,                        random.randint(0, D - 1)) for i in range(N)]
B_data = [(random.randint(0, D - 1), random.randint(0, D - 1)) for _ in range(N)]
C_data = [(random.randint(0, D - 1), i)                        for i in range(N)]

rdd_A = sc.parallelize(A_data)
rdd_B = sc.parallelize(B_data)
rdd_C = sc.parallelize(C_data)

# --- Hypercube / Shares algorithm ------------------------------------------
# Afrati–Ullman "Shares" routing for a three-way chain join in a SINGLE
# MapReduce round. Reducers are laid out on a K×K grid. Each relation is
# mapped onto the grid based on which join attributes it contains:
#
#   A(x,y) knows y  → pinned to row i* = hash(y)%K, replicated across all
#                     K columns. Replication factor K.
#   B(y,z) knows both join keys → lands in exactly one cell
#                     (hash(y)%K, hash(z)%K). No replication.
#   C(z,w) knows z  → pinned to column j* = hash(z)%K, replicated across
#                     all K rows. Replication factor K.
#
# Correctness argument: for any matching triple (A-row, B-tuple, C-row),
# B lives at its unique cell (i*, j*). A is replicated across the entire
# row i*, so it reaches that cell. C is replicated across the entire column
# j*, so it also reaches that cell. All three meet at (i*, j*) and the
# reducer computes the local join with no further shuffle. Total traffic is
# O(N·K) rather than the O(N) + O(intermediate) of two binary rounds, and
# the intermediate AB relation is never materialised on disk.
# ---------------------------------------------------------------------------

def map_A(t):
    x, y = t
    row = hash(y) % K
    return [((row, j), ('A', x, y)) for j in range(K)]

def map_B(t):
    y, z = t
    return [((hash(y) % K, hash(z) % K), ('B', y, z))]

def map_C(t):
    z, w = t
    col = hash(z) % K
    return [((i, col), ('C', z, w)) for i in range(K)]

def reduce_ternary(record):
    _key, tuples = record
    A_list, B_list, C_list = [], [], []
    for item in tuples:
        if   item[0] == 'A': A_list.append((item[1], item[2]))
        elif item[0] == 'B': B_list.append((item[1], item[2]))
        elif item[0] == 'C': C_list.append((item[1], item[2]))

    results = []
    for (y_b, z_b) in B_list:
        for x_a in [x for x, y in A_list if y == y_b]:
            for w_c in [w for z, w in C_list if z == z_b]:
                results.append((x_a, y_b, z_b, w_c))
    return results

# --- Correctness check ---
def brute_force(A, B, C):
    return {(x, yb, zb, w) for x, ya in A for yb, zb in B if ya == yb for zc, w in C if zc == zb}

random.seed(0)
A_s = [(i, random.randint(0, 4)) for i in range(10)]
B_s = [(random.randint(0, 4), random.randint(0, 4)) for _ in range(10)]
C_s = [(random.randint(0, 4), i) for i in range(10)]
ref = brute_force(A_s, B_s, C_s)

mr_out = (sc.parallelize(A_s).flatMap(map_A)
           .union(sc.parallelize(B_s).flatMap(map_B))
           .union(sc.parallelize(C_s).flatMap(map_C))
           .groupByKey().mapValues(list).flatMap(reduce_ternary).collect())

if set(mr_out) == ref:
    print(f"Correctness check passed — {len(ref)} tuples match reference.\n")
else:
    print(f"Correctness check failed!\n  Missing: {ref - set(mr_out)}\n  Extra: {set(mr_out) - ref}\n")

# --- Full run ---
print(f"Full run: N={N}, D={D}, K={K} ({K*K} reducers)")

start_time  = time.time()
result_rdd  = (rdd_A.flatMap(map_A)
               .union(rdd_B.flatMap(map_B))
               .union(rdd_C.flatMap(map_C))
               .groupByKey().mapValues(list).flatMap(reduce_ternary))
result_count = result_rdd.count()
end_time    = time.time()

print(f"\nSuccess: {result_count} result tuples.")
print(f"Execution Time: {end_time - start_time:.4f} seconds")

print("\nSample results (x, y, z, w):")
for row in result_rdd.sortBy(lambda t: (t[0], t[1], t[2], t[3])).take(5):
    print(f"  {row}")

spark.stop()
