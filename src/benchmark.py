"""
src/benchmark.py — unified benchmark runner for Parts A and B.

IMPORTANT — origin of this file
-------------------------------
Unlike every other script in this project, benchmark.py is NOT a hand-written
experiment authored alongside the algorithmic work. It was generated from
scratch by an LLM AFTER all the individual Part A / Part B scripts already
existed, specifically so the full test matrix could be executed end-to-end
from a single Spark session.

Why regenerate the algorithms instead of importing naive.py, group.py,
ternary.py, etc.? Every one of those standalone scripts creates its own
``SparkSession.builder...getOrCreate()`` AT MODULE LOAD TIME. Importing any
of them would therefore spin up (or attach to) a Spark session before this
file has a chance to configure its own, and attempting to run several of
them back-to-back would mean paying the ~15 s JVM startup cost once per
script. Duplicating the six implementations inside this single file was the
pragmatic compromise: one Spark session, one warm JVM, one set of shuffle
directories, and config-driven test matrices from ``config.BENCH_*``.

Design summary
--------------
1. A single SparkSession is created once at the top of the file and reused
   by every implementation below.
2. ``log()`` writes to stdout AND to ``benchmark_output.txt`` simultaneously
   (mode="w" so each run overwrites the previous file).
3. Part A iterates over ``config.BENCH_A_SIZES``, loading person.csv once
   and then applying ``.limit(N)`` per iteration. Naive is skipped above
   ``BENCH_A_NAIVE_MAX`` because its O(N²) shuffle makes bigger runs
   pointless.
4. Part B iterates the selectivity and scale matrices and generates
   synthetic relations from a fixed seed so result counts stay reproducible.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import config
import findspark
findspark.init()

from pyspark.sql import SparkSession, Row
from itertools import combinations, product

spark = SparkSession.builder.appName("Benchmark").getOrCreate()
sc    = spark.sparkContext

# ── Output: write to console and benchmark_output.txt simultaneously ──────────
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_output.txt")
_out = open(OUTPUT_FILE, "w", encoding="utf-8")

def log(text=""):
    print(text)
    _out.write(text + "\n")
    _out.flush()

# Capture config values into plain local ints BEFORE any mapper closure is
# defined. Spark's cloudpickle serialiser would otherwise try to ship the
# `config` module reference to executors, which fails because executors run
# in isolated Python processes without this project's sys.path. Copying the
# value here turns the closure's free variable into an ordinary integer that
# cloudpickle can serialise without issue. (See config.py for the full note.)
G = config.G
K = config.PART_B_K

# ── Part A implementations ────────────────────────────────────────────────────

def run_naive(rdd, N):
    def naive_mapper(element):
        emissions = []
        for j in range(N):
            if element < j:
                emissions.append(((element, j), element))
            elif element > j:
                emissions.append(((j, element), element))
        return emissions

    t0 = time.time()
    count = rdd.flatMap(naive_mapper).groupByKey().mapValues(list).count()
    return count, time.time() - t0


def run_group(rdd, N):
    def group_mapper(element):
        my_group = element % G
        emissions = []
        for other_group in range(G):
            if my_group < other_group:
                emissions.append(((my_group, other_group), element))
            elif my_group > other_group:
                emissions.append(((other_group, my_group), element))
            else:
                emissions.append(((my_group, my_group), element))
        return emissions

    def group_reducer(record):
        reducer_key, elements = record
        group_a_id, group_b_id = reducer_key
        group_a = [e for e in elements if e % G == group_a_id]
        group_b = [e for e in elements if e % G == group_b_id]
        if group_a_id != group_b_id:
            return [(min(a, b), max(a, b)) for a, b in product(group_a, group_b)]
        else:
            return [(min(a, b), max(a, b)) for a, b in combinations(group_a, 2)]

    t0 = time.time()
    count = (rdd.flatMap(group_mapper)
               .groupByKey().mapValues(list)
               .flatMap(group_reducer).count())
    return count, time.time() - t0


def run_sql_a(rdd):
    df = rdd.map(lambda x: Row(id=x)).toDF()
    df.createOrReplaceTempView("dataset")
    t0 = time.time()
    count = spark.sql(
        "SELECT /*+ BROADCAST(a) */ a.id AS element_1, b.id AS element_2 "
        "FROM dataset a JOIN dataset b ON a.id < b.id"
    ).count()
    return count, time.time() - t0


# ── Part B implementations ────────────────────────────────────────────────────

def run_ternary(rdd_A, rdd_B, rdd_C):
    def map_A(t):
        x, y = t
        return [((hash(y) % K, j), ('A', x, y)) for j in range(K)]
    def map_B(t):
        y, z = t
        return [((hash(y) % K, hash(z) % K), ('B', y, z))]
    def map_C(t):
        z, w = t
        return [((i, hash(z) % K), ('C', z, w)) for i in range(K)]
    def reduce_fn(record):
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

    t0 = time.time()
    count = (rdd_A.flatMap(map_A)
             .union(rdd_B.flatMap(map_B))
             .union(rdd_C.flatMap(map_C))
             .groupByKey().mapValues(list).flatMap(reduce_fn).count())
    return count, time.time() - t0


def run_binary(rdd_A, rdd_B, rdd_C):
    def map_A_r1(t):
        x, y = t; return (y, ('A', x))
    def map_B_r1(t):
        y, z = t; return (y, ('B', z))
    def reduce_AB(record):
        y, vals = record
        return [(x, y, z) for x in [v[1] for v in vals if v[0] == 'A']
                           for z in [v[1] for v in vals if v[0] == 'B']]
    def map_AB_r2(t):
        x, y, z = t; return (z, ('AB', x, y))
    def map_C_r2(t):
        z, w = t; return (z, ('C', w))
    def reduce_ABC(record):
        z, vals = record
        return [(x, y, z, w) for x, y in [v[1:] for v in vals if v[0] == 'AB']
                               for w   in [v[1]  for v in vals if v[0] == 'C']]

    AB_rdd = (rdd_A.map(map_A_r1).union(rdd_B.map(map_B_r1))
              .groupByKey().mapValues(list).flatMap(reduce_AB))

    t0 = time.time()
    count = (AB_rdd.map(map_AB_r2).union(rdd_C.map(map_C_r2))
             .groupByKey().mapValues(list).flatMap(reduce_ABC).count())
    return count, time.time() - t0


def run_sql_b(A_data, B_data, C_data):
    df_A = spark.createDataFrame([Row(x=x, y=y) for x, y in A_data])
    df_B = spark.createDataFrame([Row(y=y, z=z) for y, z in B_data])
    df_C = spark.createDataFrame([Row(z=z, w=w) for z, w in C_data])
    df_A.createOrReplaceTempView("A")
    df_B.createOrReplaceTempView("B")
    df_C.createOrReplaceTempView("C")
    t0 = time.time()
    count = spark.sql(
        "SELECT A.x, A.y, B.z, C.w "
        "FROM A JOIN B ON A.y = B.y JOIN C ON B.z = C.z"
    ).count()
    return count, time.time() - t0


# ── Part A benchmark ──────────────────────────────────────────────────────────

log("\n" + "=" * 70)
log("  PART A BENCHMARK — All-Pairs Matching")
log("=" * 70)

# Load CSV once; apply .limit(N) per iteration
df_full = spark.read.csv(config.DATASET_PATH, header=True, inferSchema=True)

results_a = []
for N in config.BENCH_A_SIZES:
    log(f"\n  N = {N}")
    rdd = df_full.limit(N).rdd.zipWithIndex().map(lambda x: x[1])
    rdd.cache()

    if N <= config.BENCH_A_NAIVE_MAX:
        cnt_n, t_n = run_naive(rdd, N)
        naive_str = f"{t_n:7.2f}s"
    else:
        cnt_n, t_n = 0, 0
        naive_str = "skipped"

    cnt_g, t_g = run_group(rdd, N)
    cnt_s, t_s = run_sql_a(rdd)

    rdd.unpersist()

    log(f"    Naive      : {cnt_n:>10} pairs  {naive_str}")
    log(f"    Group G=10 : {cnt_g:>10} pairs  {t_g:7.2f}s")
    log(f"    SQL        : {cnt_s:>10} pairs  {t_s:7.2f}s")
    results_a.append((N, cnt_g, naive_str, t_g, t_s))

# ── Part A summary ────────────────────────────────────────────────────────────
log(f"\n\n{'='*70}")
log("  PART A — SUMMARY")
log(f"{'='*70}")
log(f"  {'N':>6}  {'Pairs':>12}  {'Naive':>9}  {'Group G=10':>10}  {'SQL':>9}")
log(f"  {'-'*6}  {'-'*12}  {'-'*9}  {'-'*10}  {'-'*9}")
for N, cnt, t_n_str, t_g, t_s in results_a:
    log(f"  {N:>6}  {cnt:>12}  {t_n_str:>9}  {t_g:>9.2f}s  {t_s:>8.2f}s")
log(f"{'='*70}\n")


# ── Part B benchmark ──────────────────────────────────────────────────────────

log("\n" + "=" * 70)
log("  PART B BENCHMARK — Three-Way Join A(x,y) Join B(y,z) Join C(z,w)")
log("=" * 70)

selectivity_tests = [
    (f"Selectivity (N={N}, D={D})", N, D)
    for N, D in config.BENCH_B_SELECTIVITY
]
scale_tests = [
    (f"Scale       (N={N}, D={D})", N, D)
    for N, D in config.BENCH_B_SCALE
]

results_b = []
for label, N, D in selectivity_tests + scale_tests:
    log(f"\n  {label}")
    random.seed(42)
    A_data = [(i,                        random.randint(0, D - 1)) for i in range(N)]
    B_data = [(random.randint(0, D - 1), random.randint(0, D - 1)) for _ in range(N)]
    C_data = [(random.randint(0, D - 1), i)                        for i in range(N)]

    rdd_A = sc.parallelize(A_data)
    rdd_B = sc.parallelize(B_data)
    rdd_C = sc.parallelize(C_data)

    cnt_t, t_t = run_ternary(rdd_A, rdd_B, rdd_C)
    cnt_b, t_b = run_binary(rdd_A, rdd_B, rdd_C)
    cnt_s, t_s = run_sql_b(A_data, B_data, C_data)

    log(f"    Ternary : {cnt_t:>10} tuples  {t_t:7.2f}s")
    log(f"    Binary  : {cnt_b:>10} tuples  {t_b:7.2f}s")
    log(f"    SQL     : {cnt_s:>10} tuples  {t_s:7.2f}s")
    results_b.append((label, cnt_t, t_t, t_b, t_s))

# ── Part B summary ────────────────────────────────────────────────────────────
log(f"\n\n{'='*70}")
log("  PART B — SUMMARY")
log(f"{'='*70}")
log(f"  {'Test':<38} {'Tuples':>10}  {'Ternary':>9}  {'Binary':>9}  {'SQL':>9}")
log(f"  {'-'*38} {'-'*10}  {'-'*9}  {'-'*9}  {'-'*9}")
for label, cnt, t_t, t_b, t_s in results_b:
    log(f"  {label:<38} {cnt:>10}  {t_t:>8.2f}s  {t_b:>8.2f}s  {t_s:>8.2f}s")
log(f"{'='*70}\n")

_out.close()
spark.stop()
