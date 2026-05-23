"""
Central configuration module for the entire Spark project.

Every individual script (Part A, Part B, bonus, and benchmark) imports this
module to obtain its runtime settings. Centralising these values in one place
means experiments can be re-parameterised by editing a single file, without
touching the algorithmic code that implements each MapReduce job.

Section roadmap:
    1. System environment variables (Java / Hadoop / Spark / Python paths)
    2. Run-mode feature flag (synthetic vs real CSV)
    3. Part A variables (dataset slice, group count)
    4. Part A data-loader factory (returns the same interface in both modes)
    5. Bonus / Fano Plane constants
    6. Part B variables (hypercube side length, relation size, domain size)
    7. Benchmark test matrices consumed by src/benchmark.py
"""

import os
import sys

# --- 1. System Environment Variables ---------------------------------------
# These are set BEFORE any Spark import so that findspark / pyspark locate
# the correct JVM, Hadoop winutils, and Python interpreter. On Windows, Spark
# refuses to start unless HADOOP_HOME points at a directory containing
# winutils.exe, and PYSPARK_PYTHON must match the driver interpreter so the
# executors serialise/deserialise objects against the same Python version.
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-11.0.0.2'
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['SPARK_HOME'] = r'C:\spark'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# --- 2. Run Mode (Feature Flag) --------------------------------------------
# TEST_MODE exists so the algorithmic scripts can be exercised against a
# tiny deterministic integer range (MOCK_N) without touching disk I/O. This
# is invaluable for debugging mapper / reducer logic — a stack trace on 100
# elements is much easier to reason about than one on 2,000. 
TEST_MODE = False
DATASET_PATH = r"C:\Spark_Project\Dataset\person.csv"  # Only used if TEST_MODE is False

# --- 3. Part A Variables ----------------------------------------------------
# MOCK_N   — size of the synthetic integer RDD used when TEST_MODE is True.
# G        — number of groups for the group-based all-pairs MapReduce.
#            The number of reducers is G*(G+1)/2, so G=10 yields 55 reducers.
# LIMIT_N  — number of rows to slice off person.csv when running the
#            individual (non-benchmark) Part A scripts.
MOCK_N = 100
G = 10
LIMIT_N = 500

# --- 4. The Data Loader Factory --------------------------------------------
def get_part_a_dataset(spark, sc):
    """
    Returns (rdd, N) for Part A, abstracting away the synthetic-vs-real choice.

    Person.csv is read, sliced to LIMIT_N rows, and then each
    row is reassigned a contiguous integer ID via ``zipWithIndex()``. The
    reassignment is essential: the real CSV rows carry arbitrary string IDs
    on which ``element % G`` would be meaningless, so we replace them with
    0..N-1 before any modular arithmetic happens downstream.
    """
    if TEST_MODE:
        print(f"Test Mode: Generating mock dataset (N={MOCK_N})...")
        rdd = sc.parallelize(range(MOCK_N))
        return rdd, MOCK_N
    else:
        print(f"Dataset Mode: Loading real dataset from {DATASET_PATH}...")
        df = spark.read.csv(DATASET_PATH, header=True, inferSchema=True)
        # .limit(N) materialises only the first N rows
        df_limited = df.limit(LIMIT_N)
        rdd = df_limited.rdd.zipWithIndex().map(lambda x: x[1])
        return rdd, LIMIT_N

# --- 5. (Bonus: Afrati-Ullman / Fano Plane) --------------------------------
# The Fano Plane is the unique projective plane of order 2: 7 points arranged
# on 7 lines such that every pair of points lies on exactly one line. We use
# it as an optimal combinatorial block design for the All-Pairs problem at
# N=7 with replication rate r=3 — proven optimal by Afrati & Ullman. Each
# key in FANO_MAPPING is an element id 0..6; its value is the list of the
# three reducer ids ("lines") the element must be sent to. Because every
# pair of elements shares exactly one reducer, every unordered pair is
# generated EXACTLY once, with zero redundancy.
FANO_N = 7
FANO_MAPPING = {
    0: [0, 4, 6],
    1: [0, 1, 5],
    2: [1, 2, 6],
    3: [0, 2, 3],
    4: [1, 3, 4],
    5: [2, 4, 5],
    6: [3, 5, 6]
}

# --- 6. Part B Variables ----------------------------------------------------
# PART_B_N — number of tuples in each of the three synthetic relations A/B/C.
# PART_B_D — domain size for the join attributes y and z. Smaller D means
#            more key collisions and therefore larger join output — this is
#            the selectivity knob for the Part B benchmarks.
# PART_B_K — side length of the hypercube grid used by the ternary join.
#            The algorithm uses K² reducers, so K=10 yields 100 reducers.
PART_B_N = 200
PART_B_D = 30
PART_B_K = 10

# --- 7. Benchmark Test Matrix ----------------------------------------------
# These lists drive src/benchmark.py. Putting the test matrix in config keeps
# the benchmark runner itself generic: it iterates whatever sizes it finds
# here, so adding a new data point is a one-line change.
BENCH_A_SIZES     = [500, 1000, 2000, 5000]   # N values to test for Part A
BENCH_A_NAIVE_MAX = 5000                # skip naive above this — its O(N²)
                                        # shuffle cost makes larger runs
                                        # unproductive on a single node.

# Selectivity sweep: fix N=5000, vary D to change collision density.
BENCH_B_SELECTIVITY = [
    (2000, 50),    # Ultra-high density: D=20 means massive collisions and a huge result set.
    (2000, 200),   # High density: Significant matches.
    (2000, 800),   # Low density: Sparse matches, small output.
]
# Scale sweep: fix D=100, grow N to see how each algorithm handles size.
BENCH_B_SCALE = [
    (1000,  100),   # Baseline
    (5000,  100),   # Moderate load
    (10000, 100),   # High load: Should show clear separation in execution time.
]

