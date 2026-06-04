<p align="center">
  <img src="https://spark.apache.org/images/spark-logo-trademark.png" width="350">
</p>

<h1 align="center">🚀 Distributed Data Processing with Apache Spark</h1>

<p align="center">
  MSc Data & Web Science • Aristotle University of Thessaloniki
</p>

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.8-orange?style=for-the-badge&logo=apachespark)
![Java](https://img.shields.io/badge/Java-11-red?style=for-the-badge&logo=openjdk)
![Course](https://img.shields.io/badge/Distributed%20Data%20Processing-2025--2026-success?style=for-the-badge)

**MSc Data & Web Science — Distributed Data Processing 2025-2026**  
**Aristotle University of Thessaloniki**

👨‍💻 **Contributors**

- Georgios Varatis (AM: 219)
- Evangelia Girousi (AM: 229)

---

## 📖 Project Overview

This project implements and benchmarks distributed data processing algorithms using **Apache Spark (PySpark)**.

The project is divided into two parts:

### 🔹 Part A — All-Pairs Matching

Enumerate every unique unordered pair from a set of **N elements** using:

- 🧩 Naive MapReduce
- 📦 Group-Based MapReduce
- ⚡ Spark SQL
- 🏆 Afrati-Ullman / Fano Plane Optimal Design

### 🔹 Part B — Three-Way Relational Join

Compute:

```text
A(x,y) ⋈ B(y,z) ⋈ C(z,w)
```

using:

- 🔺 Direct Ternary Join (Shares / Hypercube)
- 🔗 Two Successive Binary Joins
- ⚡ Spark SQL Join

---

## 🏗️ Project Architecture

```mermaid
flowchart TB

    DATA["📂 Dataset<br/>person.csv"]

    SPARK["⚡ Apache Spark<br/>PySpark Execution Engine"]

    DATA --> SPARK

    subgraph A["🧩 Part A — All-Pairs Matching"]
        A1["Naive MapReduce"]
        A2["Group-Based MapReduce"]
        A3["Spark SQL"]
        A4["Afrati-Ullman<br/>Fano Plane Bonus"]
    end

    subgraph B["🔗 Part B — Three-Way Relational Join"]
        B1["Direct Ternary Join<br/>Shares / Hypercube"]
        B2["Successive Binary Joins"]
        B3["Spark SQL Join"]
    end

    SPARK --> A
    SPARK --> B

    OUTPUT["📊 Benchmark Framework<br/>benchmark.py"]

    A --> OUTPUT
    B --> OUTPUT

    RESULTS["📄 benchmark_output.txt"]

    OUTPUT --> RESULTS
```
---

## 🛠️ Prerequisites

| 📦 Dependency | 🔢 Version |
|--------------|------------|
| Java JDK | 11.0.0.2 |
| Apache Spark | 3.5.8 |
| Hadoop winutils (Windows only) | 3.x |
| Python | 3.11.0 |

### Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

---

## 📂 Project Structure

```text
Spark_Project/
├── config.py                  # Central configuration and data-loader factory
├── requirements.txt           # Python dependencies
├── Dataset/
│   └── person.csv             # Source dataset (up to 1M rows)
└── src/
    ├── benchmark.py           # Full benchmark matrix (Part A + Part B, single Spark session)
    ├── partA/
    │   ├── naive.py           # Naive MapReduce — one reducer per pair
    │   ├── group.py           # Group-Based MapReduce — G=10 bucket groups
    │   ├── sql.py             # Spark SQL with Catalyst optimizer (BROADCAST hint)
    │   └── partA_bonus.py     # Bonus — Afrati-Ullman / Fano Plane optimal design
    └── partB/
        ├── ternary.py         # Direct Ternary Join (Hypercube / Shares, K×K grid)
        ├── binary.py          # Two Successive Binary Joins (two-round hash join)
        └── join_sql.py        # Spark SQL three-way join (Catalyst SortMergeJoin)
```

---

## ⚙️ Configuration

All runtime parameters live in `config.py`. Edit this file to change dataset size, group count, hypercube side length, or benchmark matrices — no changes to the algorithmic scripts are needed.

| Variable | Default | Description |
|---|---|---|
| 🧪 `TEST_MODE` | `False` | `True` uses a synthetic integer RDD (fast debugging); `False` loads `person.csv` |
| 📊 `LIMIT_N` | `500` | Rows loaded from `person.csv` for individual Part A scripts |
| 📦 `G` | `10` | Number of groups for the group-based approach (yields 55 reducers) |
| 🔗 `PART_B_N` | `200` | Tuples per relation for individual Part B scripts |
| 🎯 `PART_B_D` | `30` | Join-attribute domain size for individual Part B scripts |
| 🧊 `PART_B_K` | `10` | Hypercube side length (K² reducers) |
| 📈 `BENCH_A_SIZES` | `[500,1000,2000,5000]` | N values for the Part A benchmark matrix |
| 📉 `BENCH_B_SELECTIVITY` | see config | (N, D) pairs for the selectivity sweep |
| 📏 `BENCH_B_SCALE` | see config | (N, D) pairs for the scale sweep |

---
