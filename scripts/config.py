"""
Settings shared by the subgraph-dataset scripts (download, build, stats).

Deviation from instructions/DATASETS_TASK.md: edges are a kNN graph, not a
cosine threshold. That is how the pregnancy graph was built
(data/pregnancy_MiniLM_L6/k_20, average degree 31.88), and raw MiniLM cosines
are so bunched up (~0.98 between random nodes) that one fixed threshold does not
carry over between datasets. The "threshold sensitivity" table is therefore a
k sensitivity table.
"""

from os.path import dirname, abspath, join

ROOT = dirname(dirname(abspath(__file__)))
RAW_DIR = join(ROOT, "data", "raw")
PROCESSED_DIR = join(ROOT, "data", "processed")
REPORTS_DIR = join(ROOT, "reports")
CSV_DIR = join(REPORTS_DIR, "csv")
FIGURES_DIR = join(REPORTS_DIR, "figures")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64  # faster than 256 on this CPU (62 vs 43 texts/s)

# Main edge rule: each node links to its K most similar other nodes (cosine),
# then the edge list is made undirected and deduplicated - same as k_20.
K = 20
EXTRA_KS = [5, 10, 40]
# kNN search: exact FAISS inner product up to this many nodes, IVF above it
# (the build records which one was used and its recall on a sample)
EXACT_MAX_NODES = 250_000
IVF_NPROBE = 64
# Average degree of the pregnancy k=20 graph, for comparison in the report.
PREGNANCY_AVG_DEGREE = 31.88

SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SCALING_FRACTIONS = [0.10, 0.25, 0.50, 1.00]
