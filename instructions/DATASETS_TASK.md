# Task: Build benchmark datasets for my subgraph-classification method and report their statistics

## My problem (read this first)

I have a method for **subgraph classification** that worked well on my own data. I now need more datasets that match my setting, so I can test how well the method **scales**.

**My setting:**

- **Node:** one *event*, e.g. a single doctor visit.
- **Color / entity:** each node belongs to exactly one entity, e.g. one patient. All nodes of the same entity form one **subgraph**.
- **Label:** a **binary label per entity**, e.g. sick / healthy. The label is constant for all nodes of that entity.
- **Edges:** built from **text**. Each event has free text (e.g. the doctor's note). The text is embedded, and an edge connects two nodes when **cosine similarity ≥ threshold**. Edges can connect nodes of the same entity or of different entities.
- **Task:** predict the entity label from its subgraph inside the global graph.

Your job:
1. Obtain datasets that have the same shape: *entities → many events with text (or text-derived vectors) → entity-level binary label*.
2. Convert each dataset into **my graph format**, using the same edge construction as mine.
3. Compute statistics.
4. Build **size-scaled versions** for scalability experiments.
5. Write a report.

---

## Configuration (edit these at the top of `scripts/config.py`)

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # TODO(me): replace with the model I use
SIM_THRESHOLD   = 0.8        # TODO(me): replace with my threshold
EXTRA_THRESHOLDS = [0.7, 0.75, 0.85, 0.9]  # for the threshold-sensitivity table
MAX_EDGES_PER_NODE = None    # optional cap (keep top-k above threshold); None = no cap
SEED = 42
```

If the placeholders are still TODO when you start, ask me for the values before building graphs.

---

## Ground rules

- **Layout:**
  - `data/raw/<dataset>/` for downloaded files
  - `data/processed/<dataset>/` for converted graphs
  - `scripts/` for code
  - `reports/` for outputs
- Add `data/` to `.gitignore`.
- **Inspect before assuming:** print file lists, column names and a few rows before writing loaders. Note any deviation from this spec in the report.
- **Downloads:**
  - If a download fails, log it and continue with the other datasets.
  - Ask me before downloading anything larger than ~2 GB.
  - Some datasets require a signed agreement or credentials. **Never try to bypass that.** Stop and give me instructions, and accept a local path from me instead.
- **Similarity edges must scale:**
  - Never materialize a dense N×N matrix for N above ~20k.
  - L2-normalize the embeddings and use **FAISS** (`IndexFlatIP` + `range_search`, or an IVF/HNSW index for very large N) or chunked matrix multiplication.
  - Store edges as an undirected, deduplicated `int64` array without self-loops.
- **Measure the graph construction itself:** record wall-clock time and peak RAM for (a) embedding and (b) edge construction.
- **Splits:** train/val/test must be **by entity** (70/15/15, stratified by label, fixed seed). No entity may appear in more than one split.

---

## Common output format (every dataset)

Write the following to `data/processed/<dataset>/`:

| File | Contents |
|---|---|
| `nodes.parquet` | `node_id` (0..N-1), `entity_id`, `timestamp` (if available), `text` (if raw text exists and the license allows storing it) |
| `embeddings.npy` | float32 array [N, d], L2-normalized |
| `edges.npy` | int64 array [E, 2] with u < v |
| `edge_weights.npy` | float32 cosine similarity for each edge |
| `entities.csv` | `entity_id`, `label` (0/1), `split`, `n_nodes` |
| `meta.json` | source, preprocessing decisions, threshold, embedding model, build time, peak RAM |

---

## Datasets

### Tier 1: freely downloadable, do these first

#### 1. JODIE Reddit and JODIE Wikipedia
- **Source:** https://github.com/claws-lab/jodie. The README has a download command that fetches `reddit.csv`, `wikipedia.csv`, `mooc.csv` and `lastfm.csv`. Follow it, or fetch the files from the URLs the script uses.
- **CSV format:** `user,item,timestamp,state_label,comma_separated_list_of_features`
- **Mapping to my setting:**
  - node = one interaction (a Reddit post, or a Wikipedia edit)
  - entity = `user`
  - entity label = 1 if the user has `state_label == 1` in any interaction (banned), else 0
- **Text:** the raw text is not included. The features are **172-dim LIWC vectors** derived from the text; use them as the node vectors.
  - Standardize each dimension (z-score), then L2-normalize, then apply the same cosine-threshold rule.
  - Because these are not sentence embeddings, my threshold may produce far too many or too few edges. Report the degree for every value in `EXTRA_THRESHOLDS`. Also propose a threshold that gives an average degree similar to my own graph (ask me for my average degree).
- **Labels:** heavily imbalanced (very few banned users). Report the entity-level class balance.
- **Skip `mooc.csv` and `lastfm.csv`:** they have no text-derived features, so text-similarity edges can't be built. Mention this in the report.

#### 2. EM-User (SubGNN), structural reference only
- **Source:** https://github.com/mims-harvard/SubGNN. The README links to a Dropbox archive with the real-world datasets.
- **What it is:** nodes are workouts and subgraphs are users; the label is the user's gender (binary). It is the closest published analogue of my setup, but its edges are **not** text-similarity edges, and there is no text.
- **What to do:** load it as-is (global graph plus subgraph membership plus labels) and compute the same statistics. Don't rebuild the edges.
- **Sanity check:** 57,333 nodes; 4,573,417 edges; 324 subgraphs; about 155.4 nodes per subgraph on average.
- **Optional:** also load PPI-BP, HPO-Neuro and HPO-Metab from the same archive and report their statistics in a separate "multi-class reference" table. They are not binary, so exclude them from the main comparison.

#### 3. Synthea (synthetic EHR): main scalability dataset
- **Source:** https://github.com/synthetichealth/synthea. It requires Java; follow its README.
- **Generate populations of increasing size:** 1,000, 5,000, 20,000 and 100,000 patients. Ask me before running 100k if it would take more than ~1 hour or produce more than ~5 GB.
- **Mapping:**
  - node = encounter (visit)
  - entity = patient
  - label = the patient has a chosen chronic condition (default: diabetes; list the most common chronic conditions in the output and let me confirm)
- **Text per visit:** first check whether this Synthea version emits clinical-note text (e.g. FHIR `DocumentReference` / notes output). If it does, use it. If it doesn't, build a text per encounter by concatenating the encounter type/reason and the descriptions of observations, procedures and medications recorded in that encounter.
- **Leakage:** remove any mention of the target condition, its codes, and its condition-specific medications from the text used for embeddings. Report exactly what was removed.

### Tier 2: requires a signed agreement (I will provide local paths)

#### 4. eRisk depression (CLEF eRisk)
- **Source:** https://erisk.irlab.org/. Access requires a user agreement; don't try to download it. Ask me for the local path.
- **Content:** Reddit users' full post histories with a user-level label (depressed vs control). The diagnosis-mention posts were removed by the organizers.
- **Mapping:**
  - node = post or comment (title + text)
  - entity = subject/user
  - label = depressed (1) / control (0)
- **Format:** XML per subject (possibly split into chunks). Merge all chunks per subject.
- **Scale:** posts per user can be in the hundreds, so N may reach millions. This is a good real-text scalability test. Build the graph with FAISS; if it's too large, report the time and memory it would need, then build with a subsample of users.

#### 5. RSDD (Reddit Self-reported Depression Diagnosis)
- **Access:** available on request from its authors (Yates et al., 2017). Ask me for the local path.
- **Size:** about 9k diagnosed users and 107k matched controls, which makes it the largest real-text option. Handle it like eRisk.

### Tier 3: medical, credentialed (only if I say so)

#### 6. MIMIC-IV-Note
- **Access:** PhysioNet with credentialed access. Ask me for the local path; never try to download it.
- **Mapping:**
  - node = one note (discharge or radiology), tied to a hospital admission
  - entity = `subject_id`
  - label = to be decided with me (e.g. a chronic diagnosis defined from ICD codes)
- **Leakage:** exclude notes and phrases that directly state the label diagnosis.

---

## Statistics to compute (per dataset and per scaled version)

### A. Global graph

Compute these at `SIM_THRESHOLD`:
- N (nodes) and E (edges)
- average degree, density
- number of connected components, size of the largest component, number of isolated nodes
- **Threshold sensitivity table:** E and average degree for every value in `EXTRA_THRESHOLDS`

### B. Subgraphs / entities
- number of entities (subgraphs)
- entity-level label balance (count and %)
- nodes per subgraph: mean ± std, median, min, max, and a histogram saved as PNG
- average **internal density** of the subgraphs
- average **number of connected components per subgraph** (using internal edges only)
- **fraction of edges that are internal** (both endpoints in the same entity) vs cross-entity
- **cross-entity label homophily:** among cross-entity edges, the % that connect entities with the same label
- the same subgraph statistics split by label (sick vs healthy): mean size, internal density, mean degree

### C. Build cost
- embedding time, edge-construction time, peak RAM, and the on-disk size of the processed graph

### D. Scalability series
- **Synthea:** use the generated sizes.
- **Every other dataset:** subsample **entities** (not nodes) at 10%, 25%, 50% and 100%, keeping all nodes of each selected entity. Rebuild the edges at each fraction.
- **For each point, record:** entities, N, E, build time, and peak RAM.
- **Outputs:** save the series to `reports/csv/scaling.csv`, and plot E vs N and build time vs N on log-log axes.

### E. Comparison with the standard benchmarks

Put this table in the report for context (values from the SubGNN / S2N papers):

| Dataset | Global nodes | Global edges | Subgraphs | Avg nodes / subgraph | Classes |
|---|---|---|---|---|---|
| PPI-BP | 17,080 | 316,951 | 1,591 | 10.2 | 6 |
| HPO-Neuro | 14,587 | 3,238,174 | 4,000 | 14.8 | 10 (multi-label) |
| HPO-Metab | 14,587 | 3,238,174 | 2,400 | 14.4 | 6 |
| EM-User | 57,333 | 4,573,417 | 324 | 155.4 | 2 |

---

## Deliverables

1. `scripts/config.py`
2. `scripts/download_datasets.py`: idempotent; logs successes and failures.
3. `scripts/build_graph.py`: shared embedding and FAISS edge construction, plus one converter per dataset. Run it as `python scripts/build_graph.py --dataset <name> [--fraction 0.25]`.
4. `scripts/compute_stats.py`: computes every statistic above for all processed datasets.
5. `reports/subgraph_datasets_report.md`, containing:
   - the setting recap and the list of datasets with their sources and access status
   - the main summary table (one row per dataset at `SIM_THRESHOLD`)
   - the threshold-sensitivity table
   - the subgraph statistics table, overall and by label
   - the scalability table and plots
   - the comparison with the standard benchmarks
   - **Notes & deviations**: column differences, text construction, leakage filtering, subsampling
   - **Needs action from me**: agreements or credentials needed, and TODO config values
   - a short recommendation of which datasets best match my setting and which are best for the scalability curve
6. `reports/csv/summary.csv`, `reports/csv/thresholds.csv`, `reports/csv/subgraphs.csv`, `reports/csv/scaling.csv`
7. `reports/figures/*.png`

Write the report in English, with numbers to 1–2 decimal places and Markdown tables.

When you finish, print a short terminal summary: which datasets were built, which are blocked (and why), and the report path.
