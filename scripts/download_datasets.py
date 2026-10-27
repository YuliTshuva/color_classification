"""
Download the freely available subgraph datasets into data/raw/<dataset>/.

Idempotent: a file that is already complete (size matches the server) is
skipped. Every success and failure goes to data/raw/download_log.txt, and a
failed download does not stop the others.

Usage:
  python scripts/download_datasets.py                 # everything below 2 GB
  python scripts/download_datasets.py --only wikipedia synthea
  python scripts/download_datasets.py --allow-large   # also the >2 GB files
  python scripts/download_datasets.py --only synthea --synthea-sizes 1000 5000 20000
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from os.path import join, exists, getsize

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DIR

LARGE = 2 * 1024 ** 3

# dataset -> list of (url, file name)
SOURCES = {
    "jodie_wikipedia": [("http://snap.stanford.edu/jodie/wikipedia.csv", "wikipedia.csv")],
    "jodie_reddit": [("http://snap.stanford.edu/jodie/reddit.csv", "reddit.csv")],
    "synthea": [("https://github.com/synthetichealth/synthea/releases/download/"
                 "master-branch-latest/synthea-with-dependencies.jar",
                 "synthea-with-dependencies.jar")],
    # The whole SubGNN archive (all four datasets) is ~15.9 GB
    "subgnn": [("https://www.dropbox.com/sh/zv7gw2bqzqev9yn/AACR9iR4Ok7f9x1fIAiVCdj3a?dl=1",
                "SubGNN_data.zip")],
}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line)
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(join(RAW_DIR, "download_log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def remote_size(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=60)
        return int(r.headers.get("Content-Length", 0)) or None
    except requests.RequestException:
        return None


def download(dataset, url, name, allow_large):
    out_dir = join(RAW_DIR, dataset)
    os.makedirs(out_dir, exist_ok=True)
    path = join(out_dir, name)
    size = remote_size(url)
    if exists(path) and size and getsize(path) == size:
        log(f"SKIP  {dataset}/{name}: already complete ({size / 1e6:.0f} MB)")
        return True
    if size and size > LARGE and not allow_large:
        log(f"HOLD  {dataset}/{name}: {size / 1e9:.2f} GB is over 2 GB, rerun with --allow-large")
        return False
    tmp = path + ".part"
    start = time.time()
    # The SNAP server stalls on long transfers, so resume from the partial file
    # with a Range request, and retry when the connection goes quiet
    for attempt in range(1, 31):
        have = getsize(tmp) if exists(tmp) else 0
        if size and have >= size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=(30, 60), headers=headers) as r:
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 ** 2):
                        f.write(chunk)
            if not size:
                break
        except Exception as e:
            log(f"RETRY {dataset}/{name}: attempt {attempt}, at {have / 1e6:.0f} MB: {e}")
            time.sleep(5)
    if exists(tmp) and (not size or getsize(tmp) == size):
        os.replace(tmp, path)
        log(f"OK    {dataset}/{name}: {getsize(path) / 1e6:.0f} MB in {time.time() - start:.0f}s")
        return True
    log(f"FAIL  {dataset}/{name}: incomplete after retries")
    return False


SYNTHEA_FILES = "patients.csv,encounters.csv,conditions.csv,observations.csv,procedures.csv," \
                "medications.csv,immunizations.csv"


def find_java():
    found = shutil.which("java")
    if found:
        return found
    candidates = sorted(glob.glob(r"C:\Program Files\Eclipse Adoptium\jdk-*\bin\java.exe"))
    return candidates[-1] if candidates else None


def generate_synthea(size, seed=42):
    """
    Generate a Synthea population of `size` living patients (Synthea adds the
    ones who died on the way) into data/raw/synthea/p<size>/csv. Only the CSV
    files the converter reads are written. Skipped if patients.csv exists.
    """
    out = join(RAW_DIR, "synthea", f"p{size}")
    if exists(join(out, "csv", "patients.csv")):
        log(f"SKIP  synthea p{size}: already generated")
        return True
    java = find_java()
    jar = join(RAW_DIR, "synthea", "synthea-with-dependencies.jar")
    if java is None or not exists(jar):
        log(f"FAIL  synthea p{size}: java ({java}) or jar ({exists(jar)}) missing")
        return False
    cmd = [java, "-Xmx12g", "-jar", jar, "-s", str(seed), "-cs", str(seed), "-p", str(size),
           f"--exporter.baseDirectory={out}/",
           "--exporter.csv.export=true", f"--exporter.csv.included_files={SYNTHEA_FILES}",
           "--exporter.csv.excluded_files=",
           "--exporter.fhir.export=false", "--exporter.hospital.fhir.export=false",
           "--exporter.practitioner.fhir.export=false", "--exporter.metadata.export=false",
           "--generate.log_patients.detail=none"]
    start = time.time()
    with open(join(RAW_DIR, "synthea", f"p{size}.log"), "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        log(f"FAIL  synthea p{size}: exit code {rc}, see data/raw/synthea/p{size}.log")
        return False
    log(f"OK    synthea p{size}: generated in {time.time() - start:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--synthea-sizes", nargs="*", type=int, default=[],
                        help="Also generate Synthea populations of these sizes, e.g. 1000 5000 20000")
    args = parser.parse_args()
    results = {d: all(download(d, u, n, args.allow_large) for u, n in SOURCES[d]) for d in args.only}
    for size in args.synthea_sizes:
        results[f"synthea_p{size}"] = generate_synthea(size)
    print("\n".join(f"{d}: {'ok' if ok else 'not downloaded'}" for d, ok in results.items()))


if __name__ == "__main__":
    main()
