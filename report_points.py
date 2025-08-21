import os

INPUT_DIRS = [
    "jpl_scripts/eval/point_evaluation/",
    "jpl_scripts/eval/jpl_points/",
]
OUTPUT_FILE = "report_points.md"


def parse_file(path):
    results = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if "H_error_ransac@" in line:
                parts = line.split(":")
                key = parts[0].strip().lstrip("{")
                val = parts[1].strip().rstrip(",}; ")
                results[key] = float(val)

            elif line.startswith("H_error_ransac"):
                val = line.split(":")[1].strip().rstrip(",}; ")
                results["H_error_ransac"] = float(val)

    return results


def format_float(val):
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def process_dir(input_dir):
    all_results = {}

    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        if os.path.isfile(fpath):
            if "ROMA" in fpath:
                continue
            metrics = parse_file(fpath)
            if metrics:
                name = fname.replace(".yaml", "").replace(".txt", "")
                all_results[name] = metrics

    return all_results


def write_table(f, title, all_results):
    if not all_results:
        f.write(f"## {title}\n\n_No results found_\n\n")
        return

    # Sort all keys
    all_keys = sorted({str(k) for metrics in all_results.values() for k in metrics.keys()})

    # Table header
    f.write(f"## {title}\n\n")
    f.write("| File | " + " | ".join(all_keys) + " |\n")
    f.write("|------|" + "|".join(["-------" for _ in all_keys]) + "|\n")

    # Rows
    for name, metrics in sorted(all_results.items()):
        row = [name] + [format_float(metrics.get(k, "")) for k in all_keys]
        f.write("| " + " | ".join(row) + " |\n")

    f.write("\n\n")


def main():
    merged_results = {}

    # Merge all folders
    for input_dir in INPUT_DIRS:
        results = process_dir(input_dir)
        merged_results.update(results)

    # Separate by dataset
    hpatches_results = {k: v for k, v in merged_results.items() if "hpatches" in k.lower()}

    # Write tables
    with open(OUTPUT_FILE, "w") as f:
        f.write("# Evaluation Results\n\n")
        write_table(f, "HPatches", hpatches_results)

    print(f"✅ Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
