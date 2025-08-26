import os
import re

def parse_file(path):
    """Extract throughput, latency, and number of parameters from a file."""
    throughput, latency, params = None, None, None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Model contains :"):
                # Example: "Model contains : 9.241e+06 parameters overall"
                parts = line.split()
                if len(parts) >= 4:
                    params = parts[3]
            elif line.startswith("Current throughput in detections/s:"):
                # Example: "Current throughput in detections/s: 3.965177022281308"
                parts = line.split(":")
                if len(parts) == 2:
                    throughput = parts[1].strip()
            elif line.startswith("Current latency in milliseconds:"):
                # Example: "Current latency in milliseconds: 2.492982769012451e-07"
                parts = line.split(":")
                if len(parts) == 2:
                    latency = parts[1].strip()
    return params, throughput, latency

def main(folder="."):
    rows = []
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        if fname in ("gpu.txt", "cpu.txt"):
            continue
        fpath = os.path.join(folder, fname)
        params, throughput, latency = parse_file(fpath)
        rows.append((fname, params, throughput, latency))

    # Generate markdown table
    md_lines = ["| File | Parameters | Throughput (detections/s) | Latency (ms) |",
                "|------|------------|--------------------------|--------------|"]
    for fname, params, throughput, latency in rows:
        md_lines.append(
            f"| {fname} | {params or 'N/A'} | "
            f"{float(throughput):.3f} | {float(latency):.3f} |"
            if throughput and latency else
            f"| {fname} | {params or 'N/A'} | {throughput or 'N/A'} | {latency or 'N/A'} |"
        )

    out_path = "result_speedup.md"
    with open(out_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"✅ Results written to {out_path}")

if __name__ == "__main__":
    main("jpl_scripts/eval/performance/")  
