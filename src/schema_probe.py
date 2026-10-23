"""Probe an OPFData tar.gz shard and print the exact JSON schema.

Run on the GPU workstation after downloading one shard (see
docs/opfdata-runbook.md). Stdlib only -- no torch/numpy needed.

    python src/schema_probe.py --tar pglib_opf_case118_ieee_19.tar.gz --n-examples 3

Prints: member filename ranges (to confirm the train/val/test split
mapping), top-level keys, per-node/per-edge feature names and vector
lengths, index ranges, and the solution/metadata structure. Paste the
output back to Muse -- it is the ground truth the data loader is written
against.
"""
import argparse
import json
import tarfile


def summarize(obj, depth=0, max_items=8):
    """Compact structural summary of a parsed JSON value."""
    pad = "  " * depth
    if isinstance(obj, dict):
        keys = list(obj.keys())
        lines = [f"{pad}dict[{len(keys)}]: {keys[:max_items]}"
                 + (" ..." if len(keys) > max_items else "")]
        for k in keys[:max_items]:
            lines.append(f"{pad}'{k}':")
            lines.append(summarize(obj[k], depth + 1, max_items))
        return "\n".join(lines)
    if isinstance(obj, list):
        if not obj:
            return f"{pad}list[0] (empty)"
        first = obj[0]
        if isinstance(first, (int, float)):
            nums = [x for x in obj if isinstance(x, (int, float))]
            lo, hi = (min(nums), max(nums)) if nums else ("?", "?")
            return f"{pad}list[{len(obj)}] of numbers, range [{lo}, {hi}]"
        if isinstance(first, list):
            return (f"{pad}list[{len(obj)}] of lists, inner len {len(first)}, "
                    f"first row {first[:max_items]}")
        if isinstance(first, dict):
            sub = list(first.keys())
            return (f"{pad}list[{len(obj)}] of dicts, keys {sub[:max_items]}"
                    + (" ..." if len(sub) > max_items else ""))
        return f"{pad}list[{len(obj)}] of {type(first).__name__}"
    return f"{pad}{type(obj).__name__}: {obj!r}"[:160]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True, help="local .tar.gz shard path")
    ap.add_argument("--n-examples", type=int, default=3)
    args = ap.parse_args()

    tf = tarfile.open(args.tar, "r:gz")
    members = [m for m in tf.getmembers() if m.isfile()]
    names = sorted(m.name for m in members)
    print(f"members: {len(names)}")
    print(f"first: {names[0]}")
    print(f"last:  {names[-1]}")

    ids = []
    for n in names:
        base = n.rsplit("/", 1)[-1]
        if base.startswith("example_") and base.endswith(".json"):
            try:
                ids.append(int(base[len("example_"):-len(".json")]))
            except ValueError:
                pass
    if ids:
        print(f"example id range in this shard: [{min(ids)}, {max(ids)}] "
              f"({len(ids)} parsed)")

    for i, name in enumerate(names[:args.n_examples]):
        print(f"\n===== example {i}: {name} =====")
        f = tf.extractfile(members[names.index(name)])
        ex = json.load(f)
        print(summarize(ex))


if __name__ == "__main__":
    main()
