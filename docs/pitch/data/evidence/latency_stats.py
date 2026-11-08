# ruff: noqa: E501, SIM115  (kept as run, for the record)
import collections
import glob
import json
import math
import statistics
import sys


def pct(xs, p):  # nearest-rank, as the team's evaluation report does
    xs = sorted(xs)
    return xs[max(0, math.ceil(p * len(xs)) - 1)]


groups = {
    "readiness+prompt trials (Mac)": glob.glob("<workspace>/walking-trial/[rv]*/*/../*.log")
    and glob.glob("<workspace>/walking-trial/[rv]*/*/events.jsonl"),
    "walking batch 02:03 (Mac, 10 teachers)": glob.glob("<workspace>/walking-trial/batch-walk-0203/c*/events.jsonl"),
    "generated fixed (8 teachers)": glob.glob(
        sys.argv[1] + "/ws1-learn/runs/learn-box_to_bin_fixed-*/workers/*/events.jsonl"
    ),
    "generated walking": glob.glob(sys.argv[1] + "/ws1-learn/runs/learn-deliver_box_to_bin-*/workers/*/events.jsonl"),
}
allact = []
allfail = 0
alln = 0
for name, files in groups.items():
    lat = []
    fail = 0
    n = 0
    ts = []
    purposes = collections.Counter()
    for f in files:
        for line in open(f):
            try:
                e = json.loads(line)
            except Exception:
                continue
            ts.append(e["t"])
            if e["type"] != "argon_call":
                continue
            d = e["data"]
            purposes[d.get("purpose")] += 1
            if d.get("purpose") not in ("act", "demonstrate", "step"):
                continue
            n += 1
            if not d.get("ok"):
                fail += 1
                continue
            lat.append(d["latency_ms"])
    allact += lat
    allfail += fail
    alln += n
    span = (max(ts) - min(ts)) / 60 if ts else 0
    if lat:
        print(
            f"{name}: files={len(files)} act_calls={n} ok={len(lat)} failed={fail} median={statistics.median(lat):.0f}ms p95={pct(lat, 0.95):.0f}ms span={span:.1f}min purposes={dict(purposes)}"
        )
    else:
        print(f"{name}: files={len(files)} purposes={dict(purposes)} span={span:.1f}min")
print(
    f"ALL act: n={alln} ok={len(allact)} failed={allfail} median={statistics.median(allact):.0f} p95={pct(allact, 0.95):.0f} p05={pct(allact, 0.05):.0f} max={max(allact):.0f}"
)
