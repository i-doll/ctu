#!/usr/bin/env python3
"""Validate PRICING table invariants: prefix shadowing + date-window sanity.

Lookup takes the first entry whose prefix and date both match, so ordering is
load-bearing and the compiler cannot check it. Run after editing the table:

    python3 scripts/check-pricing-table.py

Exits non-zero if an entry is unreachable or two windows for one model overlap.
Gaps in a model's date coverage are warnings (those dates bill $0).
"""
import re, sys, collections, pathlib

src = (pathlib.Path(__file__).resolve().parent.parent / 'src' / 'lib.rs').read_text()
start = src.index('static PRICING')
end = src.index('];', start)
body = src[start:end]

entries = []  # (prefix, from, until)
# Split per entry, not per line: rustfmt wraps long builder chains.
for line in re.split(r'(?=price!\()', body):
    m = re.match(r'price!\("([^"]+)"', line)
    if not m: continue
    pfx = m.group(1)
    f = re.search(r'\.from\("([^"]+)"\)', line)
    u = re.search(r'\.until\("([^"]+)"\)', line)
    entries.append((pfx, f.group(1) if f else None, u.group(1) if u else None))

ok = True
def overlap(a, b):
    """Do two [from, until) windows intersect?"""
    af, au = a[1] or "0000-00-00", a[2] or "9999-99-99"
    bf, bu = b[1] or "0000-00-00", b[2] or "9999-99-99"
    return af < bu and bf < au

# 1. An earlier entry must not shadow a later one for an overlapping date range.
for i, a in enumerate(entries):
    for b in entries[i+1:]:
        if b[0].startswith(a[0]) and overlap(a, b):
            print(f"UNREACHABLE: {b} shadowed by earlier {a}"); ok = False

# 2. Windows for the same prefix must not overlap each other.
by_prefix = collections.defaultdict(list)
for e in entries: by_prefix[e[0]].append(e)
for pfx, group in by_prefix.items():
    for i, a in enumerate(group):
        for b in group[i+1:]:
            if overlap(a, b):
                print(f"OVERLAPPING WINDOWS for {pfx!r}: {a} vs {b}"); ok = False

# 3. Dated windows for a prefix should tile without gaps (warn only).
for pfx, group in by_prefix.items():
    if len(group) < 2: continue
    bounds = sorted((g[1] or "0000-00-00", g[2] or "9999-99-99") for g in group)
    for (_, au), (bf, _) in zip(bounds, bounds[1:]):
        if au != bf:
            print(f"GAP for {pfx!r}: coverage ends {au}, resumes {bf} (dates in between bill $0)")

print(f"{len(entries)} entries checked;", "OK" if ok else "INVARIANTS BROKEN")
sys.exit(0 if ok else 1)
