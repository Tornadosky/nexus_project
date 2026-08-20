"""Prove that a derived config differs from its base only in the intended keys."""
import sys

import yaml


def load(p):
    with open(p) as f:
        return yaml.safe_load(f)


a, b = sys.argv[1], sys.argv[2]
expected = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
A, B = load(a), load(b)
diff = {}
for k in sorted(set(A) | set(B)):
    if A.get(k, "<MISSING>") != B.get(k, "<MISSING>"):
        diff[k] = (A.get(k, "<MISSING>"), B.get(k, "<MISSING>"))
print("--- %s  VS  %s ---" % (a.split("/")[-1], b.split("/")[-1]))
for k, (x, y) in diff.items():
    print("    %-28s %-14r -> %r" % (k, x, y))
if expected is not None:
    got = set(diff)
    if got == expected:
        print("    OK: exactly the intended keys differ (%s)" % ",".join(sorted(got)))
    else:
        print("    *** MISMATCH ***  expected=%s got=%s" % (sorted(expected), sorted(got)))
        sys.exit(1)
else:
    print("    (%d differing keys)" % len(diff))
