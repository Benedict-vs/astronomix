"""List the cross-device collectives in an XLA HLO dump and where they run.

    python hlo_collectives.py <module_*.jit__time_integration.*after_optimizations.txt>

Prints one line per (loop nesting, collective kind, source op): the bytes ONE
instance moves per device, how many such instructions there are, and the trip
counts of the while loops around them ("5x2" = inside the 5-stage LSRK loop
inside a 2-step loop; "-" = runs once, e.g. hoisted out of the loop by XLA).
A whole-state all-gather / all-to-all inside the loops is the thing to hunt:
GSPMD inserts one for e.g. `.at[static slice on a sharded axis].set(...)`.

Make the dump with
    XLA_FLAGS="--xla_dump_to=<dir> --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=NONE"
(on CPU add --xla_force_host_platform_device_count=8 and pallas_interpret=True).
"""

import re, sys, collections
F = sys.argv[1]
comp = None; calls = collections.defaultdict(list); colls = []
hdr = re.compile(r'^(ENTRY )?(%[\w.\-]+) ')
B = {'f32': 4, 's32': 4, 'u32': 4, 'f64': 8, 'bf16': 2, 'f16': 2, 'pred': 1, 's8': 1, 'u8': 1}
for line in open(F):
    if not line.startswith(' ') and '{' in line and hdr.match(line):
        comp = hdr.match(line).group(2); continue
    if comp is None: continue
    n = re.search(r'"n":"(\d+)"', line)
    for key in ('body', 'condition', 'calls', 'to_apply', 'branch_computations', 'called_computations'):
        for m in re.finditer(key + r'=(\{[^}]*\}|%[\w.\-]+)', line):
            for c in re.findall(r'%[\w.\-]+', m.group(1)):
                calls[c].append((comp, key, n.group(1) if n else None))
    m = re.search(r'= (.+?) (all-to-all|all-reduce|collective-permute|all-gather|reduce-scatter)(-start)?\(', line)
    if m:
        shapes = re.findall(r'(f32|s32|u32|f64|bf16|f16|pred|s8|u8)\[([\d,]*)\]', m.group(1))
        if m.group(3): shapes = shapes[: max(1, len(shapes) // 2)]  # -start tuples carry (in, out)
        b = sum(B[t] * eval('*'.join(s.split(',')) or '1') for t, s in shapes)
        op = re.search(r'op_name="([^"]+)"', line)
        colls.append((comp, m.group(2), b, op.group(1) if op else ''))
def chain(c):
    out, seen = [], set()
    while calls.get(c) and c not in seen:
        seen.add(c); p, k, n = calls[c][0]
        if k == 'body': out.append(n or '?')
        c = p
    return out
agg = collections.Counter(); cnt = collections.Counter()
for c, kind, b, op in colls:
    ch = 'x'.join(chain(c)) or '-'
    s = re.sub(r'jit\(_time_integration\)/', '', op).replace('while/body/closed_call/', 'W/').split(':')[0][-100:]
    agg[(ch, kind, s)] += b; cnt[(ch, kind, s)] += 1
print("MB(per instance)  count  loops(trip counts, '-'=runs once)  kind  op")
for (ch, kind, s), b in sorted(agg.items(), key=lambda x: -x[1]):
    print(f"{b/1e6:10.3f} n={cnt[(ch,kind,s)]:3d} loops={ch:6s} {kind:18s} {s}")
