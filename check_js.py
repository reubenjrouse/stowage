"""
A rough syntax check for the JavaScript inside app.html.

It strips out strings and comments, then checks the brackets balance and that
the functions and element ids the code refers to actually exist. It cannot
catch runtime errors -- only a browser can do that.
"""

import io
import re

BS = chr(92)  # backslash, written this way to survive shell quoting

src = io.open('app.html', encoding='utf-8').read()
js = src.split('<script>')[-1].split('</script>')[0]

out = []
i, n = 0, len(js)
state = None  # None | "'" | '"' | '`' | '//' | '/*'
while i < n:
    c = js[i]
    nxt = js[i + 1] if i + 1 < n else ''
    if state is None:
        if c == '/' and nxt == '/':
            state = '//'; i += 2; continue
        if c == '/' and nxt == '*':
            state = '/*'; i += 2; continue
        if c in ('"', "'", '`'):
            state = c; out.append(' '); i += 1; continue
        out.append(c); i += 1; continue
    if state == '//':
        if c == '\n':
            state = None; out.append('\n')
        i += 1; continue
    if state == '/*':
        if c == '*' and nxt == '/':
            state = None; i += 2; continue
        if c == '\n':
            out.append('\n')
        i += 1; continue
    # inside a string
    if c == BS:
        i += 2; continue
    if c == state:
        state = None
    if c == '\n':
        out.append('\n')
    i += 1

stripped = ''.join(out)

pairs = {'(': ')', '[': ']', '{': '}'}
stack, line, bad = [], 1, None
for ch in stripped:
    if ch == '\n':
        line += 1
    elif ch in pairs:
        stack.append((ch, line))
    elif ch in ')]}':
        if not stack or pairs[stack[-1][0]] != ch:
            bad = ('unexpected ' + ch + ' on line ' + str(line)); break
        stack.pop()

if bad:
    print('MISMATCH:', bad)
elif stack:
    print('UNCLOSED:', stack[-1])
else:
    print('delimiters balanced across %d lines of JS' % line)

print()
for name in ['newRound', 'stepBot', 'onClick', 'onHover', 'syncHud', 'setupRay',
             'hitStaging', 'cellUnderPointer', 'animateDrops', 'frame', 'toast',
             'refreshPickHighlight', 'boxMesh', 'aim']:
    found = re.search(r'function\s+' + name + r'\b', js) is not None
    print('  %-13s %s' % (name, 'ok' if found else 'MISSING'))

print()
# word boundary matters: applyLevel('custom') contains "el('custom')"
ids = re.findall(r"(?<![A-Za-z0-9_])el\('([a-zA-Z]+)'\)", js)
missing = sorted({i for i in set(ids) if ('id="' + i + '"') not in src})
print('element ids referenced:', len(set(ids)))
print('referenced but not in markup:', missing if missing else 'none')
