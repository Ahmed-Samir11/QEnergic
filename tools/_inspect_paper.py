p = r"e:\repos\QEnergic\paper.tex"
with open(p,'rb') as f:
    data = f.read()
needle = b"D-Wave Neal"
idx = data.find(needle)
if idx == -1:
    print('NOT FOUND')
else:
    start = max(0, idx-200)
    end = min(len(data), idx+200)
    print('SNIPPET_REPR:')
    print(repr(data[start:end]))
    lines = data.splitlines()
    for i,l in enumerate(lines,1):
        if needle in l:
            print('LINE_NO:', i)
            print('LINE_REPR:', repr(l))
            break
