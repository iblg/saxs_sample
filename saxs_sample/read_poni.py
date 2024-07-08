

def read_poni(fp):
    with open(fp, 'r') as infile:
        lines = infile.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split(': ') for line in lines]
    lines = lines[3:]
    params = {}
    for line in lines:
        if '#' not in line:
            params[line[0]] = line[1]
    for k, val in params.items():
        try:
            params[k] = float(val)
        except ValueError:
            pass

    return params
