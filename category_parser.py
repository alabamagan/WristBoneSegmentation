def parse_category_string(str):
    s = str.split('_')
    out = []
    for pairs in s:
        if pairs.find('-') > -1:
            out.extend(range(int(pairs.split('-')[0])-1, int(pairs.split('-')[1])))
        else:
            out.append(int(pairs) - 1)
    return out

def string2category(str):
    temp = []
    parts = str.split(',')
    for i, p in enumerate(parts):
        temp.append(parse_category_string(p))

    numOfIndexes = sum([len(t) for t in temp])
    out = [0] * numOfIndexes
    for i, t in enumerate(temp):
        for indexes in t:
            out[indexes] = i
    return out

def category2string(catlist):
    """category2string --> str
    Convert category list into string
    """
    d = {}
    for i in xrange(max(catlist)):
        d[i] = []

    for index, cat in enumerate(catlist):
        if not d.has_key(cat):
            d[cat] = []
        d[cat].append(index)

    # sort the index lists
    for cat in set(catlist):
        d[cat].sort()

    out = []
    for cat in d.keys():
        s = []
        start = end = None
        if len(d[cat]) == 0:
            s.append('NULL')
        else:
            for i, index in enumerate(d[cat]):
                if start is None:
                    start = index
                try:
                    if d[cat][i+1] > index + 1:
                        end = index
                except KeyError:
                    # Normal
                    pass
                except IndexError:
                    end = index
                    pass
                if not start == None and not end == None:
                    if start != end:
                        s.append('-'.join([str(start+1), str(end+1)]))
                    else:
                        s.append(str(start+1))
                    start = end = None


        s = '_'.join(s)
        out.append(s)
    out = ','.join(out)
    return out


def check_category(catlist):
    """check_category(list) --> None
    Change the input list inplace such that there are no cat 0 within other cats
    """
    state = 0
    ns = [L == 0 for L in catlist]
    for index, cat in enumerate(catlist):
        if cat == 0 and state == 0:
            continue
        elif cat > 0:
            if state == 0:
                state = 1
        elif cat == 0 and state == 1 and not all(ns[index:]):
            catlist[index] = 1
        else:
            # remaining zeros
            pass


