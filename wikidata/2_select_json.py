# 
import json

def _select_by_key(x, keys_lst=[["en", "en-gb"], ["zh-cn", "zh"]]):
    assert isinstance(x, dict)
    ret = {}
    for keys in keys_lst:
        for key in keys:
            if key in x:
                y = x[key]
                if isinstance(y, dict):
                    assert (x[key]["language"] == key)
                    ret[key] = x[key]["value"]
                elif isinstance(y, list):
                    ret[key] = [z["value"] for z in y]
                else:
                    print("y is expected to be a list or a dict, but got %s" % type(y))
                break
    return ret


def _select_with_key(x, key_lst=["enwiki", "zhwiki"]):
    assert isinstance(x, dict)
    ret = {}
    for key in key_lst:
        for x_key in x.keys():
            if key in x_key:
                ret[x_key] = x[x_key]
    return ret

def _select_claims(claims):
    new_dict = {}
    for p, vs in claims.items():
        new_lst = []
        for v in vs:
            if v["type"] == "statement":
                if ("datavalue" in v["mainsnak"]) and ("datavalue" in v["mainsnak"]):
                    new_lst.append(v["mainsnak"]["datavalue"])
        new_dict[p] = new_lst
    return new_dict


f = open("wikidata/20190520.json", encoding="utf8")
_ = f.readline()


of = open("wikidata/20190520_simplified.json", "w", encoding="utf8")
_ = of.write("[\n")
i = 0
while(True):
    if i % 10000 == 0:
        print(i)
    x = f.readline()    
    if x == "]":
        break
    x = x.strip().rstrip(",")
    a = json.loads(x) 
    b = {}
    b["type"] = a["type"]
    b["id"] = a["id"]
    b["labels"] = _select_by_key(a["labels"])
    b["descriptions"] =  _select_by_key(a["descriptions"])
    b["aliases"] = _select_by_key(a["aliases"])
    b["claims"] = _select_claims(a["claims"])
    #b["sitelinks"] = _select_with_key(a["sitelinks"], key_lst=["enwiki", "zhwiki"])
    #a["lastrevid"] = a["lastrevid"]
    t = json.dumps(b)
    _ = of.write(t + ",\n")
    i += 1

of.write("]")

f.close()
of.close()
