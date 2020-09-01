#!/usr/bin/env python3
import json
import pandas as pd
from collections import defaultdict

debug = False

if debug:
    fn = "wikidata/20190520_zh_en.json_dummy.txt"
    fn_item = "wikidata/20190520_zh_en.item_dummy.csv"
    fn_claim = "wikidata/20190520_zh_en.claim_dummy.csv"
    fn_property = "wikidata/20190520_zh_en.property_dummy.csv"
else:
    fn = "wikidata/20190520_zh_en.json"
    fn_item = "wikidata/20190520_zh_en.item.csv"
    fn_claim = "wikidata/20190520_zh_en.claim.csv"
    fn_property = "wikidata/20190520_zh_en.property.csv"

f = open(fn, encoding="utf8")
_ = f.readline()

n = 0
items_lst = []
claims_lst = []
properties_lst = []
while(True):
    if n % 100000 == 0:
        print("processing: %d" % n)
    n += 1
    x = f.readline()
    x = x.strip().rstrip(",")
    if x == "]" or not x:
        break
    e = json.loads(x)
    if e["type"] == "item":
        item = {"id": "", "zh_label": "", "en_label": "", "zh_description": "", "en_description":""}
        item["id"] = e["id"]
        if "en" in e["labels"]:
            item["en_label"] = e["labels"]["en"]
        if "zh" in e["labels"]:
            item["zh_label"] = e["labels"]["zh"]
        if "zh-cn" in e["labels"]:
            item["zh_label"] = e["labels"]["zh-cn"]
        if "en" in e["descriptions"]:
            item["en_description"] = e["descriptions"]["en"]
        if "zh" in e["descriptions"]:
            item["zh_description"] = e["descriptions"]["zh"]
        if "zh-cn" in e["descriptions"]:
            item["zh_description"] = e["descriptions"]["zh-cn"]
        # aliases are ignored
        items_lst.append(item)
        #item_str = item_format.format(**item)
        #f_item.write(item_str)
        for p, dd in e["claims"].items():
            for d in dd:
                if isinstance(d["value"], dict) and  "entity-type" in d["value"]:
                    claim = defaultdict(str)
                    claim["from"] = e["id"]
                    claim["to"] = d["value"]["id"]
                    claim["claim"] = p
                    #f_claim.write(claim_format.format(**claim))
                    claims_lst.append(claim)
                else:
                    pass # todo 
    elif e["type"] == "property":
        property = {"id": "", "zh_label": "", "en_label": "", "zh_description": "", "en_description":""}
        property["id"] = e["id"]
        if "en" in e["labels"]:
            property["en_label"] = e["labels"]["en"]
        if "zh" in e["labels"]:
            property["zh_label"] = e["labels"]["zh"]
        if "zh-cn" in e["labels"]:
            property["zh_label"] = e["labels"]["zh-cn"]
        if "en" in e["descriptions"]:
            property["en_description"] = e["descriptions"]["en"]
        if "zh" in e["descriptions"]:
            property["zh_description"] = e["descriptions"]["zh"]
        if "zh-cn" in e["descriptions"]:
            property["zh_description"] = e["descriptions"]["zh-cn"]
        # aliases are ignored
        #f_property.write(property_format.format(**property))
        properties_lst.append(property)
        for p, dd in e["claims"].items():
            for d in dd:
                if isinstance(d["value"], dict) and  "entity-type" in d["value"]:
                    claim = defaultdict(str)
                    claim["from"] = e["id"]
                    claim["to"] = d["value"]["id"]
                    claim["claim"] = p
                    #f_claim.write(claim_format.format(**claim))
                    claims_lst.append(claim)
                else:
                    pass # todo 
    else:
        print("unexpected type :" + e["type"])
        break

f.close()

df_items = pd.DataFrame(items_lst)
df_items.to_csv(fn_item, encoding="utf8")

df_properties = pd.DataFrame(properties_lst)
df_properties.to_csv(fn_property, encoding="utf8")

df_claims = pd.DataFrame(claims_lst)
df_claims.to_csv(fn_claim, encoding="utf8")


print("finished, total: %d" % n)
