#!/usr/bin/env python3
import json
from collections import defaultdict

fn = "wikidata/20190520_zh_en.json_dummy.txt"
fn_item = "wikidata/20190520_zh_en.item_dummy.csv"
fn_claim = "wikidata/20190520_zh_en.claim_dummy.csv"
fn_property = "wikidata/20190520_zh_en.property_dummy.csv"

f = open(fn, encoding="utf8")
_ = f.readline()


f_item = open(fn_item, "w", encoding="utf8")
f_item.write("id, zh_label, en_label, zh_description, en_description")
item_format = "{id}, {zh_label}, {en_label}, {zh_description}, {en_description}"

f_property = open(fn_property, "w", encoding="utf8")
f_property.write("id, zh_label, en_label, zh_description, en_description")
property_format = "{id}, {zh_label}, {en_label}, {zh_description}, {en_description}"

f_claim = open(fn_claim, "w", encoding="utf8")
f_claim.write("from, to, claim")
claim_format = "{from}, {to}, {claim}"

n = 0
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
        #print(item)
        item_str = item_format.format(**item)
        f_item.write(item_str)
        for p, dd in e["claims"].items():
            for d in dd:
                if isinstance(d["value"], dict) and  "entity-type" in d["value"]:
                    claim = defaultdict(str)
                    claim["from"] = e["id"]
                    claim["to"] = d["value"]["id"]
                    claim["claim"] = p
                    f_claim.write(claim_format.format(**claim))
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
        f_property.write(property_format.format(**property))
        for p, dd in e["claims"].items():
            for d in dd:
                if isinstance(d["value"], dict) and  "entity-type" in d["value"]:
                    claim = defaultdict(str)
                    claim["from"] = e["id"]
                    claim["to"] = d["value"]["id"]
                    claim["claim"] = p
                    f_claim.write(claim_format.format(**claim))
                else:
                    pass # todo 
    else:
        print("unexpected type :" + e["type"])
        break
        

f.close()
f_item.close()
f_claim.close()
f_property.close()

print("finished, total: %d" % n)
