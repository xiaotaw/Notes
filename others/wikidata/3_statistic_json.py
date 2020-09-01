import json
from collections import defaultdict
f = open("wikidata/20190520_zh_en.json", encoding="utf8")
#f = open("wikidata/20190520_zh.json", encoding="utf8")
_ = f.readline()
number_of_entities = 0
number_of_items = 0
number_of_properties = 0
unique_properties = defaultdict(int)
number_of_properties_lst = []
number_of_items_with_english_labels = 0
number_of_items_with_chinese_labels = 0
number_of_items_with_english_descriptions = 0
number_of_items_with_chinese_descriptions = 0
number_of_items_with_chinese_aliases = 0
number_of_items_with_english_aliases = 0
while(True):
    if number_of_entities % 100000 == 0:
        print("process %d" % number_of_entities)
    s = f.readline().strip().rstrip(",")
    if (s == "]") or (not s):
        break
    x = json.loads(s)
    if x["type"] == "item":
        number_of_items += 1
    if x["type"] == "property":
        number_of_properties += 1
    if ("en" in x["labels"]) or ("en-gb" in x["labels"]):
        number_of_items_with_english_labels += 1
    if ("zh" in x["labels"]) or ("zh-cn" in x["labels"]):
        number_of_items_with_chinese_labels += 1
    if ("en" in x["descriptions"]) or ("en-gb" in x["descriptions"]):
        number_of_items_with_english_descriptions += 1
    if ("zh" in x["descriptions"]) or ("zh-cn" in x["descriptions"]):
        number_of_items_with_chinese_descriptions += 1
    if ("en" in x["aliases"]) or ("en-gb" in x["aliases"]):
        number_of_items_with_english_aliases += 1
    if ("zh" in x["aliases"]) or ("zh-cn" in x["aliases"]):
        number_of_items_with_chinese_aliases += 1
    for k in x["claims"]:
        unique_properties[k] += 1
    number_of_properties_lst.append(len(x["claims"]))
    number_of_entities += 1 

f.close()
number_of_unique_properties = len(unique_properties)
average_number_of_properties_per_item = sum(number_of_properties_lst) / len(number_of_properties_lst)
valid_number_of_properties_lst = [ x for x in number_of_properties_lst if x != 0]
average_number_of_properties_per_item_1 = sum(valid_number_of_properties_lst) / len(valid_number_of_properties_lst)

a = list(unique_properties.items())
a.sort(key=lambda x: x[1], reverse = True)

print("number_of_entities: %d" % number_of_entities)
print("number_of_items_entities: %d" % number_of_items)
print("number_of_properties_entities: %d" % number_of_properties)
print("number_of_unique_properties : %d" % number_of_unique_properties)
print("number_of_items_with_english_labels : %d" % number_of_items_with_english_labels)
print("number_of_items_with_chinese_labels: %d" % number_of_items_with_chinese_labels)
print("number_of_items_with_english_descriptions: %d" % number_of_items_with_english_descriptions)
print("number_of_items_with_chinese_descriptions: %d" % number_of_items_with_chinese_descriptions)
print("number_of_items_with_chinese_aliases : %d" % number_of_items_with_chinese_aliases )
print("number_of_items_with_english_aliases : %d" % number_of_items_with_english_aliases )
print("average_number_of_properties_per_item : %d" % average_number_of_properties_per_item)
print("Top10_freq_properties: " + str(a[:10]))

f = open("wikidata/property_frequency.txt", "w", encoding="utf8")
for k, v in a:
    _ = f.write("%s\t%d\n" % (k, v))
        
f.close()
