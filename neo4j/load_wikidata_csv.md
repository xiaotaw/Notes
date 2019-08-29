```cypher
LOAD CSV WITH HEADERS FROM "file:///20190520_zh_en.item_dummy.csv" AS csvLine
CREATE (i:Item {id: csvLine.id, zh_label: csvLine.zh_label, en_label: csvLine.en_label,
zh_description: csvLine.zh_description, en_description: csvLine.en_description})
```
