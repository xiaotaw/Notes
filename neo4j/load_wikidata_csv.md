# Load wikidata (dummy) into csv

### preprocess
* see [wikidata数据使用：将wikidata导入neo4j](https://github.com/xiaotaw/Notes/blob/master/wikidata/README.md#%E6%95%B0%E6%8D%AE%E4%BD%BF%E7%94%A8)

### load 
```cypher

CREATE CONSTRAINT ON (i:Item) ASSERT i.id IS UNIQUE

LOAD CSV WITH HEADERS FROM "file:///20190520_zh_en.item_dummy.csv" AS csvLine
CREATE (i:Item {id: csvLine.id, zh_label: csvLine.zh_label, en_label: csvLine.en_label,
zh_description: csvLine.zh_description, en_description: csvLine.en_description})

LOAD CSV WITH HEADERS FROM "file:///20190520_zh_en.property_dummy.csv" AS csvLine
CREATE (i:Property:Item {id: csvLine.id, zh_label: csvLine.zh_label, en_label: csvLine.en_label,
zh_description: csvLine.zh_description, en_description: csvLine.en_description})

USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM "file:///20190520_zh_en.claim_dummy.csv" AS csvLine
MATCH (f:Item {id: csvLine.from}),(t:Item {id: csvLine.to}),(c:Property {id: csvLine.claim})
CREATE (f)-[:CLAIM {id: c.id, zh_label: c.zh_label, en_label: c.en_label,
zh_description: c.zh_description, en_description: c.en_description}]->(t)
```
