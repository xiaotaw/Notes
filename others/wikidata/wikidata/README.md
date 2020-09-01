## file description
* Because the original files are too big,  the dummy files upload instead.

|file|desc|details|
|----|----|-------|
|20190520.json_dummy.txt|unzipped wikidata dump file in json format|<blank>|
|20190520_zh_en.json_dummy.txt|zh_en version of wikidata, i.e. only includes entities in zh or en|result of 2_select_json.py|
|property_frequency.txt|all the properties with frequency|result of 3_statistic_json.py|
|-|<blank>|<blank>|
|20190520_zh_en.claim_dummy.csv||result of extract_2_csv.py|
|20190520_zh_en.item_dummy.csv||result of extract_2_csv.py|
|20190520_zh_en.property_dummy.csv||result of extract_2_csv.py|

## if you want to use the .py scripts, here are some guidelines
1. download the complete dump file 3 days before u get started. https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2 or latest-all.json.gz.
2. modify the file name(defualt is wikidata/20190520.json) before running.
3. use small file to test your scripts, such as 20190520.json_dummy.txt.
