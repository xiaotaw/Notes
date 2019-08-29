## file description
* Because the original files are too big,  the dummy files upload instead.

|file|desc|
|----|----|
|20190520.json_dummy.txt|unzipped wikidata dump file in json format|
|20190520_zh_en.json_dummy.txt|zh_en version of wikidata, i.e. only includes entities in zh or en|
|20190520_zh_en.json_head1000.txt|zh_en version of wikidata(only contains the first 1000 line)|
|property_frequency.txt|all the properties with frequency|

## if you want to use the .py scripts, here are some guidelines
1. download the complete dump file 3 days before u get started. https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2 or latest-all.json.gz.
2. modify the file name(defualt is wikidata/20190520.json) before running.
3. use small file to test your scripts, such as 20190520.json_dummy.txt.
