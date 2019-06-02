# wikidata


### 1. 在线使用
NULL

### 2. 离线使用
#### 2.1 数据下载
按照官方数据下载说明：<a href="https://www.wikidata.org/wiki/Wikidata:Database_download" target="_blank">Wikidata:Database download</a>，使用推荐的<a href="https://dumps.wikimedia.org/wikidatawiki/entities/" target="_blank">json格式的数据</a>。

这里使用另外一个链接，下载文件最近的一个文件。   
`wget -c --tries=0 https://dumps.wikimedia.org/other/wikidata/20190520.json.gz `   
国内网速大约下载2-3天，得到一个50G+的压缩包，解压后的json文件约700G。

#### 2.2 数据精简
700G的json文件，没法一次读入内存，不过可以按行读取数据，每一行都可以解析成一项。   

使用脚本<a href="https://github.com/xiaotaw/Notes/blob/master/wikidata/2_select_json.py" target="_blank">2_select_json.py</a>剔除“多语言”，保留中文和英文，精简“claims”的内容，得到20190520_simplified.json，文件大小约75G。

#### 2.3 数据统计
使用脚本<a href="https://github.com/xiaotaw/Notes/blob/master/wikidata/3_statistic_json.py" target="_blank">3_statistic_json.py</a>，统计20190520_simplified.json中的一些数据，如下表所示：

index | Question | Answer 
-|-|-
   | using `type` |  <x>
1  | number_of_entities | 56832908
2  | number_of_items_entities | 56826764
3  | number_of_properties_entities | 6144
   | using `labels` `descriptions` `aliases` |  <x>
4  | number_of_items_with_chinese_labels | 3063642
5  | number_of_items_with_english_labels | 43932804
6  | number_of_items_with_chinese_descriptions | 25166784
7  | number_of_items_with_english_descriptions | 42370338
8  | number_of_items_with_chinese_aliases | 435323
9  | number_of_items_with_english_aliases | 3869923
   | using `claims`  | <x>
10 | number_of_unique_properties | 5931
11 | average_number_of_properties_per_item | 7.78
12 | top_10_properties | P31(53478229),P577(22698118),P1476(22615223),P1433(21604886),P2093(21035907),P304(20326201),P478(20258231),P433(18519562),P698(17814736),P356(16913695),

简单小结一下：
1. entity一共5千万+，绝大部分是item(实物或抽象概念)，只有6144个是property(属性或者关系)。(注：从`claims`中统计得到的property数量为5931个，低于6144个，可能是在2_select_json.py中筛掉了)。
2. 有中文名称(label)的不到6%，这数值低了点。
3. Top 1 property是P31，`is instance of`。

#### 2.4 数据使用
一些可以尝试的方向(需要调研一下相关工作)：
1. 如参考资料1中所提及的，可以考虑做实体相关的，如与命名实体识别相结合。
2. 实体关系分类，预测两个item之间的property。
3. 。。。


### 参考资料
*用百度和必应搜索wikidata，整理后得到不少教程*    

index | name | remark
-|-|- 
1 | <a href="https://blog.csdn.net/qfire/article/details/79573307" target="_blank">用Wikidata做实体搜索的两种方案</a> | 推荐
2 | <a href="https://zhuanlan.zhihu.com/p/36307634" target="_blank">Wikidata从入门到放弃</a> | 
3 | <a href="https://blog.csdn.net/Wmmmdev/article/details/78333209" target="_blank">从Wikidata上面获取数据及关系的几种方法</a> | 
4 | <a href="https://www.wikidata.org/wiki/Wikidata:Data_access" target="_blank">Wikidata:Data access</a> | 推荐
5 | <a href="http://notconfusing.com/3-ways-to-access-wikidata-data-early/" target="_blank">3 WAYS TO ACCESS WIKIDATA DATA UNTIL IT CAN BE DONE PROPERLY</a> | 
6 | <a href="https://www.korrekt.org/page/How_to_use_Wikidata:_Things_to_make_and_do_with_40_million_statements" target="_blank">How to use Wikidata: Things to make and do with 40 million statements</a> | 不可用







