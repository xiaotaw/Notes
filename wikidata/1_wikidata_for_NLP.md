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
tobe continue



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







