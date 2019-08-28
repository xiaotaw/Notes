# Install
On my ubuntu 18.04, just tpye:
```bash
sudo apt-get install neo4j
```
After installed, check version by:
```bash
neo4j version
```
and get `neo4j 3.5.8`.

# Reading Documents
* One morning for official documents <a href="https://neo4j.com/docs/getting-started/current/" target="_blank">The Neo4j Getting Started Guide v3.5</a>, and the core concept lies <a href="https://neo4j.com/docs/getting-started/current/graphdb-concepts/" target="_blank">Chapter 2. Graph database concepts</a> 


# Others
### 批量导入数据方法对比

|<blank>|CREATE语句|LOAD CSV语句|Batch Inserter|Batch Import|Neo4j-import|
|-|-|-|-|-|-|
|适用场景|1-1w nodes|1w-10w nodes|>1000w nodes|>1000w nodes|>1000w nodes|
|速度|很慢(1000 nodes/s)|一般(5000 nodes/s)|非常快(数万 nodes/s)|非常快(数万 nodes/s)|非常快(数万 nodes/s)|
|优点|使用方便，可实时插入|使用方便，可以加载本地/远程csv；可实时插入|速度相比于前两个，有数量级的提升|基于Batch Inserter，可以直接运行编译好的jar包；可以在已存在数据库中导入数据|官方出品，比Batch Import占用更少的资源|
|缺点|速度慢|需要将数据转换成CSV|需要转成CSV；智能在java中使用；且插入时必须停止neo4j|需转成CSV；必须停止neo4j|需转成CSV；必须停止neo4j；智能生成新的数据库，不能再已存在的数据库中插入数据|
  
表格内容来源于blog：[学习Neo4j几小时小结~批量导入数据](https://blog.csdn.net/qiqi123i/article/details/90022799)
