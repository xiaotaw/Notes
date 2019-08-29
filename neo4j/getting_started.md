# Install
### ubuntu 18.04
On my ubuntu 18.04, just tpye:
```bash
sudo apt-get install neo4j
```
After installed, check version by:
```bash
neo4j version
```
and get `neo4j 3.5.8`.

### ubuntu 16.04（docker）
```bash
# 添加源
wget -O - https://debian.neo4j.org/neotechnology.gpg.key |  apt-key add -
echo 'deb https://debian.neo4j.org/repo stable/' |  tee /etc/apt/sources.list.d/neo4j.list

# 更新
apt-get update
# 注：发现报错"E: The method driver /usr/lib/apt/methods/http could not be found."，用apt-get install apt-transport-https修复，之后再apt-get update

# 安装
apt-get install neo4j

# 检查
neo4j version
# 得到 neo4j 3.5.8

# get the docker
docker pull registry.cn-beijing.aliyuncs.com/xt-web/web:ubuntu16.04-neo4j
```

# run
### remote connect 
```bash
# 使用docker，直接使用默认配置的目录，避免与权限有关的坑。
docker pull registry.cn-beijing.aliyuncs.com/xt-web/web:ubuntu16.04-neo4j-vim

# 运行docker，注意端口映射: 
docker run -v /data/:/data/ -p 7100:7100 -p 7101:7101  -it registry.cn-beijing.aliyuncs.com/xt-web/web:ubuntu16.04-neo4j-vim

# 修改远程连接相关配置
vi /etc/neo4j/neo4j.conf

# 第71行的 “#dbms.connector.bolt.listen_address=:7687”
# 修改为 “#dbms.connector.bolt.listen_address=0.0.0.0:7100” 
# 允许远程访问neo4j的数据库

# 第75行的“dbms.connector.http.listen_address=:7474”
# 修改为“dbms.connector.http.listen_address=0.0.0.0:7101”
# 允许远程访问neo4j的browser

# 保存退出vim

# 运行neo4j
service neo4j start

# (x.x.x.x为neo4j服务器地址，若为本机可用localhost或127.0.0.1替代)
# 在浏览器中输入 http://x.x.x.x:7101/browser，稍等一会儿即出现登录页面，类似如下信息：
:server connect
Connect to Neo4j
Database access requires an authenticated connection.
Connect URL：bolt://x.x.x.x:7100
username: neo4j
password: 

# 输入默认密码neo4j，确认，输入新密码进行重置后，即可使用neo4j。
```
# load csv
* 参考官方教程[LOAD CSV](https://neo4j.com/docs/getting-started/current/cypher-intro/load-csv/)

```bash
# 数据准备
# 在/var/lib/neo4j/import下，创建三个csv文件：

# File 1. persons.csv, a list of persons:
id,name
1,Charlie Sheen
2,Michael Douglas
3,Martin Sheen
4,Morgan Freeman

# File 2. movies.csv, a list of movies:
id,title,country,year
1,Wall Street,USA,1987
2,The American President,USA,1995
3,The Shawshank Redemption,USA,1994

# File 3. roles.csv, a list of which role was played by some of these persons in each movie:
personId,movieId,role
1,1,Bud Fox
4,1,Carl Fox
3,1,Gordon Gekko
4,2,A.J. MacInerney
3,2,President Andrew Shepherd
5,3,Ellis Boyd 'Red' Redding
```

```cypher
# 在浏览器内
CREATE CONSTRAINT ON (person:Person) ASSERT person.id IS UNIQUE

CREATE CONSTRAINT ON (movie:Movie) ASSERT movie.id IS UNIQUE

CREATE INDEX ON :Country(name)

LOAD CSV WITH HEADERS FROM "file:///persons.csv" AS csvLine
CREATE (p:Person {id: toInteger(csvLine.id), name: csvLine.name})

LOAD CSV WITH HEADERS FROM "file:///movies.csv" AS csvLine
MERGE (country:Country {name: csvLine.country})
CREATE (movie:Movie {id: toInteger(csvLine.id), title: csvLine.title, year:toInteger(csvLine.year)})
CREATE (movie)-[:MADE_IN]->(country)

USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM "file:///roles.csv" AS csvLine
MATCH (person:Person {id: toInteger(csvLine.personId)}),(movie:Movie {id: toInteger(csvLine.movieId)})
CREATE (person)-[:PLAYED {role: csvLine.role}]->(movie)

DROP CONSTRAINT ON (person:Person) ASSERT person.id IS UNIQUE

DROP CONSTRAINT ON (movie:Movie) ASSERT movie.id IS UNIQUE

MATCH (n)
WHERE n:Person OR n:Movie
REMOVE n.id

# 最后可以得到数据如图所示
MATCH (n) RETURN n
```
![load csv result](https://github.com/xiaotaw/Notes/blob/master/neo4j/pics/load_csv_result_graph.png)

# Reading Documents
* One morning for official documents <a href="https://neo4j.com/docs/getting-started/current/" target="_blank">The Neo4j Getting Started Guide v3.5</a>, and the core concept lies <a href="https://neo4j.com/docs/getting-started/current/graphdb-concepts/" target="_blank">Chapter 2. Graph database concepts</a> 

* 官方文档加载太慢了

# Others
### 批量导入数据方法对比

|<blank>|CREATE语句|LOAD CSV语句|Batch Inserter|Batch Import|Neo4j-import|
|-|-|-|-|-|-|
|适用场景|1-1w nodes|1w-10w nodes|>1000w nodes|>1000w nodes|>1000w nodes|
|速度|很慢(1000 nodes/s)|一般(5000 nodes/s)|非常快(数万 nodes/s)|非常快(数万 nodes/s)|非常快(数万 nodes/s)|
|优点|使用方便，可实时插入|使用方便，可以加载本地/远程csv；可实时插入|速度相比于前两个，有数量级的提升|基于Batch Inserter，可以直接运行编译好的jar包；可以在已存在数据库中导入数据|官方出品，比Batch Import占用更少的资源|
|缺点|速度慢|需要将数据转换成CSV|需要转成CSV；智能在java中使用；且插入时必须停止neo4j|需转成CSV；必须停止neo4j|需转成CSV；必须停止neo4j；智能生成新的数据库，不能再已存在的数据库中插入数据|
  
表格内容来源于blog：[学习Neo4j几小时小结~批量导入数据](https://blog.csdn.net/qiqi123i/article/details/90022799)
