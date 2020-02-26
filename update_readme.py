#!/usr/bin/python3
import os
import re
from glob  import glob


url = "https://github.com/xiaotaw/Notes/tree/master/"


regex = re.compile("([\s\*]*)\[([^\[\]]+)\]\(([^\(\)]+)\)")

def get_context(fn):
    context = []
    f = open(fn, encoding="utf8")
    s = f.readlines()
    f.close()
    context_flag = False

    fn = fn.lstrip("./")

    for l in s:
        if l.strip() == u"## 目录":
            context.append(l)
            context_flag = True
            continue
        if context_flag:
            if l.lstrip().startswith("*"):
                res = regex.findall(l)
                if len(res) != 1:
                    print("[error] %s: %s" % (fn, l))
                    continue
                pre, text, link = res[0]
                link = url + fn + link
                ll = '{}<a href="{}" target="_blank">{}</a>\n'.format(pre, link, text)
                context.append(ll)
            elif l.strip().startswith("#"):
                context_flag = False
                return context
    print("Info: context may not complete. plz check %s." % fn)
    return context

if __name__ == "__main__":
    
    context_lst = ["# 目录\n"]
    context_lst.append(
    "©2019-2020 xiaotaw. All Rights Reserved. \
    \n\n此目录由脚本维护：python3 update_readme.py，记录了部分技能点。\
    \n\n20191216: 通过squash、reword整理历史commit。\
    \n\n20200226: 拓展\"README.md\"为\"*.md\"，可方便将一个主题下的README.md进一步拆分成不同的文档。 \n"
    )
    
    for r, ds, fs in os.walk("."):
        for f in fs:
            if f.endswith(".md"):
                r = r.lstrip("./")
                full_path = os.path.join(r, f)
                
                if f == "README.md":
                    title = r
                else:
                    title = full_path[:-3]
                if title == "":
                    # 根目录的README.md不需要处理
                    print("skip: %s" % full_path)
                    continue
                
                context = get_context(full_path)
                if context:
                    link = url + full_path
                    ll = '## <a href="{}" target="_blank">{}</a>\n'.format(link, title)
                    context_lst.append(ll)
                    context_lst.extend(context[1: ])
                    context_lst.append("\n")

    
    context_str = "".join(context_lst)
    f = open("README.md", "w", encoding="utf8")
    f.write(context_str)
    f.close()

