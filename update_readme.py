#!/usr/bin/python3

import re
from glob  import glob


url = "https://github.com/xiaotaw/Notes/tree/master/"


r = re.compile("([\s\*]*)\[([^\[\]]+)\]\(([^\(\)]+)\)")

def get_context_2(fn):
    context = []
    f = open(fn, encoding="utf8")
    s = f.readlines()
    f.close()
    context_flag = False

    title = fn.rstrip("/README.md")

    for l in s:
        if l.strip() == u"## 目录":
            context.append(l)
            context_flag = True
            continue
        if context_flag:
            if l.lstrip().startswith("*"):
                res = r.findall(l)
                if len(res) != 1:
                    print("[error] %s: %s" % (title, l))
                    continue
                pre, text, link = res[0]
                link = url + title + link
                ll = '{}<a href="{}" target="_blank">{}</a>\n'.format(pre, link, text)
                context.append(ll)
            elif l.strip().startswith("#"):
                context_flag = False
                return context
    print("Info: context may not complete. plz check.")
    return context

if __name__ == "__main__":
    
    fn_lst_3 = glob("*/*/README.md")
    if fn_lst_3:
        print("Info: those files not included!")
        print(fn_lst_3)


    fn_lst_2 = glob("*/README.md")
    fn_lst_2.sort()
    context_1 = ["# 目录\n"]
    context_1.append(
    "此目录由脚本维护：python3 update_readme.py \n"
    )

    for fn in fn_lst_2:
        title = fn.rstrip("/README.md")
        context_1.append("## %s\n" % title)
        #
        context_2 = get_context_2(fn)
        for l in context_2[1:]:
            context_1.append(l)
        context_1.append("\n")

    context_1 = "".join(context_1)

    f = open("README.md", "w", encoding="utf8")
    f.write(context_1)
    f.close()
