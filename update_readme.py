#!/usr/bin/python3

from glob  import glob




def get_context_2(fn):
    context = []
    f = open(fn, encoding="utf8")
    s = f.readlines()
    f.close()
    context_flag = False
    for l in s:
        if l.strip() == u"## 目录":
            context.append(l)
            context_flag = True
            continue
        if context_flag:
            if l.lstrip().startswith("*"):
                context.append(l)
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
    context_1 = ["# 目录\n"]
    for fn in fn_lst_2:
        prefix = fn.rstrip("/README.md")
        context_1.append("## %s\n" % prefix)
        #
        context_2 = get_context_2(fn)
        for l in context_2[1:]:
            context_1.append(l)
        context_1.append("\n")

    context_1 = "".join(context_1)

    f = open("README.md", "w", encoding="utf8")
    f.write(context_1)
    f.close()
