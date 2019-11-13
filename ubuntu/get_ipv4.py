import os
import re
from collections import Counter

url_lst = [
"ifconfig.me", 
"ipinfo.io/ip",
"ipcanhazip", 
"ip.sb",
"ipecho.net/plain", 
"tnx.nl/ip", 

"www.trackip.net/i",

]


r = re.compile("\d+\.\d+\.\d+\.\d+")

def is_valid_ip(ip):
    x = ip.split(".")
    valid = True
    if len(x) != 4: 
        valid = False
    for y in x:
        if not (0 <= int(y) <= 255):
            valid = False
            break
    if x[0] == "192" and x[1] == "168":
        valid = False
    return valid



def get_ip(timeout):
    ips = []
    for url in url_lst:
        try:
            var = os.popen("timeout %ds curl %s" % (timeout, url))
            s = var.read()
            t = r.findall(s)
            t = list(filter(is_valid_ip, t))
            if len(t) <= 0:
                continue
            elif len(t) == 1:
                ip = t[0]
            else:
                ip = Counter(t).most_common(1)[0][0]
            print(url + " : " + ip + "\n")
            ips.append(ip)
        except :
            print(url + " failed \n") 
            continue
    if len(ips) >= 1:
        return Counter(ips).most_common(1)[0][0]
    else:
        return None


def print_result(ip):
    print("=================================ipv4===================================")
    if ip:
        print(ip)
    else:
        print("Failed")
    print("=================================ipv4===================================")
    log_name = os.path.abspath(__file__).replace(".py", ".log")
    f = open(log_name, "a+")
    import datetime
    _ = f.write("%s : %s\n" % (datetime.datetime.today(), ip))
    f.close()


if __name__ == "__main__":
    for timeout in [10, 20, 30]:
        ip = get_ip(timeout)
        if ip:
            print_result(ip)
            break
    else:
        print_result(None)
