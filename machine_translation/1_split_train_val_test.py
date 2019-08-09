# 
import sys
import nunpy as np

#assert (sys.version_info.major == 2), u"python version"

def split_corpus(corpus, val=100, test=500):
    assert isinstance(corpus, list)
    l = len(corpus)
    np.random.seed(0)
    m = np.arange(l)
    np.random.shuffle(m)
    val_lst, test_lst, train_lst = [], [], []
    for i, v in enumerate(m):
        if i < val:
            val_lst.append(corpus[m[v]])
        elif i < val + test:
            test_lst.append(corpus[m[v]])
        else:
            train_lst.append(corpus[m[v]])
    return val_lst, test_lst, train_lst
    
