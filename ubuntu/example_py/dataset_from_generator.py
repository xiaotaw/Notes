# encoding=utf8
import numpy as np
import tensorflow as tf
from collections import OrderedDict, defaultdict

# ref: https://stackoverflow.com/questions/6190331/how-to-implement-an-ordered-default-dict
class OrderedDefaultDict(OrderedDict, defaultdict):
    def __init__(self, default_factory=None, *args, **kwargs):
        #in python3 you can omit the args to super
        super(OrderedDefaultDict, self).__init__(*args, **kwargs)
        self.default_factory = default_factory

 
filenames = ["exa/ch.txt", "exa/en.txt", "exa/ch.pos", "exa/en.pos"]
unk_prob = 0.9
max_unk = 5

def data_generator():
    fs = [tf.gfile.GFile(fn) for fn in filenames]
    #np.random.seed(0)
    while True:
        src, tgt, src_pos, tgt_pos = [f.readline().strip().split() for f in fs]
        if not (src and tgt):
            _ = [f.close() for f in fs]
            fs = [tf.gfile.GFile(fn) for fn in filenames]
            print("\n\n\n reopen file \n\n\n")
            continue
        # 查找 替换的词 在句子中的位置
        src_pos_index = OrderedDefaultDict(list)
        _ = [src_pos_index[src_p] for src_p in src_pos]
        tgt_pos_index = OrderedDefaultDict(list)
        _ = [tgt_pos_index[tgt_p] for tgt_p in tgt_pos]
        
        for i, src_word in enumerate(src):
            if src_word in src_pos:
                src_pos_index[src_word].append(i)
        for i, tgt_word in enumerate(tgt):
            if tgt_word in tgt_pos:
                tgt_pos_index[tgt_word].append(i)
        available_unk_symbol = ["<unk%d>" % i for i in range(max_unk)]
        for src_p, tgt_p in zip(src_pos_index, tgt_pos_index):
            # 条件1：仅在句中出现一次（TODO：当掷n次骰子都赢了，出现n次可以全部替换）
            src_idx, tgt_idx = src_pos_index[src_p], tgt_pos_index[tgt_p]
            if len(src_idx) == 1 and len(tgt_idx) == 1:
                # 条件2：掷骰子赢了
                if np.random.random() <= unk_prob:
                    # 随机一个UNK，替换掉src和tgt中的对应token
                    available_unk_num = len(available_unk_symbol)
                    idx = int(np.random.random() * available_unk_num) % available_unk_num
                    unk_symbol = available_unk_symbol[idx]
                    src[src_idx[0]] = unk_symbol
                    tgt[tgt_idx[0]] = unk_symbol
                    available_unk_symbol.remove(unk_symbol)
                    if len(available_unk_symbol) == 0:
                        break
        yield " ".join(src), " ".join(tgt)
    _ = [f.close() for f in fs]
    
def get_training_input():
     dataset = tf.data.Dataset.from_generator(data_generator, tf.string, tf.TensorShape([2]))
     #dataset = dataset.shuffle(4)
     #dataset = dataset.repeat()
     def string_split(x):
         src, tgt = tf.split(x, [1, 1])
         src = tf.string_split(src).values
         tgt = tf.string_split(tgt).values
         return (src, tgt)
     dataset = dataset.map(string_split)
     dataset = dataset.map(
         lambda src, tgt:(
            tf.concat([src, [tf.constant("<eos>")]], axis=0),
            tf.concat([tgt, [tf.constant("<eos>")]], axis=0)
         )
     )
     iterator = dataset.make_one_shot_iterator()
     features = iterator.get_next()
     return features
     
def test_generator():
    g = data_generator()
    for src, tgt in g:
        #src, tgt = g.next()
        print(src)
        print(tgt)
    
def test_dataset():
    features = get_training_input()
    sess = tf.Session()
    for i in range(10):
        src, tgt = sess.run(features)
        print(" ".join(src) + "\t" + " ".join(tgt))
    
    

if __name__ == "__main__":
    #test_generator() 
    test_dataset()

