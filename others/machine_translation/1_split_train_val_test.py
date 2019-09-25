#!/usr/bin/env python3 
import argparse
import codecs
import logging
import os
import random
import sys

#assert (sys.version_info.major == 3), u"python version"

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description = "Split corpus into train, val, and test",
        usage = __file__ + " [<args>] [-h | --help]"
    )
    
    parser.add_argument("--filenames", type=str, nargs="+", required=True, 
        metavar="FILENAME", help="Path of corpus")
    parser.add_argument("--n_val", type=int, default=100, 
        metavar="NUM_VAL", help="Number of val")
    parser.add_argument("--n_test", type=int, default=500, 
        metavar="NUM_VAL", help="Number of test")
    parser.add_argument("--fn_format", type=str, 
        metavar="[PREFIX]{TYPE}_{BASENAME}.{EXTENSION}",
        default="{prefix}{type}_{basename}.{extension}", 
        help="output filename format")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--basename", type=str, default="",
        help="if not defined, use the basename of the input filename")
    parser.add_argument("--extension", type=str, default="",
        help="if not defined, use the extension of the input filename")
    
    return parser.parse_args(args)

def split_corpus(corpus, n_val, n_test):
    assert isinstance(corpus, list)
    l = len(corpus)
    random.seed(0)
    m = list(range(l))
    random.shuffle(m)
    val_lst, test_lst, train_lst = [], [], []
    for i, v in enumerate(m):
        if i < n_val:
            val_lst.append(corpus[m[v]])
        elif i < n_val + n_test:
            test_lst.append(corpus[m[v]])
        else:
            train_lst.append(corpus[m[v]])
    return val_lst, test_lst, train_lst

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    for fn in args.filenames:
        f = codecs.open(fn, encoding="utf8")
        s = f.read()
        s = s.replace("\r", "")
        s = s.replace(u"\u2028", " ").replace(u"\u2029", " ")
        s = [x+"\n" for x in s.strip().split("\n")]
        f.close()
        logging.info("Read %s, total lines %d ..." % (fn, len(s)))
        
        val, test, train = split_corpus(s, args.n_val, args.n_test)
        logging.info("Splited %s, val: %d, test: %d, train: %d" % (fn, len(val), 
            len(test), len(train)))
            
        filepath, filename = os.path.split(fn)
        basename, extension = os.path.splitext(filename)
        extension = extension.lstrip(".")
        if not args.basename:
            args.basename = basename 
        if not args.extension:
            args.extension = extension

        fn_format = args.fn_format.format(prefix=args.prefix, type="%s", 
            basename=args.basename, extension=args.extension)
        
        if len(val) > 0:
            f = codecs.open(fn_format % "val", "w", encoding="utf8")
            _ = f.write("".join(val))
            f.close()
        if len(test) > 0:
            f = codecs.open(fn_format % "test", "w", encoding="utf8")
            _ = f.write("".join(test))
            f.close()
        if len(train) > 0:
            f = codecs.open(fn_format % "train", "w", encoding="utf8")
            _ = f.write("".join(train))
            f.close()


