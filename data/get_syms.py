from __future__ import absolute_import
import glob, os, copy, pickle
from xml.dom import minidom
import numpy as np, cv2
from page import TablePAGE
import unidecode
import argparse, math

def transforms(line):
    line = line.replace(" ", "<space>")
    line = line.replace("&lt;space&gt;", " ")
    line = line.replace("&lt;ign&gt;", "")
    line = line.replace("<ign>", "")
    line = line.replace("<space>", " ")
    return line

def get_all_xml(path, ext="xml"):
    file_names = glob.glob(os.path.join(path, "*{}".format(ext)))
    print(os.path.join(path, "*{}".format(ext)))
    return file_names

def create_text(line, args):
    res = ""
    line = line.lower()
    print(line)
    line = transforms(line)

    line = unidecode.unidecode(line)
    print(line)
    print("--------------")
    for w in line:
        # w = re.sub(r"[^a-zA-Z0-9 ]+", '', w) #REMOVED special chars
        if w == " ":
            w = "<space>"
        res += "{} ".format(w.lower())
    return res

def make_page_txt(fname, txts, path_out, args):
    page = TablePAGE(im_path=fname)
    tls = page.get_textLines(id=True)
    basename = ".".join(fname.split(".")[:-1])
    basename = basename.split("/")[-1]
    n_deleted = 0
    for coords, text, id_line in tls:
        fname_line = "{}.{}".format(basename, id_line)
        
        text = create_text(text, args)
        # fname_line = os.path.join(path_out, "{}".format(id_line))

        txts.append([fname_line, text])
    return n_deleted

def start(args):
    path = args.path_input
    # Output
    path_out = args.path_out
    result_path_map = os.path.join(path_out, "syms.txt")

    files = get_all_xml(path)
    all_text = ""
    txts = []

    for fname in files:
        n_del = make_page_txt(fname, txts, path_out, args)
        # print(txts)
        # exit()
    for fname_line, text in txts:
        all_text += "{} ".format(text)
    print(all_text)
    exit()
    
    chars = all_text.lower().split()
    chars = list(set(chars))
    chars.insert(0, "<blank>")

    f_Result = open(result_path_map, "w")
    for i, c in enumerate(chars):
        f_Result.write("{} {}\n".format(c,i))
    f_Result.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process lines.')
    parser.add_argument('--path_input', metavar='path_input', type=str,
                    help='Input path where are the xmls and images')
    parser.add_argument('--path_out', metavar='path_out', type=str,
                    help='Output path to store the results. The dir will be created if doesnt exist')
    args = parser.parse_args()
    start(args)
