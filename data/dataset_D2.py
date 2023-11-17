from __future__ import print_function
from __future__ import division
import glob, os, re
import numpy as np, math
import torch
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from torch.utils.data import Dataset
import logging
import cv2
import functools
import time
try:
    import cPickle as pickle
except:
    import pickle  # --- To handle data imports/export
from data.page import TablePAGE
from utils.functions import load_image
from detectron2.structures import Instances, BoxMode, Boxes
from detectron2.data import DatasetCatalog
from shapely.geometry import LineString

np.seterr(divide='ignore', invalid='ignore')

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time dataset: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer

def get_all(path, ext="pkl"):
    if path[-1] != "/":
        path = path+"/"
    file_names = glob.glob(os.path.join(path,"*.{}".format(ext)))[:5]
    return file_names

def load_IG(p:str, nwords:int=2048):
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    res = []
    res_dict = {}
    for i, line in enumerate(lines[:nwords]):
        line = line.strip().upper()
        res.append(line)
        res_dict[line] = i
    return res, res_dict

class ImageDataset():
    """
    Class to handle HTR dataset feeding
    """

    def __init__(self, path, logger=None, opts=None, split="train", cfg=None):
        """
        Args:
            img_lst (string): Path to the list of images to be processed
            label_lst (string): Path to the list of label files to be processed
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.logger = logger or logging.getLogger(__name__)
        
        self.path = path
        self.split = split
        self.height, self.width = opts.img_size
        self.page_files = get_all(path, ext="xml")#[:8]
        # self.page_files = [x for x in self.page_files if "Albatross_vol009of055-065" in x]
        self.img_path = opts.img_path
        self.syms = {}
        self.HTR = False
        # self.acts = cfg.MODEL.ACTS_ON
        self.acts = False
        self.cfg = cfg
        if self.acts:
            self.IG_order, self.IG_order_dict = load_IG(cfg.MODEL.ACTS.IG_FILE, nwords=cfg.MODEL.ACTS.NWORDS)
        self.opts = opts
        self.px_height_offset = opts.px_height_offset
        self.from_baseline = opts.from_baseline
        self.textlines = False
        self.acts = False
        if "acts" in self.opts.classes:
            self.classes_dict = {"AI":0, "AM":1, "AF":2, "AC":3}
            self.acts = True
        else:
            self.classes_dict = {k:i for i,k in enumerate(self.opts.classes)}
        self.n_classes = len(self.classes_dict)
        if logger is not None:
            logger.info(self.classes_dict)
    
    def load_syms(self):
        f = open(self.path_syms, "r")
        lines = f.readlines()
        f.close()
        
        for line in lines:
            sym, value = line.strip().split(" ")
            self.syms[sym] = int(value)
        

    def __len__(self):
        return len(self.page_files)
    
    def get_all(self, ):
        res = []
        # print("--------------------------- ", len(self.page_files))
        for idx in range(len(self.page_files)):
            res.append(self.__getitem__(idx))
        return res

    def get_inclusion_box(self, coords, orig_h, orig_w, trans_width=None, trans_height=None):
        """
        x = w, 
        y = h
        coords = [[x,y]]
        """
        min_w, min_h = min([x[0] for x in coords]), min([x[1] for x in coords])
        max_w, max_h = max([x[0] for x in coords]), max([x[1] for x in coords])
        min_w = max(0, min(orig_w, min_w))
        min_h = max(0, min(orig_h, min_h))
        max_w = max(0, min(orig_w, max_w))
        max_h = max(0, min(orig_h, max_h))
        if trans_height is not None and trans_height is not None:
            min_w *= trans_width
            max_w *= trans_width
            min_h *= trans_height
            max_h *= trans_height
        return np.array([int(min_w), int(min_h), int(max_w), int(max_h)])

    def get_segm(self, coords, orig_h, orig_w):
        """ 
        x = w, 
        y = h
        coords = [[x,y]]
        """
        xs = [x[0] for x in coords]
        ys = [x[1] for x in coords]
        return xs, ys
    
    def get_orientation_bl(self, bl):
        def angle(linea):
            return math.degrees(math.atan2(linea[1][1]-linea[0][1], linea[1][0]-linea[0][0]))
        if bl is None or len(bl) == 0:
            return "h"
        # try:
        angle_ = abs(angle(bl))
        if 0 <= angle_ <= 45 or 315 <= angle_ <= 360:
            return "h"
        return "v"
    
    # @timer
    def __getitem__(self, idx):
        # print("----")
        
        fname = self.page_files[idx]
        fname_noext = fname.split("/")[-1].split(".")[0]
        
        page = TablePAGE(fname, hisClima=True)
        
        width_orig, height_orig = page.get_width(), page.get_height()
        max_width_page = width_orig
        trans_width, trans_height  = self.width / width_orig, self.height / height_orig
        fname_image = os.path.join(self.img_path, f'{fname_noext}.jpg')
        if not os.path.exists(fname_image):
            fname_image = os.path.join(self.img_path, f'{fname_noext}.png')
        if not os.path.exists(fname_image):
            fname_image = os.path.join(self.img_path, f'{fname_noext}.TIF')
        sample = {}
        lines_id, boxes = [], []
        orig_coords = []
        gt_classes = []
        gt_classes_acts = []
        texts = []
        instances = Instances(
            image_size=[height_orig, width_orig]
        )
        objs = []
        done = set()
        for class_name in self.opts.classes:
            if class_name in ["rows", "cols", "acts"]:
                continue
            if "TextLine" in class_name:
                class_name = "TextLine"
            
            textlines = page.get_Region(class_name, added_reg=True)
            for textline in textlines:
                coords, id_, text, reg = textline["coords"], textline["id"], textline["text"], textline["region"]
                row, col = textline["row"], textline["col"]
                if self.from_baseline:
                    coords_bl_b, coords_bl = build_baseline_offset(np.array(textline["baseline"]).astype(int), offset=self.px_height_offset)
                    if coords_bl_b:
                        coords = coords_bl
                    # print(coords, coords_bl)
                # print(fname, id_, row, col)
                if id_ not in done:
                    done.add(id_)
                else:
                    continue
                max_w, max_h = max([x[0] for x in coords]), max([x[1] for x in coords])
                min_w, min_h = min([x[0] for x in coords]), min([x[1] for x in coords])
                w_tl = (max_w - min_w)
                if w_tl >= max_width_page:
                    continue
                box_coords = self.get_inclusion_box(coords, height_orig, width_orig, trans_width, trans_height)
                box_coords_real = self.get_inclusion_box(coords, height_orig, width_orig)
                px, py = self.get_segm(coords, height_orig, width_orig)
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x] # [(1,2), (3,4)] -> [1,2,3,4]
                if len(poly) % 2 != 0 or len(poly) < 6:
                    print(poly)
                    print(fname, id_)
                    continue
                

                if class_name == 'TextLine' and self.opts.bl_orientation:
                    bl = page.get_baseline_fromTL(reg)
                    
                    text_encoded = []
                    if bl is None:
                        orientation = "h"
                    elif len(bl) < 2:
                        orientation = "h"
                    else:
                        orientation = self.get_orientation_bl(bl)
                    c = self.classes_dict.get(class_name + f'_{orientation}')
                    gt_classes.append(c)
                else:
                    c = self.classes_dict.get(class_name)
                    gt_classes.append(c)
                    text_encoded = []
                # print(f'box_coords {box_coords} box_coords_real {box_coords_real}')
                boxes.append(box_coords_real)
                texts.append(text_encoded)
                obj = {
                "bbox": box_coords_real,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": c,
                "segmentation": [poly],
                'text_encoded': text_encoded,
                "gt_row": row,
                "gt_col": col,

                }
                # print([poly])
               
                objs.append(obj)
        if "rows" in self.opts.classes or "cols" in self.opts.classes:
            try:
                cells, cell_by_col, cell_by_row = page.get_cells()
            except Exception as e:
                print(f'fname - {fname}')
                raise e
            if "rows" in self.opts.classes:
                class_name = "rows"
                for row_num, coords in cell_by_row.items():
                    coords_list = [x for ccords in coords for x in ccords]
                    # box_coords = self.get_inclusion_box(coords, height_orig, width_orig, trans_width, trans_height)
                    box_coords_real = self.get_inclusion_box(coords_list, height_orig, width_orig)
                   
                    coordsPr = [Polygon(ccords) for ccords in coords]
                    pR = unary_union(coordsPr)
                    try:
                        coordsR = np.array(pR.exterior.coords)
                    except Exception as e:
                        print(f'{fname} {row_num}  - pR {pR}')
                        for i in pR:
                            print(i)
                        raise 
                    px, py = self.get_segm(coordsR, height_orig, width_orig)
                    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                    poly = [p for x in poly for p in x] # [(1,2), (3,4)] -> [1,2,3,4]
                    
                    boxes.append(box_coords_real)
                    c = self.classes_dict.get(class_name)
                    gt_classes.append(c)
                    
                    obj = {
                    "bbox": box_coords_real,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": c,
                    "segmentation": [poly],
                    }
                    objs.append(obj)
            if "cols" in self.opts.classes:
                    class_name = "cols"
                    for col_num, coords in cell_by_col.items():
                        coords_list = [x for ccords in coords for x in ccords]
                        box_coords_real = self.get_inclusion_box(coords_list, height_orig, width_orig)
                        coords = [Polygon(ccords) for ccords in coords]
                        pR = unary_union(coords)
                        try:
                            coordsR = np.array(pR.exterior.coords)
                        except Exception as e:
                            print(f'{fname} {col_num}  - pR {pR}')
                            for i in pR:
                                print(i)
                            raise 
                        px, py = self.get_segm(coordsR, height_orig, width_orig)
                        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                        poly = [p for x in poly for p in x] # [(1,2), (3,4)] -> [1,2,3,4]
                        
                        boxes.append(box_coords_real)
                        c = self.classes_dict.get(class_name)
                        gt_classes.append(c)
                        
                        obj = {
                        "bbox": box_coords_real,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": c,
                        "segmentation": [poly],
                        }
                        objs.append(obj)
        elif "acts" in self.opts.classes:
            
            for type_act, coords in page.get_acts_chancery():
                c = self.classes_dict.get(type_act)
                if c is None:
                    continue
                box_coords_real = self.get_inclusion_box(coords, height_orig, width_orig)
                coordsPr = Polygon(coords)
                pR = unary_union(coordsPr)
                try:
                    coordsR = np.array(pR.exterior.coords)
                except Exception as e:
                    print(f'{fname}  - pR {pR}')
                    for i in pR:
                        print(i)
                    raise 
                px, py = self.get_segm(coordsR, height_orig, width_orig)
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x] # [(1,2), (3,4)] -> [1,2,3,4]
                boxes.append(box_coords_real)
                gt_classes.append(0)
                gt_classes_acts.append(c)
                text_encoded = []
                row, col = -1, -1
                obj = {
                "bbox": box_coords_real,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": c,
                "gt_act": c,
                "segmentation": [poly],
                'text_encoded': text_encoded,
                "gt_row": c, #TODO not use gt_row but gt_act
                "gt_col": c,
                }
                # print(obj)
                objs.append(obj)
        sample["annotations"] = objs 
        try:
            boxes = np.stack(boxes)
        except:
            boxes = np.array([])
        orig_coords = Boxes(boxes)

        instances.set("gt_boxes", orig_coords)
        instances.set("gt_classes", gt_classes)
        if self.HTR:
            instances.set("text_encoded", texts)
        sample["instances"] = instances

        if self.HTR:
            sample['text_encoded'] = texts
        sample['image_id'] = idx
        sample["height"] = height_orig
        sample["width"] = width_orig
        sample['file_name'] = fname_image
        sample['fname'] = fname_noext

        # if self.acts:
        #     idxs = load_idxs(self.cfg.MODEL.ACTS.IDX_PATH, fname_noext, size=height_orig, order_dict=self.IG_order_dict, nwords=self.cfg.MODEL.ACTS.NWORDS)
        #     sample['idxs'] = idxs
        return  sample

def build_baseline_offset(baseline, offset=50):
    """
    build a simple polygon of width $offset around the
    provided baseline, 75% over the baseline and 25% below.
    """
    try:
        line = LineString(baseline)
        up_offset = line.parallel_offset(offset * 0.75, "right", join_style=2)
        bot_offset = line.parallel_offset(offset * 0.25, "left", join_style=2)
    except:
        #--- TODO: check if this baselines can be saved
        print("Fail on try")
        return False, None
    if (up_offset.is_empty == True
        or bot_offset.is_empty == True
    ):
        print("Fail on empty")
        print(baseline)
        return False, None
    elif (up_offset.type == "MultiLineString" or bot_offset.type == "MultiLineString"):
        #--- kepp longest polygon
        if (up_offset.type == "MultiLineString"):
            outcoords = [list(i.coords) for i in up_offset]
            up_offset= LineString([i for sublist in outcoords for i in sublist])
        if (bot_offset.type == "MultiLineString"):
            outcoords = [list(i.coords) for i in bot_offset]
            bot_offset= LineString([i for sublist in outcoords for i in sublist])
        up_offset = np.array(up_offset.coords).astype(int)
        bot_offset = np.array(bot_offset.coords).astype(int)
        return True, np.vstack((up_offset, bot_offset))
    else:
        up_offset = np.array(up_offset.coords).astype(int)
        bot_offset = np.array(bot_offset.coords).astype(int)
        return True, np.vstack((up_offset, bot_offset))

# @timer
def load_idxs(path_idx:str, fname:str, min_prob:float=0.1, size:int=1024, order_dict:dict={}, nwords:int=1024):
    f = open(os.path.join(path_idx, fname+".idx"), "r")
    lines = f.readlines()[1:]
    f.close()
    res = [[0]*nwords for i in range(size)]
    for l in lines:
        word, _, prob, x1,y1,x2,y2 = l.split()
        prob = float(prob)
        if prob < min_prob:
            continue
        y_mean = (int(y1)+int(y2)) // 2
        pos = order_dict.get(word, -1)
        if pos == -1:
            continue
        res[y_mean][pos] += prob
        # res[y_mean].extend(((word, prob)))
    return res


def get_dataset(path, logger=None, opts=None, split="train", cfg=None):
    def func():
        return dataset.get_all()
    dataset = ImageDataset(path, logger, opts, split, cfg=cfg)
    return func, dataset