import glob, os, copy, pickle
from xml.dom import minidom
import numpy as np
import cv2
from baselines.utils.baselines import basic_baseline
from data.page import TablePAGE
import pycocotools.mask as mask_util
from datetime import datetime
import torch

CTC = 0

page_init = """ 
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd">
<Metadata>
<Creator>PRHLT</Creator>
<Created>{5}T00:00:00.358+01:00</Created>
<LastChange>{5}T00:00:00.506+02:00</LastChange>
</Metadata>
<Page imageFilename="{0}" imageWidth="{1}" imageHeight="{2}">
<TextRegion custom="structure {{type:full_page;}}" id="{3}">
<Coords points="0,0 0,{1} {2},{1} {2},0" />
{4}
</TextRegion>
</Page>
</PcGts>
"""

page_init_noTR = """ 
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd">
<Metadata>
<Creator>PRHLT</Creator>
<Created>{4}T12:26:16.358+01:00</Created>
<LastChange>{4}T19:27:22.506+02:00</LastChange>
</Metadata>
<Page imageFilename="{0}" imageWidth="{1}" imageHeight="{2}">
{3}
</Page>
</PcGts>
"""

str_region = """
<{3} id="{0}" custom="structure {{type:{1};}}">
<Coords points="{2}"/>
<Baseline points="{4}"/>
<TextEquiv>
    <Unicode>{5}</Unicode>
</TextEquiv>
</{3}>
"""

str_region_orientation = """
<{3} id="{0}" custom="structure {{type:{1};}}" orientation="{4}">
<Coords points="{2}"/>
<TextEquiv>
    <Unicode>{5}</Unicode>
</TextEquiv>
</{3}>
"""

str_region_orientation_colrow = """
<{3} id="{0}" custom="structure {{type:{1};col:{7};row{6};}}" orientation="{4}" col="{7}" row="{6}">
<Coords points="{2}"/>
<TextEquiv>
    <Unicode>{5}</Unicode>
</TextEquiv>
</{3}>
"""

str_region_tables = """
<{3} id="{0}" custom="structure {{type:{1};col:{5};row:{6};colspan:{7};rowspan:{8};}}">
<Coords points="{2}"/>
<Baseline points="{4}"/>
</{3}>
"""

str_region_noBL = """
<{3} id="{0}" custom="structure {{type:{1};}}">
<Coords points="{2}"/>
<TextEquiv>
    <Unicode>{4}</Unicode>
</TextEquiv>
</{3}>
"""

str_region_act = """
<{3} id="{0}" custom="structure {{type:{1};probBbox:{5};probType:{6};}}" probs="{7}">
<Coords points="{2}"/>
<TextEquiv>
    <Unicode>{4}</Unicode>
</TextEquiv>
</{3}>
"""
classes_dict_acts = {"AI":0, "AM":1, "AF":2, "AC":3}
classes_dict_acts_inv = {v:k for k,v in classes_dict_acts.items()}
#419,2162 617,2162 617,2290 419,2290

from PIL import Image  
def rle_to_poly(rle_data):
    # mask = mask_util.decode(rle_data)
    #--- force copy array, so CV2 can access it on correct format
    mask = rle_data.cpu().numpy().copy()
    mask = mask.astype("uint8")
    # mask = mask.copy()

    res_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # res_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #--- check len to support CV versions
    if len(res_) == 2:
        contours, hierarchy = res_
    else:
        _, contours, hierarchy = res_
    if len(contours) > 1:
        print("Warning: more than one polygon found on RLE mask, using the bigger one")
        cs = [(cv2.contourArea(contour), contour) for contour in contours]
        try:
            cs.sort(key=lambda x: x[0])
        except Exception as e:
            print(cs)
            raise e
        contours = [cs[-1][1]]
    if len(contours) == 0:
        return np.array([])
    else:
        return contours[0]

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def decode(outputs_argmax, ids):
    res = []
    align = []
    last_char = ''
    for i, out in enumerate(outputs_argmax):
        if out != CTC and last_char != out:
            # last_char = out
            res.append(ids[out])

            align.append(i)
        last_char = out
    return res, align

def save_to_file(data, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def trellis_to_syms(t:torch.Tensor, syms:dict, join:bool=True, argmax=True, tensorToNumpy=True): 
    syms_inv = {k:v for v,k in syms.items()}
    if tensorToNumpy:
        t = tensor_to_numpy(t)
    if argmax:
        t = np.argmax(t, axis=-1)
    res, align = decode(t, syms_inv)
    if join:
        res = "".join(res)
        res = res.replace("<space>", " ").replace("!print", "!print ").replace("!manuscript", "!manuscript ")
    return res, align

def createPage(outputs, img_name, opts, dir_output, im=None, hisClima=True, saveLattice=False, path_lattices=""):
    # mask = False #TODO
    try:
        pred_mask_i = outputs[0].pred_masks
        mask = True #TODO
    except:
        mask = False #TODO
    classes = opts.classes
    img_name_ = img_name.split(".")[0]
    h, w = im.shape[:2]
    regions = []
    path_img = os.path.join(opts.img_path, img_name)
    Bls = False
    #TODO
    if "new" in opts.model.lower():
        syms = load_syms(opts)
    # for i, pred_mask_i in enumerate(outputs.pred_masks):
    for i, pred_box_i in enumerate(outputs.pred_boxes):
        pred_class_i = outputs.pred_classes[i]
        score = outputs.scores[i]
        if mask:
            pred_mask_i = outputs.pred_masks[i]
        if "new" in opts.model.lower():
            text_encoded_i = outputs.text_encoded[i]
            text, align_text = trellis_to_syms(text_encoded_i, syms)
        else:
            text = ""
        
        if opts.structure:
            cc_row_i = 0
            # cc_row_i = int(tensor_to_numpy(outputs.cc_rows[i]))
            cc_col_i = int(tensor_to_numpy(outputs.cc_cols[i]))
            

        x1, y1, x2, y2 = [int(x) for x in tensor_to_numpy(pred_box_i)]
        
        if mask:
            poly = rle_to_poly(pred_mask_i)
            poly_str = ""
            for z in poly:
                x,y = z[0]
                poly_str += f'{x},{y} '
            poly = [z[0] for z in poly]
        else:
            Lpoly = [[(x1, y1)], [(x2, y2)]]
            poly = [(x1, y1), (x2,y1), (x2,y2), (x1,y2)]
            poly_str = ""
            for x,y in poly:
                poly_str += f'{x},{y} '
        
        poly = np.array(poly).astype(int)
        pred_class_i = int(tensor_to_numpy(pred_class_i))
        if classes == ["acts"]:
            # scores_hierarchical_i = tensor_to_numpy(outputs.scores_hierarchical[i])
            scores_all = tensor_to_numpy(outputs.scores_all)[i]
            scores_all_str = ""
            for name_class, num_class_ in classes_dict_acts.items():
                scores_all_str += f"{name_class}:{scores_all[num_class_]};"
            scores_hierarchical_i = tensor_to_numpy(outputs.scores)
            pred_classes_i = tensor_to_numpy(outputs.pred_classes)[i]
            # scores_hierarchical_i_argmax = np.argmax(scores_hierarchical_i)
            scores_hierarchical_i = scores_hierarchical_i[i]
            classes_dict = {"AI":0, "AM":1, "AF":2, "AC":3}
            classes_dict_rev = {v:k for k,v in classes_dict.items()}
            type_ = "TextRegion"
            prob_bbox = tensor_to_numpy(score)
            # print(f"{classes_dict_rev[pred_class_i]} - {scores_hierarchical_i[pred_class_i]}  [{scores_hierarchical_i}]")
            id_=f'{img_name_}_{str(i)}'
            regions.append(str_region_act.format(id_, classes_dict_rev[pred_classes_i],poly_str, type_, text, prob_bbox, scores_hierarchical_i, scores_all_str))
        elif classes[pred_class_i] == "TextLine":
            type_ = "TextLine"
            if opts.optim_line:
                trace, aproxLin = basic_baseline(im, poly, opts)
                if not trace:
                    print("No baseline found for line {}".format(f'{img_name}_{str(i)}'))
                    continue
                    
            else:
                largo = (x2 - x1) / w
                ancho = (y2 - y1) / h
                if largo*opts.th_vert >= ancho:
                    aproxLin = [[x1, y2], [x2, y2]]
                else:
                    aproxLin = [[x2, y2], [x2, y1]]
                trace = True

            baseline_coords = create_coords_str(aproxLin)
            if not baseline_coords or baseline_coords == "":
                print(i, " --- ", f'{img_name}_{str(i)}', classes[pred_class_i], x1, y1, x2, y2, poly)
                print(trace, aproxLin, baseline_coords)
                exit()
            id_ = f'{img_name_}_{str(i)}'
            if not opts.structure:
                regions.append(str_region.format(id_, classes[pred_class_i], poly_str, type_, baseline_coords, text))
            else:
                # TODO orientation
                regions.append(str_region.format(id_, classes[pred_class_i], poly_str, type_, baseline_coords, text))
            Bls = True
        elif "TextLine" in classes[pred_class_i]: #TODO orientation
            type_ = "TextLine"
            if "_v" in classes[pred_class_i]:
                orientation = "v"
            else:
                orientation = "h" 
            
            if not opts.structure:
                id_=f'{img_name_}_{str(i)}'
                regions.append(str_region_orientation.format(id_, classes[pred_class_i], poly_str, type_, orientation, text))
            else:
                id_=f'{img_name_}_{str(i)}_row{cc_row_i}_col{cc_col_i}'
                regions.append(str_region_orientation_colrow.format(id_, classes[pred_class_i], poly_str, type_, orientation, text, cc_row_i, cc_col_i))
            Bls = True

        else:
            type_ = "TextRegion"
            id_=f'{img_name_}_{str(i)}'
            regions.append(str_region_noBL.format(id_, classes[pred_class_i],poly_str, type_, text))
        
        if saveLattice:
            path_output_lattices = os.path.join(path_lattices, f'{id_}.pkl')
            save_to_file(text_encoded_i, path_output_lattices)

    regions = "\n".join(regions)
    if Bls:
        page = page_init.format(img_name, w, h, f'textregion_{img_name_}',regions, datetime.now().strftime('%Y-%m-%d')) # 2020-02-10
    else:
        page = page_init_noTR.format(img_name, w, h,regions, datetime.now().strftime('%Y-%m-%d'))
    path_output = os.path.join(dir_output, f'{img_name_}.xml')
    f = open(path_output, "w")
    f.write(page)
    f.close()
    # exit()

page_init_Table = """ 
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd">
<Metadata>
<Creator>PRHLT</Creator>
<Created>{5}T00:00:100.358+01:00</Created>
<LastChange>{5}T00:00:00.506+02:00</LastChange>
</Metadata>
<Page imageFilename="{0}" imageWidth="{1}" imageHeight="{2}">
    {3}
    {4}
</Page>
</PcGts>
"""

coords_str = """<Coords points="{0}" />"""

# tableRegion_str = """
#     {2}
# """

# tableCell_str = """
# <TextRegion row="{0}" col="{1}" rowSpan="{2}" colSpan="{3}" id="{4}">
#     {5}
# </TextRegion>
# """

tableRegion_str = """
<TableRegion id="{0}"  custom="readingOrder {{index:{3};}}">>
    {1}
    {2}
</TableRegion>
"""

tableCell_str = """
<TableCell row="{0}" col="{1}" rowSpan="{2}" colSpan="{3}" id="{4}">
    {5}
    <CornerPts>0 1 2 3</CornerPts>
</TableCell>
"""

def createPage_GTTablesTextLines(outputs, img_name, opts, dir_output,  im=None, train = False, hisClima=True):
    type_ = "TextLine"
    img_name_ = img_name.split(".")[0]
    # print(outputs)
    # print(outputs.pred_masks.shape)
    # print(outputs.pred_boxes.shape)
    h, w = outputs.image_size
    path_img = os.path.join(opts.img_path, img_name)
    table_top, table_bot = [], [] # Only Hisclima
    table_top_coordsX, table_top_coordsY = [], []
    table_bot_coordsX, table_bot_coordsY = [], []
    if train:
        xml_path = os.path.join(opts.tr_data, f'{img_name_}.xml')
    else:
        xml_path = os.path.join(opts.te_data, f'{img_name_}.xml')
    path_output = os.path.join(dir_output, f'{img_name_}.xml')
    page_table = TablePAGE(xml_path, hisClima=True)
    # print(xml_path)
    tls_GT = page_table.get_textLinesFromCell()
    if not tls_GT:
        page = page_init_Table.format(img_name, h, w, "", "", datetime.now().strftime('%Y-%m-%d'))
        f = open(path_output, "w")
        f.write(page)
        f.close()
        return
    for i, pred_mask_i in enumerate(outputs.pred_masks):
        pred_class_i = outputs.pred_classes[i]
        pred_box_i = outputs.pred_boxes[i]
        x1, y1, x2, y2 = [int(x) for x in pred_box_i.tensor.cpu().numpy()[0]]
        poly = rle_to_poly(pred_mask_i)
        poly_str = ""
        for z in poly:
            x,y = z[0]
            poly_str += f'{x},{y} '
        poly = [z[0] for z in poly]
        poly = np.array(poly).astype(int)

        if opts.optim_line:
            trace, aproxLin = basic_baseline(im, poly, opts)
            if not trace:
                continue
        else:
            largo = (x2 - x1) / w
            ancho = (y2 - y1) / h
            if largo*opts.th_vert >= ancho:
                aproxLin = [[x1, y2], [x2, y2]]
            else:
                aproxLin = [[x2, y2], [x2, y1]]
            trace = True

        baseline_coords = create_coords_str(aproxLin)

        meanY = np.mean([y1, y2])
        meanY = meanY / h
        if meanY > 0.5:
            table_top.append((pred_class_i, pred_box_i, x1, y1, x2, y2, baseline_coords, poly_str))
            table_top_coordsY.extend([y1, y2])
            table_top_coordsX.extend([x1, x2])
        else:
            table_bot.append((pred_class_i, pred_box_i, x1, y1, x2, y2, baseline_coords, poly_str))
            table_bot_coordsY.extend([y1, y2])
            table_bot_coordsX.extend([x1, x2])
    
    # print("---------")
    regions = []
    for i, (pred_class_i, pred_box_i, x1, y1, x2, y2, baseline_coords, poly_str) in enumerate(table_top):
        iou, cell_gt = match_IoU_TL([x1, y1, x2, y2], tls_GT)
        # print(f'    IoU -> {iou}')
        # if iou < 0.1:
        #     exit()
        _, _, _, info  = tls_GT[cell_gt]
        col, row, colspan, rowspan = info['col'], info['row'], info['colspan'], info['rowspan']
        id_ = f'cell_top_row{row}col{col}rspan{rowspan}cspan{colspan}_{i}'
        if row == -1 or col == -1:
            id_ = f'notcell_row{row}col{col}rspan{rowspan}cspan{colspan}_{i}_1'
        textline_i = str_region_tables.format(id_, "TextLine", poly_str, type_, baseline_coords, col, row, colspan, rowspan)
        regions.append(textline_i)
    # res_tablecells_top = "\n".join(res_tablecells_top)

    
    for i, (pred_class_i, pred_box_i, x1, y1, x2, y2, baseline_coords, poly_str) in enumerate(table_bot):
        iou, cell_gt = match_IoU_TL([x1, y1, x2, y2], tls_GT)
        # print(f'    IoU -> {iou}')
        _, _, _, info  = tls_GT[cell_gt]
        col, row, colspan, rowspan = info['col'], info['row'], info['colspan'], info['rowspan']
        id_ = f'cell_bot_row{row}col{col}rspan{rowspan}cspan{colspan}_{i}'
        if row == -1 or col == -1:
            id_ = f'notcell_row{row}col{col}rspan{rowspan}cspan{colspan}_{i}_2'
        textline_i = str_region_tables.format(id_, "TextLine", poly_str, type_, baseline_coords, col, row, colspan, rowspan)
        regions.append(textline_i)
    regions = "\n".join(regions)

    page = page_init.format(img_name, w, h, f'textregion_{img_name_}',regions, datetime.now().strftime('%Y-%m-%d'))

    
    f = open(path_output, "w")
    f.write(page)
    f.close()
    # exit()


def createPage_GTTables(outputs, img_name, opts, dir_output, im=None, train = False, hisClima=True):
    classes = opts.classes
    img_name_ = img_name.split(".")[0]
    # print(outputs)
    # print(outputs.pred_masks.shape)
    # print(outputs.pred_boxes.shape)
    h, w = outputs.image_size
    path_img = os.path.join(opts.img_path, img_name)
    table_top, table_bot = [], [] # Only Hisclima
    table_top_coordsX, table_top_coordsY = [], []
    table_bot_coordsX, table_bot_coordsY = [], []
    if train:
        xml_path = os.path.join(opts.tr_data, f'{img_name_}.xml')
    else:
        xml_path = os.path.join(opts.te_data, f'{img_name_}.xml')
    path_output = os.path.join(dir_output, f'{img_name_}.xml')
    page_table = TablePAGE(xml_path, hisClima=True)
    # print(xml_path)
    cells_GT = page_table.get_cells_RAW()
    if not cells_GT:
        page = page_init_Table.format(img_name, h, w, "", "". datetime.now().strftime('%Y-%m-%d'))
        f = open(path_output, "w")
        f.write(page)
        f.close()
        return
    for i, pred_mask_i in enumerate(outputs.pred_masks):
        pred_class_i = outputs.pred_classes[i]
        pred_box_i = outputs.pred_boxes[i]
        x1, y1, x2, y2 = [int(x) for x in pred_box_i.tensor.cpu().numpy()[0]]
        poly = rle_to_poly(pred_mask_i)
        poly_str = ""
        for z in poly:
            x,y = z[0]
            poly_str += f'{x},{y} '
        poly = [z[0] for z in poly]
        poly = np.array(poly).astype(int)
        meanY = np.mean([y1, y2])
        meanY = meanY / h
        if meanY > 0.5:
            table_top.append((pred_class_i, pred_box_i, x1, y1, x2, y2, poly_str))
            table_top_coordsY.extend([y1, y2])
            table_top_coordsX.extend([x1, x2])
        else:
            table_bot.append((pred_class_i, pred_box_i, x1, y1, x2, y2, poly_str))
            table_bot_coordsY.extend([y1, y2])
            table_bot_coordsX.extend([x1, x2])
    top_table_y = [np.min(table_top_coordsY), np.max(table_top_coordsY)]
    top_table_x = [np.min(table_top_coordsX), np.max(table_top_coordsX)]
    bot_table_x = [np.min(table_top_coordsX), np.max(table_top_coordsX)]
    bot_table_y = [np.min(table_bot_coordsY), np.max(table_bot_coordsY)]
    
    # print("---------")
    res_tablecells_top = []
    for i, (pred_class_i, pred_box_i, x1, y1, x2, y2, poly_str) in enumerate(table_top):
        iou, cell_gt = match_IoU([x1, y1, x2, y2], cells_GT)
        # print(f'    IoU -> {iou}')
        _, col, row, colspan, rowspan = cells_GT[cell_gt]
        coords_i = coords_str.format(poly_str)
        tableCell_i = tableCell_str.format(row, col, rowspan, colspan, f'cell_top_row{row}col{col}rspan{rowspan}cspan{colspan}_{i}', coords_i)
        res_tablecells_top.append(tableCell_i)
    res_tablecells_top = "\n".join(res_tablecells_top)
    coords_table_top = coords_str.format(top_table_x[0], top_table_y[0], top_table_x[1], top_table_y[1])
    tableRegion_top = tableRegion_str.format(f'tableTop', coords_table_top, res_tablecells_top, 1)
    
    res_tablecells_bot = []
    for i, (pred_class_i, pred_box_i, x1, y1, x2, y2, poly_str) in enumerate(table_bot):
        iou, cell_gt = match_IoU([x1, y1, x2, y2], cells_GT)
        # print(f'    IoU -> {iou}')
        _, col, row, colspan, rowspan = cells_GT[cell_gt]
        coords_i = coords_str.format(poly_str)
        tableCell_i = tableCell_str.format(row, col, rowspan, colspan, f'cell_bot_row{row}col{col}rspan{rowspan}cspan{colspan}_{i}', coords_i)
        res_tablecells_bot.append(tableCell_i)
    res_tablecells_bot = "\n".join(res_tablecells_bot)
    coords_table_bot = coords_str.format(bot_table_x[0], bot_table_y[0], bot_table_x[1], bot_table_y[1])
    tableRegion_bot = tableRegion_str.format(f'tableBot', coords_table_bot, res_tablecells_bot, 2)

   
    page = page_init_Table.format(img_name, w, h, tableRegion_top, tableRegion_bot, datetime.now().strftime('%Y-%m-%d'))

    
    f = open(path_output, "w")
    f.write(page)
    f.close()
    # exit()

def match_IoU(hyp_BB, cells_GT):
    res = []
    hyp_BB = np.array(hyp_BB)
    # print("-> ", hyp_BB, cells_GT)
    for i, (coords, col, row, colspan, rowspan) in enumerate(cells_GT):
        coords = np.array(coords)
        coords = np.array([coords[:,0].min(), coords[:,1].min(), coords[:,0].max(), coords[:,1].max()])
        iou = IoU(hyp_BB, coords)
        res.append([iou, i])
    # print(res)
    # print("----------")
    res.sort()
    # print(res)
    # exit()
    return res[-1]

def match_IoU_TL(hyp_BB, tls_GT):
    res = []
    hyp_BB = np.array(hyp_BB)
    # print("-> ", hyp_BB, cells_GT)
    for i, (coords,_,_,_) in enumerate(tls_GT):
        coords = np.array(coords)
        coords = np.array([coords[:,0].min(), coords[:,1].min(), coords[:,0].max(), coords[:,1].max()])
        iou = IoU(hyp_BB, coords)
        res.append([iou, i])
    # print(res)
    # print("----------")
    res.sort()
    # print(res)
    # exit()
    return res[-1]

def IoU(box1: np.ndarray, box2: np.ndarray):
    """
    calculate intersection over union cover percent
    :param box1: box1 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    :param box2: box2 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    :return: IoU ratio if intersect, else 0
    """
    # first unify all boxes to shape (N,4)
    if box1.shape[-1] == 2 or len(box1.shape) == 1:
        box1 = box1.reshape(1, 4) if len(box1.shape) <= 2 else box1.reshape(box1.shape[0], 4)
    if box2.shape[-1] == 2 or len(box2.shape) == 1:
        box2 = box2.reshape(1, 4) if len(box2.shape) <= 2 else box2.reshape(box2.shape[0], 4)
    point_num = max(box1.shape[0], box2.shape[0])
    b1p1, b1p2, b2p1, b2p2 = box1[:, :2], box1[:, 2:], box2[:, :2], box2[:, 2:]

    # mask that eliminates non-intersecting matrices
    base_mat = np.ones(shape=(point_num,))
    base_mat *= np.all(np.greater(b1p2 - b2p1, 0), axis=1)
    base_mat *= np.all(np.greater(b2p2 - b1p1, 0), axis=1)

    # I area
    intersect_area = np.prod(np.minimum(b2p2, b1p2) - np.maximum(b1p1, b2p1), axis=1)
    # U area
    union_area = np.prod(b1p2 - b1p1, axis=1) + np.prod(b2p2 - b2p1, axis=1) - intersect_area
    # IoU
    intersect_ratio = intersect_area / union_area

    return base_mat * intersect_ratio

def create_coords_str(coords):
    res = ""
    for x,y in coords:
        res += f'{x},{y}, '
    res = res[:-2]
    return res

def load_syms(opts):
    f = open(opts.path_syms, "r")
    lines = f.readlines()
    f.close()
    syms = {}
    for line in lines:
        sym, value = line.strip().split(" ")
        syms[sym] = int(value)
    return syms