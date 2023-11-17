from __future__ import print_function
from __future__ import division
from numpy.core.fromnumeric import resize
import torch
import os, re
import pickle
import numpy as np
from utils import metrics
import cv2, time
import matplotlib.pyplot as plt

def timing_val(func):
    def wrapper(*arg, **kw):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        return (t2 - t1), res, func.__name__
    return wrapper


def load_image(path, dim=None):
    #dim = (width, height)
    if os.path.exists(path+".jpg"):
        p = path+".jpg"
    elif os.path.exists(path+".JPG"):
        p = path+".JPG"
    elif os.path.exists(path+".png"):
        p = path+".png"
    elif os.path.exists(path+".PNG"):
        p = path+".PNG"
    else:
        print("Problem with image {}".format(path))
        return None
    image = cv2.imread(p)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    return image



def check_inputs(opts, logger):
    """
    check if some inputs are correct
    """
    n_err = 0
    # --- check if input files/folders exists
    if opts.do_train:
        if not (os.path.isdir(opts.tr_data) and os.access(opts.tr_data, os.R_OK)):
            n_err = n_err + 1
            logger.error(
                "Folder {} does not exists or is unreadable".format(opts.tr_data)
            )


    if opts.do_test:
        if not (os.path.isdir(opts.te_data) and os.access(opts.te_data, os.R_OK)):
            n_err = n_err + 1
            logger.error(
                "Folder {} does not exists or is unreadable".format(opts.te_data)
            )

    return n_err

def save_to_file(data, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def create_segmentation_img(arr, opts, gt=False, min_prob=0.5):
    width, height = opts.img_size
    res = []
    # if gt:
    # arr = [v['instances'].get_fields()['pred_boxes'] for v in arr]
    arr_ = []
    for i,_ in enumerate(arr):
        v = arr[i]['instances'].get_fields()
        # print(v)
        if not gt:
            scores = v['scores']
            classes = v['pred_classes']
            pred_boxes = v['pred_boxes']
        else:
            pred_boxes = v['gt_boxes']
            classes = torch.ones(len(pred_boxes))
            scores = torch.ones(len(pred_boxes))
        arr_.append((pred_boxes, scores, classes))
    arr = arr_
       

    for pred_boxes, scores, classes in arr:
        # print("img_bboxes ", img_bboxes.size())
        img = np.zeros([width, height])
        for i, [x0, y0, x1, y1] in enumerate(pred_boxes):
            score = scores[i]
            c = classes[i]
            if score >= min_prob:
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                img[x0:x1, y0:y1] = 1 #TODO change per class
        res.append(img)
    return res

def save_img(img, title="", path=""):
    """
    Show the image
    :return:
    """
    plt.title(title)
    # img = np.array(img)
    if type(img) == cv2.UMat:
        img = cv2.UMat.get(img)
    # print(img.shape, type(img))
    
    plt.imshow(img)
    # print(path)
    plt.savefig(path, format='jpg', dpi=400)
    plt.close()

def create_and_save_img(img, bboxes,scores, opts, path_save, name_img, min_th=0.5):
    name_img = name_img.split("/")[-1].split(".")[0]
    height, width = opts.img_size
    # img = load_image(path=os.path.join(opts.img_path, name_img), dim=(width, height))
    color = (255, 0, 0)
    thickness = 1
    for i, coords in enumerate(bboxes):
        x0, y0, x1, y1 = tensor_to_numpy(coords)
        score = scores[i]
        print(name_img, [x0, y0, x1, y1], score)
        # min_th = 0
        if score > min_th:
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            start_point = (x0, y0)
            end_point = (x1, y1)
            # print(start_point, end_point, targets['lines_id'][i], targets['lines_id'][i], targets['orig_coords'][i])
            img = cv2.rectangle(img, start_point, end_point, color, thickness)
            # break
    save_img(img, title=name_img, path=os.path.join(path_save, f'{name_img}.jpg') )

def save_batch_img(targets, boxes, opts, path_save, min_th=0.5):
    i_total = 0
    for i, target in enumerate(boxes):
        i_total += 1
        # print(targets[i].keys())
        img = targets[i]['image']#.transpose(1,2,0)
        img = np.transpose(tensor_to_numpy(img), (1, 2, 0))
        # print(type(img))
        file_name = targets[i]['file_name']
        v = target['instances'].get_fields()
        scores = v['scores']
        bboxes = v['pred_boxes']
        create_and_save_img(img, bboxes, scores, opts, path_save, file_name)


def evaluate(net, dataloader, opts, device, epoch, writer, logger, split, path_save):
    net.eval()
    with torch.no_grad():
        pix_accs, mean_accs, mean_ius, freq_w_ius = [], [], [], []
        loss_objectness_total = 0
        loss_rpn_box_reg_total = 0
        total_loss = 0
        boxes_arr, targets_arr = [], []
        for batch, targets in enumerate(dataloader):
            # targets = [{k: v for k, v in t.items()} for t in targets]
            boxes = net(device, targets)
            # print(boxes)
            img_segm = create_segmentation_img(boxes, opts)
            img_segm_gt = create_segmentation_img(targets, opts, gt=True)
            pix_acc, mean_acc, mean_iu, freq_w_iu = metrics.calc_metrics(img_segm, img_segm_gt)
            pix_accs.extend(pix_acc)
            mean_accs.extend(mean_acc)
            mean_ius.extend(mean_iu)
            freq_w_ius.extend(freq_w_iu)
            
            boxes_arr.extend(boxes)
            targets_arr.extend(targets)

            if ((split.lower() == "train" or split.lower() == "dev") and epoch > 0 and epoch % opts.save_train == 0) or split.lower() == "test":
                save_batch_img(targets, boxes, opts, path_save)
        metrics_detection = metrics.calc_detection_metric(boxes_arr, targets_arr)
        pix_acc = np.mean(pix_accs)
        mean_acc = np.mean(mean_accs)
        mean_iu = np.mean(mean_ius)
        freq_w_iu = np.mean(freq_w_iu)
        writer.add_scalar(f'{split}/pix_acc', pix_acc, epoch)
        writer.add_scalar(f'{split}/mean_acc', mean_acc, epoch)
        writer.add_scalar(f'{split}/mean_iu', mean_iu, epoch)
        writer.add_scalar(f'{split}/freq_w_iu', freq_w_iu, epoch)
        logger.info(f'{split} - Epoch {epoch} pix_acc {pix_acc}')
        logger.info(f'{split} - Epoch {epoch} mean_acc {mean_acc}')
        logger.info(f'{split} - Epoch {epoch} mean_iu {mean_iu}')
        logger.info(f'{split} - Epoch {epoch} freq_w_iu {freq_w_iu}')
        for metric_name, value in metrics_detection.items():
            logger.info(f'{split} - Epoch {epoch} {metric_name} {value}')
            writer.add_scalar(f'{split}/{metric_name}', value, epoch)
        
        writer.flush()

        return total_loss
