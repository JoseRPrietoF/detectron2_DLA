#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import time, os, logging
from utils.functions import check_inputs
from utils.optparse import Arguments as arguments
import numpy as np
import torch
from data.dataset_D2 import get_dataset
# from torchvision.models.detection.
from detectron2.engine import  launch
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
# torch.autograd.set_detect_anomaly(True)
from detectron2.utils.visualizer import Visualizer
import cv2
from utils.functions import save_img
from detectron2.utils.logger import setup_logger
from utils.createPage import createPage
from detectron2.engine import DefaultTrainer
from detectron2.engine.defaults import DefaultPredictor
from models import model
from pathlib import Path
VISUALIZE_ON = False

def prepare():
    """
    Logging and arguments
    :return:
    """

    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    # --- keep this logger at DEBUG level, until aguments are processed
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --- Get Input Arguments
    in_args = arguments(logger)
    opts = in_args.parse()
    if check_inputs(opts, logger):
        logger.critical("Execution aborted due input errors...")
        exit(1)

    fh = logging.FileHandler(opts.log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # --- restore ch logger to INFO
    ch.setLevel(logging.INFO)
    return logger, opts

def setup(opts):
    """ 
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add config to detectron2
    if opts.config_file:
        print(f'opts.config_file {opts.config_file}')
        cfg.merge_from_file(opts.config_file)

    return cfg 

def main():
    global_start = time.time()
    logger, opts = prepare()
    # --- set device
    device = torch.device("cuda:{}".format(opts.gpu) if opts.use_gpu else "cpu")
    # --- Init torch random
    # --- This two are suposed to be merged in the future, for now keep boot
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    # --- Init model variable
    torch.set_default_tensor_type("torch.FloatTensor")
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)

    path_res = os.path.join(opts.work_dir, "results")
    if not os.path.exists(path_res):
        os.mkdir(path_res)
    path_res_train = os.path.join(path_res, "train")
    if not os.path.exists(path_res_train):
        os.mkdir(path_res_train)
    path_res_test = os.path.join(path_res, "test")
    if not os.path.exists(path_res_test):
        os.mkdir(path_res_test)
    # --- configure TensorBoard display
    opts.img_size = np.array(opts.img_size, dtype=int)
    logger.info(opts)

    setup_logger(output=opts.work_dir)
    # --------------------------------------------------------------------------
    # -----  TRAIN STEP
    # --------------------------------------------------------------------------
    cfg = setup(opts)
    if opts.do_train:
        train_start = time.time()
        logger.info("Working on training stage...")            
        

        cfg.DATASETS.TRAIN = ("train",)
        cfg.DATASETS.TEST = ("test", )
        cfg.OUTPUT_DIR = opts.work_dir
        cfg.SOLVER.IMS_PER_BATCH = opts.batch_size

        cfg.SOLVER.REFERENCE_WORLD_SIZE = opts.WORLD_SIZE #https://github.com/facebookresearch/detectron2/blob/66d658de02a2579d9516a72d94e98a394e2f0ccf/detectron2/engine/launch.py#L52

        # cfg.SOLVER.GAMMA = 1
        logger.info(cfg)
         # --- Get Train Data
        dataset_function_tr, dataset_tr = get_dataset(path=opts.tr_data, opts=opts, split="train", logger=logger, cfg=cfg)
        DatasetCatalog.register("train", dataset_function_tr)
        if dataset_tr.acts:
            MetadataCatalog.get("train").set(thing_classes=["AI", "AM", "AF", "AC"])
        else:
            MetadataCatalog.get("train").set(thing_classes=opts.classes)


        if opts.do_train:
            if opts.WORLD_SIZE != 0:
                print(f"More than one gpu world size {opts.WORLD_SIZE}")
                launch(
                    m,
                    num_gpus_per_machine=opts.WORLD_SIZE,
                    num_machines=opts.num_machines,
                    machine_rank=opts.machine_rank,
                    dist_url=opts.dist_url,
                    args=(cfg,opts),
                )
            else:
                trainer = DefaultTrainer(cfg) 
                trainer.resume_or_load(resume=True)
                trainer.train()

        ####EVALUATE - Train

        path_res_train_imgs = os.path.join(path_res_train, "inference")
        os.makedirs(path_res_train_imgs, exist_ok=True)
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") 
        metadata = MetadataCatalog.get("train")
        output_page_train = os.path.join(path_res_train_imgs, "page")

        os.makedirs(output_page_train, exist_ok=True)
        
        ####EVALUATE - Test

        path_res_test_imgs = os.path.join(path_res_test, "inference")
        os.makedirs(path_res_test_imgs, exist_ok=True)
        output_page_test = os.path.join(path_res_test_imgs, "page")
        output_lattice_test = os.path.join(path_res_test_imgs, "lattice")
        
        
        os.makedirs(output_page_test, exist_ok=True)
        os.makedirs(output_lattice_test, exist_ok=True)
        if opts.WORLD_SIZE != 0:
            print(f'More than one gpu world size {opts.WORLD_SIZE}')
            launch(
                inference,
                opts.WORLD_SIZE,
                num_machines=1,
                machine_rank=0,
                dist_url="auto",
                args=(opts,output_page_test, output_lattice_test, logger, cfg),
            )
        else: 
            inference(opts,output_page_test, output_lattice_test, logger, cfg, acts=dataset_tr.acts )


def inference(opts,output_page_test, output_lattice_test, logger, cfg, acts=False):
    dataset_function_te, dataset_te = get_dataset(path=opts.te_data, opts=opts, split="test", logger=logger, cfg=cfg)
    DatasetCatalog.register("test", dataset_function_te)
    if acts:
        MetadataCatalog.get("test").set(thing_classes=["AI", "AM", "AF", "AC"], num_classes=4)
    else:
        MetadataCatalog.get("test").set(thing_classes=opts.classes)
    path_res = os.path.join(opts.work_dir, "results")
    path_res_test = os.path.join(path_res, "test")
    path_res_test_imgs = os.path.join(path_res_test, "visualizer")
    Path(path_res_test_imgs).mkdir(parents=True, exist_ok=True)

    print("MODEL --- \n\n\n")

    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get("test")

    for d in dataset_function_te():
            img_name = d["file_name"].split("/")[-1]
            print(f'" =========== Processing image {img_name} =========== ')
            im = cv2.imread(d["file_name"])
            im2 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            instances = outputs["instances"].to("cpu")#.get_fields()
            instances = instances[instances.scores > opts.min_w]
            if VISUALIZE_ON:
                v = Visualizer(im, # [:, :, ::-1]
                    metadata=metadata, 
                    scale=0.5,
                )
                out = v.draw_instance_predictions(instances)
                save_img(out.get_image(), title="", path=os.path.join(path_res_test_imgs, img_name))
            
            createPage(instances, img_name, opts,im=im, dir_output = output_page_test, saveLattice=False, path_lattices=output_lattice_test)
            del outputs
            del im
            del im2
            del instances
    print("Evaluating with COCO metrics") # MODIFIED cocoeval.py pycototools/cocoeval.py
    # COCOEvaluator()
    evaluator = COCOEvaluator("test", output_dir= os.path.join(opts.work_dir, "coco_output"))
    val_loader = build_detection_test_loader(cfg, "test")
    # TODO FIX NUM CLASSES acts
    ev = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(ev)
    logger.info(ev)
            
def m(cfg, opts):
    dataset_function_tr, dataset_tr = get_dataset(path=opts.tr_data, opts=opts, split="train", logger=None, cfg=cfg)
    DatasetCatalog.register("train", dataset_function_tr)
    if dataset_tr.acts:
            MetadataCatalog.get("train").set(thing_classes=["AI", "AM", "AF", "AC"])
    else:
        MetadataCatalog.get("train").set(thing_classes=opts.classes)

    dataset_function_te, dataset_te = get_dataset(path=opts.te_data, opts=opts, split="test", logger=None, cfg=cfg)
    DatasetCatalog.register("test", dataset_function_te)
    if dataset_tr.acts:
        MetadataCatalog.get("test").set(thing_classes=["AI", "AM", "AF", "AC"])
    else:
        MetadataCatalog.get("test").set(thing_classes=opts.classes)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    return trainer.train()

if __name__ == "__main__":
    main()
