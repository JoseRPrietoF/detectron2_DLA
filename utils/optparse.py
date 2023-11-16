from __future__ import print_function
from __future__ import division

import numpy as np
from collections import OrderedDict
import argparse
import os

# from math import log
import multiprocessing
import logging

# from evalTools.metrics import levenshtein


class Arguments(object):
    """
    """

    def __init__(self, logger=None):
        """
        """
        self.logger = logger or logging.getLogger(__name__)
        parser_description = """
        NN Implentation for Layout Analysis
        """
        n_cpus = multiprocessing.cpu_count()

        self.parser = argparse.ArgumentParser(
            description=parser_description,
            fromfile_prefix_chars="@",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        # ----------------------------------------------------------------------
        # ----- Define general parameters
        # ----------------------------------------------------------------------
        general = self.parser.add_argument_group("General Parameters")
        general.add_argument(
            "--config", default=None, type=str, help="Use this configuration file"
        )
        general.add_argument(
            "--test_lst", default=None, type=str, help="Use this configuration file"
        )
        general.add_argument(
            "--train_lst", default=None, type=str, help="Use this configuration file"
        )
        general.add_argument(
            "--exp_name",
            default="rpn",
            type=str,
            help="""Name of the experiment. Models and data 
                                       will be stored into a folder under this name""",
        )
        general.add_argument(
            "--path_syms",
            default="rpn",
            type=str,
            help="""Name of the experiment. Models and data 
                                       will be stored into a folder under this name""",
        )
        
        general.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
        general.add_argument(
            "--work_dir",  default="./work/", type=str, help="Where to place output data"
        )
        general.add_argument(
            "--loss_function", default="OHEM", type=str,
            help="arr_result,arr_result_easy,arr_result_more_easy.too_much_easy "
        )
        general.add_argument(
            "--difficulty", default="arr_result", type=str,
            help="arr_result,arr_result_easy,arr_result_more_easy.too_much_easy "
        )
        # --- Removed, input data should be handled by {tr,val,te,prod}_data variables
        # general.add_argument('--data_path', default='./data/',
        #                     type=self._check_in_dir,
        #                     help='path to input data') 
        general.add_argument(
            "--log_level",
            default="INFO",
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level",
        )
        general.add_argument(
            "--optimizer",
            default="Adam",
            type=str,
            choices=["Adam", "RMSprop"],
            help="Logging level",
        )
        # general.add_argument('--baseline_evaluator', default=baseline_evaluator_cmd,
        #                     type=str, help='Command to evaluate baselines')
        general.add_argument(
            "--num_workers",
            default=n_cpus,
            type=int,
            help="""Number of workers used to proces 
                                  input data. If not provided all available
                                  CPUs will be used.
                                  """,
        )
        general.add_argument(
            "--gpu",
            default=0,
            type=int,
            help=(
                "GPU id. Use -1 to disable. "
            ),
        )
        general.add_argument(
            "--seed",
            default=0,
            type=int,
            help="Set manual seed for generating random numbers",
        )
        general.add_argument(
            "--dims",
            default=16,
            type=int,
            help="Set manual seed for generating random numbers",
        )
        general.add_argument(
            "--pools",
            default=5,
            type=int,
            help="Set the num of pools.",
        )
        general.add_argument(
            "--ratio_OHEM",
            default=3,
            type=int,
            help="Set manual seed for generating random numbers",
        )
        general.add_argument(
            "--no_display",
            default=False,
            action="store_true",
            help="Do not display data on TensorBoard",
        )
        general.add_argument(
            "--only_preprocess",
            default=False,
            action="store_true",
            help="only_preprocess",
        )
        general.add_argument(
            "--use_global_log",
            default="",
            type=str,
            help="Save TensorBoard log on this folder instead default",
        )
        general.add_argument(
            "--log_comment",
            default="",
            type=str,
            help="Add this commaent to TensorBoard logs name",
        )
        general.add_argument(
            "--classes",
            default="",
            type=self._str_to_list,
            help="Add this commaent to TensorBoard logs name",
        )
        general.add_argument(
            "--steps",
            default="",
            type=self._str_to_list_int,
            help="Steps for the solver",
        )
        general.add_argument(
            "--POST_NMS_TOPK_TEST",
            default=1000,
            type=int,
            help="Steps for the solver",
        )
        general.add_argument(
            "--POST_NMS_TOPK_TRAIN",
            default=1000,
            type=int,
            help="Steps for the solver",
        )
        general.add_argument(
            "--DETECTIONS_PER_IMAGE",
            default=1000,
            type=int,
            help="Steps for the solver",
        )
        general.add_argument(
            "--PRE_NMS_TOPK_TRAIN",
            default=2000,
            type=int,
            help="Steps for the solver",
        )
        general.add_argument(
            "--PRE_NMS_TOPK_TEST",
            default=2000,
            type=int,
            help="Steps for the solver",
        )
        general.add_argument(
            "--SCORE_THRESH_TEST",
            default=0.05,
            type=float,
            help="Steps for the solver",
        )
        general.add_argument(
            "--px_height_offset", 
            default=30,
            type=int,
            help="Steps for the solver",
        )
        general.add_argument(
            "--BATCH_SIZE_PER_IMAGE_RPN",
            default=1000,
            type=int,
            help="Steps for the solver",
        )
        general.add_argument(
            "--ROI_HEADS_IN_FEATURES",
            default=["p2", "p3", "p4", "p5"],
            type=self._str_to_list,
            help="Steps for the solver",
        )
        general.add_argument(
            "--BATCH_SIZE_PER_IMAGE_ROI",
            default=1000,
            type=int,
            help="Steps for the solver",
        )
        general.add_argument(
            "--WORLD_SIZE",
            default=0,
            type=int,
            help="Steps for the solver",
        )
        general.add_argument(
            "--min_w",
            default=0.5,
            type=float,
            help="Steps for the solver",
        )
        general.add_argument(
            "--img_path",
            default="",
            type=str,
            help="Add this commaent to TensorBoard logs name",
        )
        general.add_argument(
            "--hyp_text",
            default="",
            type=str,
            help="Add this commaent to TensorBoard logs name",
        )
        # ----------------------------------------------------------------------
        # ----- Define distributed parameters
        distributed = self.parser.add_argument_group("distributed Parameters")
        distributed.add_argument(
            "--num_machines", default=1, type=int, help=""
        )
        distributed.add_argument(
            "--machine_rank", default=0, type=int, help=""
        )
        distributed.add_argument(
            "--dist_url", default="auto", type=str, help=""
        )
        
        # ----------------------------------------------------------------------
        # ----- Define processing data parameters
        # ----------------------------------------------------------------------
        data = self.parser.add_argument_group("Data Related Parameters")
        data.add_argument(
            "--img_size",
            default=[1024, 768],
            nargs=2,
            type=self._check_to_int_array,
            help="Scale images to this size. Format --img_size H W",
        )


        # ----------------------------------------------------------------------
        # ----- Define dataloader parameters
        # ----------------------------------------------------------------------
        loader = self.parser.add_argument_group("Data Loader Parameters")
        loader.add_argument(
            "--batch_size", default=6, type=int, help="Number of images per mini-batch"
        )
        l_meg1 = loader.add_mutually_exclusive_group(required=False)
        l_meg1.add_argument(
            "--shuffle_data",
            dest="shuffle_data",
            action="store_true",
            help="Suffle data during training",
        )
        l_meg1.add_argument(
            "--no-shuffle_data",
            dest="shuffle_data",
            action="store_false",
            help="Do not suffle data during training",
        )
        l_meg1.set_defaults(shuffle_data=True)
        l_meg2 = loader.add_mutually_exclusive_group(required=False)
        l_meg2.add_argument(
            "--pin_memory",
            dest="pin_memory",
            action="store_true",
            help="Pin memory before send to GPU",
        )
        l_meg2.add_argument(
            "--no-pin_memory",
            dest="pin_memory",
            action="store_false",
            help="Pin memory before send to GPU",
        )
        l_meg2.set_defaults(pin_memory=True)
        l_meg3 = loader.add_mutually_exclusive_group(required=False)
        l_meg3.add_argument(
            "--flip_img",
            dest="flip_img",
            action="store_true",
            help="Randomly flip images during training",
        )
        l_meg3.add_argument(
            "--no-flip_img",
            dest="flip_img",
            action="store_false",
            help="Do not randomly flip images during training",
        )
        l_meg3.set_defaults(flip_img=False)

        elastic_def = loader.add_mutually_exclusive_group(required=False)
        elastic_def.add_argument(
            "--elastic_def",
            dest="elastic_def",
            action="store_true",
            help="Use elastic deformation during training",
        )
        elastic_def.add_argument(
            "--no-elastic_def",
            dest="elastic_def",
            action="store_false",
            help="Do not Use elastic deformation during training",
        )
        elastic_def.set_defaults(elastic_def=True)

        loader.add_argument(
            "--e_alpha",
            default=0.045,
            type=float,
            help="alpha value for elastic deformations",
        )
        loader.add_argument(
            "--e_stdv",
            default=4,
            type=float,
            help="std dev value for elastic deformations",
        )

        affine_trans = loader.add_mutually_exclusive_group(required=False)
        affine_trans.add_argument(
            "--affine_trans",
            dest="affine_trans",
            action="store_true",
            help="Use affine transformations during training",
        )
        affine_trans.add_argument(
            "--no-affine_trans",
            dest="affine_trans",
            action="store_false",
            help="Do not Use affine transformations during training",
        )
        affine_trans.set_defaults(affine_trans=True)

        only_table = loader.add_mutually_exclusive_group(required=False)
        only_table.add_argument(
            "--only_table",
            dest="only_table",
            action="store_true",
            help="Use affine transformations during training",
        )
        only_table.add_argument(
            "--no-only_table",
            dest="only_table",
            action="store_false",
            help="Do not Use affine transformations during training",
        )

        loader.add_argument(
            "--t_stdv",
            default=0.05,
            type=float,
            help="std deviation of normal dist. used in translate",
        )
        loader.add_argument(
            "--r_kappa",
            default=220,
            type=float,
            help="concentration of von mises dist. used in rotate",
        )
        loader.add_argument(
            "--sc_stdv",
            default=0.1,
            type=float,
            help="std deviation of log-normal dist. used in scale",
        )
        loader.add_argument(
            "--sh_kappa",
            default=225,
            type=float,
            help="concentration of von mises dist. used in shear",
        )
        loader.add_argument(
            "--trans_prob",
            default=0.5,
            type=float,
            help="probability to perform a transformation",
        )
        loader.add_argument(
            "--do_prior",
            default=False,
            type=bool,
            help="Compute prior distribution over classes",
        )
        loader.add_argument(
            "--bl_orientation",
            default=False,
            type=bool,
            help="bl_orientation",
        )
        loader.add_argument(
            "--create_GT_tables",
            default=False,
            type=bool,
            help="create_GT_tables",
        )
        loader.add_argument(
            "--only_F",
            default=False,
            type=bool,
            help="Compute prior distribution over classes",
        )
        loader.add_argument(
            "--IMF",
            default="False",
            type=self._str_to_bool,
            help="IMF calc",
        )
        loader.add_argument(
            "--optim_line",
            default="True",
            type=self._str_to_bool,
            help="IMF calc",
        )
        loader.add_argument(
            "--from_baseline",
            default="False",
            type=self._str_to_bool,
            help="IMF calc",
        )
        loader.add_argument(
            "--pp_BL",
            default="True",
            type=self._str_to_bool,
            help="Post process by BLs",
        )
        # ----------------------------------------------------------------------
        # ----- Define NN parameters
        # ----------------------------------------------------------------------
        net = self.parser.add_argument_group("Neural Networks Parameters")
        net.add_argument(
            "--input_channels",
            default=3,
            type=int,
            help="Number of channels of input data",
        )
        net.add_argument(
            "--item_to_search",
            default=2,
            type=int,
            help="Item to search as output. Default 2 (Chancery)",
        )
        net.add_argument(
            "--output_channels",
            default=2,
            type=int,
            help="Number of channels for the output",
        )
        net.add_argument(
            "--heads_att",
            default=8,
            type=int,
            help="Number of channels of input data",
        )
        net.add_argument(
            "--dk",
            default=32,
            type=int,
            help="Number of channels of input data",
        )
        net.add_argument(
            "--dv",
            default=32,
            type=int,
            help="Number of channels of input data",
        )
        net.add_argument(
            "--cnn_ngf", default=12, type=int, help="Number of filters of CNNs"
        )
        n_meg = net.add_mutually_exclusive_group(required=False)

        net.add_argument(
            "--g_loss",
            default="CTC",
            type=str,
            choices=["BCE", "NLL", "CTC"],
            help="Loss function",
        )

        # ----------------------------------------------------------------------
        # ----- Define Optimizer parameters
        # ----------------------------------------------------------------------
        optim = self.parser.add_argument_group("Optimizer Parameters")
        optim.add_argument(
            "--lr",
            default=0.001,
            type=float,
            help="Initial Lerning rate for ADAM opt",
        )
        optim.add_argument(
            "--scale_ctc",
            default=5,
            type=float,
            help="Initial Lerning rate for ADAM opt",
        )
        optim.add_argument(
            "--momentum",
            default=0.9,
            type=float,
            help="Initial Lerning rate for ADAM opt",
        )
        optim.add_argument(
            "--weight_decay",
            default=0.0001,
            type=float,
            help="Initial Lerning rate for ADAM opt",
        )
        optim.add_argument(
            "--lr_backbone",
            default=0.0001,
            type=float,
            help="Initial Lerning rate for backbone",
        )
        optim.add_argument(
            "--step_scheduler",
            default=30,
            type=int,
            help="Initial Lerning rate for ADAM opt",
        )
        optim.add_argument(
            "--n_blocks",
            default=6,
            type=int,
            help="Initial Lerning rate for ADAM opt",
        )
        optim.add_argument(
            "--gamma_scheduler",
            default=0.1,
            type=float,
            help="Initial Lerning rate for ADAM opt",
        )
        optim.add_argument(
            "--adam_beta1",
            default=0.5,
            type=float,
            help="First ADAM exponential decay rate",
        )
        optim.add_argument(
            "--adam_beta2",
            default=0.999,
            type=float,
            help="Secod ADAM exponential decay rate",
        )
        optim.add_argument(
            "--alpha_mae",
            default=0.001,
            type=float,
            help="Alpha to ponderate the loss function on skewing",
        )
        # ----------------------------------------------------------------------
        # ----- Define Train parameters
        # ----------------------------------------------------------------------
        train = self.parser.add_argument_group("Training Parameters")
        tr_meg = train.add_mutually_exclusive_group(required=False)
        tr_meg.add_argument(
            "--do_train", dest="do_train", action="store_true", help="Run train stage"
        )
        tr_meg.add_argument(
            "--no-do_train",
            dest="do_train",
            action="store_false",
            help="Do not run train stage",
        )
        tr_meg.set_defaults(do_train=True)

        skew = train.add_mutually_exclusive_group(required=False)
        skew.add_argument(
            "--do_skew", dest="do_skew", action="store_true", help="Run train stage"
        )
        skew.add_argument(
            "--no-skew",
            dest="do_skew",
            action="store_false",
            help="Do not run train stage",
        )
        skew.set_defaults(do_skew=True)

        debug = train.add_mutually_exclusive_group(required=False)
        debug.add_argument(
            "--debug", dest="debug", action="store_true", help="Run train stage"
        )
        debug.add_argument(
            "--no-debug",
            dest="debug",
            action="store_false",
            help="Do not run train stage",
        )
        debug.set_defaults(debug=False)

        debug.add_argument(
            "--print_img", dest="print_img", action="store_true", help="Run train stage"
        )
        debug.add_argument(
            "--no-print_img",
            dest="print_img",
            action="store_false",
            help="Do not run train stage",
        )
        debug.set_defaults(print_img=False)

        only_blines = train.add_mutually_exclusive_group(required=False)
        only_blines.add_argument(
            "--only_blines", dest="only_blines", action="store_true", help="Run train stage"
        )
        only_blines.add_argument(
            "--not-only_blines",
            dest="only_blines",
            action="store_false",
            help="Do not run train stage",
        )
        only_blines.set_defaults(only_blines=False)

        with_lines = train.add_mutually_exclusive_group(required=False)
        with_lines.add_argument(
            "--with_lines", dest="with_lines", action="store_true", help="Run train stage"
        )
        with_lines.add_argument(
            "--not-with_lines",
            dest="with_lines",
            action="store_false",
            help="Do not run train stage",
        )
        with_lines.set_defaults(with_lines=False)

        with_projection = train.add_mutually_exclusive_group(required=False)
        with_projection.add_argument(
            "--with_projection", dest="with_projection", action="store_true", help="Run train stage"
        )
        with_projection.add_argument(
            "--not-with_projection",
            dest="with_projection",
            action="store_false",
            help="Do not run train stage",
        )
        with_projection.set_defaults(with_projection=False)

        only_cols = train.add_mutually_exclusive_group(required=False)
        only_cols.add_argument(
            "--only_cols", dest="only_cols", action="store_true", help="only_cols and the table shape"
        )
        only_cols.add_argument(
            "--not-only_cols",
            dest="only_cols",
            action="store_false",
            help="Do not run train stage",
        )
        only_cols.set_defaults(only_cols=False)

        train.add_argument( 
            "--rc_data",
            default="./data/train/",
            type=str,
            help="""Train rc_data folder. Pkl's""",
        )
        train.add_argument(
            "--type_block",
            default="residual",
            type=str,
            help="""Residual or ContractiveBlock |conv""",
        )
        train.add_argument(
            "--type_seq",
            default="lstm",
            type=str,
            help="""lstm | mha | transformer""",
        )

        train.add_argument(
            "--cont_train",
            default=False,
            action="store_true",
            help="Continue training using this model",
        )

        train.add_argument(
            "--soft_labels",
            default=False,
            action="store_true",
            help="soft_labels",
        )

        train.add_argument(
            "--count_acts",
            default=False,
            action="store_true",
            help="count_acts",
        )

        train.add_argument(
            "--next_page",
            default=False,
            action="store_true",
            help="Use next page",
        )

        train.add_argument( 
            "--prev_model",
            default=None,
            type=str,
            help="Use this previously trainned model",
        )
        train.add_argument(
            "--model",
            default="get_config_mask_rcnn_R_50_FPN_3x",
            type=str,
            help="normal or reduced (16)",
        )
        train.add_argument(
            "--structure",
            default="False",
            type=self._str_to_bool,
            help="normal or reduced (16)",
        )
        train.add_argument(
            "--backbone",
            default="resnet18 or resnet50 by the momment",
            type=str,
            help="backbone resnet18 or resnet50 | efficientnet_fpn_b0 | fcos_efficientnet_fpn_b0 ",
        ) 
        train.add_argument(
            "--fusion_model",
            default="GMU",
            type=str,
            help="normal or reduced (16)", 
        )
        train.add_argument(
            "--NN",
            default="normal",
            type=str,
            help="normal, residual or BiLSTM. nextpagev2",
        )
        train.add_argument(
            "--save_rate",
            default=10,
            type=int,
            help="Save checkpoint each --save_rate epochs",
        )
        train.add_argument(
            "--lstm_layers",
            default=3,
            type=int,
            help="Number of layers of lstm",
        )
        train.add_argument(
            "--hsize_lstm",
            default=10,
            type=int,
            help="Number of hidden size of lstm",
        )
        train.add_argument(
            "--max_width",
            default=1,
            type=float,
            help="Number of hidden size of lstm",
        )
        train.add_argument(
            "--hidden_dim",
            default=256,
            type=int,
            help="Number of hidden size of lstm",
        )
        train.add_argument(
            "--conv_blocks",
            default=5,
            type=int,
            help="Number of convs",
        )
        train.add_argument(
            "--dim_feedforward",
            default=2048,
            type=int,
            help="Number of heads for transformer and mha. Must be input multiple of mha or transformer",
        )
        train.add_argument(
            "--nheads",
            default=6,
            type=int,
            help="Number of heads for transformer and mha. Must be input multiple of mha or transformer",
        )
        train.add_argument(
            "--nlayers_t",
            default=6,
            type=int,
            help="Number of layers of transformer encoder",
        )
        train.add_argument(
            "--enc_layers",
            default=6,
            type=int,
            help="Number of layers of transformer encoder",
        )
        train.add_argument(
            "--dec_layers",
            default=6,
            type=int,
            help="Number of layers of transformer encoder",
        )
        train.add_argument(
            "--show_test",
            default=5,
            type=int,
            help="Save checkpoint each --save_rate epochs",
        )
        train.add_argument(
            "--show_train",
            default=1,
            type=int,
            help="Save checkpoint each --save_rate epochs",
        )
        train.add_argument(
            "--save_train",
            default=100,
            type=int,
            help="Save images",
        )
        train.add_argument(
            "--tr_data",
            default="./data/train/",
            type=str,
            help="""Train data folder. Train images are
                                   expected there, also PAGE XML files are
                                   expected to be in --tr_data/page folder""",
        )
        train.add_argument(
            "--tr_xmls",
            default="/data/chancery2/labelled_volumes/train_page/",
            type=str,
            help="""Train data folder. Train images are
                                           expected there, also PAGE XML files are
                                           expected to be in --tr_data/page folder""",
        )
        train.add_argument(
            "--te_xmls",
            default="/data/chancery2/labelled_volumes/test_page/",
            type=str,
            help="""Train data folder. Train images are
                                           expected there, also PAGE XML files are
                                           expected to be in --tr_data/page folder""",
        )
        train.add_argument(
            "--context",
            default="False",
            type=self._str_to_bool,
            help="use prev and next page as context. Uses 'context_model' as model to use different pages",
        )
        train.add_argument(
            "--fusion",
            default="middle",
            type=str,
            help="middle or late",
        )
        train.add_argument(
            "--context_model",
            default="shared_backbone",
            type=str,
            help="""shared_backbone = shared parameteres for the backbone of prev and next page """,
        )
        train.add_argument(
            "--idxs",
            default="False",
            type=self._str_to_bool,
            help="use prev and next page as context. Uses 'context_model' as model to use different pages",
        )
        train.add_argument(
            "--idxs_path",
            default="/data/chancery2/labelled_volumes/idxs",
            type=str,
            help="""shared_backbone = shared parameteres for the backbone of prev and next page """,
        )
        train.add_argument(
            "--words_used",
            default=512,
            type=int,
            help="Save checkpoint each --save_rate epochs",
        )
        train.add_argument(
            "--order_words_path",
            default="/data/chancery2/labelled_volumes/words_per_class_train/IG",
            type=str,
            help="""shared_backbone = shared parameteres for the backbone of prev and next page """,
        )
        train.add_argument(
            "--epochs", default=100, type=int, help="Number of training epochs"
        )

        train.add_argument(
            "--fix_class_imbalance",
            default=False,
            type=bool,
            help="use weights at loss function to handle class imbalance.",
        )
        train.add_argument(
            "--deep_loss",
            default=False,
            type=bool,
            help="deep_loss",
        )
        train.add_argument(
            "--idx_image",
            default=False,
            type=bool,
            help="idx_image (channels +1)",
        )
        train.add_argument(
            "--idx_vector",
            default=0,
            type=int,
            help="0 no vector. 1 vector mean (divided). 2 two vector separateds",
        )
        train.add_argument(
            "--deep_loss_alpha",
            default=0.1,
            type=float,
            help="deep_loss_alpha multiplier",
        )
        train.add_argument(
            "--alpha_OHEM",
            default=0.1,
            type=float,
            help="alpha_OHEM multiplier",
        )
        train.add_argument(
            "--weight_const",
            default=1.02,
            type=float,
            help="weight constant to fix class imbalance",
        )
        train.add_argument(
            "--droput_blstm",
            default=0,
            type=float,
            help="Dropout on the lstm layers",
        )
        # ----------------------------------------------------------------------
        # ----- Define Test parameters
        # ----------------------------------------------------------------------
        test = self.parser.add_argument_group("Test Parameters")
        te_meg = test.add_mutually_exclusive_group(required=False)
        te_meg.add_argument(
            "--do_test", dest="do_test", action="store_true", help="Run test stage"
        )
        te_meg.add_argument(
            "--no-do_test",
            dest="do_test",
            action="store_false",
            help="Do not run test stage",
        )
        te_meg.set_defaults(do_test=False)
        te_save = test.add_mutually_exclusive_group(required=False)
        te_save.add_argument(
            "--save_test", dest="save_test", action="store_true", help="Save the result as pickle file"
        )
        te_save.add_argument(
            "--no-save_test",
            dest="save_test",
            action="store_false",
            help="Dont Save the result as pickle file",
        )
        te_save.set_defaults(save_test=False)
        te_save_res = test.add_mutually_exclusive_group(required=False)
        te_save_res.add_argument(
            "--te_save_res", dest="te_save_res", action="store_true", help="Save the result as pickle file"
        )
        te_save_res.add_argument(
            "--no-te_save_res",
            dest="te_save_res",
            action="store_false",
            help="Dont Save the result as pickle file",
        )
        te_save_res.set_defaults(save_test=False)
        test.add_argument(
            "--te_data",
            default="./data/test/",
            type=str,
            help="""Test data folder. Test images are
                                 expected there, also PAGE XML files are
                                 expected to be in --te_data/page folder
                                 """,
        )
        test.add_argument(
            "--output_dir",
            default="./data/output/",
            type=str,
            help="""Test data folder. Test images are
                                 expected there, also PAGE XML files are
                                 expected to be in --te_data/page folder
                                 """,
        )

        # ----------------------------------------------------------------------
        # ----- Define PRODUCTION parameters
        # ----------------------------------------------------------------------
        prod = self.parser.add_argument_group("Prod Parameters")
        prod_meg = prod.add_mutually_exclusive_group(required=False)
        prod_meg.add_argument(
            "--do_prod", dest="do_prod", action="store_true", help="Run test stage"
        )
        prod_meg.add_argument(
            "--no-do_prod",
            dest="do_prod",
            action="store_false",
            help="Do not run test stage",
        )
        prod_meg.set_defaults(do_prod=False)
        prod.add_argument(
            "--prod_data",
            default="./data/prod/",
            type=str,
            help="""Prod data folder.""",
        )
        prod.add_argument(
            "--dpi",
            default=300,
            type=int,
            help="""Prod data folder.""",
        )
        prod.add_argument(
            "--num_segments",
            default=4,
            type=int,
            help="""Prod data folder.""",
        )
        prod.add_argument(
            "--th_vert",
            default=2,
            type=float,
            help="""Prod data folder.""",
        )
        prod.add_argument(
            "--max_vertex",
            default=30,
            type=int,
            help="""Prod data folder.""",
        )
        prod.add_argument(
            "--approx_alg",
            default="optimal",
            type=str,
            help="""Prod data folder.""",
        )
        # ----------------------------------------------------------------------
        # ----- Define Validation parameters
        # ----------------------------------------------------------------------
        validation = self.parser.add_argument_group("Validation Parameters")
        v_meg = validation.add_mutually_exclusive_group(required=False)
        v_meg.add_argument(
            "--do_val", dest="do_val", action="store_true", help="Run Validation stage"
        )
        v_meg.add_argument(
            "--no-do_val",
            dest="do_val",
            action="store_false",
            help="do not run Validation stage",
        )
        v_meg.set_defaults(do_val=False)
        validation.add_argument(
            "--val_data",
            default="./data/val/",
            type=str,
            help="""Validation data folder. Validation images are
                                 expected there, also PAGE XML files are
                                 expected to be in --te_data/page folder
                                 """,
        )


        # ----------------------------------------------------------------------
        # ----- Define Evaluation parameters
        # ----------------------------------------------------------------------
        evaluation = self.parser.add_argument_group("Evaluation Parameters")
        evaluation.add_argument(
            "--target_list",
            default="",
            type=str,
            help="List of ground-truth PAGE-XML files",
        )
        evaluation.add_argument(
            "--hyp_list", default="", type=str, help="List of hypotesis PAGE-XMLfiles"
        )

    def _convert_file_to_args(self, arg_line):
        return arg_line.split(" ")

    def _str_to_bool(self, data):
        """
        Nice way to handle bool flags:
        from: https://stackoverflow.com/a/43357954
        """
        if data.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif data.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    def _str_to_list(self, data):
        """
        str splitted with comas
        """
        data = data.split(",")
        return data
    
    def _str_to_list_int(self, data):
        """
        str splitted with comas
        """
        data = data.split(",")
        data = [int(x) for x in data]
        return data

    def _check_out_dir(self, pointer):
        """ Checks if the dir is wirtable"""
        if os.path.isdir(pointer):
            # --- check if is writeable
            if os.access(pointer, os.W_OK):
                if not (os.path.isdir(pointer + "/checkpoints")):
                    os.makedirs(pointer + "/checkpoints")
                    self.logger.debug(
                        "Creating checkpoints dir: {}".format(pointer + "/checkpoints")
                    )
                return pointer
            else:
                raise argparse.ArgumentTypeError(
                    "{} folder is not writeable.".format(pointer)
                )
        else:
            try:
                os.makedirs(pointer)
                self.logger.debug("Creating output dir: {}".format(pointer))
                os.makedirs(pointer + "/checkpoints")
                self.logger.debug(
                    "Creating checkpoints dir: {}".format(pointer + "/checkpoints")
                )
                return pointer
            except OSError as e:
                raise argparse.ArgumentTypeError(
                    "{} folder does not exist and cannot be created\n{}".format(e)
                )

    def _check_in_dir(self, pointer):
        """check if path exists and is readable"""
        if os.path.isdir(pointer):
            if os.access(pointer, os.R_OK):
                return pointer
            else:
                raise argparse.ArgumentTypeError(
                    "{} folder is not readable.".format(pointer)
                )
        else:
            raise argparse.ArgumentTypeError(
                "{} folder does not exists".format(pointer)
            )

    def _check_to_int_array(self, data):
        """check is size is 256 multiple"""
        data = int(data)
        if data > 0 and data % 256 == 0:
            return data
        else:
            raise argparse.ArgumentTypeError(
                "Image size must be multiple of 256: {} is not".format(data)
            )
    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def parse(self):
        """Perform arguments parsing"""
        # --- Parse initialization + command line arguments
        # --- Arguments priority stack:
        # ---    1) command line arguments
        # ---    2) config file arguments
        # ---    3) default arguments
        self.opts, unkwn = self.parser.parse_known_args()
        if unkwn:
            msg = "unrecognized command line arguments: {}\n".format(unkwn)
            print(msg)
            self.parser.error(msg)

        # --- Parse config file if defined
        if self.opts.config != None:
            self.logger.info("Reading configuration from {}".format(self.opts.config))
            self.opts, unkwn_conf = self.parser.parse_known_args(
                ["@" + self.opts.config], namespace=self.opts
            )
            if unkwn_conf:
                msg = "unrecognized  arguments in config file: {}\n".format(unkwn_conf)
                self.parser.error(msg)
            self.opts = self.parser.parse_args(namespace=self.opts)
        # --- Preprocess some input variables
        # --- enable/disable
        self.opts.use_gpu = self.opts.gpu != -1
        if self.opts.bl_orientation:
            self.opts.classes.remove('TextLine')
            self.opts.classes.append('TextLine_h') 
            self.opts.classes.append('TextLine_v')
        # --- make sure to don't use pinned memory when CPU only, DataLoader class
        # --- will copy tensors into GPU by default if pinned memory is True.
        if not self.opts.use_gpu:
            self.opts.pin_memory = False
        # --- set logging data
        self.opts.log_level_id = getattr(logging, self.opts.log_level.upper())
        self.opts.log_file = self.opts.work_dir + "/" + self.opts.exp_name + ".log"
        # --- add merde regions to color dic, so parent and merged will share the same color

        # --- TODO: Move this create dir to check inputs function
        self._check_out_dir(self.opts.work_dir)
        self.opts.checkpoints = os.path.join(self.opts.work_dir, "checkpoints/")
        if self.opts.debug:
            self.create_dir(os.path.join(self.opts.work_dir, "debug/"))
            self.create_dir(os.path.join(self.opts.work_dir, "debug/test"))
            self.create_dir(os.path.join(self.opts.work_dir, "debug/train"))
            self.create_dir(os.path.join(self.opts.work_dir, "debug/dev"))

        # if self.opts.do_class:
        #    self.opts.line_color = 1
        # --- define network output channels based on inputs
        return self.opts

    def __str__(self):
        """pretty print handle"""
        data = "------------ Options -------------"
        try:
            for k, v in sorted(vars(self.opts).items()):
                data = data + "\n" + "{0:15}\t{1}".format(k, v)
        except:
            data = data + "\nNo arguments parsed yet..."

        data = data + "\n---------- End  Options ----------\n"
        return data

    def __repr__(self):
        return self.__str__()
