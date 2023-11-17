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
            "--exp_name",
            default="rpn",
            type=str,
            help="""Name of the experiment. A log file will be saved under this name""",
        )

        general.add_argument("--config_file", default="", metavar="FILE", help="Use this configuration file [yaml]")
        general.add_argument(
            "--work_dir",  default="./work/", type=str, help="Where to place output data"
        )

        general.add_argument(
            "--log_level",
            default="INFO",
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level",
        )

        general.add_argument(
            "--classes",
            default="",
            type=self._str_to_list,
            help="Add this commaent to TensorBoard logs name",
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
            "--config", default=None, type=str, help="Use this configuration file"
        )
        general.add_argument(
            "--seed",
            default=0,
            type=int,
            help="Set manual seed for generating random numbers",
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
            help="Min probability to accept an instance",
        )
        general.add_argument(
            "--img_path",
            default="",
            type=str,
            help="Add this commaent to TensorBoard logs name",
        )

        # ----------------------------------------------------------------------
        # ----- Define distributed parameters
        distributed = self.parser.add_argument_group("Distributed Parameters")
        distributed.add_argument(
            "--num_machines", default=1, type=int, help="Number of machines"
        )
        distributed.add_argument(
            "--machine_rank", default=0, type=int, help="Machine rank"
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
        
        loader.add_argument(
            "--bl_orientation",
            default=False,
            type=bool,
            help="bl_orientation",
        )
        
        
        loader.add_argument(
            "--from_baseline",
            default="False",
            type=self._str_to_bool,
            help="IMF calc",
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
        
        train.add_argument(
            "--tr_data",
            default="./data/train/",
            type=str,
            help="""Train data folder. Train images are
                                   expected there, also PAGE XML files are
                                   expected to be in --tr_data/page folder""",
        )

        test.add_argument(
            "--te_data",
            default="./data/test/",
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
      
        

        prod.add_argument(
            "--th_vert",
            default=2,
            type=float,
            help="""Parameters for baseline calculation.""",
        )

        prod.add_argument(
            "--px_height_offset", 
            default=30,
            type=int,
            help="Parameters for baseline calculation.",
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
        # --- set logging data
        self.opts.log_level_id = getattr(logging, self.opts.log_level.upper())
        self.opts.log_file = self.opts.work_dir + "/" + self.opts.exp_name + ".log"
        # --- add merde regions to color dic, so parent and merged will share the same color

        # --- TODO: Move this create dir to check inputs function
        self._check_out_dir(self.opts.work_dir)
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
