from yacs.config import CfgNode as CN

_C = CN()

# -------------------------------------------------------- #
#                           Input                          #
# -------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.DATASET = "CUHK-SYSU"
_C.INPUT.DATA_ROOT = "/data/PersonSearch/CUHK-SYSU"
_C.INPUT.TDATASET = "PRW"
_C.INPUT.TDATA_ROOT = "/data/PersonSearch/PRW"

# Size of the smallest side of the image
_C.INPUT.MIN_SIZE = 900
# Maximum size of the side of the image
_C.INPUT.MAX_SIZE = 1500

# TODO: support aspect ratio grouping
# Whether to use aspect ratio grouping for saving GPU memory
# _C.INPUT.ASPECT_RATIO_GROUPING_TRAIN = False

# Number of images per batch
_C.INPUT.BATCH_SIZE_TRAIN = 2
_C.INPUT.BATCH_SIZE_TEST = 1

# Number of data loading threads
_C.INPUT.NUM_WORKERS_TRAIN = 5
_C.INPUT.NUM_WORKERS_TEST = 1

# -------------------------------------------------------- #
#                          Solver                          #
# -------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 20

# Learning rate settings
_C.SOLVER.BASE_LR = 0.0024

# TODO: add config option WARMUP_EPOCHS
_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
# _C.SOLVER.WARMUP_EPOCHS = 1

# The epoch milestones to decrease the learning rate by GAMMA
_C.SOLVER.LR_DECAY_MILESTONES = [16]
_C.SOLVER.GAMMA = 0.1

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.SGD_MOMENTUM = 0.9

# Loss weight of RPN regression
_C.SOLVER.LW_RPN_REG = 1
# Loss weight of RPN classification
_C.SOLVER.LW_RPN_CLS = 1
# Loss weight of proposal regression
_C.SOLVER.LW_PROPOSAL_REG = 10
# Loss weight of proposal classification
_C.SOLVER.LW_PROPOSAL_CLS = 1
# Loss weight of box regression
_C.SOLVER.LW_BOX_REG = 1
# Loss weight of box classification
_C.SOLVER.LW_BOX_CLS = 1
# Loss weight of box OIM (i.e. Online Instance Matching)
_C.SOLVER.LW_BOX_REID = 1
_C.SOLVER.LW_BOX_REID_T = 0.1
_C.SOLVER.LW_DA_INS = 0.02

# Set to negative value to disable gradient clipping
_C.SOLVER.CLIP_GRADIENTS = 10.0


# -------------------------------------------------------- #
#                            RPN                           #
# -------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.RPN = CN()
# NMS threshold used on RoIs
_C.MODEL.RPN.NMS_THRESH = 0.7
# Number of anchors per image used to train RPN
_C.MODEL.RPN.BATCH_SIZE_TRAIN = 256
# Target fraction of foreground examples per RPN minibatch
_C.MODEL.RPN.POS_FRAC_TRAIN = 0.5
# Overlap threshold for an anchor to be considered foreground (if >= POS_THRESH_TRAIN)
_C.MODEL.RPN.POS_THRESH_TRAIN = 0.7
# Overlap threshold for an anchor to be considered background (if < NEG_THRESH_TRAIN)
_C.MODEL.RPN.NEG_THRESH_TRAIN = 0.3
# Number of top scoring RPN RoIs to keep before applying NMS
_C.MODEL.RPN.PRE_NMS_TOPN_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOPN_TEST = 6000
# Number of top scoring RPN RoIs to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOPN_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOPN_TEST = 300

# -------------------------------------------------------- #
#                         RoI head                         #
# -------------------------------------------------------- #
_C.MODEL.ROI_HEAD = CN()
# Whether to use bn neck (i.e. batch normalization after linear)
_C.MODEL.ROI_HEAD.BN_NECK = True
# Number of RoIs per image used to train RoI head
_C.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN = 128
# Target fraction of foreground examples per RoI minibatch
_C.MODEL.ROI_HEAD.POS_FRAC_TRAIN = 0.5
# Overlap threshold for an RoI to be considered foreground (if >= POS_THRESH_TRAIN)
_C.MODEL.ROI_HEAD.POS_THRESH_TRAIN = 0.5
# Overlap threshold for an RoI to be considered background (if < NEG_THRESH_TRAIN)
_C.MODEL.ROI_HEAD.NEG_THRESH_TRAIN = 0.5
# Minimum score threshold
_C.MODEL.ROI_HEAD.SCORE_THRESH_TEST = 0.5
# NMS threshold used on boxes
_C.MODEL.ROI_HEAD.NMS_THRESH_TEST = 0.4
# Maximum number of detected objects
_C.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST = 300

# -------------------------------------------------------- #
#                  Update factor                           #
# -------------------------------------------------------- #
_C.MODEL.UPDATE_FACTOR = CN()
_C.MODEL.UPDATE_FACTOR.ONLINE = 0.2
_C.MODEL.UPDATE_FACTOR.OFFLINE = 0.2
# -------------------------------------------------------- #
#                           Loss                           #
# -------------------------------------------------------- #
_C.MODEL.DA_HEADS = CN()
_C.MODEL.DA_HEADS.DA_IMG_LOSS_WEIGHT = 1.0
_C.MODEL.DA_HEADS.DA_INS_LOSS_WEIGHT = 1.0
_C.MODEL.DA_HEADS.DA_CST_LOSS_WEIGHT = 0.1
_C.MODEL.DA_HEADS.LW_DA_INS = 0.02
# -------------------------------------------------------- #
#                        Evaluation                        #
# -------------------------------------------------------- #
# The period to evaluate the model during training
_C.EVAL_PERIOD = 10
# Evaluation with GT boxes to verify the upper bound of person search performance
_C.EVAL_USE_GT = False
# Fast evaluation with cached features
_C.EVAL_USE_CACHE = False
# Evaluation with Context Bipartite Graph Matching (CBGM) algorithm
_C.EVAL_USE_CBGM = False

# -------------------------------------------------------- #
#                           Miscs                          #
# -------------------------------------------------------- #
# Save a checkpoint after every this number of epochs
_C.CKPT_PERIOD = 1
# The period (in terms of iterations) to display training losses
_C.DISP_PERIOD = 10
# Whether to use tensorboard for visualization
_C.TF_BOARD = True
# The device loading the model
_C.DEVICE = "cuda:0"
# Set seed to negative to fully randomize everything
_C.SEED = 1
# Directory where output files are written
_C.OUTPUT_DIR = "/dockerdata/tjunjieli/data/project_data/SeqNet-DA"

_C.TARGET_REID_START = 0

_C.HM_THRESH = 0.5

_C.EPS_P = 0.98

_C.LOCAL_RANK = -1


def get_default_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
