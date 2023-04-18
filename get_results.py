# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import mss
import pickle

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, "third_party/CenterNet2/")
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo
from detectron2.data import MetadataCatalog

# constants
WINDOW_NAME = "Detic"


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"  # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=["lvis", "openimages", "objects365", "coco", "custom"],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="cake,bread,fish,orange,tomato,lemon,squash,pumpkin,strawberry,grapesgrape,pear,egg,hamburger,cucumber,ice_cream,peach,cabbage,cookie,fries,coconut,asparagusgreen_onion,sushi,burrito,spring_rolls,pasta,noodles,crab,shellfish,sandwich,apple,donutdoughnut,tangerine,grapefruit,orange,pizza,broccoli,banana,carrot,hot_dog,basket,canned,candy,pepper,potato,sausage,pie,mango,onion,plum,pine_apple,watermelon,green_beans,garlic,avocado,kiwi_fruit,cherry,green_vegetables,nuts,corn,eggplant,dates,rice,lettuce,meat_balls,mushroom,egg_tart,pomegranate,cheese,papaya,chips,steak,radish,bread,bun,red_cabbage,okra,durian,asparagus,pasta,guacamole,mangosteen,pitaya,scallop,lobster,taco,zucchini,bell_pepper,croissant,pancake,vegetable,tart,dessert,fruit,juice,beer,waffle,artichoke,drink,pumpkin,muffin,pretzel,wine,tea,winter_melon,submarine_sandwich,seafood,pineapple,cocktail,bagel,coffee,common_fig,salad,popcorn,hamimelon,pomelo,baozi,cannon,missile,weapon,shotgun,gun,rifle,handgun,dagger,sword,tank,bow_and_arrow,chainsaw,plant,tree,frisbee,rugby_ball,american_football,tennis_ball,golf_ball,skis,skates,baseball,cricket_ball,volleyball,dumbbell,treadmill,table_tennis_paddle,soccer,ball,baseball_glove,baseball_bat,skateboard,surfboard,tennis_racket,snowboard,hockey,billiards,basketball,golf_club,barbell,punching_bag,canoe,stationary_bicycle,swim_cap,football_helmet,bat,racket,cue,helmet,scoreboard,trophy,hurdle,paddle,parachute,elastic,stop_sign,traffic_sign,traffic_cone,crosswalk_sign,fire_hydrant,parking_meter,traffic_light,street_lightsstreet_light,house,building,lighthouse,tower,office_building,swimming_pool,fountain,castle,convenience_store,porch,skyscraper,tent,sculpture,billboard,bronze_sculpture,swing,slide,stairs,Tire,wheel,_bicycle_wheel,rickshaw,ambulance,truck,car,bicycle,motorcycle,bus,train,suv,van,pickup_truck,machinery_vehicle,sports_car,crane,fire_truck,race_car,limousine,vehicle,golf_cart,snowmobile,taxi,land_vehicle,carriage,tricycle,hoverboardsegway,boat,watercraft,bargeship,jet_ski,gondola,sailboat,boat,Helicopter,airplane,aircraft,rocket,hot_air_balloon,drum,guitar,cello,saxophone,trumpet,trombone,flute,musical_keyboard,piano,violin,trombone,tuba,french_horn,harp,harpsichord,trumpet,musical_instrument,accordion,organ,oboe,piano,cymbal,horn,lifejacket,seat_belt,life_saver,fire_extinguisher,wheelchair,stroller,stretcher,crutch,washing_machine,radiator,air_conditioner,home_appliance,snowplow,lawn_mover,iron,sewing_machine,wrench,nail,pliers,hammer,tape_measure,ruler,saw,electric_drill,screwdriver,power_tool,trolley,ladder,flashlight,torch,binoculars,lantern,power_tool,knitting_needles,tea_pot,kettle,cutting_board,spatula,scissors,pot,pan,kitchen_knife,tong,measuring_cup,frying_pan,wok,teapot,tin_can,knife,storage_boxbox,spatulashovel,tableware,bowl,chopsticks,fork,spoon,plate,platter,serving_tray,drinking_straw,saucer,pitcher,mug,jug,coffee_cup,wine_glass,bottle,cup,beaker,kitchen_appliance,pressure_cooker,stove,rice_cooker,wood-burning_stove,food_processor,slow_cooker,microwave_oven,blender,toaster,oven,microwave,kettle,coffee_maker,barrel,refrigerator,dining_table,desktable,chair,couch,loveseat,bed,bench,bucket,coffee_table,nightstand,dining_table,bed,billiard_table,bookcase,desk,furniture,couch,studio,_sofa_bed,dog_bed,cupboard,bathroom_cabinet,cabinetry,shelf,filing_cabinet,chest_of_drawers,drawer,pillow,vase,candle,clock,lamp,wall_clock,curtain,stool,carpet,picture_frame,alarm_clock,mechanical_fan,digital_clock,hanger,door_handle,mirror,ceiling_fan,vent,whiteboard,blackboard,power_outlet,door,countertop,window,blinds,light_switch,plumbing_fixture,fireplace,window,camera,printer,camera,router,modem,projector,tripod,microphone,remote,cd,mouse,television,computer,keyboard,laptop,monitor,headphones,speaker,earphone,radio,power_plugs_and_sockets,extension_cord,converter,tablet,mobile_phone,cell_phone,telephone,trashcan,broom,brush,mop,cleaning_products,paper_towel,toilet_paper,towel,napkin,tissue,toothpaste,toothbrush,comb,hair_dryer,facial_cleanser,toiletries,shampoo,body_wash,soap,urinal,faucet,tap,toilet,bathtub,bidet,pen,pencil,tape,folder,marker,calculator,stapler,scale,microscope,pencil_case,eraser,binder,ruler,envelope,paper,office_supplies,book,poster,toy,sock,dress,uniform,coat,skirt,swimwear,jeans,miniskirt,scarf,shorts,jacket,shirt,suit,trousers,bra,hat,shoes,heels,boots,bracelet,goggles,sunglasses,earrings,ring,watch,necklace,glove,mask,belt,tie,glasses,helmet,clutch,luggage,bags,handbag,suitcase,backpack,briefcase,purse,sink,kite,umbrella,paint_brush,fishing_rod,key,globe,medal,poker_card,lighter,game_board,coin,dice,picnic_basket,plastic_bag,flag,balloon,cigar,target,bust,snowman",
        help="",
    )
    parser.add_argument("--pred_all_class", action="store_true")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[
            "MODEL.WEIGHTS",
            "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        ],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    class_names = metadata.thing_classes

    if args.vocabulary == "custom":
        class_names = args.custom_vocabulary.split(",")

    demo = VisualizationDemo(cfg, args)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                    out_filename_pkl = os.path.join(
                        args.output, os.path.basename(path).split(".")[0] + "_detic.pkl"
                    )
                else:
                    assert (
                        len(args.input) == 1
                    ), "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)

                # save predictions
                class_ids = predictions["instances"].pred_classes.cpu().numpy()

                preds = {
                    "boxes": predictions["instances"].pred_boxes.tensor.cpu().numpy(),
                    "scores": predictions["instances"].scores.cpu().numpy(),
                    "classes": predictions["instances"].pred_classes.cpu().numpy(),
                    "class_names": np.array([class_names[i] for i in class_ids]),
                }
                with open(out_filename_pkl, "wb") as file:
                    pickle.dump(preds, file)
