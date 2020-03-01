# from joblib import Parallel, delayed
import re
import getopt
import sys
# from os import listdir
import string  
import json
import glob

import pytesseract
from Levenshtein import distance

import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import numpy as np

from skimage import io, img_as_uint
from skimage.filters import threshold_minimum
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops# compare_ssim
from skimage.morphology import  square, binary_opening, binary_dilation, remove_small_objects, area_opening, binary_erosion, binary_closing
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.util import invert
from skimage.feature import match_template #canny, shape_index, 
from scipy import ndimage as ndi

from time import gmtime, strftime

usd = io.imread('./currency/USD.png', as_gray=True)
eur = io.imread('./currency/EUR.png', as_gray=True)
rub = io.imread('./currency/RUB.png', as_gray=True)

JSONMODE = False
IMGMODE = False

try:
    opts, args = getopt.getopt(sys.argv[1:], "ij")
except getopt.GetoptError:
    print('Wrong command line options')
    sys.exit(2)
for opt, arg in opts:
    print(opt, arg)
    if opt == '-i':
        IMGMODE = True
    elif opt == '-j':
        JSONMODE = True

print(JSONMODE, IMGMODE)
def checkCurrency(currency):
    if (match_template(currency, rub) >  0.7):
        return 'RUB'
    if (match_template(currency, usd) > 0.7):
        return 'USD'
    if (match_template(currency, eur) > 0.7):
        return 'EUR'

    return 'UNKNOWN'

def checkIfValid(img, gray_crop, item_name):
    if (np.count_nonzero(gray_crop) < 70):
        return (0, 'No offers found (less than 70 bright pixels)')
    thresh = 0.6
    crop = img[45:1020, 0:700]
    mask = binary_opening(ndi.binary_fill_holes(crop > thresh), square(20))
    
    regions, num = label(mask, return_num=True)

    if num < 1:
        return (0, 'No item selected')
    
    if num > 1:
        return(0, 'Too many items selected')

    props = regionprops(regions)
    minr, minc, maxr, maxc = props[0].bbox

    ocr_box = crop[minr:maxr, minc + 38:maxc] > 0.47
    ocr_box = np.pad(ocr_box, (10,5), constant_values=1)

    name_compare = ''.join([word for word in item_name if word not in string.punctuation])
    name_compare = ' '.join(name_compare.split())

    config = ("-l eng2 --oem 1 --psm 7")
    print('Tesseract text')
    text = pytesseract.image_to_string(img_as_uint(ocr_box), config=config)
    text = text.replace('@', '0')
    text = text.replace('¢', 'x')
    blacklist = ('§', '—', "'", '"')
    text = ''.join([word for word in text if word not in string.punctuation and word not in blacklist])
    text = text.strip().split()
    if (len(name_compare) < 40):
        text = text[:-1]
    text = ' '.join(text)

    text = text[:40]
    name_compare = name_compare[:40]
    score = distance(text.lower(), name_compare.lower())
    
    plt.text(100, 770, text)
    plt.text(100, 800, name_compare)

    if score > 4:
        return (0, 'Match score to high: '+str(score))

    return (1, 'Match score: '+str(score))


def ocr(item):
    print('Starting ocr')
    plt.rcParams['text.color'] = 'white'
    fig, ax = plt.subplots(figsize=(10, 6))
    

    item_name = item['title']
    item_path = item['filePath']
    readPath = glob.glob('/media/xetro/Faster/dev/'+item_path+'--*.png')[0]
    writePath = "processed"+readPath[6:]
 
    regex = r"--(\d+).png"
    item["timestamp"] = re.search(regex, readPath).group(1)

    print(item_name)

    orig_image = io.imread(readPath)
    gray_img = rgb2gray(orig_image)
    ax.imshow(orig_image)
    gray_crop = gray_img[45:1020, 1150:1550]
    ocr_crop =  gray_crop > 0.6


    valid = checkIfValid(gray_img, ocr_crop, item_name)


    if valid[0] < 1:
        if (IMGMODE == True):
            plt.text(500, 500, '########## NOT A VALID IMAGE! ##########')
            ax.set_axis_off()
            plt.tight_layout()
            fig.savefig(writePath)
            plt.close()

        print('Opening log file')
        f = open("logfile.txt", "a")
        print('Log file opened')
        f.write(item_name+" NOT A VALID IMAGE "+valid[1])
        f.write('\n')
        print('Log file written')
        f.close()
        print('Log file closed')
        return []


    # ocr_crop =  gray_crop > 0.8
    bulk_crop = gray_crop < 0.25
    thresh = threshold_minimum(gray_img)

    red_mask = orig_image[45:1020, 1150:1550, 0] > 70
    red_mask = binary_opening(red_mask, square(6))

    ocr_crop[red_mask] = 0

    dilated_crop = binary_dilation(ocr_crop, square(16))

    border_mask = np.ones((975, 400), dtype=bool)
    border_mask[:, 370:] = False
    border_mask[:, :5] = False

    cleared = clear_border(dilated_crop, mask=border_mask)
    labeled, num = label(cleared, return_num=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.imshow(img_as_uint(ocr_crop))

    # fig, ax = plt.subplots(figsize=(10, 6))

    # ax.imshow(img_as_uint(binary_opening(ocr_crop, square(4))))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.imshow(img_as_uint(cleared))

    print('non zero: ', np.count_nonzero(ocr_crop))

    if num < 1:
        if (IMGMODE == True):
            plt.text(500, 500, '########## NO OFFERS ##########')
            ax.set_axis_off()
            plt.tight_layout()
            fig.savefig(writePath)
            plt.close()

        print('Opening log file')
        f = open("logfile.txt", "a")
        print('Log file opened')
        f.write(item_name+" NO OFFERS")
        f.write('\n')
        print('Log file written')
        f.close()
        print('Log file closed')
        return []

    regions = regionprops(labeled)
    last_region = None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img_as_uint(cleared))

    prices = []
    found = 0
    for region in regions:
        # if found >= 3:
        #     break

        print('HOLA KARJOLA')

        center = region.centroid
        if last_region is None or (center[0] - last_region.centroid[0]) > 60:
            minr, minc, maxr, maxc = region.bbox
            print(minr, minc, maxr, maxc)

            bulk_crop2 = bulk_crop[maxr - 5:maxr + 15, int(center[1]) - 100:int(center[1]) + 100]
            # print(maxr - 5, maxr + 15, int(center[1]) - 100, int(center[1]) + 100)

            config = ("-l eng2 --oem 1 --psm 7")
            bulk_text = pytesseract.image_to_string(img_as_uint(bulk_crop2), config=config)
            bulk_text = ''.join([word for word in bulk_text if word not in string.punctuation and not word.isdigit()])
            score1 = distance(bulk_text, 'per item')
            score2 = distance(bulk_text, 'per pack  items')

            if (score2 < score1):
                last_region = region
                continue

            currency = clear_border(ocr_crop[minr:minr + 40, maxc - 35:maxc])

            price_crop = ocr_crop[minr:maxr, minc:maxc - 15]
            border_mask = np.zeros((price_crop.shape), dtype=bool)
            border_mask[:, :-12] = True
            # fig, ax = plt.subplots(figsize=(10, 6))
            # ax.imshow(price_crop)
            # fig, ax = plt.subplots(figsize=(10, 6))
            # ax.imshow(border_mask)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(currency)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(price_crop)
            price_crop = clear_border(price_crop, mask=border_mask)

            currency = checkCurrency(currency)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(price_crop)

            config = ("-l bender --oem 1 --psm 7")

            print('Tesseract price')
            price = pytesseract.image_to_string(img_as_uint(invert(price_crop)), config=config)
            plt.show()

            print(price)

            price = int(price.replace(" ", ""))
                        
            if (currency == "USD"):
                price = price * 108
            
            if (currency == "EUR"):
                price = price * 118

            prices.append(price)
            found += 1


            
            if (IMGMODE == True):
                plt.text(1150, minr+50, price)

        last_region = region

    if (IMGMODE == True):
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(writePath)
        plt.close()
    print(prices)

    prices.sort()
    return prices


log_file = open("logfile.txt","r+")
log_file.truncate(0)
log_file.close()

item = {
  "title": "5.7x28 mm L191",
  "filePath": "5_7x28_mm_l191"
}

# item = {
#   "title": "20/70 6.2mm Buckshot",
#   "filePath": "20_70_6_2mm_buckshot"
# }

# item = {
#   "title": "5.45x39 mm SP",
#   "filePath": "5_45x39_mm_sp"
# }

# item = {
#   "title": "9x18 mm PM PSV",
#   "filePath": "9x18_mm_pm_psv"
# }

# item = {
#   "title": "5-round .308 AICS M700 magazine",
#   "filePath": "5_round__308_aics_m700_magazine"
# }

ocr(item)

# for category in ("magazines", "762x25", "9x18", "9x19", "9x21", "46x30", "57x28", "545x39", "556x45", "762x39", "762x51", "762x54", "9x39", "366", "127x55", "12x70", "20x70"):
#     final = []
# # "weapons", "loot", "suppressors", "reflex_sights", "compact_reflex_sights", "iron_sights", "scopes", "assault_scopes", "special_scopes", "medical", "injectors", "provisions", "keys_factory", "keys_customs", "keys_woods", "keys_shoreline", "keys_interchange", "keys_labs", "keys_reserve", "containers", "headwear", "armor_vests", "armored_chest_rigs", "additional_armor", "thermal_vision_devices", "visors", "headsets", "eyewear", "backpacks", "chest_rigs", "face_cover", "helmet_mounts", "helmet_headsets", "night_vision_devices", "helmet_vanity", "foregrips", "pistol_grips", "handguards", "tactical_combo_devices", "stocks_chassis", "auxiliary_parts", "flashlights", "laser_target_pointers", "barrels", "bipods", "muzzle_adapters", "flash_hiders_muzzle_brakes", "charging_handles", "mounts", "gas_blocks", "receivers_slides", 
#     print('#####'+category+'########')
#     print('Opening read file')

#     with open('../tarkov-AH-scrapper/data/wiki/'+category+'-data.json') as json_file:
#         print('Read file opened!')
#         data = json.load(json_file)
#         print('Json loaded')
#         for item in data:
#             prices = ocr(item)
#             if len(prices) == 0:
#                 price = -1
#             elif len(prices) > 2:
#                 price = int(round(sum(prices[:2]) / 2))
#             else:
#                 price = prices[0]
#             print(price)

#             if price > 0:
#                 item["price_avg"] = price
#                 item["price_per_slot"] = round(price  / (int(item["size"]["width"]) * int(item["size"]["height"])))
#                 final.append(item)
    
#     if (JSONMODE == True):
#         time = strftime("%Y%m%d%H%M%S", gmtime())

#         print('Opening write file')
#         with open('./json_output/'+category+'-data-'+time+'.json', 'w') as json_file:
#             print('Write file opened!')
#             json.dump(final, json_file)
            # print('Json dumped')

             

