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
from skimage.morphology import  square, binary_opening, binary_dilation
from skimage.color import rgb2gray
# from skimage.transform import rescale
# from skimage.util import invert
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

def checkIfValid(img, item_name):
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
    
    if (IMGMODE == True):
        plt.text(100, 770, text)
        plt.text(100, 800, name_compare)

    if score > 4:
        return (0, 'Match score to high: '+str(score))

    return (1, 'Match score: '+str(score))


def ocr(item):
    if (IMGMODE == True):
        plt.rcParams['text.color'] = 'white'
        fig, ax = plt.subplots(figsize=(10, 6))


    item_name = item['title']
    item_path = item['filePath']
    readPath = glob.glob('images/'+item_path+'--*.png')[0]
    writePath = "processed"+readPath[6:]
 
    regex = r"--(\d+).png"
    item["timestamp"] = re.search(regex, readPath).group(1)

    print(item_name)

    orig_image = io.imread(readPath)
    gray_img = rgb2gray(orig_image)
    valid = checkIfValid(gray_img, item_name)

    if (IMGMODE == True):
        ax.imshow(orig_image)
        plt.text(100, 700, valid[1])


    if valid[0] < 1:
        if (IMGMODE == True):
            plt.text(500, 500, '########## NOT A VALID IMAGE! ##########')
            ax.set_axis_off()
            plt.tight_layout()
            fig.savefig(writePath)
            plt.close()

        f = open("logfile.txt", "a")
        f.write(item_name+" NOT A VALID IMAGE "+valid[1])
        f.write('\n')
        f.close()
        return []



    ocr_crop = gray_img[45:1020, 1150:1550] > 0.6
    thresh = threshold_minimum(gray_img)

    dilated = binary_dilation(gray_img > thresh, square(16))
    dilated_crop = dilated[45:1020, 1150: 1550]


    border_mask = np.ones((975, 400), dtype=bool)
    border_mask[:, 370:] = False
    border_mask[:, :5] = False

    cleared = clear_border(dilated_crop, mask=border_mask)
    labeled, num = label(cleared, return_num=True)

    if num < 1:
        if (IMGMODE == True):
            plt.text(500, 500, '########## NO OFFERS ##########')
            ax.set_axis_off()
            plt.tight_layout()
            fig.savefig(writePath)
            plt.close()

        f = open("logfile.txt", "a")
        f.write(item_name+" NO OFFERS")
        f.write('\n')
        f.close()
        return []

    regions = regionprops(labeled)
    last_region = None

    prices = []
    found = 0
    for region in regions:
        if found >= 3:
            break
        if last_region is None or (region.centroid[0] - last_region.centroid[0]) > 60:
            minr, minc, maxr, maxc = region.bbox
            currency = clear_border(ocr_crop[minr:minr + 40, maxc - 35:maxc])
            price_crop = clear_border(ocr_crop[minr:maxr, minc:maxc - 15])

            currency = checkCurrency(currency)

            config = ("-l bender --oem 1 --psm 7")
            price = pytesseract.image_to_string(img_as_uint(price_crop), config=config)

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

    prices.sort()
    return prices

for category in ("foregrips", "pistol_grips", "handguards", "containers", "headwear", "armor_vests", "armored_chest_rigs", "additional_armor", "thermal_vision_devices", "visors", "headsets", "eyewear", "backpacks", "chest_rigs", "face_cover", "helmet_mounts", "helmet_headsets", "night_vision_devices", "helmet_vanity"):
    final = []


    print('#####'+category+'########')

    with open('../tarkov-AH-scrapper/data/wiki/'+category+'-data.json') as json_file:
        data = json.load(json_file)
        for item in data:
            prices = ocr(item)
            if len(prices) == 0:
                price = -1
            elif len(prices) > 2:
                price = int(round(sum(prices[:2]) / 2))
            else:
                price = prices[0]
            print(price)

            if price > 0:
                item["price_avg"] = price
                item["price_per_slot"] = round(price  / (int(item["size"]["width"]) * int(item["size"]["height"])))
                final.append(item)
    
    if (JSONMODE == True):
        time = strftime("%Y%m%d%H%M%S", gmtime())

        with open('./json_output/'+category+'-data-'+time+'.json', 'w') as json_file:
            json.dump(final, json_file)

            



