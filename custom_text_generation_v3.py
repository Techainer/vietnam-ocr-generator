import PIL
import os
import cv2
import random
import argparse
import glob
import copy
from tqdm import tqdm
import blend_modes
import string

import albumentations as A
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import imgaug.parameters as iap
import scipy.ndimage as ndi
import pandas as pd

from PIL import Image, ImageFont, ImageDraw, ImageFilter
import numpy as np
import warnings
warnings.filterwarnings("ignore")

LOWER_LETTER = list(string.ascii_lowercase)
UPPER_LETTER = list(string.ascii_uppercase)
DIGITS = list(string.digits)


address_db_path = 'data/address.csv'
address_db = pd.read_csv(address_db_path)
tinh_tp = pd.unique(address_db['Tỉnh Thành Phố'])
quan_huyen = pd.unique(address_db['Quận Huyện'])
phuong_xa = pd.unique(address_db['Phường Xã'])


def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description='Generate synthetic text data for text recognition.')
    parser.add_argument(
        "--string",
        type=str,
        nargs="?",
        help="The input string",
        default="1645 Phạm Thế Hiển",
    )
    
    parser.add_argument(
        "--font",
        type=str,
        nargs="?",
        help="The path to font",
        default="fonts/cccd/Arial.ttf",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug",
        default=False
    )
    
    parser.add_argument(
        "--text_from_file",
        type=str,
        help="text from file",
        default="",
    )
    
    parser.add_argument(
        "--type_card",
        type=str,
        help="text from file",
        default="old_cccd",
    )
    
    parser.add_argument(
        "--debug_dir",
        type=str,
        help="debug dir",
        default="debugs",
    )
    
    parser.add_argument(
        "--input_type",
        type=str,
        help="input type",
        default="front",
    )
    
    return parser.parse_args()

    
def get_rand_address():
    have_prefix = True if random.random() <= 0.5 else False
    special_address = ""
    ratio_special = random.random()

    if ratio_special < 0.05:
        special_address = "Quốc lộ {}{}".format(random.choice(UPPER_LETTER), random.randint(1, 20))
        if random.random() < 0.2:
            special_address = "Km{} {}".format(random.randint(1, 100), special_address)

    elif ratio_special < 0.25:
        list_kdt = ['KĐT', 'Khu Đô Thị', 'Chung cư', "Khu Công nghiệp", 'KCN', "Cụm công nghiệp", 
                        "Khu tái định cư", "Khu dân cư", "Khu tập thể", "KTT"]

        toa_nha_code = "".join(random.sample(UPPER_LETTER + DIGITS, random.randint(2, 4)))

        if random.random() < 0.25:
            toa_nha_code = "{}-{}".format(toa_nha_code, "".join(random.sample(UPPER_LETTER + LOWER_LETTER + DIGITS, random.randint(2, 4))))
        
        if random.random() < 0.5:
            special_address = "{} {}".format(toa_nha_code, random.choice(list_kdt))
        else:
            special_address = "{}".format(random.choice(list_kdt))

        special_address = "{} {}".format(special_address, random.choice(phuong_xa))
        special_address = special_address.replace('Xã ', '').replace('Phường ', '').replace('Thị trấn ', '').replace('Thị xã ', '')
        
    # So nha
    
    if random.random() < 0.5:
        have_address_number = 0
    else:
        have_address_number = random.randint(1, 2)
        
    if have_address_number > 0:
        if random.random() < 0.85:
            slash = random.choice(["-", ".", "/"])
            address_number = slash.join([str(random.randint(1, 1000))
                                for _ in range(have_address_number)])
        else:
            address_number = "".join(random.sample(UPPER_LETTER + DIGITS, k=random.randint(2, 8)))
    
        if have_prefix:
            # if random.random() < 0.5:
            address_number = '{} {}'.format(
                    random.choice(['Số', 'Lô', 'Ấp', 'Đội', "Tổ", "Số nhà", "Khu", "Xóm", "Tiểu khu", "Thôn", "Lầu", "Căn hộ", "Tầng", "Phòng"]),
                    address_number)
            # else:
            #     address_number = '{} {}{}'.format(
            #             random.choice(['Số', 'Lô', 'Ấp', 'Đội', "Tổ", "Số nhà", "Khu", "Xóm", "Tiểu khu", "Thôn", "Lầu", "Căn hộ", "Tầng", "Phòng"]),
            #             address_number,
            #             random.choice([".", ","]))

            if random.random() < 0.5:
                address_number = '{}, {} {}'.format(
                            address_number,
                            random.choice(['Lô', 'Ấp', 'Đội', "Tổ", "Khu", "Xóm", "Tiểu khu", "Thôn"]),
                            random.randint(0, 9))
    else:
        address_number = ""

    street = random.choice(phuong_xa)
    if not have_prefix:
        street = street.replace('Xã ', '').replace(
            'Phường ', '').replace('Thị trấn ', '').replace('Thị xã ', '')
    else:
        prefix_street = random.choice(['Đường ', 'Phố ', 'Thôn ', 'Quốc lộ ', "Khu phố "])
        street = street.replace('Xã ', prefix_street).replace(
            'Phường ', prefix_street).replace('Thị trấn ', prefix_street).replace('Thị xã ', prefix_street)

    address = address_db.sample(n=1, random_state=np.random.RandomState())

    base_address = '{}, {}, {}'.format(
        address['Phường Xã'].values[0], address['Quận Huyện'].values[0], address['Tỉnh Thành Phố'].values[0])
    
    if not have_prefix:
        base_address = base_address.replace('Tỉnh ', '').replace('Thành phố ', '').replace('Quận ', '').replace(
            'Huyện ', '').replace('Xã ', '').replace('Phường ', '').replace('Thị trấn ', '').replace('Thị xã ', '')
    else:
        short_case = {
            'Thị xã ': 'Tx.',
            'Quận ': 'Q.',
            'Thị trấn ': 'Tt.',
            'Thành phố ': 'TP.',
            'Huyện ': 'H.',
            'Phường ': 'P.',
            'Tỉnh ': 'T.'
        }
        for each in short_case:
            if random.uniform(0, 1) <= 0.65:
                base_address = base_address.replace(each, short_case[each])


    if have_address_number > 0:
        last_address_1 = "{} {}".format(special_address, address_number).strip()
        if random.random() < 0.5:
            if random.random() < 0.75:
                last_address_1 = "{},".format(last_address_1)
            else:
                last_address_1 = "{}.".format(last_address_1)
        last_address_2 = "{}, {}".format(street, base_address)
    else:
        last_address_1 = "{} {}".format(special_address, street).strip()
        if random.random() < 0.5:
            if random.random() < 0.75:
                last_address_1 = "{},".format(last_address_1)
            else:
                last_address_1 = "{}.".format(last_address_1)
        last_address_2 = base_address
    
    return (last_address_1, last_address_2)


def get_rand_nguyenquan():
    have_prefix = True if random.uniform(0, 1) <= 0.35 else False
    address = address_db.sample(n=1, random_state=np.random.RandomState())

    address_1 = "{}".format(address['Phường Xã'].values[0])
    address_2 = "{}, {}".format(address['Quận Huyện'].values[0], address['Tỉnh Thành Phố'].values[0])
    if not have_prefix:
        address_1 = address_1.replace('Xã ', '').replace('Phường ', '').replace('Thị trấn ', '').replace('Thị xã ', '')
        address_2 = address_2.replace('Tỉnh ', '').replace('Thành phố ', '').replace('Quận ', '').replace('Huyện ', '')

    else:
        short_case = {
            'Thị xã ': 'Tx.',
            'Quận ': 'Q.',
            'Thị trấn ': 'Tt.',
            'Thành phố ': 'TP.',
            'Huyện ': 'H.',
            'Phường ': 'P.',
            'Tỉnh ': 'T.'
        }
        for each in short_case:
            if random.uniform(0, 1) <= 0.85:
                address_1 = address_1.replace(each, short_case[each])
                address_2 = address_2.replace(each, short_case[each])

    return address_1, address_2


def generate(text, font, text_color, height):
    # print(text, font, text_color)
    # image_font = ImageFont.truetype(font="/Library/Fonts/Arial Unicode.ttf", size=32)
    image_font = ImageFont.truetype(font=font, size=height)
    text_width, text_height = image_font.getsize(text)

    # text = u'日産コーポレート/個人ゴールドJBC123JAL'
    txt_img = Image.new('L', (text_width, text_height), 255)

    txt_draw = ImageDraw.Draw(txt_img)

    txt_draw.text((0, 0), u'{0}'.format(text), fill=random.randint(1, 80) if text_color < 0 else text_color, font=image_font)

    return txt_img

def augment_for_ocr():
    aug = iaa.Sequential([
        # Custom augmentation
        # iaa.Sometimes(0.1, LightFlare()),
        # iaa.Sometimes(0.1, ParallelLight()),
        # iaa.Sometimes(0.1, SpotLight()),
        # iaa.Sometimes(0.1, RandomLine()),
        # iaa.Sometimes(0.05, Blob()),
        # iaa.Sometimes(0.05, WarpTexture()),
        
        # Change the brightness and color

        iaa.Sometimes(0.75, 
            iaa.OneOf(
                [
                    # iaa.Add((-30, 30)),
                    iaa.AddToHueAndSaturation((-20, 20)),
                    iaa.LinearContrast((0.4, 2.5)),
                    iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.1, 0.9), per_channel=True),
                    iaa.ReplaceElementwise(0.05, iap.Normal(128, 0.4*128), per_channel=0.5),
                    iaa.Emboss(alpha=(0.1, 0.5), strength=(0.8, 1.2)),
                    # iaa.BlendAlphaSimplexNoise(iaa.Multiply(iap.Choice([0.9,1.1]), per_channel=True)),
                    iaa.ChangeColorTemperature((3000, 15000)),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    iaa.Sometimes(0.1, iaa.CoarseDropout(0.02, size_percent=0.1, per_channel=True)),
                    # iaa.ChannelShuffle(1),
                    # iaa.Invert(0.01, per_channel=True),
                ]
            )
        ),
        
        # Noise and change background
        iaa.Sometimes(0.6,
            iaa.OneOf([
                # iaa.pillike.FilterSmoothMore((100, 170)),
                # iaa.imgcorruptlike.Spatter(severity=(1,3)),
                # iaa.pillike.EnhanceSharpness(),
                # iaa.AdditiveLaplaceNoise(scale=(0.07 * 255, 0.08 * 255)),
                # iaa.AdditiveLaplaceNoise(scale=(0.07 * 255, 0.08 * 255), per_channel=True),
                # iaa.AdditiveGaussianNoise(scale=(0.02 * 255, 0.1 * 255)),
                # iaa.AdditiveGaussianNoise(scale=(0.02 * 255, 0.1 * 255), per_channel=True),
                # iaa.SaltAndPepper(p=0.01),
                iaa.Sharpen(alpha=(0.1, 0.5)),
                iaa.MultiplyElementwise((0.8, 1.2), per_channel=0.5),
                iaa.GaussianBlur(sigma=(1.0, 3.5)),
                iaa.AverageBlur(k=(1, 5)),
                iaa.MotionBlur(k=(3, 9), angle=(-90, 90)),
                iaa.Dropout((0, 0.1), per_channel=True),
                iaa.ElasticTransformation(alpha=(1, 1.5), sigma=(0.5, 1)),
            ])
        ),
        # compress image
        iaa.Sometimes(0.2,
            iaa.OneOf([
                iaa.JpegCompression(compression=(90, 99)),
                # iaa.imgcorruptlike.Pixelate(severity=(4,5)),
                # iaa.UniformColorQuantization((10,200)),
                # iaa.AveragePooling((2,5)),
            ])
        ),
    ])
    return aug

def generate_data(background: np.array,
                  text: str=None,
                  font: str=None,
                  debug: bool=True,
                  debug_dir: str=None,
                  type_text: str=None,
                  type_card: str="old_cccd"):
    #######################
    # Generate text image #
    #######################
    background = cv2.cvtColor(background,cv2.COLOR_BGR2RGB)
    if type_card == "old_cccd":
        top_left = (203,368)
    elif type_card == "new_cccd":
        top_left = (25,35)
    else:
        print("Only support for cccd_chip and the cccd!!!")
        
    if type_text == "hokhauthuongtru":
        if type_card == "old_cccd":
            top_left = (203,368)
        elif type_card == "new_cccd":
            top_left = (25,35)
        else:
            print("Only support for cccd_chip and the cccd!!!")
            
    if type_text == "nguyenquan":
        if type_card == "old_cccd":
            top_left = (203,368)
        elif type_card == "new_cccd":
            top_left = (25,95)
        else:
            print("Only support for cccd_chip and the cccd!!!")
            
    # if option_2
    # font = "/home/jon/vietocr/VietNamese-OCR-DataGenerator/fonts/cccd/Roboto-Black.ttf"
    
    text_image = generate(text,font,1,20) #Image
    np_text_image = np.array(text_image) # np.array
    # if option_2:
    # np_text_image =  cv2.erode(np_text_image,(3,3))
    # text_image = Image.fromarray(np_text_image)
    
    mask = 255-np_text_image 
    
    text = text.replace("/","---")
    if debug:
        os.makedirs(os.path.join(debug_dir,text),exist_ok=True)
    
    if type_card == 'old_cccd':
        if np_text_image.shape[1] < 680-470:
            top_left = (470,340)
    elif type_card == 'new_cccd':
        if np_text_image.shape[1] < 680-400:
            top_left = (235,10)
    else:
        print("Only support for cccd_chip and the cccd!!!")

    if debug:
        text_image.save(os.path.join(debug_dir,text,"text_image.png"))
        cv2.imwrite(os.path.join(debug_dir,text,"mask.png"),mask)
    
    h,w = mask.shape
    
    ####################
    #       Blend      #
    ####################
    blur_text_image = cv2.GaussianBlur(np_text_image,(3,3),0)
    
    
    np_crop_background = np.array(background)[top_left[1]:top_left[1]+h,top_left[0]:top_left[0]+w]
    
    crop_and_paste_background = Image.fromarray(np_crop_background)
    crop_and_paste_background.paste(text_image,(0,0),mask=Image.fromarray(mask))
    # crop_and_paste_background.paste(Image.fromarray(blur_text_image),(0,0),mask=Image.fromarray(mask))
    np_crop_and_paste_background = np.array(crop_and_paste_background)
    
    assert np_crop_and_paste_background.shape == np_crop_background.shape , "Shape is not equally!!!"
    
    # Generate weight matrix
    h, w, _ = np_crop_and_paste_background.shape
    weight_matrix = np.random.uniform(low=0.7, high=1, size=(h,w,1))
    weight_matrix = np.concatenate((weight_matrix,weight_matrix,weight_matrix),axis=2)
    
    blend_text_image = weight_matrix*np_crop_and_paste_background+(1-weight_matrix)*np_crop_background
    blend_text_image = blend_text_image.astype(np.uint8)

    if debug:
        plt.subplot(1,4,1)
        plt.title("mask")
        plt.imshow(mask)
        
        plt.subplot(1,4,2)
        plt.title("crop background")
        plt.imshow(np_crop_background)
        
        plt.subplot(1,4,3)
        plt.title("paste background")
        plt.imshow(np_crop_and_paste_background)
        
        plt.subplot(1,4,4)
        plt.title("blend image")
        plt.imshow(blend_text_image)
        
        plt.savefig(os.path.join(debug_dir,text,"blend_process.png"),dpi=120)
        plt.close()

        cv2.imwrite(os.path.join(debug_dir,text,"blend_image.png"),blend_text_image)
    
    
    #############################
    # Paste image in background #
    #############################
    # Original background    
    # background = Image.fromarray(background)
    # background.paste(text_image,top_left,mask=Image.fromarray(mask))

    background[top_left[1]:top_left[1]+h,top_left[0]:top_left[0]+w][mask>0] = blend_text_image[mask>0]
    background = Image.fromarray(background)
       
    if debug:
        background.save(os.path.join(debug_dir,text,"background_after_paste.png"))
  
    ##############
    # Crop Image #
    ##############

    crop_image = np.array(background)[top_left[1]:top_left[1]+h,top_left[0]:top_left[0]+w]
    crop_image = cv2.cvtColor(crop_image,cv2.COLOR_BGR2RGB)
    if debug:
        cv2.imwrite(os.path.join(debug_dir,text,"crop_image.png"),crop_image)
   
    ####################
    # Start augmenting #
    ####################

    # get edge image
    edge = cv2.Canny(mask,100,200)
    # if option_2:
    # edge = cv2.dilate(edge,(3,3))
    # edge = cv2.dilate(edge,(3,3))

    if debug:
        cv2.imwrite(os.path.join(debug_dir,text,"edge.png"),edge)

    blur = cv2.blur(crop_image,(3,3))
    blur = cv2.blur(blur,(3,3)) 
    # transform = A.Blur(blur_limit=(3, 5), p=1.0)
    # blur = transform(image=crop_image)
    # blur = blur["image"]
    # blur = transform(image=blur)
    # blur = blur["image"]
    # if option_2:
    # blur = cv2.blur(blur,(3,3))
    # blur = cv2.blur(blur,(3,3))
    # blur = cv2.blur(blur,(3,3))
    # blur = cv2.blur(blur,(3,3))
    # blur = cv2.blur(blur,(3,3))
    # blur = cv2.blur(blur,(3,3))
    # blur = cv2.blur(blur,(3,3))
    # blur = cv2.blur(blur,(3,3))
    # blur = cv2.blur(blur,(3,3))
    # blur = cv2.blur(blur,(3,3))
    # blur = cv2.blur(blur,(3,3))
    # blur = cv2.blur(blur,(3,3))
    # blur = cv2.blur(blur,(3,3))
    # blur = cv2.blur(blur,(3,3))



    res_image = crop_image.copy()
    # Only blur edge of text
    res_image[edge>0] = blur[edge>0]
    final_image = cv2.GaussianBlur(res_image,(3,3),0)
    final_image = cv2.GaussianBlur(final_image,(3,3),0)

    
    #######################
    # Add some experiment #
    #######################

    # aug = augment_for_ocr()
    # final_image = aug.augment_image(final_image)
    
    if debug:
        plt.subplot(2,2,1)
        plt.title("Edge image")
        plt.imshow(edge)
        
        plt.subplot(2,2,2)
        plt.title("Original Image")
        plt.imshow(crop_image)
        
        plt.subplot(2,2,3)
        plt.title("Edge blur image")
        plt.imshow(res_image)
        
        plt.subplot(2,2,4)
        plt.title("Final image")
        plt.imshow(final_image)

        plt.savefig(os.path.join(debug_dir,text,"algorithm.png"),dpi=120)
        plt.close()

    return final_image


def random_background(type_card:str="old_cccd"):
    if type_card == "old_cccd":
        background_paths = glob.glob(os.path.join("/home/jon/vietocr/VietNamese-OCR-DataGenerator/background","*.*"))
        background_image = cv2.imread(random.choice(background_paths))
    elif type_card == "new_cccd":
        background_paths = glob.glob(os.path.join("/home/jon/vietocr/VietNamese-OCR-DataGenerator/background_cccd_new","*.*"))
        background_image = cv2.imread(random.choice(background_paths))
    else:
        print("Only support for cccd_chip and the cccd!!!")
        
    return background_image

def main():
    args = parse_arguments()
    
    #######################
    #        Init         #
    #######################
    font = args.font
    type_card = args.type_card
    print("font: ",font)
    print("type card: ",type_card)
    print("Start")
    if args.text_from_file == "":
        text = args.string
        background = random_background(type_card)
        background = cv2.resize(background,(680,401))
        print("Text: ",text)
        final_image = generate_data(background=background,text=text,font=font,debug=args.debug,debug_dir=args.debug_dir,type_card=type_card)
        cv2.imwrite("output_img/test_img.png",final_image)
        
    else:
        with open(args.text_from_file,'r') as f:
            content = f.readlines()
        texts = []
        for line in tqdm(content[25000:]):
            background = random_background(type_card)
            background = cv2.resize(background,(680,401))

            image, text = line.split("\t")
            text = text.replace("\n","")
            temp_text = text.replace("/","---")
            
            texts.append(f"/home/jon/vietocr/data_ocr_5/gen_val_image/{temp_text}.png\t"+text+"\n")

            type_text = None
            if "hokhauthuongtru" in image:
                type_text = "hokhauthuongtru"
            if "nguyenquan" in image:
                type_text = "nguyenquan"
            try:
                final_image = generate_data(background=background,text=text,font=font,debug=args.debug,debug_dir=args.debug_dir,type_text=type_text,type_card=type_card)
            except:
                print(image)
            cv2.imwrite(f"/home/jon/vietocr/VietNamese-OCR-DataGenerator/output_img/gen_val_image/{temp_text}.png",final_image)
            
        with open("/home/jon/vietocr/VietNamese-OCR-DataGenerator/output_img/val_label.txt",'w') as f:
            for line in texts:
                f.write(line)

if __name__ == "__main__":
    main()