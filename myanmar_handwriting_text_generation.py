import cv2
import os
import glob
import random
import argparse
import shutil
from natsort import natsorted
from tqdm import tqdm
import numpy as np


def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description='Generate synthetic text data for myanmar text recognition.')
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug",
        default=False
    )
    parser.add_argument(
        "--zoom_in",
        action="store_true",
        help="zoom in",
        default=False
    )
    parser.add_argument(
        "--zoom_out",
        action="store_true",
        help="zoom out",
        default=False
    )
    parser.add_argument(
        "--save_path",
        type=str,
        nargs="?",
        help="save label path",
        default="label.txt",
    )

    parser.add_argument(
        "--save_image_path",
        type=str,
        nargs="?",
        help="save image path",
        default="myanmar_project/augment_crop_data_bg",
    )
    return parser.parse_args()
    
def get_dot_from_image(image: np.array=None, max_size:int = 25):
    '''
        Get an mask image(uint8/gray_scale)->remove all area that have S>25 pixel
        https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv
    '''
    
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(image)
    sizes = stats[:, cv2.CC_STAT_AREA]
    im_result = np.zeros_like(im_with_separated_blobs)
    for index_blob in range(1, nb_blobs):
        if sizes[index_blob] <= max_size:
            im_result[im_with_separated_blobs == index_blob] = 255
    return im_result

def get_background(text_type:str="ID"):
    background_paths = glob.glob(os.path.join(f"/home/jon/vietocr/VietNamese-OCR-DataGenerator/myanmar_project/crop_bg_image/{text_type}","*.*"))
    backgrounds = []
    for background_path in background_paths:
        background = cv2.imread(background_path)
        backgrounds.append(background)
    return backgrounds
    
def augment(image: np.array=None,
            mask:np.array=None,
            top:int=0,
            down:int=0,
            left:int=0,
            right:int=0):
    h,w = image.shape[:2]
    
    assert image[top:h-down,left:w-right].shape[:2] == mask.shape[:2], "image and mask must have same size but image have size {} and mask have size {}".format(image[top:h-down,left:w-right].shape[:2],mask.shape[:2])

    edge = cv2.Canny(mask,100,200)
    # cv2.imwrite("/home/jon/vietocr/VietNamese-OCR-DataGenerator/myanmar_project/debugs_myanmar/edge.png",edge)
    blur = cv2.blur(image,(3,3))
    blur = cv2.blur(blur,(3,3)) 

    res_image = image.copy()
    res_image[top:h-down,left:w-right][edge>0] = blur[top:h-down,left:w-right][edge>0]
    final_image = cv2.GaussianBlur(res_image,(3,3),0)
    final_image = cv2.GaussianBlur(final_image,(3,3),0)
    return final_image

def zoom_out_fc(text_image: np.array=None,
             background:np.array=None):
    # resize background image so that it is larger than text image
    top = random.randint(0,5)
    down = random.randint(0,5)
    left = random.randint(0,5)
    right = random.randint(0,5)
    background = cv2.resize(background,(text_image.shape[1]+left+right,text_image.shape[0]+top+down))
    return background,top,down,left,right

def zoom_in_fc(text_image:np.array=None):
    # resize text image to be bigger then crop it to original shape
    top = random.randint(0,5)
    down = random.randint(0,5)
    left = random.randint(0,5)
    right = random.randint(0,5)
    text_image = cv2.resize(text_image,(text_image.shape[1]+left+right,text_image.shape[0]+top+down))
    
    txt_h,txt_w = text_image.shape[:2]
    text_image = text_image[top:txt_h-down,left:txt_w-right]
    return text_image

def generate_image(image_path: str=None,
                   background: np.array=None,
                   debug_dir: str="myanmar_project/debugs_myanmar",
                   debug: bool=False,
                   zoom_out:bool=False,
                   zoom_in:bool=False):
    # Load text image and background image
    text_image = cv2.imread(image_path)
    h,w = text_image.shape[:2]
    file_infor = os.stat(image_path)
    # if invalid image -> return text image
    if h == 100 and w == 100 and file_infor.st_size == 214:
        return text_image
    
    # resize background image to text image size
    background = cv2.resize(background,(text_image.shape[1],text_image.shape[0]))
    
    bg_h,bg_w,bg_c = background.shape
    txt_h,txt_w,txt_c = text_image.shape
    assert bg_h == txt_h and bg_w == txt_w and bg_c == txt_c, "background and text image must have same size"
    
    # init top,down,left,right for paste text image to background image 
    top,down,left,right = 0,0,0,0

    # resize background image so that it is larger than text image
    if zoom_out:
        background,top,down,left,right = zoom_out_fc(text_image,background)
    
    # resize text image to be bigger then crop it to original shape
    if zoom_in:
        text_image = zoom_in_fc(text_image)
        bg_h,bg_w,bg_c = background.shape
        txt_h,txt_w,txt_c = text_image.shape
        assert bg_h == txt_h and bg_w == txt_w and bg_c == txt_c, "background and text image must have same size"
    
    
    if debug:
        cv2.imwrite(f"{debug_dir}/text_image.png",text_image)

        
    # Get mask from text_image
    gray_text = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
    _,binary_text_image = cv2.threshold(gray_text,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    if debug:
        cv2.imwrite(f"{debug_dir}/gray_text_image.png",gray_text)
        cv2.imwrite(f"{debug_dir}/binary_text_image.png",binary_text_image)
    
    # Paste text image to background image
    if zoom_out:
        bg_h,bg_w = background.shape[:2]
        background[top:bg_h-down,left:bg_w-right][binary_text_image>0] = text_image[binary_text_image>0]
    else:
        background[binary_text_image>0] = text_image[binary_text_image>0]
            
    if debug:
        cv2.imwrite(f"{debug_dir}/combined_image.png",background)
    
    # Augment
    final_image = augment(background,binary_text_image, top, down, left, right)
    if debug:
        cv2.imwrite(f"{debug_dir}/final_image.png",final_image)
    
    return final_image
    
    

def main():
    args = parse_arguments()
    debug,save_path, zoom_out,zoom_in, save_image_path = args.debug, args.save_path, args.zoom_out, args.zoom_in, args.save_image_path

    # Cropped text image directory
    dir_path = "/home/jon/vietocr/VietNamese-OCR-DataGenerator/myanmar_project/crop_data"
    images = glob.glob(os.path.join(dir_path,"*.*"))
    images = natsorted(images)
    cnt_image = 0
    actual_images = 0

    # Read label of cropped text image
    with open("myanmar_label.txt","r") as f:
        labels = f.readlines()
    
    myanmar_images = []
    
    # For each text image
    for image in tqdm(images):

        # choice background suitable with text image 
        # text images is splited into 9 fields "ID","Name","ISSUDE DATE","RELIGION","HEIGHT","GENDER","DOB","FATHER NAME","BLOOD"
        # each filed has 5 different backgrounds that mean each text image will be sampled 5 times
        image_name = os.path.basename(image).replace(".png","")
        text_type = os.path.basename(image).split("_")[-1].replace(".png","")
        backgrounds = get_background(text_type)

        for idx,background in enumerate(backgrounds):

            final_image = generate_image(image,background=background, debug=debug, zoom_out=zoom_out, zoom_in=zoom_in)

            # Save image            
            os.makedirs(f"{save_image_path}",exist_ok=True)
            cv2.imwrite(f"{save_image_path}/{image_name}_bg{idx}.png",final_image)

            myanmar_images.append(f"zoomin_augment_crop_data_bg/{image_name}_bg{idx}.png")
            cnt_image += 1
    print("Number of images: ",cnt_image)
    
    # Double check
    assert len(myanmar_images)/5 == len(labels), "Number of images and labels must be the same but images have {} and labels have {}".format(len(myanmar_images),len(labels))
    
    # write image and label to file
    for i in range(len(labels)):
        with open(save_path,"a") as f:
            # Loop though 5 images with difference background of each text image
            for j in range(5):
                try:
                    im = cv2.imread("myanmar_project/"+myanmar_images[5*i+j]) 
                    h,w = im.shape[:2]
                    # remove image with label is "N/A" or image with label is "Male" or image with median of pixel is 0 (invalid image)
                    if labels[i].strip() == "Male" and labels[i].strip() == "N/A" or np.median(im) == 0 or h == 100 and w == 100:
                        remove_name = os.path.basename("myanmar_project/"+myanmar_images[5*i+j])
                        shutil.copy("myanmar_project/"+myanmar_images[5*i+j],os.path.join("myanmar_project/remove_image",remove_name))
                        os.remove("myanmar_project/"+myanmar_images[5*i+j])
                        continue
                    f.write(myanmar_images[5*i+j]+"\t"+labels[i])
                    actual_images+=1
                except:
                    with open("error.txt",'a') as err_f:
                        err_f.write(myanmar_images[5*i+j]+"\n")
    
    print("Number of actual images: ",actual_images)
    assert len(glob.glob(os.path.join(f"{save_image_path}","*.*"))) == actual_images, "Number of images and actual images must be the same but images have {} and actual images have {}".format(len(glob.glob(os.path.join(f"{save_image_path}","*.*"))),actual_images)
        
    
if __name__ == "__main__":
    main()