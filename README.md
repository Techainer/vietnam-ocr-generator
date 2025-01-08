## Project structure
>This project consist of 3 main file corresponding with specific function and some helper folder

**File**
- myanmar_handwriting_text_generation.py (used to generate ```myanmar handwriting text``` image)
- custom_text_generation_v2.py (generate ```vietnamese text image``` with custom erode operation)
- text_generation.py (generate random text)

**Directory:**
- background_cccd_new (background for the cccd)
- background (background for cccd chip)
- data (data to generate custom text)
- fonts (font for text)
- myanmar_project/crop_bg_image (background for myanmar)
- myanmar_project/crop_data (myanmar cropped text image)
###  1. myanmar_handwriting_text_generation.py
> Following below step to run 

Step 1: Prepare 

- **myanmar_label.txt** file

- myanmar images in **myanmar_project/crop_data** 

- myanmar background in **myanmar_project/crop_bg_image**

Step 2: Run
```
python myanmar_handwriting_text_generation.py 
```

The output image and label will appear in your ```save_image_path``` and ```save_path``` 

### 2. custom_text_generation_v2.py 
> Following below step to run 

Step 1: Prepare new background (the cccd) in **background_cccd_new** and former background (cccd_chip) in **background**

Step 2: Run this file following  below instructions
 
 This file have 2 mode:
- generate custom text (use ```string``` argument).
Run this file by following command:

```
python custom_text_generation_v3.py 
```

- generate exist text from ```.txt``` file.
Run this file by following command:
```
python custom_text_generation_v3.py --text_from_file <text file>
```

Note: This file only support ```hokhauthuongtru``` and ```nguyenquan``` field

### 2. text_generation.py 
> This file is used for generate .txt file as input of custom_text_generation_v2.py file. You can simply run this by:

```
python text_generation.py
```
a ```synthetic_data.txt``` file will be created