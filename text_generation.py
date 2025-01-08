import cv2
from tqdm import tqdm
from custom_text_generation_v3 import get_rand_address, get_rand_nguyenquan
with open("synthetic_data.txt",'a') as f:
    for i in tqdm(range(int(10000))):
        hktt1,hktt2 = get_rand_address()
        nq1,nq2 = get_rand_nguyenquan()
        nq = nq1+ "," + nq2
        f.write("hokhauthuongtru"+"\t"+hktt1 + "\n")
        f.write("hokhauthuongtru"+"\t"+hktt2 + "\n")
        f.write("nguyenquan"+"\t"+nq + "\n")
# with open("nq.txt",'r') as f:
#     content = f.readlines()
# new_content = []
# for line in content:
#     new_line = "nguyenquan\t"+line
#     new_content.append(new_line)

# with open("nq.txt",'w') as f:
#     for line in new_content:
#         f.write(line)