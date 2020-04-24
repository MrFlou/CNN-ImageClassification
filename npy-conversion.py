import numpy as np
import os
from PIL import Image
from io import StringIO
import pandas as pd
import csv

# for npyfilename in filenames:
images = np.load("D:\\CNN-ImageClassification\\Data\\poke_image_data.npy")
csv = np.genfromtxt("D:\\CNN-ImageClassification\\Data\\names_and_strengths.csv", delimiter=',',dtype=None)
#data = pd.read_csv("Data/names_and_strengths.csv")
# assuming arr.shape is (W,H,C) for Width and Height in pixels, and C channels (such as 3 for RGB)
# also assuming that values in each array position are in range(0, 256) - if not, see PIL's convert modes
# if these assumptions don't hold, you need to first reshape and normalize

print(len(images))
print(csv[1,0].decode("utf-8"))
x = 0
while x < len(images):
    im = Image.fromarray(images[x])
    if not os.path.exists("Pokemon\\"+csv[x+1,0].decode("utf-8")):
        os.mkdir("Pokemon\\"+csv[x+1,0].decode("utf-8"))
    pngfilename = "Pokemon\\"+csv[x+1,0].decode("utf-8")+"\\"+csv[x+1,0].decode("utf-8")+"."+str(x)+".png"
    im.save(pngfilename)
    x +=1
