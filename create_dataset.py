import os
import cv2
import numpy as np
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

print("Buat dataset disini")
jumlah_class = input("Masukkan Jumlah Class")

if not os.path.exists("dataset"):
    os.mkdir("dataset")

for i in range(int(jumlah_class)):
    nama_class = input("Masukkan Nama Class Ke-" + str(i+1))
    folder = "dataset/" + str(nama_class).upper()
    if not os.path.exists(folder):
      os.mkdir(folder)

    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    x, y, w, h = 350, 150, 200, 200

    jumlah_file_dalam_folder = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
    no =  1 if jumlah_file_dalam_folder == 0 else jumlah_file_dalam_folder+1

    while True:

        # for i in os.listdir("dataset"):
        #     jumlah_file = len([name for name in os.listdir("dataset/" + i) if os.path.isfile(os.path.join("dataset/" + i, name))])
        #     print(i, jumlah_file, "," , end='')
        # print('')

        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Video", frame)

        gambar_crop = frame[y:y+h, x:x+w]
        gambar_crop = cv2.cvtColor(gambar_crop, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Video Crop", gambar_crop)

        keypress = cv2.waitKey(1)

        if keypress == 27:
            break
        elif keypress == ord('a'):
            gambar_crop = cv2.resize(gambar_crop, (28,28))
            cv2.imwrite("{}/{}.jpg".format(folder,no),gambar_crop)
            print("Gambar Class {} Ke-{} Tersimpan".format(nama_class, no))
            no += 1
        elif keypress == ord('c'):
            no = 1
            shutil.rmtree(folder, ignore_errors=False, onerror=None)
            os.mkdir(folder)
            print("Berhasil Di clear untuk folder class" + nama_class)


    camera.release()
    cv2.destroyAllWindows()