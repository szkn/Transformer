# クラスを記述したスクリプト、Kinectからいつも使っているデータへの変換を行える

# 動画の性質取得
import json
import cv2
import numpy as np
import os
import glob
import csv
from tqdm import tqdm

# Kinectで撮影した映像を画像に落とし込むためのクラス
class kinect2py():
    def __init__(self):
        self.index = 0
        self.index_2d = 0

    #画像から線を描画して動画として出力する
    #基本的には，この関数だけ使っておけばOK
    def write_video(self, csvfile, img_folder):
        self.index_2d = 0
        self.date = csvfile.split("/")[-1][:-4]
        self.rename(img_folder) #ゼロ詰めにして，リネームすることで関節点情報と，動画のフレームの順番の整合性をとる
        IMG_WIDTH = 1280
        IMG_HEIGHT = 720
        fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
        video  = cv2.VideoWriter(self.date+'.mp4', fourcc, 25.0, (IMG_WIDTH, IMG_HEIGHT))
        img_files = sorted(glob.glob(img_folder + "/" + self.date + "/*.jpg"))

        num_csv_rows = self.count_rows(csvfile)

        print(f"Processing {self.date} video")
        print(f'number of files {len(img_files)}')
        print(f'number of csv rows {num_csv_rows}')

        for i, img_file in enumerate(tqdm(img_files)):
            img = cv2.imread(img_file)

            # csvファイルが尽きたら終わり
            if i < num_csv_rows:
                l = self.get_2d_keypoints(csvfile)
                
                self.draw_line(Nose_2d, Head_2d, img)
                self.draw_line(Head_2d, Neck_2d,img)
                self.draw_line(Neck_2d, SpineChest_2d, img)
                self.draw_line(SpineChest_2d, SpineNaval_2d, img)

                self.draw_line(Nose_2d, REye_2d, img)
                self.draw_line(REye_2d, REar_2d, img)
                self.draw_line(Nose_2d, LEye_2d, img)
                self.draw_line(LEye_2d, LEar_2d, img)
                
                self.draw_line(SpineChest_2d, RClavicle_2d, img)
                self.draw_line(RClavicle_2d, RShoulder_2d, img)
                self.draw_line(RShoulder_2d, RElbow_2d, img)
                self.draw_line(RElbow_2d, RWrist_2d, img)
                self.draw_line(RWrist_2d, RHand_2d, img)
                self.draw_line(RHand_2d, RHandTip_2d, img)
                self.draw_line(RHand_2d, RThumb_2d, img)

                self.draw_line(SpineChest_2d, LClavicle_2d, img)
                self.draw_line(LClavicle_2d, LShoulder_2d, img)
                self.draw_line(LShoulder_2d, LElbow_2d, img)
                self.draw_line(LElbow_2d, LWrist_2d, img)
                self.draw_line(LWrist_2d, LHand_2d, img)
                self.draw_line(LHand_2d, LHandTip_2d, img)
                self.draw_line(LHand_2d, LThumb_2d, img)
                
                # self.draw_point(SpineChest_2d, "SpineChest", img)
                # self.draw_point(SpineNaval_2d, "SpineNaval", img)
                # self.draw_point(RThumb_2d, "Rthumb", img)
                # self.draw_point(LThumb_2d, "Lthumb", img)
                # self.draw_point(RClavicle_2d, "Rclav", img)
                # self.draw_point(LClavicle_2d, "Lclav", img)
                # self.draw_point(RShoulder_2d, "Rshoulder", img)
                # self.draw_point(LShoulder_2d, "Lshoulder", img)
                # self.draw_point(RElbow_2d, "Relbow", img)
                # self.draw_point(LElbow_2d, "Lelbow", img)
                # self.draw_point(RWrist_2d, "Rwrist", img)
                # self.draw_point(LWrist_2d, "Lwrist", img)
                # self.draw_point(LEar_2d, "LEar", img)
                # self.draw_point(REar_2d, "REar", img)
                # self.draw_point(LEye_2d, "LEye", img)
                # self.draw_point(REye_2d, "REye", img)
                # self.draw_point(LHandTip_2d, "LHandTip", img)
                # self.draw_point(RHandTip_2d, "RHandTip", img)
                # self.draw_point(LHand_2d, "LHand", img)
                # self.draw_point(RHand_2d, "RHand", img)

                video.write(img)

            else:
                break

        # print("complete!!")
        video.release()
        self.index=0

    # csvファイルを読み込んですべてのデータを含んだベクトルを返すメソッド.内部メソッドなので、外部では用いない
    def read_csv(self, csvfile):
        l_vectors = []
        with open(csvfile) as file:
            reader = csv.reader(file, delimiter=",")
            l_vectors = file.readlines()
        # print(l_vectors.shape)
        return l_vectors

    # 一回実行すると、1フレーム分の関節点を返すメソッド
    # データは処理用の3Dのデータ
    def get_3d_keypoints(self, csvfile):
        global Pelvis, SpineNaval, SpineChest, Neck, LClavicle, LShoulder, LElbow, LWrist, LHand,  LHandTip, \
            LThumb, RClavicle, RShoulder, RElbow, RWrist, RHand, RHandTip, RThumb, LHip,\
            LKnee, LAnkle, LFoot, RHip, RKnee, RAnkle, RFoot, Head, Nose, LEye, LEar, REye,  REar
        l_vectors = self.read_csv(csvfile)
        l_keypoints = l_vectors[self.index].split(",")
        for i in range(100):
            l_keypoints[i] = float(l_keypoints[i])

        Pelvis = np.array([l_keypoints[2], l_keypoints[3], l_keypoints[4]])
        SpineNaval = np.array([l_keypoints[5], l_keypoints[6], l_keypoints[7]])
        SpineChest = np.array([l_keypoints[8], l_keypoints[9], l_keypoints[10]])
        Neck = np.array([l_keypoints[11], l_keypoints[12], l_keypoints[13]])
        LClavicle = np.array([l_keypoints[14], l_keypoints[15], l_keypoints[16]])
        LShoulder = np.array([l_keypoints[17], l_keypoints[18], l_keypoints[19]])
        LElbow = np.array([l_keypoints[20], l_keypoints[21], l_keypoints[22]])
        LWrist = np.array([l_keypoints[23], l_keypoints[24], l_keypoints[25]])
        LHand = np.array([l_keypoints[26], l_keypoints[27], l_keypoints[28]])
        LHandTip = np.array([l_keypoints[29], l_keypoints[30], l_keypoints[31]])
        LThumb = np.array([l_keypoints[32], l_keypoints[33], l_keypoints[34]])
        RClavicle = np.array([l_keypoints[35], l_keypoints[36], l_keypoints[37]])
        RShoulder = np.array([l_keypoints[38], l_keypoints[39], l_keypoints[40]])
        RElbow = np.array([l_keypoints[41], l_keypoints[42], l_keypoints[43]])
        RWrist = np.array([l_keypoints[44], l_keypoints[45], l_keypoints[46]])
        RHand = np.array([l_keypoints[47], l_keypoints[48], l_keypoints[49]])
        RHandTip = np.array([l_keypoints[50], l_keypoints[51], l_keypoints[52]])
        RThumb = np.array([l_keypoints[53], l_keypoints[54], l_keypoints[55]])
        LHip = np.array([l_keypoints[56], l_keypoints[57], l_keypoints[58]])
        LKnee = np.array([l_keypoints[59], l_keypoints[60], l_keypoints[61]])
        LAnkle = np.array([l_keypoints[62], l_keypoints[63], l_keypoints[64]])
        LFoot = np.array([l_keypoints[65], l_keypoints[66], l_keypoints[67]])
        RHip = np.array([l_keypoints[68], l_keypoints[69], l_keypoints[70]])
        RKnee = np.array([l_keypoints[71], l_keypoints[72], l_keypoints[73]])
        RAnkle = np.array([l_keypoints[74], l_keypoints[75], l_keypoints[76]])
        RFoot = np.array([l_keypoints[77], l_keypoints[78], l_keypoints[79]])
        Head = np.array([l_keypoints[80], l_keypoints[81], l_keypoints[82]])
        Nose = np.array([l_keypoints[83], l_keypoints[84], l_keypoints[85]])
        LEye = np.array([l_keypoints[86], l_keypoints[87], l_keypoints[88]])
        LEar = np.array([l_keypoints[89], l_keypoints[90], l_keypoints[91]])
        REye = np.array([l_keypoints[92], l_keypoints[93], l_keypoints[94]])
        REar = np.array([l_keypoints[95], l_keypoints[96], l_keypoints[97]])

        self.index = self.index + 1

        return [Pelvis, SpineNaval, SpineChest, Neck, LClavicle, LShoulder, LElbow, LWrist, LHand,  LHandTip, \
            LThumb, RClavicle, RShoulder, RElbow, RWrist, RHand, RHandTip, RThumb, LHip,\
            LKnee, LAnkle, LFoot, RHip, RKnee, RAnkle, RFoot, Head, Nose, LEye, LEar, REye,  REar]
    
    # 一回実行すると、1フレーム分の関節点を返すメソッド
    # データは画面描写用の2Dのコード
    # 他の関数から使われるので，単体で用いることはない．
    def get_2d_keypoints(self, csvfile):
        global Pelvis_2d, SpineNaval_2d, SpineChest_2d, Neck_2d, LClavicle_2d, LShoulder_2d, LElbow_2d, LWrist_2d, LHand_2d,  \
            LHandTip_2d, LThumb_2d, RClavicle_2d, RShoulder_2d, RElbow_2d, RWrist_2d, RHand_2d, RHandTip_2d, RThumb_2d, LHip_2d,\
            LKnee_2d, LAnkle_2d, LFoot_2d, RHip_2d, RKnee_2d, RAnkle_2d, RFoot_2d, Head_2d, Nose_2d, LEye_2d, LEar_2d, REye_2d,  REar_2d
        l_vectors = self.read_csv(csvfile)
        l_keypoints = l_vectors[self.index_2d].split(",")
        # print(l_keypoints[20])
        for i in range(66):
            l_keypoints[i] = float(l_keypoints[i])

        Pelvis_2d = np.array([l_keypoints[2], l_keypoints[3]])
        SpineNaval_2d = np.array([l_keypoints[4], l_keypoints[5]])
        SpineChest_2d = np.array([l_keypoints[6], l_keypoints[7]])
        Neck_2d = np.array([l_keypoints[8], l_keypoints[9]])
        LClavicle_2d = np.array([l_keypoints[10], l_keypoints[11]])
        LShoulder_2d = np.array([l_keypoints[12], l_keypoints[13]])
        LElbow_2d = np.array([l_keypoints[14], l_keypoints[15]])
        LWrist_2d = np.array([l_keypoints[16], l_keypoints[17]])
        LHand_2d = np.array([l_keypoints[18], l_keypoints[19]])
        LHandTip_2d = np.array([l_keypoints[20], l_keypoints[21]])
        LThumb_2d = np.array([l_keypoints[22], l_keypoints[23]])
        RClavicle_2d = np.array([l_keypoints[24], l_keypoints[25]])
        RShoulder_2d = np.array([l_keypoints[26], l_keypoints[27]])
        RElbow_2d = np.array([l_keypoints[28], l_keypoints[29]])
        RWrist_2d = np.array([l_keypoints[30], l_keypoints[31]])
        RHand_2d = np.array([l_keypoints[32], l_keypoints[33]])
        RHandTip_2d = np.array([l_keypoints[34], l_keypoints[35]])
        RThumb_2d = np.array([l_keypoints[36], l_keypoints[37]])
        LHip_2d = np.array([l_keypoints[38], l_keypoints[39]])
        LKnee_2d = np.array([l_keypoints[40], l_keypoints[41]])
        LAnkle_2d = np.array([l_keypoints[42], l_keypoints[43]])
        LFoot_2d = np.array([l_keypoints[44], l_keypoints[45]])
        RHip_2d = np.array([l_keypoints[46], l_keypoints[47]])
        RKnee_2d = np.array([l_keypoints[48], l_keypoints[49]])
        RAnkle_2d = np.array([l_keypoints[50], l_keypoints[51]])
        RFoot_2d = np.array([l_keypoints[52], l_keypoints[53]])
        Head_2d = np.array([l_keypoints[54], l_keypoints[55]])
        Nose_2d = np.array([l_keypoints[56], l_keypoints[57]])
        LEye_2d = np.array([l_keypoints[58], l_keypoints[59]])
        LEar_2d = np.array([l_keypoints[60], l_keypoints[61]])
        REye_2d = np.array([l_keypoints[62], l_keypoints[63]])
        REar_2d = np.array([l_keypoints[64], l_keypoints[65]])

        self.index_2d = self.index_2d + 1
        
    def draw_point(self, joint, name,  img):
        cv2.circle(img,(int(joint[0]),int(joint[1])), 3, (255,0,0), -1)
        cv2.putText(img, name, (int(joint[0]),int(joint[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=1,lineType=cv2.LINE_4)

    def draw_line(self, joint1, joint2, img):
        cv2.line(img, (int(joint1[0]),int(joint1[1])), (int(joint2[0]),int(joint2[1])), color=(255,0,255),thickness=2)

    def rename(self, img_folder):
        files = glob.glob(img_folder + "/" + self.date + "/*.jpg")
        # ファイルを連番に変換する
        for file in files:
            number = file.split("/")[-1][:-4]
            p_number = number.zfill(5)
            os.rename(file, img_folder + "/" + self.date+"/"+p_number+".jpg")

    # csvファイルの長さを返す関数
    def count_rows(self, file_path):
        with open(file_path) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        return len(rows)