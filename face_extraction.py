import cv2
import os

# photoフォルダに入っているフォルダ名を抽出する。
SRC_DIR_PATH = "./Photos"
in_photos = os.listdir(SRC_DIR_PATH)
photo_folders = [f for f in in_photos if os.path.isdir(os.path.join(SRC_DIR_PATH, f))]

# 使用するカスケード分類器を読み込む。分類器を変えたいときは、Cascadesフォルダから、使用したい分類器を選ぶ。
cascade_path = "./Cascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)

# 抽出した顔画像のファイル名とラベルを記録するテキストファイル
file = open("path_label.txt", "w+")
file.close()

# ファイルセットを作成する。
label_id = 0
for folder in photo_folders:

    src_file_list = os.listdir(SRC_DIR_PATH + "/" + folder)   # 写真のリスト
    id = 0

    for file in src_file_list:
        # 不要なファイルに判定を行わない。
        if file.startswith(".") or file.endswith(".txt"):
            continue

        try:
            # 画像の読み込み
            image = cv2.imread(os.path.join(SRC_DIR_PATH + "/" + folder, file))
            # 画像を読み込めなかった場合は、処理をスキップ
            if image is None:
                continue
            # 最初からグレースケールであれば変換の必要なし
            if len(image.shape) == 2:
                image_gray = image
                continue
            # 上記の場合以外はグレースケール変換を行う。
            else:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(e, file)
            continue

        # 画像のスケール調整
        resize_image_gray = cv2.resize(image_gray, (512, 512))
        # 元画像とのスケール比
        scale_height = 512.0 / image.shape[0]
        scale_width = 512.0 / image.shape[1]

        print("グレースケール変換が終わりました。")
        minsize = (int(resize_image_gray.shape[0]*0.1), int(resize_image_gray.shape[1]*0.1))

        try:
            print("顔画像を検出します。")
            # 顔領域の探索
            facerect = cascade.detectMultiScale(resize_image_gray, scaleFactor=1.11, minNeighbors=3, minSize=minsize)

            # 検出領域が2つ以上ある場合も、検出できなかったことにする。
            if len(facerect) == 0 or len(facerect) > 1:
                print("検出できませんでした。")
                coutinue
            # 切り出し範囲
            for rect in facerect:
                min_height = int(rect[0]/scale_height)
                min_width = int(rect[1]/scale_width)
                max_height = int(rect[2]/scale_height) + min_height
                max_width = int(rect[3]/scale_width) + min_width

            # 抽出した顔画像の保管先
            DST_DIR_PATH = "./extracted_faces_" + folder + "/"
            if not os.path.exists(DST_DIR_PATH):
                os.makedirs(DST_DIR_PATH)
            # カラー画像として保存する。
            photo_saved = os.path.join(DST_DIR_PATH, "image_{}.jpg".format(str(id).zfill(4)))
            cv2.imwrite(photo_saved, image[min_height:max_height, min_width:max_width])

            # 抽出した顔画像のファイル名とラベルをテキストファイルに記録
            with open("path_label.txt", "a") as file:
                file.write(photo_saved + " " + str(label_id) + "\n")

            id += 1

        except Exception as e:
            print(e)
    label_id += 1
