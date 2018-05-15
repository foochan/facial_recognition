# facial_recognition
顔画像

Cascades
顔認識をするためのカスケード分類器が入っているフォルダです。
OpenCVのGithubページ
https://github.com/opencv/opencv/tree/master/data/haarcascades
https://github.com/opencv/opencv/tree/master/data/lbpcascades
にある、Haar-like特徴を使った分類器とLBP特徴を使った分類器をここに入れてあります。
使用する分類器を変えるときは、


Photos
　識別に使いたい人物の写真を入れておくフォルダです。人物ごとにフォルダを分けています。このページでは一昔前に現役だった
サッカー選手3人(ベッカム、ジダン、ロナウド(※ブラジルの方))のフォルダを入れてあります。

face_extraction.py
　画像から顔の部分を抽出⇒トリミングし、切り出した画像を別途ファイルとして保存するコードです。
人物別にフォルダに分ける。

model.py
　画像データをクラス分類出力に変換するためのCNNモデルが記述されているコードです。
 
flags.py
　Tensorflowのメソッドが使えなくなったので、代用するためのファイルです。

training.py
　トレーニングを行うコードです。
