# facial_recognition
顔画像

◆Cascades<br/>
顔領域を認識するためのカスケード分類器が入っているフォルダです。<br/>
OpenCVのGithubページ<br/>
https://github.com/opencv/opencv/tree/master/data/haarcascades<br/>
https://github.com/opencv/opencv/tree/master/data/lbpcascades<br/>
にある、Haar-like特徴を使った分類器とLBP特徴を使った分類器をここに入れてあります。<br/>
使用する分類器を変えるときは、face_extraction.py内の変数cascade_pathを書き換えます。<br/>


◆Photos<br/>
識別に使いたい人物の写真を入れておくフォルダです。人物ごとにフォルダを分けています。<br/>
このページでは一昔前に現役だったサッカー選手3人(ベッカム、ジダン、(ブラジルの)ロナウド)のフォルダを入れてあります。<br/>

◆face_extraction.py<br/>
画像から顔の部分を抽出⇒トリミングし、切り出した画像を別途ファイルとして保存するコードです。<br/>
人物別にフォルダに分ける。<br/>

◆model.py<br/>
画像データをクラス分類出力に変換するためのCNNモデルが記述されているコードです。<br/>
 
◆flags.py<br/>
Tensorflowのメソッドが使えなくなったので、代用するためのファイルです。<br/>

◆training.py<br/>
トレーニングを行うコードです。<br/>
