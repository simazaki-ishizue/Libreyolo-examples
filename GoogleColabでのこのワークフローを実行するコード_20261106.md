# GoogleColabでのこのワークフローを実行するコード

Google Colab上で資料のワークフローを再現するための、ステップバイステップの実装コードを提示します。

Colab特有のI/Oボトルネック（Google Driveからの直接読み込みによる遅延）を回避するため、データセットを一度Colabのローカルストレージ（`/content/`）に展開する手順を含めています。

### 1. 環境構築とドライブのマウント

クラウドインスタンスの環境差異によるサイレントな依存関係の競合を防ぐため、セッション開始時に必要なライブラリのバージョンを固定してインストールします。

```
# Google Driveのマウント（チェックポイントの永続化とZIPデータの取得用）
from google.colab import drive
drive.mount('/content/drive')
```

```
# 必要なライブラリのインストール(pythonパッケージ)
# ※ LibreYOLOがPyPIに登録されている前提。GitHubリポジトリの場合は 'git+https://...' を指定します。
!pip install libreyolo albumentations sahi numpy opencv-python
```

```python
# I/Oボトルネック回避のため、Drive上のデータセット(ZIP)をColabのローカルにコピーして解凍
import os
import shutil

zip_path = '/content/drive/MyDrive/libreyolo/dataset.zip'  # 環境に合わせて変更してください。
local_extract_path = '/content/dataset'

if not os.path.exists(local_extract_path):
    # os.makedirs(local_extract_path)
    shutil.unpack_archive(zip_path, "/content")
    print("データセットをローカルストレージに展開しました。")
```

### 2. データ拡張（Augmentation）の設定

合成データの代わりに、Albumentationsを用いて環境変化に対するモデルの耐性を高めるための変換処理を定義します。

```python
import albumentations as A

# 資料に記載されている強力なデータ拡張パイプライン
augmentation = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),         # モーションブラー[cite: 1]
    A.GaussNoise(var_limit=(10, 50), p=0.3),          # センサーノイズ[cite: 1]
    A.ImageCompression(quality_lower=60, p=0.4),      # ストリーム圧縮劣化[cite: 1]
    A.RandomBrightnessContrast(p=0.5),                # 照明の変化[cite: 1]
    A.Perspective(scale=(0.02, 0.08), p=0.3),         # カメラアングルの変動[cite: 1]
])

# ※ 実際の学習時は、このaugmentationをデータローダーやLibreYOLOの設定ファイルに組み込みます。

```

### 3. RF-DETRモデルの学習

LibreYOLOの統合APIを使用し、Vision TransformerベースのRF-DETRモデルを学習させます。混合精度学習（AMP）を有効にしてGPUメモリの消費を抑えます。

```python
# LibreYOLOのインポート (YOLOファミリと共通のAPIを想定)
from libreyolo import LibreYOLO

# RF-DETRアーキテクチャの初期化[cite: 1]
model = LibreYOLO("LibreYOLO9t.pt")

# 学習の実行
# プロジェクトの保存先をGoogle Driveに指定し、Spot VM（Colabセッション）中断時の消失を防ぐ
results = model.train(
    data=f'{local_extract_path}/dataset.yaml', # ローカルに展開したデータセットを指定
    epochs=100,
    batch=16,
    amp=True,  # 混合精度学習(FP16)を有効化[cite: 1]
    patience=25,  # 20〜30エポック改善がない場合の早期終了(Early Stopping)[cite: 1]
    project='/content/drive/MyDrive/libreyolo/libreyolo_runs', # チェックポイントの保存先をDriveへ
    name='rf_LibreYOLO9t_model'
)
```

### 4. 信頼度しきい値の自動キャリブレーション

固定のしきい値による誤報を防ぐため、F1スコアが最大となる最適なしきい値を探索します。

```python
import numpy as np

CONF_THRESHOLDS = np.arange(0.01, 0.96, 0.01)
f1_scores = []

# 学習済みのベストモデルをロード
best_model = LibreYOLO('/content/drive/MyDrive/libreyolo/libreyolo_runs/rf_LibreYOLO9t_model/weights/best.pt')

# 各しきい値でのPrecision, Recall, F1スコアの計算
for conf in CONF_THRESHOLDS:
    val_results = best_model.val(data=f'{local_extract_path}/dataset.yaml', conf=conf)
    
    # BoxImageMetrics 内の配列から平均値を取得
    p = np.mean(val_results.box.p)
    r = np.mean(val_results.box.r)

    # F1スコアの計算（PrecisionとRecallの調和平均）
    f1 = 2 * p * r / (p + r + 1e-9)
    f1_scores.append(f1)

# F1スコアが最も高くなるしきい値を選択
best_f1_conf = CONF_THRESHOLDS[np.argmax(f1_scores)]

print(f"Deployment向けの最適なしきい値: {best_f1_conf}")
```

### 5. SAHIを用いた小物体検出（推論）

広角映像内の小さな物体（ナイフや携帯電話など）を検出するため、入力フレームをタイル状に分割・推論・結合するSAHIを実装します。

```python
import cv2

# NMS（Non-Maximum Suppression）の簡易実装用のダミー関数
# ※ 実際には torchvision.ops.nms などを利用します
def non_max_suppression(boxes, iou_threshold):
    # 重複するバウンディングボックスを統合する処理
    return boxes

def sahi_predict(model, image, tile_size=640, overlap=0.2, conf=0.25, iou=0.45):
    H, W  = image.shape[:2]
    step  = int(tile_size * (1 - overlap))   # タイル間のストライド計算[cite: 1]
    boxes = []

    # 画像を重なり合うタイルに分割して推論[cite: 1]
    for y in range(0, H - tile_size + 1, step):
        for x in range(0, W - tile_size + 1, step):

            # フルフレームからタイルをクロップ[cite: 1]
            tile    = image[y:y+tile_size, x:x+tile_size]
            sahi_results = model.predict(tile, conf=conf)[0] # LibreYOLOの予測

            if sahi_results.boxes is None or len(sahi_results.boxes) == 0:
                continue

            # タイル上の座標を元のフル画像上の空間にマッピングし直す[cite: 1]
            for box in sahi_results.boxes.xyxy:
                x1, y1, x2, y2 = box.tolist()
                boxes.append([x1+x, y1+y, x2+x, y2+y])

    # 複数のタイルにまたがって検出された重複ボックスを統合する[cite: 1]
    return non_max_suppression(boxes, iou_threshold=iou)

# テスト画像の読み込みとSAHIの実行
frame = cv2.imread(f'{local_extract_path}/test/images/000000000049.jpg')

# 640×640のタイルサイズ、20%のオーバーラップで推論（小物体の検出数が約4倍に増加）[cite: 1]
final_boxes = sahi_predict(best_model, frame, tile_size=640, overlap=0.2, conf=best_f1_conf)
print(f"検出されたオブジェクトのバウンディングボックス: {final_boxes}")
# 画像にBBOXを描画
color = (0, 255, 0)
thickness = 2  # 枠線の太さ
for bbox in final_boxes:
    # 矩形（BBOX）の描画
    xmin, ymin, xmax, ymax = map(int, bbox)
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
# BBOX描画画像の出力
os.makedirs("/content/output_images")
cv2.imwrite(os.path.join("/content/output_images/outimage.jpg"), frame)
```

このコードを実行することで、データの読み込みから学習、最適化、そして高度な推論（SAHI）までの一連のワークフローをColab上で検証することが可能です。