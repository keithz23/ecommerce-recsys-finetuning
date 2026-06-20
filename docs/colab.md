# Huong Dan Chay Toan Bo Du An Tren Google Colab

Tai lieu nay huong dan cach chay project recommendation system tren Google Colab voi dataset moi.

Notebook nen dung:

```text
notebooks/combined_recommendation_system.ipynb
```

Notebook nay da duoc dong nhat de doc dataset tu:

```text
data/input/interaction_train.csv
data/input/interaction_test.csv
```

va sinh file trung gian cho Popularity-C tai:

```text
data/interim/popularity_train_dataset.csv
```

## 1. Chuan Bi Tren Google Drive

Nen tao thu muc tren Google Drive:

```text
MyDrive/ecommerce-recsys-finetuning/
```

Ben trong nen co:

```text
ecommerce-recsys-finetuning/
  data/
    input/
      interaction_train.csv
      interaction_test.csv
  notebooks/
    combined_recommendation_system.ipynb
  requirements.txt
  src/
  models/
  reports/
```

Neu project dang o local, co the zip ca folder project roi upload len Drive.

## 2. Mount Google Drive

Trong Colab, chay:

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 3. Dua Project Vao Runtime `/content`

Nen copy project tu Drive vao `/content` de doc file nhanh hon. Doc truc tiep file lon tu Drive thuong cham hon.

### Cach A: Neu Project La File Zip

```bash
!unzip "/content/drive/MyDrive/ecommerce-recsys-finetuning.zip" -d /content/
%cd /content/ecommerce-recsys-finetuning
```

### Cach B: Neu Project La Folder Tren Drive

```bash
!cp -r "/content/drive/MyDrive/ecommerce-recsys-finetuning" /content/ecommerce-recsys-finetuning
%cd /content/ecommerce-recsys-finetuning
```

### Cach C: Clone Tu GitHub

```bash
!git clone <REPO_URL> /content/ecommerce-recsys-finetuning
%cd /content/ecommerce-recsys-finetuning
```

Sau do copy dataset tu Drive vao project neu dataset khong nam trong repo:

```bash
!mkdir -p data/input data/interim reports/popularity/charts

!cp "/content/drive/MyDrive/datasets/interaction_train.csv" data/input/interaction_train.csv
!cp "/content/drive/MyDrive/datasets/interaction_test.csv" data/input/interaction_test.csv
```

## 4. Kiem Tra File Dataset

Chay:

```bash
!ls -lh data/input
!head -n 2 data/input/interaction_train.csv
!head -n 2 data/input/interaction_test.csv
```

Can thay 2 file:

```text
interaction_train.csv
interaction_test.csv
```

Header can co it nhat:

```text
UserId,ProductId,Rating
```

Header hien tai nen la:

```text
UserId,ProductId,Rating,Timestamp,Date,VerifiedPurchase,HelpfulVote,SourceCategory,IsRelevant,Split
```

## 5. Cai Dependencies

Chay:

```bash
!pip install -r requirements.txt
```

Neu loi `surprise` hoac notebook khong import duoc `surprise`, chay them:

```bash
!pip install scikit-surprise
```

Neu Colab bao can restart runtime sau khi cai package, bam:

```text
Runtime -> Restart runtime
```

roi chay lai tu dau cac cell import.

## 6. Chay Notebook Chinh

Mo notebook:

```text
notebooks/combined_recommendation_system.ipynb
```

Chay tu tren xuong duoi.

Notebook nay da dung path theo project root:

```text
data/input/interaction_train.csv
data/input/interaction_test.csv
data/interim/popularity_train_dataset.csv
reports/popularity/
```

Khong can dung path cu:

```text
/content/data/...
```

## 7. Luu Output Ve Google Drive

Sau khi chay xong, copy output quan trong ve Drive:

```bash
!mkdir -p "/content/drive/MyDrive/ecommerce_outputs"

!cp -f data/processed/processed_data.pkl "/content/drive/MyDrive/ecommerce_outputs/processed_data.pkl"
!cp -f models/final_model_svd.pkl "/content/drive/MyDrive/ecommerce_outputs/final_model_svd.pkl"
!cp -r reports "/content/drive/MyDrive/ecommerce_outputs/reports"
!cp -r data/interim "/content/drive/MyDrive/ecommerce_outputs/interim"
```

Neu mot file chua duoc tao, lenh `cp` co the bao loi. Khi do hay kiem tra cell export trong notebook da chay chua.

## 8. File Output Quan Trong

Sau khi chay full, nen co:

```text
data/processed/processed_data.pkl
models/final_model_svd.pkl
data/interim/popularity_train_dataset.csv
reports/figures/*.png
reports/popularity/*.csv
reports/popularity/charts/*.png
```

Y nghia:

| File/Folder | Y nghia |
|---|---|
| `data/processed/processed_data.pkl` | Dataset da xu ly cho model/demo |
| `models/final_model_svd.pkl` | Model SVD da train lai |
| `data/interim/popularity_train_dataset.csv` | Product-level features cho Popularity-C |
| `reports/figures/` | Hinh EDA/model comparison |
| `reports/popularity/` | Ket qua Popularity A/B/C |

## 9. Luu Y Ve RAM Va Thoi Gian Chay

File hien tai rat lon:

```text
interaction_train.csv gan 1.9GB
interaction_test.csv gan 300MB
```

Vi vay tren Colab free co the gap:

```text
- Chay cham
- Het RAM
- Runtime bi reset
```

De test nhanh pipeline, co the dung sample nho truoc:

```python
train_raw = pd.read_csv(INTERACTION_TRAIN_FILE_MAIN, usecols=main_usecols, nrows=200_000)
test_raw = pd.read_csv(INTERACTION_TEST_FILE_MAIN, usecols=main_usecols, nrows=50_000)
```

Chi dung sample de test code. Khi can ket qua chinh thuc thi bo `nrows`.

## 10. Co Can Chay `demo_app.py` Tren Colab Khong?

Khong bat buoc.

Colab phu hop cho:

```text
- training
- evaluation
- tao report
- tao model artifact
```

`demo_app.py` la local web app, nen chay tren may local se tien hon.

Neu van muon chay demo tren Colab, can mo port/tao tunnel rieng. Phan nay khong can cho pipeline notebook.

## 11. Loi Thuong Gap Tren Colab

### Loi: Khong Tim Thay Dataset

Kiem tra:

```bash
!pwd
!ls -lh data/input
```

Phai dang o project root:

```text
/content/ecommerce-recsys-finetuning
```

va co:

```text
data/input/interaction_train.csv
data/input/interaction_test.csv
```

### Loi: `ModuleNotFoundError: No module named 'src'`

Ban dang khong o project root. Chay:

```python
%cd /content/ecommerce-recsys-finetuning
```

### Loi: `ModuleNotFoundError: No module named 'surprise'`

Chay:

```bash
!pip install scikit-surprise
```

Sau do restart runtime neu Colab yeu cau.

### Loi: Het RAM

Giai phap:

```text
- Dung Colab Pro/High-RAM runtime.
- Test bang sample voi nrows.
- Chi doc cot can thiet.
- Chay tung section thay vi Run all.
```

### Loi: Output Mat Sau Khi Runtime Reset

Colab xoa `/content` khi runtime reset. Hay copy output ve Drive:

```bash
!cp -r reports "/content/drive/MyDrive/ecommerce_outputs/reports"
```

## 12. Quy Trinh Chay Khuyen Nghi

1. Mount Drive.
2. Copy project vao `/content`.
3. Copy dataset vao `data/input`.
4. Cai dependencies.
5. Chay `combined_recommendation_system.ipynb`.
6. Kiem tra output trong `data/processed`, `models`, `reports`.
7. Copy output ve Drive.

