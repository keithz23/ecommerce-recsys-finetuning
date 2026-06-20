# Huong Dan Thay The Dataset Cho Product Recommendation Notebook

Tai lieu nay huong dan cach thay dataset dau vao cho cac model trong notebook:

```text
notebooks/product_recommendation_system.ipynb
notebooks/combined_recommendation_system.ipynb
```

Trong repo hien tai, nen uu tien dung notebook da gop va da dong nhat logic:

```text
notebooks/combined_recommendation_system.ipynb
```

## 1. Dataset Moi Nen Dat O Dau?

Dataset moi nen dat trong:

```text
data/input/interaction_train.csv
data/input/interaction_test.csv
```

Hai file nay la input chinh cho notebook.

Schema hien tai:

```text
UserId, ProductId, Rating, Timestamp, Date, VerifiedPurchase, HelpfulVote, SourceCategory, IsRelevant, Split
```

Cot bat buoc cho cac model trong notebook product:

```text
UserId
ProductId
Rating
```

Cot nen co neu muon dung day du popularity/rank-based:

```text
Timestamp
Date
VerifiedPurchase
IsRelevant
SourceCategory
```

## 2. Notebook Product Dang Can Schema Nao?

Phan lon code cu trong notebook dung ten cot:

```text
user_id
prod_id
rating
```

Vi dataset moi dung:

```text
UserId
ProductId
Rating
```

nen can rename:

```python
UserId -> user_id
ProductId -> prod_id
Rating -> rating
```

Trong `combined_recommendation_system.ipynb`, phan nay da duoc sua san.

## 3. Cach Load Dataset Moi Trong Notebook Product

Neu sua `product_recommendation_system.ipynb` goc, thay cell doc `../data/raw` bang doan sau:

```python
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data" / "input").exists() and (PROJECT_ROOT.parent / "data" / "input").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_INPUT_DIR = PROJECT_ROOT / "data" / "input"
INTERACTION_TRAIN_FILE = DATA_INPUT_DIR / "interaction_train.csv"
INTERACTION_TEST_FILE = DATA_INPUT_DIR / "interaction_test.csv"

main_usecols = ["UserId", "ProductId", "Rating"]

train_raw = pd.read_csv(INTERACTION_TRAIN_FILE, usecols=main_usecols, low_memory=False)
test_raw = pd.read_csv(INTERACTION_TEST_FILE, usecols=main_usecols, low_memory=False)

raw_data = pd.concat([train_raw, test_raw], ignore_index=True)

df = raw_data.rename(
    columns={
        "UserId": "user_id",
        "ProductId": "prod_id",
        "Rating": "rating",
    }
)

df = df[["user_id", "prod_id", "rating"]].copy()
df["user_id"] = df["user_id"].astype(str).str.strip()
df["prod_id"] = df["prod_id"].astype(str).str.strip()
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["user_id", "prod_id", "rating"])
df = df[(df["user_id"].str.len() > 0) & (df["prod_id"].str.len() > 0)].copy()

df.shape
```

## 4. Cac Model Trong Product Notebook Dung File Nao?

Tat ca model trong notebook product deu di qua DataFrame chuan:

```text
df[["user_id", "prod_id", "rating"]]
```

Bang map cu the:

| Model | Input can co | File goc sau khi thay dataset |
|---|---|---|
| EDA | `df` | `data/input/interaction_train.csv` + `data/input/interaction_test.csv` |
| Rank-Based / Popularity-C | product-level score tu train interactions | `data/input/interaction_train.csv`, co the sinh `data/interim/popularity_train_dataset.csv` |
| User-User CF | `df[["user_id", "prod_id", "rating"]]` | train + test gop lai thanh `df` |
| Item-Item CF | `df[["user_id", "prod_id", "rating"]]` | train + test gop lai thanh `df` |
| SVD | `df[["user_id", "prod_id", "rating"]]` | train + test gop lai thanh `df` |
| Hybrid | SVD + Rank-Based/Popularity-C | `df` + rank scores |
| Content-Based | user history + product metadata | `df` + optional `data/processed/product_metadata.csv` |

## 5. Rank-Based Da Duoc Thay Bang Popularity-C Nhu The Nao?

Trong notebook gop:

```text
notebooks/combined_recommendation_system.ipynb
```

model rank-based cu da duoc thay bang:

```python
PopularityRankRecommendationSystem
```

No van giu bien:

```python
model_rank
```

de cac cell sau khong bi vo:

```python
model_rank.evaluate(...)
model_rank.recommend(...)
HybridRecommendationSystem(..., model_rank=model_rank, ...)
```

Logic moi su dung cong thuc tu popularity notebook:

```text
Score_C = popularity_weight * PopularityDecayNorm + bayesian_weight * BayesianNorm
```

Trong do:

```text
PopularityDecayNorm = diem pho bien co time decay da normalize
BayesianNorm = Bayesian average da normalize
```

Output rank score duoc scale ve khoang rating-like `1-5` de dung duoc voi evaluation va hybrid.

## 6. File Trung Gian Cho Rank-Based / Popularity-C

Popularity-C co the dung file trung gian:

```text
data/interim/popularity_train_dataset.csv
```

Neu file nay chua co, notebook gop co the tu sinh tu:

```text
data/input/interaction_train.csv
```

Schema file trung gian:

```text
ProductId
ViewCount
CartCount
PurchaseCount
AvgRating
RatingCount
LastTimestamp
LastDate
```

Y nghia:

| Cot | Cach tinh tu dataset moi |
|---|---|
| `ProductId` | Ma san pham |
| `ViewCount` | So interaction/rating cua product trong train |
| `CartCount` | Mac dinh `0` neu dataset khong co cart signal |
| `PurchaseCount` | Tong `VerifiedPurchase` |
| `AvgRating` | Trung binh `Rating` |
| `RatingCount` | So rating |
| `LastTimestamp` | Timestamp moi nhat |
| `LastDate` | Date moi nhat |

## 7. Co Nen Gop Train + Test Khong?

Trong notebook product goc, code thuong dung:

```python
train_test_split(data, test_size=0.2, random_state=42)
```

Neu ban gop:

```text
interaction_train.csv + interaction_test.csv -> df
```

roi de notebook tu `train_test_split`, thi split goc trong cot `Split` se khong duoc giu lai.

Co 2 cach:

### Cach A: De notebook tu chia lai

Dung khi:

```text
- Muon giu code cu chay nhanh nhat.
- Khong bat buoc dung split co san.
```

Luc nay chi can gop train + test thanh `df`.

### Cach B: Giu split goc cua dataset

Dung khi:

```text
- Muon danh gia dung theo train/test ban dau.
- Muon so sanh metric nghiem tuc hon.
```

Luc nay can sua sau hon cac cell tao:

```python
trainset
testset
```

voi Surprise. Khong chi thay cell load dataset la xong.

## 8. Output Can Tao Lai Sau Khi Thay Dataset

Sau khi thay dataset, can regenerate:

```text
data/processed/processed_data.pkl
models/final_model_svd.pkl
reports/figures/*.png
data/interim/popularity_train_dataset.csv
reports/popularity/*.csv
reports/popularity/charts/*.png
```

Ly do:

| Output | Vi sao can tao lai |
|---|---|
| `processed_data.pkl` | Demo app uu tien dung file nay |
| `final_model_svd.pkl` | Model SVD cu hoc user/product cua dataset cu |
| `reports/figures/*.png` | Hinh EDA/metrics thay doi theo dataset moi |
| `popularity_train_dataset.csv` | Rank-Based/Popularity-C can product-level features moi |
| `reports/popularity/*.csv` | Metric popularity phu thuoc dataset moi |

## 9. Demo App Co Tu Dung Dataset Moi Khong?

Khong tu dong neu:

```text
data/processed/processed_data.pkl
```

van la file cu.

`demo_app.py` uu tien doc:

```text
data/processed/processed_data.pkl
```

neu file nay ton tai. Vi vay sau khi thay dataset, phai chay notebook va export lai:

```text
data/processed/processed_data.pkl
```

neu khong demo van co the hien ket qua cua dataset cu.

## 10. Checklist Thay Dataset

1. Dat file moi vao:

```text
data/input/interaction_train.csv
data/input/interaction_test.csv
```

2. Kiem tra header:

```powershell
Get-Content data\input\interaction_train.csv -TotalCount 2
Get-Content data\input\interaction_test.csv -TotalCount 2
```

3. Dam bao co cot bat buoc:

```text
UserId, ProductId, Rating
```

4. Chay notebook gop:

```text
notebooks/combined_recommendation_system.ipynb
```

5. Neu dung notebook goc `product_recommendation_system.ipynb`, thay cell load `../data/raw` bang loader trong muc 3.

6. Tao lai:

```text
data/processed/processed_data.pkl
models/final_model_svd.pkl
data/interim/popularity_train_dataset.csv
```

7. Chay lai evaluation cua cac model:

```text
Rank-Based / Popularity-C
User-User CF
Item-Item CF
SVD
Hybrid
Content-Based
```

8. Chay lai demo app neu can:

```powershell
python demo_app.py
```

## 11. Loi Thuong Gap

### Loi: `FileNotFoundError`

Kiem tra file co dung vi tri khong:

```text
data/input/interaction_train.csv
data/input/interaction_test.csv
```

### Loi: missing columns

Dataset moi phai co it nhat:

```text
UserId
ProductId
Rating
```

Neu ten cot khac, sua mapping rename trong cell load dataset.

### Notebook chay qua lau

File train hien tai rat lon. Nen:

```text
- Chi doc cot can thiet cho model chinh.
- Dung chunksize khi build popularity_train_dataset.csv.
- Neu test nhanh, co the sample truoc khi train full.
```

### Metric thay doi manh

Day la binh thuong vi:

```text
- Dataset moi co user/product khac.
- Rating distribution khac.
- Split train/test khac.
- Popularity score moi co them recency va VerifiedPurchase.
```

