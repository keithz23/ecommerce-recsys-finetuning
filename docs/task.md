# Huong dan giai thich model SVD trong du an

Tai lieu nay dung de viet phan bao cao/slide cho model SVD trong du an recommendation system. Noi dung tap trung vao: kien truc pipeline, nguyen ly hoat dong, rang buoc bai toan, input/output, ly do ket hop voi Collaborative Filtering va cach trinh bay ket qua that.

## 1. Kien truc tong the he thong

Pipeline tong the cua du an:

```text
Raw Data
-> Data Cleaning
-> Feature Engineering / Dataset Filtering
-> Model Training
-> Recommendation Engine
-> Evaluation
-> Output / Model File
```

| Bước                  | Input                   | Xử lý                                   | Output                 | Công cụ                          |
| --------------------- | ----------------------- | --------------------------------------- | ---------------------- | -------------------------------- |
| Raw Data              | `data/input/*.csv`      | Nhận dữ liệu gốc                        | File CSV gốc           | CSV                              |
| Data Cleaning         | interaction/product raw | chuẩn hóa cột, bỏ thiếu, ép kiểu rating | dataframe sạch         | Pandas                           |
| Dataset Filtering     | full interaction        | lọc theo từng model để tránh OOM        | `data/model_input/...` | `scripts/filter_model_inputs.py` |
| Model Training        | SVD train file          | train Surprise SVD                      | `model_svd`            | Surprise                         |
| Recommendation Engine | user id + model         | dự đoán sản phẩm chưa tương tác         | Top-N list             | Python/Surprise                  |
| Evaluation            | testset                 | tính RMSE, Recall@K...                  | metrics CSV            | custom eval                      |
| Output                | model + metrics         | lưu file                                | `.pkl`, `.csv`         | Pickle/CSV                       |

Trong du an nay, du lieu goc duoc dua vao thu muc:

```text
data/input/interaction_train.csv
data/input/interaction_test.csv
data/input/products.csv
```

Sau do chay script loc du lieu:

```bash
python scripts/filter_model_inputs.py
```

Script nay tao ra cac tap du lieu nho hon cho tung nhom model de tranh tran RAM tren Colab/Kaggle:

```text
data/model_input/cf/
data/model_input/svd/
data/model_input/content_based/
data/model_input/popularity/
```

Rieng model SVD su dung:

```text
Train: data/model_input/svd/interaction_train.csv
Test : data/model_input/svd/interaction_test.csv
```

Ly do khong dung truc tiep full dataset la vi file interaction goc co kich thuoc rat lon. Neu dua toan bo vao Surprise/SVD hoac GridSearchCV thi thoi gian train rat lau va de tran RAM.

### 1.1. Giai thich chi tiet kien truc he thong theo rubric 3.1

Kien truc giai phap de xuat co the trinh bay theo pipeline day du nhu sau:

```text
Raw CSV
-> Data Cleaning
-> Dataset Filtering
-> User-Item Matrix
-> Model Training
-> Model Files .pkl
-> Recommendation Engine
-> Streamlit UI / Notebook Demo
-> Top-N Output
```

Y nghia tung buoc:

| Buoc                         | Vai tro                                 | Input                                                                                            | Xu ly chinh                                                                                       | Output                                                                                                             | Cong cu                          |
| ---------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------- |
| Raw CSV                      | Luu du lieu goc cua bai toan            | `data/input/interaction_train.csv`, `data/input/interaction_test.csv`, `data/input/products.csv` | Doc du lieu interaction va product                                                                | Dataframe ban dau                                                                                                  | CSV, Pandas                      |
| Data Cleaning                | Lam sach va chuan hoa du lieu           | Raw CSV                                                                                          | Chuan hoa ten cot, ep kieu `UserId`, `ProductId`, `Rating`, loai dong loi/thieu neu co            | Du lieu sach de dua vao model                                                                                      | Pandas                           |
| Dataset Filtering            | Giam kich thuoc du lieu theo tung model | Full interaction/product data                                                                    | Loc user/product/interaction theo profile cua tung model de tranh tran RAM                        | `data/model_input/svd/`, `data/model_input/cf/`, `data/model_input/content_based/`, `data/model_input/popularity/` | `scripts/filter_model_inputs.py` |
| User-Item Matrix             | Tao bieu dien user-product-rating       | `UserId`, `ProductId`, `Rating`                                                                  | Dua du lieu ve dang ma tran rating thua, trong do hang la user, cot la product, gia tri la rating | Surprise Dataset / Trainset                                                                                        | Surprise                         |
| Model Training               | Train model recommendation              | `data/model_input/svd/interaction_train.csv`                                                     | SVD hoc latent factors cua user va product tu rating                                              | `model_svd` da train                                                                                               | Surprise SVD                     |
| Model Files .pkl             | Luu model da train                      | `model_svd.model`                                                                                | Serialize model de tai su dung khi demo/deploy                                                    | `models/final_model_svd.pkl`                                                                                       | Pickle                           |
| Recommendation Engine        | Sinh goi y ca nhan hoa                  | User ID, model da train, danh sach product user chua tuong tac                                   | Du doan predicted score cho tung product, sap xep giam dan                                        | Danh sach Top-N product                                                                                            | Python, Surprise                 |
| Streamlit UI / Notebook Demo | Giao dien nhap user va xem ket qua      | User nhap `UserId`                                                                               | Load model, goi recommendation engine, hien thi ket qua                                           | Bang recommendation                                                                                                | Streamlit hoac Jupyter Notebook  |
| Top-N Output                 | Ket qua cuoi cung cho user              | Predicted scores da sap xep                                                                      | Lay N san pham co score cao nhat                                                                  | `UserId`, `ProductId`, `PredictedScore`, `Rank`                                                                    | Pandas/CSV                       |

Luồng request khi co giao dien UI:

```text
User nhap UserId vao UI
-> Ung dung kiem tra UserId co ton tai trong interaction data hay khong
-> Ung dung load model SVD da train tu models/final_model_svd.pkl
-> Lay danh sach product user chua tuong tac
-> SVD predict score cho tung product
-> Sap xep predicted score giam dan
-> Lay Top-N product
-> Tra ve bang UserId | ProductId | PredictedScore | Rank
```

Neu chua kip lam Streamlit UI, co the minh hoa luong tren bang Jupyter Notebook. Khi do Notebook dong vai tro demo UI: nguoi dung chon hoac nhap `UserId`, cell goi ham recommendation, sau do in ra bang Top-N. Yeu cau quan trong la output phai duoc sinh tu model da train tren data that, khong dung mock data.

Vai tro cua cac component chinh:

| Component                | Vai tro trong he thong                                | Input                                | Output                                  |
| ------------------------ | ----------------------------------------------------- | ------------------------------------ | --------------------------------------- |
| `interaction_train.csv`  | Du lieu train cho SVD                                 | Lich su user-product-rating          | Surprise trainset                       |
| `interaction_test.csv`   | Du lieu evaluate                                      | Rating that cua user-product         | Metrics nhu RMSE, MAE, Recall@K         |
| `products.csv`           | Thong tin san pham cho Content-Based/bo sung hien thi | Product metadata                     | Ten/category/description cua product    |
| `filter_model_inputs.py` | Tao data nho theo tung model                          | Full dataset                         | `data/model_input/...`                  |
| `model_svd`              | Model SVD trong notebook                              | Surprise trainset                    | Predicted rating                        |
| `final_model_svd.pkl`    | Model da luu                                          | Model SVD da train                   | File co the load lai de demo            |
| Recommendation function  | Tao Top-N recommendation                              | User ID + model + product candidates | Bang Top-N                              |
| Metrics export           | Luu ket qua danh gia                                  | Ket qua evaluate cua model           | `reports/model_metrics/svd_metrics.csv` |

Cong cu tuong ung trong pipeline:

```text
Pandas     : doc CSV, lam sach data, xu ly dataframe, xuat metrics CSV
Surprise   : tao Dataset/Trainset, train SVD, predict rating
Pickle     : luu va load model `.pkl`
Streamlit  : tao UI nhap UserId va hien thi Top-N recommendation
Notebook   : train, evaluate, demo output that neu chua co UI
```

## 1.2. Khao sat toi thieu 03 mo hinh/thuat toan lien quan

Phan nay dung de dap ung yeu cau khao sat it nhat 03 model/thuat toan lien quan. Trong bai toan recommendation system, co the trinh bay 04 nhom model chinh: Popularity, Collaborative Filtering KNN, SVD va Content-Based. Trong do SVD la model trong tam cua phan hien thuc.

### 1. Popularity / Rank-Based Recommendation

Nguyen ly:

Popularity goi y san pham dua tren muc do pho bien cua san pham trong tap train. Diem pho bien co the duoc tinh tu so luong rating, diem rating trung binh, Bayesian average va time decay. Model nay khong ca nhan hoa theo tung user.

Input:

```text
data/model_input/popularity/interaction_train.csv
data/model_input/popularity/popularity_train_dataset.csv
```

Output:

```text
ProductId | PopularityScore | Rank
```

Uu diem:

- Don gian, de giai thich va chay nhanh.
- Phu hop voi cold-start user vi user moi chua co lich su van co the nhan goi y san pham pho bien.
- Lam baseline tot de so sanh voi cac model phuc tap hon.

Nhuoc diem:

- Khong ca nhan hoa theo so thich tung user.
- De uu tien san pham da pho bien san, lam giam kha nang kham pha san pham moi.
- Khong xu ly tot truong hop user co khau vi rieng.

Kha nang ap dung cho nhom:

Popularity duoc dung lam baseline va fallback khi user moi khong co lich su rating. Trong du an, no cung duoc dung de thay the rank-based model cu bang Popularity-C.

### 2. User-User / Item-Item Collaborative Filtering

Nguyen ly:

Collaborative Filtering KNN dua tren do tuong dong. User-User CF tim nhung user co hanh vi rating giong nhau, sau do goi y san pham ma user tuong dong da thich. Item-Item CF tim cac san pham co pattern rating tuong dong, sau do goi y san pham giong voi san pham user da tuong tac.

Input:

```text
data/model_input/cf/interaction_train.csv
data/model_input/cf/interaction_test.csv
```

Output:

```text
UserId | ProductId | PredictedScore | Rank
```

Uu diem:

- Co tinh ca nhan hoa tot hon Popularity.
- De giai thich: goi y dua tren user giong ban hoac item giong item ban da thich.
- Phu hop lam baseline CF de so sanh voi SVD.

Nhuoc diem:

- Rat ton RAM khi so user/product lon vi can tinh similarity matrix.
- Gap van de voi sparse data vi ma tran user-item co nhieu o trong.
- Khong xu ly tot cold-start user/item.

Kha nang ap dung cho nhom:

Model nay duoc dung de minh hoa Collaborative Filtering theo huong memory-based. Tuy nhien khi chay tren Colab/Kaggle can dung filtered dataset nho trong `data/model_input/cf/` de tranh tran RAM.

### 3. SVD / Matrix Factorization

Nguyen ly:

SVD la model-based Collaborative Filtering. Model factorize ma tran user-item thanh latent factors cua user va product. Sau khi train, SVD du doan rating ma user co the cho product chua tuong tac, sau do sap xep predicted score de tao Top-N recommendation.

Input:

```text
data/model_input/svd/interaction_train.csv
data/model_input/svd/interaction_test.csv
```

Output:

```text
UserId | ProductId | PredictedScore | Rank
```

Uu diem:

- Ca nhan hoa tot hon Popularity.
- Xu ly sparse matrix tot hon KNN CF vi hoc latent factors thay vi tinh similarity truc tiep tren toan bo ma tran.
- Phu hop voi bai toan rating prediction va Top-N recommendation.

Nhuoc diem:

- Van can lich su rating, nen khong xu ly tot cold-start user/item.
- Train va grid search co the lau neu dataset lon hoac param grid qua rong.
- Kho giai thich truc quan hon Popularity/KNN vi latent factors la dac trung an.

Kha nang ap dung cho nhom:

SVD la model chinh cua nhom vi can bang giua kha nang ca nhan hoa va kha nang chay duoc tren filtered dataset. SVD cung phu hop de export model `.pkl` va demo recommendation cho user cu the.

### 4. Content-Based Recommendation

Nguyen ly:

Content-Based goi y san pham dua tren noi dung cua product, vi du title, category, description, features. Model tao vector noi dung cho san pham, sau do tim cac san pham tuong tu voi san pham user da tuong tac.

Input:

```text
data/model_input/content_based/products.csv
data/model_input/content_based/interaction_train.csv
data/model_input/content_based/interaction_test.csv
```

Output:

```text
UserId | ProductId | SimilarityScore | Rank
```

Uu diem:

- Co the xu ly cold-start item neu product moi co noi dung mo ta.
- Khong phu thuoc hoan toan vao rating cua cong dong user.
- De ket hop voi CF/SVD trong hybrid system.

Nhuoc diem:

- Chat luong phu thuoc vao do day du va sach cua thong tin product.
- Co the goi y cac san pham qua giong nhau, it tao su da dang.
- Khong hoc duoc hanh vi cong dong manh nhu CF/SVD.

Kha nang ap dung cho nhom:

Content-Based duoc dung lam model bo tro cho SVD, dac biet trong truong hop product moi chua co nhieu rating. Trong hybrid system, Content-Based giup giam han che cold-start item.

### Ket luan khao sat

Bang tom tat:

```text
Model              Nguyen ly chinh                  Vai tro trong du an
Popularity         Goi y theo do pho bien             Baseline, fallback user moi
User/Item CF       Goi y theo similarity              Baseline CF, de so sanh voi SVD
SVD                Matrix factorization               Model chinh cho personalized recommendation
Content-Based      Goi y theo noi dung product        Fallback item moi, bo tro hybrid
```

Qua khao sat, SVD duoc chon lam model trong tam vi no van thuoc Collaborative Filtering nhung co kha nang tong quat hoa tot hon KNN CF tren du lieu thua. Tuy nhien SVD khong thay the hoan toan cac model khac, nen he thong van can Popularity va Content-Based de xu ly cold-start va tang do on dinh.

## 2. Nguyen tac thiet ke pipeline SVD

Model SVD duoc thiet ke dua tren cac nguyen tac sau:

1. Chi dung explicit feedback

   SVD trong du an nay hoc tu cot `Rating`, tuc la diem danh gia cua user cho product. Model khong dung text mo ta san pham hay anh san pham.

2. Tach train/test ro rang

   File `interaction_train.csv` dung de train model. File `interaction_test.csv` dung de danh gia lai kha nang du doan/recommend cua model tren du lieu chua hoc.

3. Loc du lieu truoc khi train

   SVD co the xu ly nhieu dong hon User-User CF va Item-Item CF, nhung full dataset van qua lon cho Colab/Kaggle. Vi vay can dung `data/model_input/svd/` thay vi full `data/input/`.

4. Dung Surprise de train model

   Thu vien Surprise ho tro SVD/SVD++ cho bai toan collaborative filtering dua tren rating. Du lieu duoc dua ve dang:

   ```text
   UserId, ProductId, Rating
   ```

5. Output phai la ket qua that

   Model phai train tren data that va sinh Top-N recommendation that, khong dung mock data. Ket qua can co user ID, product ID, predicted score va rank.

## 3. SVD hoat dong nhu the nao trong du an

SVD la mot mo hinh Matrix Factorization. Y tuong chinh la bien ma tran User-Item rat lon thanh hai ma tran an nho hon:

```text
User-Item Rating Matrix ~= User Latent Matrix x Item Latent Matrix
```

Neu co:

```text
User A danh gia Product X = 5 sao
User A danh gia Product Y = 4 sao
User B danh gia Product X = 5 sao
```

SVD se hoc cac dac trung an, vi du:

```text
User A co xu huong thich laptop gaming
Product X co dac trung gaming, gia cao, rating tot
Product Y co dac trung van phong, gia vua
```

Nhung cac dac trung nay khong phai cot co san trong dataset. Model tu hoc chung tu pattern rating cua nhieu user va nhieu product.

Cong thuc du doan rating co the mo ta don gian:

```text
predicted_rating(user, item)
= global_mean
+ user_bias
+ item_bias
+ user_latent_vector dot item_latent_vector
```

Trong do:

- `global_mean`: diem rating trung binh toan bo dataset.
- `user_bias`: xu huong user cho diem cao/thap hon trung binh.
- `item_bias`: xu huong product duoc cham diem cao/thap hon trung binh.
- `user_latent_vector`: vector dac trung an cua user.
- `item_latent_vector`: vector dac trung an cua product.
- Tich vo huong giua hai vector cho biet user co phu hop voi product hay khong.

Sau khi train, voi moi user, model du doan diem cho cac product user chua tuong tac. Cac product co predicted score cao nhat se duoc tra ve thanh Top-N recommendation.

## 4. Tai sao SVD van la Collaborative Filtering

SVD khong phai model tach khoi Collaborative Filtering. SVD la mot dang Collaborative Filtering theo huong model-based.

Co hai nhom CF chinh:

```text
Memory-based CF:
- User-User CF
- Item-Item CF
- Dua tren similarity truc tiep giua user/item

Model-based CF:
- SVD
- SVD++
- Hoc latent factors tu ma tran user-item
```

Vi vay trong du an, SVD duoc dat chung voi CF la dung, vi:

1. SVD chi dung lich su tuong tac/rating cua user-item.
2. SVD khong can noi dung san pham nhu title, category, description.
3. Muc tieu cua SVD van la du doan user nao se thich item nao dua tren hanh vi cua cong dong user.
4. SVD giai quyet cung bai toan voi User-User CF va Item-Item CF, nhung bang cach factorize ma tran thay vi tinh similarity truc tiep.

Noi cach khac:

```text
SVD khong chay rieng ngoai CF.
SVD la mot phuong phap Collaborative Filtering nang cao.
```

## 5. Tai sao khong chi dung SVD mot minh

Trong he thong recommendation thuc te, khong nen chi dung SVD mot minh vi SVD co cac han che:

1. Cold-start user

   Neu user moi chua co lich su rating, SVD khong co vector user de du doan chinh xac. Khi do can fallback sang Popularity.

2. Cold-start item

   Neu product moi chua co rating, SVD khong co vector item. Khi do Content-Based co the dung thong tin product nhu title, category, description de goi y.

3. Sparse data

   Ma tran user-item thuong rat rong, vi moi user chi tuong tac voi mot phan rat nho san pham. SVD xu ly sparsity tot hon User-User CF, nhung van can du rating de hoc vector on dinh.

4. Can so sanh voi baseline

   Popularity, User-User CF, Item-Item CF va Content-Based la cac baseline quan trong. Neu SVD tot hon baseline thi moi co co so chung minh mo hinh co gia tri.

5. Hybrid giup he thong on dinh hon

   Hybrid co the ket hop diem tu SVD voi Popularity/Content-Based de giam rui ro khi SVD gap user/item it du lieu.

Vi vay trong du an, SVD la model chinh cho personalized recommendation, nhung van can dat trong pipeline co CF, Popularity va Content-Based de xu ly day du cac tinh huong.

## 6. Input va output cua model SVD

### Input train

```text
data/model_input/svd/interaction_train.csv
```

Cac cot quan trong:

```text
UserId
ProductId
Rating
```

Y nghia:

- `UserId`: ID nguoi dung.
- `ProductId`: ID san pham.
- `Rating`: diem danh gia cua user cho product.

Trong notebook, du lieu nay duoc chuyen ve dang Surprise:

```python
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user_id", "prod_id", "rating"]], reader)
trainset = data.build_full_trainset()
```

### Input test/evaluate

```text
data/model_input/svd/interaction_test.csv
```

File test dung de danh gia model sau khi train. Ket qua co the gom:

```text
RMSE
MAE
Precision@K
Recall@K
F1@K
HitRate@K
MAP@K
MRR@K
NDCG@K
```

### Output recommendation

Output Top-N nen co dang:

```text
UserId | ProductId | PredictedScore | Rank
```

Vi du:

```text
UserId     ProductId     PredictedScore     Rank
123        B00001        4.82               1
123        B00002        4.76               2
123        B00003        4.61               3
```

Trong notebook, output metrics duoc ghi ra:

```text
reports/model_metrics/svd_metrics.csv
reports/model_metrics/all_model_metrics.csv
```

Model SVD sau khi train co the export ra:

```text
models/final_model_svd.pkl
```

File `.pkl` nay dung de load lai model khi can deploy hoac demo recommendation ma khong phai train lai tu dau.

## 7. Quy trinh train SVD trong notebook

Quy trinh trong `notebooks/combined_recommendation_system.ipynb`:

```text
1. Doc data/model_input/svd/interaction_train.csv
2. Doc data/model_input/svd/interaction_test.csv
3. Rename cot ve user_id, prod_id, rating
4. Tao Surprise Dataset
5. Chon SVD/SVD++ va tham so
6. Train model_svd
7. Evaluate bang testset
8. Sinh recommendation Top-N
9. Export metrics va model file
```

Pipeline co the trinh bay tren slide:

```text
Filtered SVD Interactions
-> Surprise Dataset
-> Trainset/Testset
-> SVD Training
-> Predicted Rating
-> Top-N Recommendation
-> Metrics CSV + final_model_svd.pkl
```

## 8. Giai thich GridSearch va tai sao can giam tham so

Doan grid search ban dau:

```python
base_svd_param_grid = {
    "n_factors": [20, 40, 60],
    "n_epochs": [10, 20, 30],
    "lr_all": [0.003, 0.005, 0.007],
    "reg_all": [0.02, 0.03, 0.04],
    "random_state": [42],
}
```

So cau hinh:

```text
3 * 3 * 3 * 3 * 1 = 81 cau hinh / model
```

Neu chay ca SVD va SVD++:

```text
81 * 2 = 162 cau hinh
```

Neu cross-validation 3 fold:

```text
162 * 3 = 486 lan train
```

SVD++ nang hon SVD vi no dung them implicit feedback. Do do tren Colab/Kaggle, grid nay co the chay nhieu gio hoac hon 1 ngay.

De demo va bao cao, nen dung grid gon hon:

```python
base_svd_param_grid = {
    "n_factors": [20, 40],
    "n_epochs": [10],
    "lr_all": [0.005],
    "reg_all": [0.02],
    "random_state": [42],
}

svd_param_grids = {
    "SVD": base_svd_param_grid,
}

svd_algorithms = {
    "SVD": SVD,
}
```

Nguyen tac la:

- Chay nhanh truoc de dam bao pipeline dung.
- Chi mo rong grid khi co du thoi gian va tai nguyen.
- Uu tien SVD truoc SVD++ vi SVD nhe hon va phu hop hon cho demo.
- Neu can SVD++, chi train 1 cau hinh co dinh sau khi da xac nhan pipeline.

## 9. Rang buoc va luat cua bai toan

1. Cold-start user

   User moi khong co lich su rating thi SVD/CF khong hoat dong tot. Giai phap fallback: Popularity.

2. Cold-start item

   Product moi khong co rating thi SVD/CF khong hoc duoc latent vector. Giai phap fallback: Content-Based.

3. Sparse matrix

   User-item matrix rat thua vi moi user chi rating mot so it product. SVD duoc chon vi no giam chieu ma tran va hoc latent factors tu du lieu thua.

4. Rating scale

   Rating trong dataset nam trong khoang 1-5. Khi dua vao Surprise can khai bao dung:

   ```python
   Reader(rating_scale=(1, 5))
   ```

5. Chi co explicit feedback

   Du an chu yeu dung rating. Cac hanh vi nhu click/view khong phai tin hieu chinh, nen do phu du lieu bi gioi han.

6. Gioi han RAM

   User-User CF va Item-Item CF can ma tran similarity, rat ton RAM. SVD nhe hon KNN CF, nhung grid search tren data lon van rat nang. Vi vay can dung filtered dataset.

## 10. Cach trinh bay output that

Khi bao cao, can chup man hinh output that tu notebook, khong dung anh mock.

Output nen co:

```text
UserId
ProductId
PredictedScore
Rank
```

Neu demo tren Streamlit chua kip, co the dung output truc tiep tu Jupyter Notebook. Dieu quan trong la:

- Model da train xong.
- User ID la user co that trong dataset.
- Product ID la product co that trong dataset.
- Score la diem du doan tu model.
- Rank la thu tu sap xep theo predicted score giam dan.

## 11. Noi dung co the dua vao slide

### Slide: Kien truc pipeline SVD

```text
data/model_input/svd/interaction_train.csv
-> Data preprocessing
-> Surprise Dataset
-> SVD training
-> Predict rating for unseen products
-> Top-N recommendation
-> Evaluate metrics
-> Export svd_metrics.csv + final_model_svd.pkl
```

### Slide: SVD la gi?

SVD la model-based Collaborative Filtering. Model factorize ma tran user-item thanh cac latent factors cua user va product. Sau khi hoc, model du doan rating cho nhung product user chua tuong tac, sau do sap xep diem du doan de tao Top-N recommendation.

### Slide: Vi sao ket hop voi CF/Popularity/Content-Based?

SVD thuoc Collaborative Filtering vi no hoc tu lich su rating user-item. Tuy nhien SVD khong xu ly tot user moi va product moi. Do do he thong can Popularity de fallback cho cold-start user, Content-Based de fallback cho cold-start item, va cac model CF baseline de so sanh hieu qua.

### Slide: Input/Output

```text
Input train:
data/model_input/svd/interaction_train.csv

Input test:
data/model_input/svd/interaction_test.csv

Required columns:
UserId, ProductId, Rating

Output:
UserId, ProductId, PredictedScore, Rank

Metrics:
reports/model_metrics/svd_metrics.csv

Model file:
models/final_model_svd.pkl
```

## 12. Tom tat ngan gon

SVD trong du an nay la mot model Collaborative Filtering dua tren rating. Model hoc latent factors cua user va product tu du lieu interaction da loc, sau do du doan diem cho cac product user chua mua/danh gia va tra ve Top-N recommendation. SVD khong nen dung mot minh vi gap cold-start user/item, nen can dat trong pipeline co Popularity, Content-Based va cac CF baseline. Input chinh cua SVD la `data/model_input/svd/interaction_train.csv`, output la recommendation Top-N, metrics CSV va file model `.pkl`.
