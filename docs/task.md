# Huong dan giai thich pipeline, Item-Item CF va macro-architecture

Tai lieu nay dung de viet phan bao cao/slide cho recommendation system. Noi dung tap trung vao: Transparent Pipeline - Data Transformation, 5 rang buoc du lieu, Item-Item CF trong nhom Collaborative Filtering, SVD lam model-based CF doi chieu, System Macro-Architecture & Tech Stack, input/output va cach trinh bay ket qua that.

## 0. Phan thuyet trinh can nam: Pipeline, Item-Item CF, rang buoc du lieu, Macro-Architecture

Theo file slide/PDF, phan can trinh bay co 3 cum chinh:

```text
1. The Transparent Pipeline - Data Transformation
2. Cac Models khao sat - tap trung vao Item-Item CF trong nhom Collaborative Filtering, dat canh voi User-User CF va SVD
3. System Macro-Architecture & Tech Stack
```

Neu bi hoi "phan cua em lam gi?", co the tra loi ngan:

```text
Em phu trach giai thich duong di cua du lieu tu raw Amazon reviews den model-ready data, dac biet buoc filter_model_inputs.py va cac rang buoc du lieu. Sau do em trinh bay Item-Item CF trong nhom Collaborative Filtering, so sanh voi User-User CF va SVD. Cuoi cung em noi cach cac component ghep thanh he thong: data pipeline, model training, evaluation, recommendation engine va UI demo.
```

### 0.1. The Transparent Pipeline - Data Transformation

Pipeline du lieu nen duoc giai thich theo logic sau:

```text
Raw Amazon JSONL reviews + metadata
-> build_master_dataset.py
-> 01_build_interaction.py
-> 02_build_products.py
-> 04_split_train_test.py
-> 05_build_popularity_train.py
-> filter_model_inputs.py
-> data/model_input/{cf, svd, content_based, popularity}/
```

Y nghia tung buoc:

| Buoc | Script / Xu ly | Tai sao can lam |
| ---- | -------------- | --------------- |
| 0 | `build_master_dataset.py` scan raw review + metadata, giu data gan nhat, merge review voi product metadata, xu ly `Price` thieu bang median | Tao mot bang tong hop day du ca interaction va thong tin san pham; tranh moi model phai doc raw JSONL lon nhieu lan |
| 1 | `01_build_interaction.py` dedup theo `(UserId, ProductId, Timestamp)`, chuan hoa `VerifiedPurchase` | Mot user co the co duplicate review; dedup giup rating khong bi dem lap va popularity khong bi thoi phong |
| 2 | `02_build_products.py` tao `CombinedText = Title + Category + Description + Features + Store` | Content-Based/TF-IDF can mot cot text tong hop de tinh cosine similarity giua san pham |
| 3 | `04_split_train_test.py` temporal split, tao `IsRelevant = Rating >= 4`, bo gap rows neu co | Split theo thoi gian giong thuc te: train tren qua khu, test tren tuong lai; `IsRelevant` dung cho HitRate/MAP/MRR |
| 4 | `05_build_popularity_train.py` build popularity chi tu train data | Tranh data leakage: khong duoc dung test de tinh item nao pho bien |
| 5 | `filter_model_inputs.py` tao profile rieng cho CF/SVD/Popularity/Content-Based | Moi model co gioi han RAM khac nhau; CF can tap nho hon vi similarity matrix tang theo O(n^2), SVD chiu duoc lon hon |

Doi chieu voi repo hien tai:

- `scripts/filter_model_inputs.py` dang tao 4 profile that: `cf`, `svd`, `content_based`, `popularity`.
- Profile `cf` gioi han 5,000 users va 5,000 products vi User-User CF va Item-Item CF can similarity matrix.
- Profile `svd` va `popularity` gioi han 100,000 users/products; profile `content_based` gioi han 30,000 users va 50,000 products.
- Output da co trong `data/model_input/filter_summary.json`: CF 56,106 train rows / 3,231 test rows; SVD 1,000,000 train rows / 71,483 test rows; Content-Based 500,000 train rows / 29,549 test rows; Popularity 1,000,000 train rows / 71,483 test rows.

5 rang buoc du lieu trong PDF va cach giai thich:

| Rang buoc | Giai phap trong pipeline | Dua vao dau de noi |
| --------- | ------------------------ | ------------------ |
| Cold-start user | Fallback sang Popularity model | User moi chua co rating nen CF/SVD khong co vector/user history; popularity chi can thong ke product tu train |
| Cold-start item | Fallback sang Content-Based neu co metadata | Item moi chua co rating nen CF/SVD khong co item vector; Content-Based dung text/category/brand de tinh similarity |
| Sparsity ~99%+ | Dung SVD latent factors va filter profile rieng cho KNN CF | User-item matrix rat thua; KNN similarity kem on dinh va ton RAM, SVD tong quat hoa tot hon |
| Rating bias ve 4-5 | Dung `IsRelevant >= 4`, Bayesian average/Popularity C va doc metric can than | Review dataset thien ve rating cao, nen ranking metric co the dep neu model day item pho bien len top |
| Data leakage | Popularity va model chi hoc tu train; test chi evaluate | PDF tach `popularity_train_dataset.csv` voi `popularity_dataset.csv`; repo co ca `data/input/*` va `data/model_input/popularity/popularity_train_dataset.csv` |

Luu y dong bo slide/code: PDF ghi proxy `Cart = Rating >= 4` va `Purchase = Rating == 5 OR VerifiedPurchase == 1`, nhung `scripts/filter_model_inputs.py` hien tai dang build `CartCount = 0` va `PurchaseCount = VerifiedPurchase`. Khi bao cao, nen noi day la cong thuc slide/huong thiet ke; neu lay metric chinh thuc theo cong thuc do thi can cap nhat code build popularity truoc.

Ly do lam pipeline theo nhieu buoc:

- De minh bach: moi buoc co input/output ro, hoi den dau giai thich den do.
- De tranh leakage: test set chi dung de evaluate, khong dung tinh popularity, Bayesian average, TF-IDF profile hay train model.
- De tranh crash RAM: User-User CF/Item-Item CF can tinh similarity matrix, nen phai loc user/item; SVD dung latent factors nen chay duoc tren tap lon hon.
- De dung voi bai toan thuc te: he thong recommendation khong duoc hoc tu du lieu tuong lai.

### 0.2. Cac models khao sat - phan can nhan manh Item-Item CF

Trong slide/PDF co nhac nhieu model: Popularity, Content-Based, User-User CF, Item-Item CF va SVD. Neu phan cua ban la Item-Item CF thi nen dat no trong boi canh nay:

```text
Em tap trung vao Item-Item Collaborative Filtering:
- Item-Item CF: memory-based CF, tim san pham co co-rating pattern giong nhau.
- User-User CF: baseline doi chieu, tim user co hanh vi rating giong nhau.
- SVD: model-based CF, hoc latent factors va xu ly sparse matrix tot hon KNN CF.
```

Bang so sanh ngan cho phan cua ban:

| Model | Nguyen ly | Vi sao khao sat | Han che |
| ----- | --------- | --------------- | ------- |
| User-User CF | Tao vector rating cho moi user, tinh cosine similarity giua user, lay Top-K user giong nhat va aggregate rating cua ho | Day la baseline Collaborative Filtering de giai thich truc quan: "nguoi dung giong ban cung thich san pham nay" | Cold-start user kem, du lieu sparse lam similarity kem on dinh, memory/time tang nhanh khi so user lon |
| Item-Item CF | Tao vector rating cho moi product, tinh similarity giua product dua tren cac user cung rating, goi y item giong nhung item user da thich | Day la baseline CF gan voi ecommerce: "san pham giong san pham ban da thich" | Can similarity matrix theo product nen van ton RAM O(n_items^2), cold-start item kem, de nghieng ve item pho bien |
| SVD | Factorize user-item matrix thanh latent factors cua user va item, du doan `r_hat(u,i) = mu + b_u + b_i + q_i^T p_u` | Day la model chinh vi xu ly sparse matrix tot hon KNN CF va cho RMSE tot hon trong thuc nghiem | Kho giai thich truc quan hon, van can lich su rating, khong tu xu ly duoc user/item moi |

Tai sao can lam ca User-User CF, Item-Item CF va SVD:

- User-User CF la baseline de chung minh cach Collaborative Filtering co ban hoat dong.
- Item-Item CF la bien the CF phu hop voi ecommerce vi san pham thuong on dinh hon user base.
- SVD la ban nang cao cua Collaborative Filtering, khong tinh similarity truc tiep tren toan bo ma tran ma hoc latent factors.
- Neu SVD tot hon User-User/Item-Item CF ve RMSE/HitRate/MRR thi co co so noi rang factorization phu hop hon voi du lieu sparse.
- Khi bi hoi "SVD co phai CF khong?", tra loi: co, SVD la model-based Collaborative Filtering vi no hoc tu rating user-item, khong dung noi dung san pham.

### 0.3. System Macro-Architecture & Tech Stack

Macro-architecture dung voi repo hien tai nen trinh bay theo cac layer:

```text
Data Source
-> Data Preparation / Filtering
-> Model Input Store
-> Model Training & Evaluation
-> Model Artifact / Metrics
-> Recommendation Engine
-> Local UI Demo
```

Ghep voi file/code that trong repo:

| Layer | File / Thanh phan trong repo | Vai tro |
| ----- | ---------------------------- | ------- |
| Data Source | `data/input/interaction_train.csv`, `data/input/interaction_test.csv`, `data/input/products.csv` | Du lieu interaction da split train/test va product catalog |
| Data Preparation / Filtering | `scripts/filter_model_inputs.py` | Tao cac tap nho rieng cho `cf`, `svd`, `content_based`, `popularity` de tranh tran RAM |
| Model Input Store | `data/model_input/...` | Luu train/test da loc theo tung model profile |
| Model Training & Evaluation | `notebooks/combined_recommendation_system.ipynb`, `src/cf_recommender.py`, `src/rank_recommender.py`, `src/content_based_recommendation.py`, `src/hybrid_recommender.py`, `src/model_eval_functions.py` | Train/evaluate Rank-Based, User-User CF, Item-Item CF, SVD, Content-Based va Hybrid |
| Model Artifact / Metrics | `models/final_model_svd.pkl`, `reports/model_metrics/all_model_metrics.csv` neu da export | Luu model da train va bang metric de demo/bao cao |
| Recommendation Engine | `demo_app.py` + cac ham trong `src/*` | Sinh Top-N recommendation cho user theo Content-Based, Rank-Based, SVD hoac Hybrid |
| Local UI Demo | `demo_app.py` | Web demo local o `localhost:8000`, chon user/model/top-N va hien thi ket qua |

Bang tech stack dung de giai thich khi thuyet trinh:

| Layer | Cong cu | Vi sao dung |
| ----- | ------- | ----------- |
| Data Preparation | Python, Pandas, chunking/filtering | Python/Pandas de xu ly CSV linh hoat; filtering theo profile giup moi model chay duoc trong gioi han RAM |
| Popularity | Pandas custom scoring | Popularity la aggregation theo product nen Pandas du nhanh, de debug va de giai thich |
| Content-Based | Scikit-learn `TfidfVectorizer`, `cosine_similarity` | TF-IDF phu hop voi text metadata nhu title/category/description; cosine similarity de tim san pham noi dung tuong tu |
| User-User CF | Surprise `KNNBasic` / cosine similarity | Surprise co san API cho rating-based CF, de train/evaluate voi RMSE va Top-N |
| Item-Item CF | Surprise `KNNBasic` voi `sim_options.user_based=False` | Dung cung user-item rating matrix, nhung similarity tinh theo item thay vi theo user |
| SVD | Surprise `SVD` | Surprise SVD co cong thuc bias + latent factors, phu hop bai toan explicit rating 1-5 |
| Evaluation | Custom Python metrics | Can tinh ca rating metrics (`RMSE`, `MAE`) va ranking metrics (`HitRate@K`, `MAP@K`, `MRR@K`) |
| UI Demo | `demo_app.py` local HTTP server bang `BaseHTTPRequestHandler`, hoac notebook demo | Repo hien tai khong dung Streamlit; demo chay tai `http://localhost:8000` |

Ly do kien truc nay hop ly:

- Tach pipeline voi model: data xu ly mot lan, nhieu model dung lai duoc.
- Tach train voi evaluate: dam bao metric cong bang va tranh leakage.
- Tach recommendation engine voi UI: UI chi goi ham recommend, khong chua logic train phuc tap.
- Dung model phu hop tung tinh huong: Popularity cho user moi, Content-Based cho item moi co metadata, User-User/Item-Item CF va SVD cho personalized recommendation.

Luu y khi doi chieu voi slide PDF: slide co nhac `SQLite/WAL mode` va cac script ETL nhu `build_master_dataset.py`, `01_build_interaction.py`, `02_build_products.py`, `04_split_train_test.py`, `05_build_popularity_train.py`. Cac thanh phan do hop ly o muc pipeline tong quat/upstream, nhung trong repo hien tai khong thay cac file script do. Khi thuyet trinh theo repo nay, nen noi phan implemented ro nhat la `data/input`, `scripts/filter_model_inputs.py`, `notebooks/combined_recommendation_system.ipynb`, `src/*`, `models/final_model_svd.pkl` va `demo_app.py`.

## 0.4. Doi chieu rubric bat buoc trong anh

Bang nay dung de kiem tra nhanh cac task trong rubric da co noi dung va bang chung trong du an hay chua.

| Muc rubric | Trang thai trong `task.md` | Noi dung can trinh bay | Bang chung trong repo |
| ---------- | -------------------------- | ---------------------- | --------------------- |
| 1.2 - Khao sat toi thieu 03 mo hinh/thuat toan lien quan | Da co, can nhan manh 4 nhom model | Popularity, User-User/Item-Item CF, SVD, Content-Based. Moi model can co nguyen ly, uu/nhuoc diem, kha nang ap dung cho nhom | `src/rank_recommender.py`, `src/cf_recommender.py`, `src/content_based_recommendation.py`, `reports/model_metrics/all_model_metrics.csv` |
| 1.3 - Trinh bay nguyen ly hoat dong hoac kien truc cua model da khao sat | Da co rai rac, nen noi theo pipeline tong the | Raw data -> cleaning/filtering -> user-item matrix -> training -> recommendation engine -> demo UI. Mo ta input/output cua tung buoc va tech stack | `scripts/filter_model_inputs.py`, `notebooks/combined_recommendation_system.ipynb`, `demo_app.py` |
| 2.4 - Trinh bay rang buoc, luat, thuoc tinh tinh toan can thiet cua bai toan | Da co | Cold-start user, cold-start item, sparsity, rating bias, explicit feedback, RAM/OOM, data leakage, metadata constraint | `data/model_input/filter_summary.json`, `scripts/filter_model_inputs.py`, `reports/model_metrics/all_model_metrics.csv` |
| 3.1 - Trinh bay kien truc/quy trinh hoat dong cua giai phap de xuat | Da co, can gan voi demo that | Data source -> model input store -> training/evaluation -> model artifact/metrics -> recommendation engine -> local UI demo | `models/final_model_svd.pkl`, `demo_app.py`, `reports/model_metrics/all_model_metrics.csv` |
| 3.3 - Mo hinh/thuat toan co the thuc thi va tao ket qua dau ra | Da co, can chup output that | It nhat 1 model tao Top-N output that. Output can co `User ID`, `Product ID`, `Predicted Score`, `Rank`. Khong dung mock data | `demo_app.py`, cell cuoi `notebooks/combined_recommendation_system.ipynb`, `models/final_model_svd.pkl` |

### 1.2 - Khao sat toi thieu 03 mo hinh/thuat toan lien quan

De dat diem muc 1.2, khong chi liet ke ten model ma can noi du 4 y: model, nguyen ly, uu/nhuoc diem, ap dung cho nhom.

| Model | Nguyen ly | Uu diem | Nhuoc diem | Ap dung cho nhom |
| ----- | --------- | ------- | ---------- | ---------------- |
| Popularity / Rank-Based | Xep hang product bang thong ke tren train data nhu rating count, average rating, Bayesian average, popularity score | Don gian, nhanh, de giai thich, dung tot cho user moi | Khong ca nhan hoa, de day san pham pho bien len top | Baseline va fallback cho cold-start user |
| User-User CF | Tim user co vector rating tuong tu, lay item user tuong tu da thich de goi y | Truc quan, co ca nhan hoa, khong can metadata | Ton RAM khi nhieu user, kem voi sparse data, cold-start user kem | Baseline Collaborative Filtering de so sanh voi SVD |
| Item-Item CF | Tim product co co-rating pattern tuong tu, goi y item giong item user da thich | Phu hop ecommerce/product-to-product, on dinh hon User-User khi user thay doi nhanh | Ton RAM theo so item, cold-start item kem, de nghieng ve item pho bien | Phan can nhan manh trong model survey; dung `sim_options.user_based=False` |
| SVD / Matrix Factorization | Hoc latent factors cua user va item tu user-item rating matrix, du doan rating cho item chua tuong tac | Xu ly sparse matrix tot hon KNN CF, RMSE tot, phu hop Top-N personalized recommendation | Kho giai thich truc quan, van can lich su rating, khong tu xu ly cold-start | Model chinh cho recommendation ca nhan hoa va export `.pkl` |
| Content-Based | Dung metadata product nhu title/category/description/brand de tinh similarity | Giai thich duoc, ho tro cold-start item neu co metadata | Phu thuoc chat luong metadata, co the goi y qua giong nhau | Fallback/bo tro cho product moi va hybrid |

### 1.3 - Nguyen ly hoat dong/kien truc cua cac model da khao sat

Co the trinh bay theo hai lop: kien truc he thong va nguyen ly model.

Kien truc he thong:

```text
Raw Amazon reviews + product metadata
-> Data cleaning / dedup / temporal split
-> filter_model_inputs.py tao profile theo tung model
-> User-item matrix / Surprise Dataset
-> Train Popularity, User-User CF, Item-Item CF, SVD, Content-Based, Hybrid
-> Evaluate bang RMSE, MAE, Precision@K, Recall@K, HitRate@K, MAP@K, MRR@K
-> Export metrics/model
-> demo_app.py tao Top-N recommendation that
```

Nguyen ly model can noi ngan:

- Popularity hoc thong ke product tu train data, khong dung lich su rieng cua user.
- User-User CF va Item-Item CF la memory-based CF, tinh similarity truc tiep tren user-item matrix.
- SVD la model-based CF, hoc latent factors thay vi tinh similarity truc tiep tren toan bo ma tran.
- Content-Based dung metadata product, khong phu thuoc hoan toan vao rating cua cong dong.
- Hybrid ket hop diem SVD voi Rank-Based de can bang personalization va do on dinh.

Tech stack dung voi repo hien tai:

```text
Python + Pandas          : doc CSV, cleaning, aggregation, filtering
SQLite/chunking upstream : xu ly raw data lon trong slide/PDF
Scikit-learn             : TF-IDF / similarity cho Content-Based
Surprise                 : KNNBasic cho User-User/Item-Item CF, SVD cho Matrix Factorization
Custom metrics           : HitRate, MRR, MAP, RMSE...
Pickle                   : luu/load final_model_svd.pkl
demo_app.py              : local HTTP UI, khong phai Streamlit trong repo hien tai
```

### 2.4 - Rang buoc, luat va thuoc tinh tinh toan can thiet

Muc nay nen trinh bay theo dang "rang buoc -> vi sao co -> cach giai quyet":

| Rang buoc | Vi sao co trong bai toan | Cach giai quyet trong du an |
| --------- | ------------------------ | --------------------------- |
| Cold-start user | User moi chua co rating nen CF/SVD khong co vector/user history | Fallback sang Popularity/Rank-Based |
| Cold-start item | Product moi chua co rating nen CF/SVD khong hoc duoc item vector | Fallback sang Content-Based neu item co metadata |
| Sparsity | User-item matrix rat thua, PDF noi sparse ~99%+ | Loc data theo profile; dung SVD latent factors de tong quat hoa tot hon KNN |
| Rating bias | Review thuong thien ve 4-5 sao, model de uu tien item pho bien | Dung `IsRelevant >= 4`, Bayesian average, va giai thich metric khong nen doc rieng le |
| Explicit feedback only | Dataset chu yeu la rating/review, khong co log click/view/cart/purchase that | Cac signal view/cart/purchase chi la proxy; can noi ro khi bao cao |
| Gioi han RAM/OOM | KNN CF can similarity matrix O(n_users^2) hoac O(n_items^2) | `filter_model_inputs.py` tao profile `cf` nho hon SVD/Popularity |
| Data leakage | Neu tinh popularity/rating avg tren ca test thi model thay tuong lai | Popularity/model/statistics chi hoc tu train; test chi evaluate |
| Metadata constraint | Content-Based chi tot khi co title/category/description/brand | Neu item khong co metadata thi fallback ve Popularity/cho den khi co interaction |

### 3.1 - Kien truc/quy trinh hoat dong cua giai phap de xuat

So do de dua vao slide:

```text
Data Source
-> Data Preparation / Filtering
-> Model Input Store
-> Model Training & Evaluation
-> Model Artifact / Metrics
-> Recommendation Engine
-> Local UI Demo
-> Top-N Output
```

Mo ta luong request trong demo:

```text
User chon User ID + Method + Top N
-> demo_app.py nhan request /api/recommend
-> lay cac product user chua tuong tac
-> tinh score bang Rank-Based/SVD/Hybrid/Content-Based
-> sort score giam dan
-> tra ve Top-N bang User ID | Product ID | Predicted Score | Rank
```

Input/output tung component:

| Component | Input | Output | Ly do ton tai |
| --------- | ----- | ------ | ------------- |
| `data/model_input/svd/interaction_train.csv` | UserId, ProductId, Rating | Surprise trainset/model-ready data | Train SVD bang data da loc, tranh OOM |
| `data/model_input/cf/interaction_train.csv` | UserId, ProductId, Rating | CF train data | User-User/Item-Item CF can tap nho hon vi similarity matrix |
| `data/model_input/content_based/products.csv` | Product metadata | Feature text/category/brand | Dung cho Content-Based va hien thi demo |
| `models/final_model_svd.pkl` | Model SVD da train | Loaded model trong demo | Khong can train lai khi demo |
| `reports/model_metrics/all_model_metrics.csv` | Metric export | Bang so sanh model | Chung minh model da evaluate |
| `demo_app.py` | User ID, method, top N | Top-N recommendation table | Chung minh he thong co output that |

### 3.3 - Mo hinh/thuat toan co the thuc thi va tao ket qua dau ra

Muc nay la bat buoc: phai co output that, khong mock data.

Trong repo hien tai co 2 cach dap ung:

1. Dung local UI:

   ```bash
   conda activate product-recommendation-system-env
   python demo_app.py
   ```

   Sau do mo URL ma terminal in ra, chon user that va bam Generate.

2. Dung notebook:

   - Cell cuoi `notebooks/combined_recommendation_system.ipynb` da them section `Required Real Top-N Output for Slides`.
   - Cell nay load `models/final_model_svd.pkl`, load interaction data that, chon user that, predict Top-N va hien thi dung cot rubric yeu cau.

Output can chup man hinh:

```text
User ID | Product ID | Predicted Score | Rank
```

Bang chung nen dua vao slide:

- User ID cu the co that trong dataset.
- Product ID cu the co that trong dataset.
- Score la diem model tinh ra, khong tu dien tay.
- Rank la thu tu sau khi sort predicted score giam dan.
- Neu dung UI thi nen chup bang trong `demo_app.py`; neu UI loi moi dung output notebook.

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
-> Local HTTP UI / Notebook Demo
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
| Local HTTP UI / Notebook Demo | Giao dien nhap user va xem ket qua      | User nhap `UserId`                                                                               | Load model, goi recommendation engine, hien thi ket qua                                           | Bang recommendation                                                                                                | `demo_app.py` hoac Jupyter Notebook  |
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

Neu chua kip chay local HTTP UI trong `demo_app.py`, co the minh hoa luong tren bang Jupyter Notebook. Khi do Notebook dong vai tro demo UI: nguoi dung chon hoac nhap `UserId`, cell goi ham recommendation, sau do in ra bang Top-N. Yeu cau quan trong la output phai duoc sinh tu model da train tren data that, khong dung mock data.

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
Local HTTP : `demo_app.py` tao UI nhap UserId va hien thi Top-N recommendation tai localhost:8000
Notebook   : train, evaluate, demo output that neu chua co UI
```

## 1.2. Khao sat toi thieu 03 mo hinh/thuat toan lien quan

Phan nay dung de dap ung yeu cau khao sat it nhat 03 model/thuat toan lien quan. Trong bai toan recommendation system, co the trinh bay 04 nhom model chinh: Popularity, Collaborative Filtering KNN, SVD va Content-Based. Trong do SVD la model trong tam cua phan hien thuc.

Luu y cho phan thuyet trinh cua ban: neu nhom da chia phan "model survey" va ban chi phu trach Item-Item CF trong bang khao sat, thi chi can noi ngan Popularity/Content-Based/User-User/SVD de dat boi canh. Phan can giai thich sau la Item-Item CF va vi sao no duoc dat trong nhom Collaborative Filtering:

```text
User-User CF = memory-based CF baseline
Item-Item CF = memory-based CF dua tren item similarity, phu hop product-to-product recommendation
SVD          = model-based CF / matrix factorization, model chinh
```

Item-Item CF nen duoc noi ro neu PDF/slide giao phan model survey cho ban: no dung cung input `user_id`, `prod_id`, `rating` voi User-User CF, nhung trong Surprise can de `sim_options.user_based=False`.

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
reader = Reader(rating_scale=(0, 5))  # Notebook hien tai dung (0, 5); rating thuc te cua Amazon nam trong 1-5
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
   Reader(rating_scale=(0, 5))  # Notebook hien tai dung (0, 5); rating thuc te nam trong 1-5
   ```

5. Chi co explicit feedback

   Du an chu yeu dung rating. Cac hanh vi nhu click/view khong phai tin hieu chinh, nen do phu du lieu bi gioi han.

6. Gioi han RAM

   User-User CF va Item-Item CF can ma tran similarity, rat ton RAM. SVD nhe hon KNN CF, nhung grid search tren data lon van rat nang. Vi vay can dung filtered dataset.

7. Data leakage

   Cac thong ke dung de train/rank nhu `PopularityScore`, `BayesianAvg`, `RatingCount`, `AvgRating`, `global_mean` va cac latent factors cua SVD chi duoc tinh tu tap train. Tap test chi dung de danh gia sau khi model da hoc xong. Neu tinh popularity hoac average rating tren ca train + test thi model da nhin thay thong tin tu tuong lai, lam metric dep gia va khong con cong bang.

   Trong pipeline, can tach ro:

   ```text
   interaction_train.csv -> train model / tinh thong ke ranking
   interaction_test.csv  -> evaluate only
   ```

   Vi vay file `popularity_train_dataset.csv` phai duoc build tu `data/model_input/popularity/interaction_train.csv`, khong build tu test hoac full data da tron train-test.

8. Rating bias

   Dataset review thuong bi thien ve diem cao, dac biet rating 4-5. Dieu nay lam model de hoc xu huong cho diem cao va de uu tien cac san pham da pho bien/co nhieu rating tot. He qua la cac metric ranking nhu Precision@K, MAP, MRR, HitRate@K co the cao hon cam nhan thuc te vi model chi can day cac item pho bien len top.

   Cach xu ly/giai thich trong du an:

   - SVD co `global_mean`, `user_bias`, `item_bias` de dieu chinh xu huong user cho diem cao/thap va item duoc cham diem cao/thap.
   - Popularity dung Bayesian average de tranh truong hop san pham moi co it rating 5 sao nhung bi day len top qua nhanh.
   - Khi bao cao metric can noi ro ket qua bi anh huong boi sparsity va positive rating bias, khong nen chi dua vao mot metric.

9. Popularity signal chi la proxy

   Dataset chinh cua du an la review/rating, khong phai log e-commerce day du nhu click, view, add-to-cart, purchase. Vi vay cac tin hieu trong Popularity model duoc hieu la proxy:

   ```text
   ViewCount     = 1 cho moi review/interaction
   CartCount     = 1 neu Rating >= 4
   PurchaseCount = 1 neu Rating == 5 hoac VerifiedPurchase == 1
   ```

   Cong thuc minh hoa:

   ```text
   PopularityScore = 1.0 * ViewCount + 3.0 * CartCount + 5.0 * PurchaseCount
   ```

   Trong slide co the trinh bay cong thuc tren, nhung khi bi hoi can noi ro: day khong phai log hanh vi that nhu view/cart/purchase tren website. Day la tin hieu gia lap tu review dataset:

   - `View` nghia la product co mot interaction/review.
   - `Cart` duoc proxy bang rating cao (`Rating >= 4`) vi rating cao the hien user co y dinh/tin hieu tich cuc.
   - `Purchase` duoc proxy bang rating cuc cao (`Rating == 5`) hoac cot `VerifiedPurchase`.

   Luu y implementation: neu dung truc tiep `scripts/filter_model_inputs.py` hien tai, can kiem tra lai vi script dang build `CartCount = 0` va `PurchaseCount = VerifiedPurchase`. Neu slide su dung cong thuc `Cart = Rating >= 4`, `Purchase = Rating == 5 hoac VerifiedPurchase == 1` thi code build popularity can dong bo theo cong thuc nay truoc khi lay metric chinh thuc.

10. Metadata constraint cho Content-Based

   Content-Based chi xu ly cold-start item tot khi product moi co metadata nhu title, category, brand, description hoac feature text. Neu product moi chi co `ProductId` ma khong co thong tin noi dung, Content-Based cung khong co du tin hieu de tinh similarity. Khi do he thong chi co the fallback ve Popularity hoac cho den khi item co tuong tac dau tien.

### 9.1. Cau tra loi nhanh khi bi hoi xoay

1. Vi sao SVD khong xu ly duoc user moi?

   SVD hoc vector an cho user tu lich su rating. User moi chua co rating thi khong co vector user on dinh, nen du doan ca nhan hoa khong dang tin. Vi vay fallback sang Popularity.

2. Vi sao product moi fallback sang Content-Based?

   Product moi chua co rating thi SVD/CF khong hoc duoc vector item. Neu product co metadata, Content-Based van co the so sanh title/category/description voi cac item user da thich.

3. Vi sao khong dung full dataset?

   Full train co hang chuc trieu dong. KNN CF can similarity matrix rat lon, SVD/GridSearch cung ton RAM va thoi gian. Vi vay pipeline loc du lieu theo tung model de chay duoc tren Colab/Kaggle ma van giu dung logic train/evaluate.

4. Vi sao metric co the bi ao?

   Du lieu sparse va rating bi thien ve 4-5 sao. Neu nhieu user cung rating cac san pham pho bien, model de day item pho bien len top va dat ranking metric cao. Vi vay can giai thich kem sparsity, rating bias va baseline comparison.

5. Lam sao tranh data leakage?

   Tat ca feature/thong ke ranking va model parameter chi hoc tu train. Test set chi dung sau cung de tinh RMSE, MAE, Precision@K, Recall@K... Khong dung test de tinh popularity, Bayesian average hay chon item top.

## 10. Cach trinh bay output that

Khi bao cao, can chup man hinh output that tu notebook, khong dung anh mock.

Output nen co:

```text
UserId
ProductId
PredictedScore
Rank
```

Neu demo tren `demo_app.py` chua kip, co the dung output truc tiep tu Jupyter Notebook. Dieu quan trong la:

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
