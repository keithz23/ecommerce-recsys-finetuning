import pandas as pd


def calculate_dataset_sparsity(interaction_file_path: str) -> None:
    # 1. Đọc file dữ liệu tương tác (ví dụ tập train của thuật toán CF)
    df = pd.read_csv(interaction_file_path)

    # 2. Đếm số lượng User duy nhất và Product duy nhất
    num_unique_users = df["UserId"].nunique()
    num_unique_products = df["ProductId"].nunique()

    # 3. Tổng số lượng tương tác thực tế đang có
    actual_interactions = len(df)

    # 4. Tính toán tổng số ô tối đa nếu ma trận chứa đầy đủ dữ liệu
    total_possible_interactions = num_unique_users * num_unique_products

    # 5. Tính tỷ lệ phần trăm ô trống (Sparsity)
    sparsity_ratio = (
        1 - (actual_interactions / total_possible_interactions)
    ) * 100

    # In kết quả ra màn hình
    print("=" * 50)
    print(f"THỐNG KÊ ĐỘ THƯA THỚT (SPARSITY) CỦA FILE:")
    print(f" Đường dẫn: {interaction_file_path}")
    print("-" * 50)
    print(f"- Số lượng Users duy nhất  : {num_unique_users:,}")
    print(f"- Số lượng Products duy nhất: {num_unique_products:,}")
    print(f"- Tổng số ô trong ma trận  : {total_possible_interactions:,}")
    print(f"- Số tương tác thực tế có  : {actual_interactions:,}")
    print(f"- TỶ LỆ SPARSITY (TRỐNG)   : {sparsity_ratio:.4f}%")
    print(
        f"- Tỷ lệ Density (Có dữ liệu): {100 - sparsity_ratio:.4f}%"
    )
    print("=" * 50)


# --- Cách chạy thử nghiệm ---
# Bạn truyền đường dẫn file tương tác của mô hình 'cf' hoặc 'svd' vừa tạo ở bước trước vào đây
calculate_dataset_sparsity("data/model_input/content_based/interaction_train.csv")