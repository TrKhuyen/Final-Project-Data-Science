import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# 1. LOAD DỮ LIỆU
input_file = 'data/anime_data.csv'
df = pd.read_csv(input_file, encoding='utf-8-sig')
print(f"  ✓ Đã load: {input_file}")
print(f"  ✓ Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  ✓ Columns: {df.columns.tolist()}")

# 2. XỬ LÝ MISSING VALUES
# Hiển thị missing values trước khi xử lý
missing_before = df.isnull().sum()
total_missing_before = missing_before.sum()
print(f"  Tổng missing values trước xử lý: {total_missing_before:,}")

# 2.1 Xử lý TEXT columns - Fill với empty string
text_columns = ['synopsis', 'genres', 'themes', 'demographics', 'studios', 
                'source', 'status', 'type', 'rating']
for col in text_columns:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col] = df[col].fillna('')

# 2.2 Xử lý title_english - Fill với title gốc
if 'title_english' in df.columns:
    missing_count = df['title_english'].isnull().sum()
    if missing_count > 0:
        df['title_english'] = df['title_english'].fillna('None')

# 2.3 Xử lý episodes (NUMERIC) - Fill với median
if 'episodes' in df.columns:
    missing_count = df['episodes'].isnull().sum()
    if missing_count > 0:
        median_val = df['episodes'].median()
        df['episodes'] = df['episodes'].fillna(median_val)

# 2.4 Xử lý year (NUMERIC) - Fill với mode
if 'year' in df.columns:
    missing_count = df['year'].isnull().sum()
    if missing_count > 0:
        mode_val = df['year'].mode()[0]
        df['year'] = df['year'].fillna(mode_val)

# 2.5 Xử lý URL columns - Fill với empty string
url_columns = ['images_url', 'trailer_url']
for col in url_columns:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col] = df[col].fillna('')

# Verify
total_missing_after = df.isnull().sum().sum()

# 3. XOÁ FEATURES KHÔNG CẦN THIẾT

features_to_remove = [
    'rank', 'popularity', 'aired_from', 'aired_to', 
    'season', 'duration', 'is_score_outlier'
]

# Chỉ xoá những features tồn tại
existing_to_remove = [f for f in features_to_remove if f in df.columns]
df_clean = df.drop(columns=existing_to_remove, errors='ignore')

# 4. XỬ LÝ OUTLIERS

before_count = len(df_clean)

# Điều kiện: scored_by >= 500 AND score > 0
df_clean = df_clean[(df_clean['scored_by'] >= 500) & (df_clean['score'] > 0)]

after_count = len(df_clean)
removed_count = before_count - after_count

# 5. CHUẨN BỊ TEXT CHO BERT EMBEDDINGS

# Kết hợp synopsis + genres
df_clean['text_for_embedding'] = (
    df_clean['synopsis'].fillna('') + ' ' + 
    df_clean['genres'].fillna('')
).str.strip()
for i in range(min(2, len(df_clean))):
    text = df_clean['text_for_embedding'].iloc[i]

# 6. VECTOR HOÁ BẰNG BERT (sentence-transformers)

bert_success = False

try:
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embeddings = model.encode(
        df_clean['text_for_embedding'].tolist(),
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    # Save embeddings
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    embeddings_file = output_dir / 'bert_embeddings.pkl'
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Drop text_for_embedding column (không cần lưu vào CSV)
    df_clean = df_clean.drop(columns=['text_for_embedding'])
    
    # Save model for later use
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    model_file = model_dir / 'bert_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    bert_success = True
    
except ImportError:
    print("   sentence-transformers chưa được cài đặt!")
    print("     Chạy: pip install sentence-transformers")
    
except Exception as e:
    print(f"   Lỗi khi encoding: {e}")

#LƯU DỮ LIỆU ĐÃ LÀM SẠCH

output_dir = Path('data/processed')
output_dir.mkdir(exist_ok=True, parents=True)

output_file = output_dir / 'anime_data_final.csv'
df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
