import pandas as pd
import numpy as np
from pathlib import Path
import pickle

print("BƯỚC 2: LÀM SẠCH VÀ CHUẨN BỊ DỮ LIỆU")

# 1. LOAD DỮ LIỆU
input_file = 'data/anime_data.csv'
df = pd.read_csv(input_file, encoding='utf-8-sig')
print(f"  Đã load: {input_file}")
print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Columns: {df.columns.tolist()}")

# 2. XOÁ FEATURES KHÔNG CẦN THIẾT

features_to_remove = [
    'rank', 'popularity', 'aired_from', 'aired_to', 
    'season', 'duration', 'is_score_outlier'
]

# Chỉ xoá những features tồn tại
existing_to_remove = [f for f in features_to_remove if f in df.columns]
df_clean = df.drop(columns=existing_to_remove, errors='ignore')

# 3. XỬ LÝ OUTLIERS

before_count = len(df_clean)
print(f"Số anime trước khi lọc: {before_count}")

# Điều kiện: scored_by >= 500 AND score > 0
df_clean = df_clean[(df_clean['scored_by'] >= 500) & (df_clean['score'] > 0)]

after_count = len(df_clean)
removed_count = before_count - after_count

print(f"Điều kiện: scored_by >= 500 AND score > 0")
print(f"✓ Đã loại bỏ: {removed_count} anime ({removed_count/before_count*100:.1f}%)")
print(f"✓ Còn lại: {after_count} anime")

# 4. CHUẨN BỊ TEXT CHO BERT EMBEDDINGS
# Kết hợp synopsis + genres
df_clean['text_for_embedding'] = (
    df_clean['synopsis'].fillna('') + ' ' + 
    df_clean['genres'].fillna('')
).str.strip()

print(f"✓ Đã tạo text_for_embedding từ: synopsis + genres")
print(f"\nVí dụ text (100 ký tự đầu):")
for i in range(min(2, len(df_clean))):
    text = df_clean['text_for_embedding'].iloc[i]
    print(f"  {i+1}. {text[:100]}...")

# 5. VECTOR HOÁ BẰNG BERT (sentence-transformers)

bert_success = False

try:
    from sentence_transformers import SentenceTransformer
    
    print("Loading BERT model: all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Encoding {len(df_clean)} texts (có thể mất vài phút)...")
    embeddings = model.encode(
        df_clean['text_for_embedding'].tolist(),
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    print(f"✓ Embeddings shape: {embeddings.shape}")
    print(f"  Mỗi anime được biểu diễn bằng vector {embeddings.shape[1]} chiều")
    
    # Save embeddings
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    embeddings_file = output_dir / 'bert_embeddings.pkl'
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"✓ Saved embeddings to: {embeddings_file}")
    
    # Drop text_for_embedding column (không cần lưu vào CSV)
    df_clean = df_clean.drop(columns=['text_for_embedding'])
    print(f"✓ Đã loại bỏ text_for_embedding (chỉ dùng cho BERT encoding)")
    
    # Save model for later use
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    model_file = model_dir / 'bert_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved BERT model to: {model_file}")
    
    bert_success = True
    
except ImportError:
    print("   sentence-transformers chưa được cài đặt!")
    print("   Chạy: pip install sentence-transformers")
    
except Exception as e:
    print(f"   Lỗi khi encoding: {e}")

# 6. LƯU DỮ LIỆU ĐÃ LÀM SẠCH

output_dir = Path('data/processed')
output_dir.mkdir(exist_ok=True, parents=True)

output_file = output_dir / 'anime_data_final.csv'
df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✓ Đã lưu: {output_file}")
print(f"  Rows: {len(df_clean)}")
print(f"  Columns: {len(df_clean.columns)}")
