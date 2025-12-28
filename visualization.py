import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import Counter

# Cấu hình matplotlib
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Load dữ liệu đã clean
input_file = 'data/processed/anime_data_final.csv'
df = pd.read_csv(input_file, encoding='utf-8-sig')

# Tạo folder visualizations
viz_dir = Path('visualizations')
viz_dir.mkdir(exist_ok=True)

# BIỂU ĐỒ 1: PHÂN BỐ RATING (HISTOGRAM)

plt.figure(figsize=(10, 6))
plt.hist(df['score'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
plt.title('Phân bố điểm Score của Anime', fontsize=14, fontweight='bold')
plt.xlabel('Score (điểm đánh giá)')
plt.ylabel('Số lượng anime')
plt.xlim(6.5, 10) 
plt.axvline(df['score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["score"].mean():.2f}')
plt.axvline(df['score'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["score"].median():.2f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
output_file = viz_dir / '01_score_distribution.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')

# BIỂU ĐỒ 2: TOP 10 ANIME CÓ NHIỀU NGƯỜI XEM NHẤT (scored_by)

top_10 = df.nlargest(10, 'scored_by')[['title', 'score', 'scored_by', 'genres']]

plt.figure(figsize=(14, 8))
plt.barh(range(10), top_10['scored_by'].values, color='coral', edgecolor='black')
plt.yticks(range(10), top_10['title'].values)
plt.xlabel('Số người đánh giá (scored_by)', fontsize=12)
plt.title('Top 10 Anime có nhiều người xem nhất', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# Thêm giá trị và score
for i, (scored, score) in enumerate(zip(top_10['scored_by'].values, top_10['score'].values)):
    plt.text(scored + 10000, i, f'{scored:,.0f} | ⭐{score:.2f}', 
             va='center', fontsize=9, fontweight='bold')

plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
output_file = viz_dir / '02_top_10_anime.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')

# BIỂU ĐỒ 3: TẦN SUẤT GENRE
# Parse genres (cột genres có dạng "Action, Drama, Fantasy")
all_genres = []
for genres_str in df['genres'].dropna():
    genres_list = [g.strip() for g in str(genres_str).split(',')]
    all_genres.extend(genres_list)

# Đếm tần suất
genre_counts = Counter(all_genres)
top_genres = dict(genre_counts.most_common(15))

plt.figure(figsize=(12, 8))
plt.barh(range(len(top_genres)), list(top_genres.values()), color='skyblue', edgecolor='black')
plt.yticks(range(len(top_genres)), list(top_genres.keys()))
plt.xlabel('Số lượng anime', fontsize=12)
plt.title('Top 15 Genre phổ biến nhất', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# Thêm giá trị
for i, count in enumerate(top_genres.values()):
    plt.text(count + 5, i, f'{count}', va='center', fontsize=10, fontweight='bold')

plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
output_file = viz_dir / '03_genre_frequency.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')

# BIỂU ĐỒ 4: HEATMAP CORRELATION (score vs scored_by)

# Chọn các cột số để tính correlation
numeric_cols = ['score', 'scored_by', 'members', 'favorites', 'year', 'episodes']
corr_df = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Ma trận tương quan (Correlation Matrix)', fontsize=14, fontweight='bold')
plt.tight_layout()
output_file = viz_dir / '04_correlation_heatmap.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')