import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎌 Anime AI Recommender", layout="wide", page_icon="🎌")

@st.cache_resource
def load_assets():
    """Load dữ liệu và mô hình"""
    try:
        df = pd.read_csv("data/processed/anime_data_final.csv", encoding='utf-8-sig')
        
        with open("data/processed/bert_embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        
        cosine_sim_file = Path("models/cosine_similarity_matrix.pkl")
        if cosine_sim_file.exists():
            with open(cosine_sim_file, "rb") as f:
                cosine_sim = pickle.load(f)
        else:
            cosine_sim = cosine_similarity(embeddings, embeddings)
            cosine_sim_file.parent.mkdir(exist_ok=True, parents=True)
            with open(cosine_sim_file, "wb") as f:
                pickle.dump(cosine_sim, f)
        
        model_file = Path("models/bert_model.pkl")
        if model_file.exists():
            with open(model_file, "rb") as f:
                model = pickle.load(f)
        else:
            st.info("Đang load BERT model...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
        
        C = df['score'].mean()
        m = df['scored_by'].quantile(0.7)
        df['weighted_rating'] = (df['scored_by'] / (df['scored_by'] + m)) * df['score'] + (m / (df['scored_by'] + m)) * C
        
        return df, embeddings, cosine_sim, model
    
    except Exception as e:
        st.stop()

df, embeddings, cosine_sim, bert_model = load_assets()

if 'history' not in st.session_state:
    st.session_state.history = []

def search_by_description(description, df, embeddings, bert_model, top_n=10, min_score=7.0):
    """Tìm anime theo mô tả sử dụng BERT semantic search"""
    user_vector = bert_model.encode([description])
    
    sim_scores = cosine_similarity(user_vector, embeddings)[0]
    
    results = df.copy()
    results['similarity_score'] = sim_scores
    results = results[results['score'] >= min_score]
    
    if len(results) == 0:
        return None
    
    results['similarity_normalized'] = (results['similarity_score'] - results['similarity_score'].min()) / (results['similarity_score'].max() - results['similarity_score'].min() + 1e-10)
    results['quality_normalized'] = (results['weighted_rating'] - results['weighted_rating'].min()) / (results['weighted_rating'].max() - results['weighted_rating'].min() + 1e-10)
    results['hybrid_score'] = 0.7 * results['similarity_normalized'] + 0.3 * results['quality_normalized']
    
    return results.sort_values('hybrid_score', ascending=False).head(top_n)

def get_all_matches(anime_title, df):
    """Tìm tất cả anime match và sắp xếp theo độ ưu tiên"""
    try:
        matches = df[df['title'].str.contains(anime_title, case=False, na=False)].copy()
        if len(matches) == 0:
            return None
        
        matches['match_score'] = matches['title'].apply(
            lambda x: 100 if x.lower() == anime_title.lower() else 
                     50 if x.lower().startswith(anime_title.lower()) else 
                     10
        )
        
        matches = matches.sort_values(by=['match_score', 'weighted_rating'], ascending=[False, False])
        
        return matches
    except:
        return None

def get_hybrid_recommendations(anime_title, df, cosine_sim, top_n=10, min_score=7.0):
    try:
        idx = df[df['title'].str.contains(anime_title, case=False, na=False)].index[0]
    except IndexError:
        return None, None
    
    input_anime = df.iloc[idx]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
    
    anime_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[anime_indices].copy()
    recommendations['similarity_score'] = [i[1] for i in sim_scores]
    recommendations = recommendations[recommendations['score'] >= min_score]
    
    if len(recommendations) == 0:
        return input_anime, None
    
    recommendations['similarity_normalized'] = (recommendations['similarity_score'] - recommendations['similarity_score'].min()) / (recommendations['similarity_score'].max() - recommendations['similarity_score'].min() + 1e-10)
    recommendations['quality_normalized'] = (recommendations['weighted_rating'] - recommendations['weighted_rating'].min()) / (recommendations['weighted_rating'].max() - recommendations['weighted_rating'].min() + 1e-10)
    recommendations['hybrid_score'] = 0.7 * recommendations['similarity_normalized'] + 0.3 * recommendations['quality_normalized']
    
    return input_anime, recommendations.sort_values('hybrid_score', ascending=False).head(top_n)

def calculate_precision_at_k(anime_title, recommendations, df, k=10):
    """Tính Precision@K dựa trên genre overlap"""
    try:
        idx = df[df['title'].str.contains(anime_title, case=False, na=False)].index[0]
        input_genres_str = str(df.iloc[idx]['genres'])
        if pd.isna(input_genres_str) or input_genres_str == 'nan':
            return 0.0
        
        # Split và clean genres
        input_genres = set([g.strip() for g in input_genres_str.split(',') if g.strip()])
        
        if recommendations is None or len(recommendations) == 0:
            return 0.0
        
        relevant = 0
        for _, row in recommendations.head(k).iterrows():
            rec_genres_str = str(row['genres'])
            if pd.notna(rec_genres_str) and rec_genres_str != 'nan':
                rec_genres = set([g.strip() for g in rec_genres_str.split(',') if g.strip()])
                # Có ít nhất 1 genre chung thì tính là relevant
                if len(input_genres & rec_genres) > 0:
                    relevant += 1
        
        return relevant / k
    except Exception as e:
        return 0.0

def calculate_recall_at_k(anime_title, recommendations, df, k=10):
    """Tính Recall@K dựa trên genre overlap"""
    try:
        idx = df[df['title'].str.contains(anime_title, case=False, na=False)].index[0]
        input_genres_str = str(df.iloc[idx]['genres'])
        if pd.isna(input_genres_str) or input_genres_str == 'nan':
            return 0.0
        
        input_genres = set([g.strip() for g in input_genres_str.split(',') if g.strip()])
        
        # Tính tổng số anime relevant trong dataset (có ít nhất 1 genre chung)
        total_relevant = 0
        for _, row in df.iterrows():
            rec_genres_str = str(row['genres'])
            if pd.notna(rec_genres_str) and rec_genres_str != 'nan':
                rec_genres = set([g.strip() for g in rec_genres_str.split(',') if g.strip()])
                if len(input_genres & rec_genres) > 0:
                    total_relevant += 1
        
        total_relevant -= 1  # Trừ chính anime đầu vào
        
        if recommendations is None or len(recommendations) == 0 or total_relevant == 0:
            return 0.0
        
        # Đếm số relevant trong top-k recommendations
        retrieved_relevant = 0
        for _, row in recommendations.head(k).iterrows():
            rec_genres_str = str(row['genres'])
            if pd.notna(rec_genres_str) and rec_genres_str != 'nan':
                rec_genres = set([g.strip() for g in rec_genres_str.split(',') if g.strip()])
                if len(input_genres & rec_genres) > 0:
                    retrieved_relevant += 1
        
        return retrieved_relevant / min(total_relevant, k)
    except Exception as e:
        return 0.0

def calculate_rmse_mae(anime_title, recommendations, df):
    """Tính RMSE và MAE dựa trên score similarity"""
    try:
        idx = df[df['title'].str.contains(anime_title, case=False, na=False)].index[0]
        input_score = df.iloc[idx]['score']
        
        if recommendations is None or len(recommendations) == 0:
            return 0.0, 0.0
        
        rec_scores = recommendations['score'].values
        errors = rec_scores - input_score
        
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        
        return rmse, mae
    except Exception as e:
        return 0.0, 0.0

st.sidebar.title("🎌 Bộ lọc & Cài đặt")

st.sidebar.subheader("📍 Lọc theo ngữ cảnh")
year_range = st.sidebar.slider("Năm phát hành", int(df['year'].min()), int(df['year'].max()), (2000, int(df['year'].max())))
min_score = st.sidebar.slider("Score tối thiểu", 0.0, 10.0, 7.0, 0.1)
top_n = st.sidebar.slider("Số lượng gợi ý", 5, 20, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("📜 Lịch sử tìm kiếm")
if st.session_state.history:
    for item in reversed(st.session_state.history[-5:]):
        if st.sidebar.button(f"🔄 {item}", key=f"hist_{item}"):
            st.session_state.search_query = item
    if st.sidebar.button("🗑️ Xóa lịch sử"):
        st.session_state.history = []
        st.rerun()
else:
    st.sidebar.caption("Chưa có lịch sử.")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Thông tin Dataset")
st.sidebar.metric("Tổng số anime", len(df))
st.sidebar.metric("Score trung bình", f"{df['score'].mean():.2f}")

st.title("🎌 Hệ thống Gợi ý Anime Thông minh")
st.markdown("*Powered by BERT Embeddings & Hybrid Filtering*")

tab1, tab2, tab3 = st.tabs(["🔍 Gợi ý Anime", "📊 Phân tích Dữ liệu", "🎯 Đánh giá Mô hình"])

with tab1:
    st.subheader("🎬 Tìm anime phù hợp với sở thích của bạn")
    
    col_input, col_random = st.columns([4, 1])
    
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    
    user_input = col_input.text_input(
        "Nhập tên anime hoặc mô tả:", 
        value=st.session_state.search_query,
        placeholder="VD: Naruto, Attack on Titan, anime về samurai..."
    )
    
    random_btn = col_random.button("🎲 Ngẫu nhiên")
    
    if st.button("🔍 Tìm kiếm ngay", type="primary") or random_btn:
        target = user_input if not random_btn else df.sample(1).iloc[0]['title']
        
        if target:
            if target not in st.session_state.history:
                st.session_state.history.append(target)
            
            with st.spinner("Đang phân tích..."):
                all_matches = get_all_matches(target, df)
                
                if all_matches is not None and len(all_matches) > 1:
                    st.info(f"🔍 Tìm thấy **{len(all_matches)}** anime phù hợp với '{target}'. Vui lòng chọn:")
                    
                    for i in range(0, len(all_matches), 5):
                        cols = st.columns(5)
                        for j, (idx, row) in enumerate(all_matches.iloc[i:i+5].iterrows()):
                            with cols[j]:
                                if pd.notna(row['images_url']) and row['images_url']:
                                    st.image(row['images_url'], use_container_width=True)
                                else:
                                    st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
                                
                                st.markdown(f"**{row['title'][:40]}**")
                                year_str = int(row['year']) if pd.notna(row['year']) else 'N/A'
                                st.caption(f"⭐ {row['score']:.2f} | 📅 {year_str}")
                                st.caption(f"🎯 {row['weighted_rating']:.2f}")
                                
                                if pd.notna(row['genres']):
                                    genres = str(row['genres']).split(',')[:2]
                                    st.info(" • ".join([g.strip() for g in genres]))
                                
                                if pd.notna(row.get('trailer_url')) and row['trailer_url']:
                                    with st.expander("▶️ Xem Trailer"):
                                        st.video(row['trailer_url'])
                                else:
                                    st.caption("⚠️ Không có trailer")
                    
                
                elif 'selected_anime' in st.session_state or (all_matches is not None and len(all_matches) == 1):
                    if 'selected_anime' in st.session_state:
                        target = st.session_state.selected_anime
                        del st.session_state.selected_anime
                    
                    input_anime, recommendations = get_hybrid_recommendations(target, df, cosine_sim, top_n, min_score)
                    
                    if input_anime is None or recommendations is None or len(recommendations) == 0:
                        st.warning("⚠️ Không có anime phù hợp với bộ lọc!")
                    else:
                        st.success(f"✅ Tìm thấy: **{input_anime['title']}**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Score", f"{input_anime['score']:.2f}/10")
                        col2.metric("W.Rating", f"{input_anime['weighted_rating']:.2f}")
                        col3.metric("Scored by", f"{input_anime['scored_by']:,}")
                        col4.metric("Year", int(input_anime['year']) if pd.notna(input_anime['year']) else 'N/A')
                        
                        with st.expander("📖 Xem thông tin chi tiết"):
                            col_img, col_detail = st.columns([1, 2])
                            with col_img:
                                if pd.notna(input_anime['images_url']) and input_anime['images_url']:
                                    st.image(input_anime['images_url'], width=200)
                                else:
                                    st.info("Không có ảnh")
                            
                            with col_detail:
                                if pd.notna(input_anime['genres']):
                                    st.markdown(f"**Genres:** {input_anime['genres']}")
                                if pd.notna(input_anime.get('themes')) and input_anime['themes']:
                                    st.markdown(f"**Themes:** {input_anime['themes']}")
                                if pd.notna(input_anime.get('demographics')) and input_anime['demographics']:
                                    st.markdown(f"**Demographics:** {input_anime['demographics']}")
                                if pd.notna(input_anime.get('studios')) and input_anime['studios']:
                                    st.markdown(f"**Studios:** {input_anime['studios']}")
                                
                                if pd.notna(input_anime.get('trailer_url')) and input_anime['trailer_url']:
                                    st.link_button("▶️ Xem Trailer", input_anime['trailer_url'])
                                
                                if pd.notna(input_anime.get('synopsis')) and input_anime['synopsis']:
                                    with st.expander("📝 Synopsis"):
                                        st.write(input_anime['synopsis'][:500] + "..." if len(str(input_anime['synopsis'])) > 500 else input_anime['synopsis'])
                        
                        st.markdown("---")
                        st.subheader(f"🎯 Top {min(len(recommendations), top_n)} Gợi ý")
                        
                        filtered = recommendations[(recommendations['year'] >= year_range[0]) & (recommendations['year'] <= year_range[1])]
                        
                        if len(filtered) == 0:
                            st.warning("⚠️ Không có anime phù hợp với năm đã chọn. Hiển thị gốc:")
                            filtered = recommendations
                        
                        for i in range(0, len(filtered), 5):
                            cols = st.columns(5)
                            for j, (idx, row) in enumerate(filtered.iloc[i:i+5].iterrows()):
                                with cols[j]:
                                    if pd.notna(row['images_url']) and row['images_url']:
                                        st.image(row['images_url'], use_container_width=True)
                                    else:
                                        st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
                                    
                                    st.markdown(f"**{i+j+1}. {row['title'][:40]}**")
                                    year_str = int(row['year']) if pd.notna(row['year']) else 'N/A'
                                    st.caption(f"⭐ {row['score']:.2f} | 📅 {year_str}")
                                    st.caption(f"🎯 {row['hybrid_score']:.3f}")
                                    
                                    if pd.notna(row['genres']):
                                        genres = str(row['genres']).split(',')[:2]
                                        st.info(" • ".join([g.strip() for g in genres]))
                                    
                                    if pd.notna(row['trailer_url']) and row['trailer_url']:
                                        st.link_button("▶️ Trailer", row['trailer_url'], use_container_width=True)
                                    
                                    with st.expander("Chi tiết"):
                                        st.write(f"**W.Rating:** {row['weighted_rating']:.2f}")
                                        st.write(f"**Similarity:** {row['similarity_score']:.3f}")
                
                else:
                    st.info(f"🔍 Không tìm thấy anime tên '{target}'. Đang tìm theo mô tả...")
                    
                    description_results = search_by_description(target, df, embeddings, bert_model, top_n, min_score)
                    
                    if description_results is not None and len(description_results) > 0:
                        st.success(f"✅ Tìm thấy {len(description_results)} anime phù hợp với mô tả!")
                        
                        for i in range(0, len(description_results), 5):
                            cols = st.columns(5)
                            for j, (idx, row) in enumerate(description_results.iloc[i:i+5].iterrows()):
                                with cols[j]:
                                    if pd.notna(row['images_url']) and row['images_url']:
                                        st.image(row['images_url'], use_container_width=True)
                                    else:
                                        st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
                                    
                                    st.markdown(f"**{i+j+1}. {row['title'][:40]}**")
                                    year_str = int(row['year']) if pd.notna(row['year']) else 'N/A'
                                    st.caption(f"⭐ {row['score']:.2f} | 📅 {year_str}")
                                    st.caption(f"🎯 Hybrid: {row['hybrid_score']:.3f}")
                                    
                                    if pd.notna(row['genres']):
                                        genres = str(row['genres']).split(',')[:2]
                                        st.info(" • ".join([g.strip() for g in genres]))
                                    
                                    if pd.notna(row['trailer_url']) and row['trailer_url']:
                                        st.link_button("▶️ Trailer", row['trailer_url'], use_container_width=True)
                                    
                                    with st.expander("Chi tiết"):
                                        st.write(f"**W.Rating:** {row['weighted_rating']:.2f}")
                                        st.write(f"**Similarity:** {row['similarity_score']:.3f}")
                                        if pd.notna(row.get('synopsis')):
                                            st.write(f"**Synopsis:** {str(row['synopsis'])[:200]}...")
                    else:
                        st.error(f"❌ Không tìm thấy anime phù hợp với: {target}")

with tab2:
    st.header("Trực quan hóa Dữ liệu")
    
    viz_dir = Path('visualizations')
    if viz_dir.exists():
        col1, col2 = st.columns(2)
        
        with col1:
            if (viz_dir / '01_score_distribution.png').exists():
                st.image(str(viz_dir / '01_score_distribution.png'))
            
            if (viz_dir / '03_genre_frequency.png').exists():
                st.image(str(viz_dir / '03_genre_frequency.png'))
        
        with col2:
            if (viz_dir / '02_top_10_anime.png').exists():
                st.image(str(viz_dir / '02_top_10_anime.png'))
            
            if (viz_dir / '04_correlation_heatmap.png').exists():
                st.image(str(viz_dir / '04_correlation_heatmap.png'))
    else:
        st.warning("Chưa có biểu đồ. Chạy 03_eda_visualization.py!")

with tab3:
    st.header("Đánh giá Hiệu năng Mô hình")
    st.markdown(f"*Đánh giá trên {len(df)} anime*")
    
    test_anime = ['Naruto', 'Death Note', 'One Piece', 'Attack on Titan', 'Dragon Ball Z']
    precisions = []
    recalls = []
    rmse_list = []
    mae_list = []
    valid_anime = []
    
    with st.spinner("Đang tính toán metrics..."):
        for anime in test_anime:
            _, recs = get_hybrid_recommendations(anime, df, cosine_sim, 10, 0)
            
            if recs is not None and len(recs) > 0:
                precision = calculate_precision_at_k(anime, recs, df, 10)
                recall = calculate_recall_at_k(anime, recs, df, 10)
                rmse, mae = calculate_rmse_mae(anime, recs, df)
                
                precisions.append(precision)
                recalls.append(recall)
                rmse_list.append(rmse)
                mae_list.append(mae)
                valid_anime.append(anime)
    
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_rmse = np.mean(rmse_list) if rmse_list else 0
    avg_mae = np.mean(mae_list) if mae_list else 0
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    # Row 1: Main metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision@10", f"{avg_precision*100:.1f}%", help="Tỷ lệ gợi ý có cùng genre trong top-10")
    col2.metric("Recall@10", f"{avg_recall*100:.1f}%", help="Tỷ lệ anime relevant được gợi ý trong top-10")
    col3.metric("RMSE", f"{avg_rmse:.3f}", help="Root Mean Square Error - Sai số bình phương trung bình")
    col4.metric("MAE", f"{avg_mae:.3f}", help="Mean Absolute Error - Sai số tuyệt đối trung bình")
    
    # Row 2: Additional info
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("F1-Score", f"{f1_score*100:.1f}%", help="Harmonic mean của Precision và Recall")
    col6.metric("Số anime test", len(valid_anime), help="Số anime đã test thành công")
    col7.metric("Dataset size", f"{len(df):,}", help="Tổng số anime")
    col8.metric("Embedding dim", "384", help="BERT vector dimensions")
    
    st.markdown("---")
    
    col_plot, col_info = st.columns([2, 1])
    
    with col_plot:
        st.subheader("📊 Metrics từng anime")
        
        if precisions and recalls:
            test_names = valid_anime
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Chart 1: Precision & Recall
            x = np.arange(len(test_names))
            width = 0.35
            
            bars1 = axes[0].bar(x - width/2, [p*100 for p in precisions], width, label='Precision@10', color='steelblue')
            bars2 = axes[0].bar(x + width/2, [r*100 for r in recalls[:len(precisions)]], width, label='Recall@10', color='coral')
            
            axes[0].set_ylabel('Score (%)', fontsize=11)
            axes[0].set_xlabel('Anime', fontsize=11)
            axes[0].set_title('Precision@10 vs Recall@10', fontsize=12, fontweight='bold')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(test_names, rotation=45, ha='right')
            axes[0].legend()
            axes[0].set_ylim(0, 110)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            for bar in bars1:
                height = bar.get_height()
                axes[0].annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                axes[0].annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
            # Chart 2: RMSE & MAE
            if rmse_list and mae_list:
                bars3 = axes[1].bar(x - width/2, rmse_list[:len(test_names)], width, label='RMSE', color='forestgreen')
                bars4 = axes[1].bar(x + width/2, mae_list[:len(test_names)], width, label='MAE', color='gold')
                
                axes[1].set_ylabel('Error Score', fontsize=11)
                axes[1].set_xlabel('Anime', fontsize=11)
                axes[1].set_title('RMSE vs MAE (Score Similarity)', fontsize=12, fontweight='bold')
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(test_names, rotation=45, ha='right')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3, axis='y')
                
                for bar in bars3:
                    height = bar.get_height()
                    axes[1].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                for bar in bars4:
                    height = bar.get_height()
                    axes[1].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with col_info:
        st.subheader("📋 Giải thích Metrics")
        
        with st.expander("📊 Precision@K", expanded=True):
            st.markdown("""**Precision@K** = Relevant trong top-K / K
            
Đo lường: Trong K anime được gợi ý, có bao nhiêu % thực sự liên quan (cùng genre).""")
        
        with st.expander("📊 Recall@K"):
            st.markdown("""**Recall@K** = Relevant trong top-K / Tổng relevant
            
Đo lường: Trong tất cả anime liên quan, hệ thống đã gợi ý được bao nhiêu % trong top-K.""")
        
        with st.expander("📊 RMSE & MAE"):
            st.markdown("""**RMSE** = √(Σ(error)² / n)
            
**MAE** = Σ|error| / n
            
Đo độ chênh lệch score giữa anime gốc và anime được gợi ý. Giá trị thấp = gợi ý có chất lượng tương đương.""")
        
        with st.expander("📊 F1-Score"):
            st.markdown("""**F1** = 2 × (Precision × Recall) / (Precision + Recall)
            
Là trung bình điều hòa của Precision và Recall, cân bằng giữa độ chính xác và độ bao phủ.""")
        
        st.markdown("---")
        
        st.success("""**✅ Ưu điểm:**
- BERT embeddings hiểu ngữ nghĩa tốt
- Hybrid score cân bằng similarity & quality""")
        
        st.warning("""**💡 Cải thiện:**
- Thêm collaborative filtering
- Fine-tune BERT trên anime domain""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎌 <b>Anime Recommendation System</b> | Powered by BERT & Hybrid Filtering</p>
    <p>Dataset: {n_anime:,} anime | Model: Content-based + Quality Filter</p>
</div>
""".format(n_anime=len(df)), unsafe_allow_html=True)
