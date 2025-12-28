import requests
import json
import time
import pandas as pd
from pathlib import Path

def crawl_top_anime(pages=5, start_page=1):
    """
    Crawl top anime data from Jikan API
    
    Args:
        pages: Number of pages to crawl (25 anime per page)
        start_page: Starting page number (for resuming)
    """
    base_url = "https://api.jikan.moe/v4/top/anime"
    all_anime = []
    
    for page in range(start_page, start_page + pages):
        try:
            print(f"\nFetching page {page}...")
            params = {'page': page}
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                anime_list = data.get('data', [])
                all_anime.extend(anime_list)
            else:
                print(f"  Error: Status code {response.status_code}")
            
            if page < (start_page + pages - 1):
                time.sleep(1)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            continue
    
    return all_anime

def save_data(anime_data, output_dir='data'):
    """Save crawled data in multiple formats"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    json_file = output_path / 'anime_data_raw.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(anime_data, f, ensure_ascii=False, indent=2)
    
    csv_data = []
    for anime in anime_data:
        row = {
            'mal_id': anime.get('mal_id'),
            'title': anime.get('title'),
            'title_english': anime.get('title_english'),
            'type': anime.get('type'),
            'episodes': anime.get('episodes'),
            'status': anime.get('status'),
            'score': anime.get('score'),
            'scored_by': anime.get('scored_by'),
            'rank': anime.get('rank'),
            'popularity': anime.get('popularity'),
            'members': anime.get('members'),
            'favorites': anime.get('favorites'),
            'source': anime.get('source'),
            'rating': anime.get('rating'),
            'season': anime.get('season'),
            'year': anime.get('year'),
            'duration': anime.get('duration'),
            'synopsis': anime.get('synopsis'),  
        }
        
        images = anime.get('images', {})
        jpg_images = images.get('jpg', {})
        row['images_url'] = jpg_images.get('image_url', '')
        
        trailer = anime.get('trailer', {})
        row['trailer_url'] = trailer.get('embed_url', '')
        
        genres = anime.get('genres', [])
        row['genres'] = ', '.join([g.get('name', '') for g in genres])
        
        studios = anime.get('studios', [])
        row['studios'] = ', '.join([s.get('name', '') for s in studios])
        
        themes = anime.get('themes', [])
        row['themes'] = ', '.join([t.get('name', '') for t in themes])
        
        demographics = anime.get('demographics', [])
        row['demographics'] = ', '.join([d.get('name', '') for d in demographics])
        
        aired = anime.get('aired', {})
        row['aired_from'] = aired.get('from')
        row['aired_to'] = aired.get('to')
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_file = output_path / 'anime_data.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    return df

def main():
    anime_data = crawl_top_anime(pages=200)  
    
    if not anime_data:
        print("\n No data collected!")
        return
    
    df = save_data(anime_data)
    
if __name__ == "__main__":
    main()
