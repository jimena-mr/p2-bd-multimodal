import csv
from backend.indexing.spimi import SPIMIIndexer

def load_documents_from_csv(csv_path):
    docs = {}
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = f"{row['track_id']} {row['track_name']} {row['track_artist']} {row['lyrics']} {row['track_popularity']} {row['track_album_id']} {row['track_album_name']} {row['track_album_release_date']} {row['playlist_name']} {row['playlist_id']} {row['playlist_genre']} {row['playlist_subgenre']} {row['danceability']} {row['energy']} {row['key']} {row['loudness']} {row['mode']} {row['speechiness']} {row['acousticness']} {row['instrumentalness']} {row['liveness']} {row['valence']} {row['tempo']} {row['duration_ms']} {row['language']}"
            docs[row['track_id']] = text
    return docs

if __name__ == "__main__":
    csv_path = "test/spotify_songs.csv"
    docs = load_documents_from_csv(csv_path)
    indexer = SPIMIIndexer(output_path="data/Audio/index.json")
    indexer.index_documents(docs)
    print("✔ Índice invertido construido y guardado en data/index.json")
