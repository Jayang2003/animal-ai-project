from icrawler.builtin import BingImageCrawler
import os

breeds = {
    "Marwari": "Marwari horse India",
    "Kathiawari": "Kathiawari horse Gujarat",
    "Arabian": "Arabian horse",
    "Friesian": "Friesian horse black",
    "Mustang": "Mustang horse wild"
}

BASE_DIR = r"D:\animal_ai_project\datasets\horse\breed"

for breed_name, keyword in breeds.items():
    save_dir = os.path.join(BASE_DIR, breed_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Downloading {breed_name} images...")

    crawler = BingImageCrawler(storage={'root_dir': save_dir})
    crawler.crawl(
        keyword=keyword,
        max_num=100
    )

print("✅ All breeds downloaded successfully")