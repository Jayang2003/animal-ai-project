import shutil
from pathlib import Path

from icrawler.builtin import BingImageCrawler
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = Path(r"D:\animal_ai_project\datasets\dog\age\labrador")
REJECTED_DIR = BASE_DIR / "_rejected"

TARGET_IMAGES_PER_CLASS = 120
MAX_IMAGES_PER_QUERY = 60
MIN_WIDTH = 256
MIN_HEIGHT = 256
MIN_FILE_SIZE_BYTES = 15 * 1024
MAX_ASPECT_RATIO = 2.2
MIN_COLOR_VARIATION = 18
HASH_SIZE = 8

NEGATIVE_TERMS = (
    "-ai -generated -art -illustration -painting -drawing -cartoon "
    "-studio -portrait -model -fashion -dress -celebrity -person -people "
    "-man -woman -child -human -interview -redcarpet -event -football "
    "-soccer -basketball -wedding -selfie -instagram"
)

BING_FILTERS = {
    "type": "photo",
    "size": "large",
}

CLASSES = {
    "0_5_years": [
        f"labrador retriever puppy outdoor real photo {NEGATIVE_TERMS}",
        f"young labrador retriever playing outside real photo {NEGATIVE_TERMS}",
        f"1 year old labrador retriever dog outdoor real photo {NEGATIVE_TERMS}",
        f"3 year old labrador retriever candid outdoor photo {NEGATIVE_TERMS}",
        f"4 year old yellow labrador retriever park real photo {NEGATIVE_TERMS}",
    ],
    "5_10_years": [
        f"6 year old labrador retriever outdoor real photo {NEGATIVE_TERMS}",
        f"8 year old labrador retriever natural light real photo {NEGATIVE_TERMS}",
        f"adult labrador retriever walking outside candid photo {NEGATIVE_TERMS}",
        f"adult yellow labrador retriever park real photo {NEGATIVE_TERMS}",
        f"adult black labrador retriever outdoor real photo {NEGATIVE_TERMS}",
    ],
    "10_15_years": [
        f"10 year old labrador retriever real photo outdoor {NEGATIVE_TERMS}",
        f"12 year old senior labrador retriever real photo {NEGATIVE_TERMS}",
        f"14 year old old labrador retriever outdoor candid photo {NEGATIVE_TERMS}",
        f"senior yellow labrador retriever natural light real photo {NEGATIVE_TERMS}",
        f"old black labrador retriever outside real photo {NEGATIVE_TERMS}",
    ],
}


def average_hash(image: Image.Image, hash_size: int = HASH_SIZE) -> str:
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels)
    return "".join("1" if pixel >= avg else "0" for pixel in pixels)


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return sum(bit_a != bit_b for bit_a, bit_b in zip(hash_a, hash_b))


def ensure_dirs() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    for class_name in CLASSES:
        (BASE_DIR / class_name).mkdir(parents=True, exist_ok=True)
        (REJECTED_DIR / class_name).mkdir(parents=True, exist_ok=True)


def count_images(folder: Path) -> int:
    return sum(1 for file_path in folder.iterdir() if file_path.is_file())


def download_images(class_name: str, keywords: list[str]) -> None:
    save_path = BASE_DIR / class_name
    crawler = BingImageCrawler(
        storage={"root_dir": str(save_path)},
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=4,
    )

    for keyword in keywords:
        existing_count = count_images(save_path)
        if existing_count >= TARGET_IMAGES_PER_CLASS:
            print(f"[{class_name}] Target reached ({existing_count} images).")
            break

        remaining = TARGET_IMAGES_PER_CLASS - existing_count
        max_num = min(MAX_IMAGES_PER_QUERY, remaining)
        if max_num <= 0:
            break

        print(f"[{class_name}] Downloading up to {max_num} images for: {keyword}")
        crawler.crawl(keyword=keyword, max_num=max_num, filters=BING_FILTERS)


def reject_image(file_path: Path, class_name: str, reason: str) -> None:
    rejected_folder = REJECTED_DIR / class_name / reason
    rejected_folder.mkdir(parents=True, exist_ok=True)
    destination = rejected_folder / file_path.name

    counter = 1
    while destination.exists():
        destination = rejected_folder / f"{file_path.stem}_{counter}{file_path.suffix}"
        counter += 1

    shutil.move(str(file_path), str(destination))
    print(f"[{class_name}] Rejected {file_path.name}: {reason}")


def image_has_enough_detail(image: Image.Image) -> bool:
    small = image.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())

    max_spread = 0
    for channel in range(3):
        values = [pixel[channel] for pixel in pixels]
        spread = max(values) - min(values)
        max_spread = max(max_spread, spread)

    return max_spread >= MIN_COLOR_VARIATION


def validate_and_clean_class(class_name: str) -> None:
    folder = BASE_DIR / class_name
    hashes: list[tuple[str, Path]] = []

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue

        try:
            if file_path.stat().st_size < MIN_FILE_SIZE_BYTES:
                reject_image(file_path, class_name, "too_small_file")
                continue

            with Image.open(file_path) as image:
                image.load()
                width, height = image.size

                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    reject_image(file_path, class_name, "too_small_resolution")
                    continue

                aspect_ratio = max(width / height, height / width)
                if aspect_ratio > MAX_ASPECT_RATIO:
                    reject_image(file_path, class_name, "bad_aspect_ratio")
                    continue

                if not image_has_enough_detail(image):
                    reject_image(file_path, class_name, "low_detail")
                    continue

                current_hash = average_hash(image)

            is_duplicate = False
            for saved_hash, saved_path in hashes:
                if hamming_distance(current_hash, saved_hash) <= 5:
                    reject_image(file_path, class_name, f"duplicate_of_{saved_path.stem}")
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            hashes.append((current_hash, file_path))

        except (UnidentifiedImageError, OSError, ValueError):
            reject_image(file_path, class_name, "invalid_image")


def rename_images(class_name: str) -> None:
    folder = BASE_DIR / class_name
    valid_files = [file_path for file_path in sorted(folder.iterdir()) if file_path.is_file()]

    temp_paths = []
    for index, file_path in enumerate(valid_files, start=1):
        temp_path = folder / f"tmp_{index:04d}{file_path.suffix.lower()}"
        if file_path != temp_path:
            file_path.rename(temp_path)
        temp_paths.append(temp_path)

    for index, temp_path in enumerate(temp_paths, start=1):
        final_path = folder / f"{class_name}_{index:04d}{temp_path.suffix.lower()}"
        temp_path.rename(final_path)


def summarize() -> None:
    print("\nFinal dataset summary")
    print("-" * 40)
    for class_name in CLASSES:
        total = count_images(BASE_DIR / class_name)
        print(f"{class_name}: {total} clean images")


def main() -> None:
    ensure_dirs()

    for class_name, keywords in CLASSES.items():
        print(f"\nProcessing class: {class_name}")
        download_images(class_name, keywords)
        validate_and_clean_class(class_name)
        rename_images(class_name)

    summarize()
    print(f"\nRejected images were moved to: {REJECTED_DIR}")


if __name__ == "__main__":
    main()
