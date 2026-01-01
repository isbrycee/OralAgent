import requests
from bs4 import BeautifulSoup
import time
import json
from tqdm import tqdm


def get_response(url):
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.54"
    }
    return requests.get(url, headers=headers)

def get_case_numbers_from_page(page):
    url = f"https://www.eurorad.org/advanced-search?sort_by=published_at&sort_order=ASC&page={page}&filter%5B0%5D=section%3A40"

    # Remove proxy usage since it's likely triggering the protection
    response = get_response(url)
    print(response.text)

    soup = BeautifulSoup(response.text, "html.parser")
    spans = soup.find_all("span", class_="case__number small")

    # Remove '#' from the span text and strip extra whitespace
    numbers = [span.text.strip().replace("#", "").strip() for span in spans]
    return numbers


def main():
    total_pages = 107  # Pages 0 through 106
    all_numbers = []

    for page in tqdm(range(total_pages)):
        numbers = get_case_numbers_from_page(page)
        all_numbers.extend(numbers)

        if page != total_pages - 1 and len(numbers) != 9:
            print(f"Warning: Page {page} returned {len(numbers)} cases instead of 9")

        # Be kind to the server – avoid hitting it too fast
        time.sleep(1)
        break

    with open('case_numbers.json', 'w') as f:
        json.dump(all_numbers, f)

    print(f"Saved {len(all_numbers)} case numbers to case_numbers.json")


if __name__ == "__main__":
    main()
