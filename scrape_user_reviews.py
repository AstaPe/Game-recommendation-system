import requests
import csv

# Define the API URL and parameters
base_url = "https://backend.metacritic.com/composer/metacritic/pages/games-user-reviews/{game_slug}/platform/{platform_slug}/web"
api_key = "1MOZgmNFxvmljaQR1X9KAij9Mo4xAY3u"

# Function to fetch API data
def fetch_api_data(api_key, base_url, game_slug, platform_slug):
    url = f"{base_url.format(game_slug=game_slug, platform_slug=platform_slug)}?filter=all&sort=date&apiKey={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        print(f"Failed to fetch data. Status Code: 404 for {game_slug} on platform {platform_slug}")
        return None
    else:
        print(f"Failed to fetch data. Status Code: {response.status_code} for {game_slug} on platform {platform_slug}")
        return None


# Function to fetch user reviews
def fetch_user_reviews(game_slug, platform_slug, api_key):
    data = fetch_api_data(api_key, base_url, game_slug, platform_slug)

    if data:
        # Extract reviews from the 'components' list (review data)
        components = data.get("components", [])
        if len(components) > 1:
            items = components[1].get("data", {}).get("item", [])

            if isinstance(items, list) and items:
                # Extract first review item (summary)
                item = items[0]
                return extract_item_data(item)
            elif isinstance(items, dict):
                return extract_item_data(items)
    return {
        "User Score": "N/A",
        "User Review Count": "N/A",
        "User Positive Count": "N/A",
        "User Neutral Count": "N/A",
        "User Negative Count": "N/A",
        "User Sentiment": "N/A",
        "User Review URL": "N/A"
    }


# Function to extract item data
def extract_item_data(item):
    return {
        "User Score": item.get("score", "N/A"),
        "User Review Count": item.get("reviewCount", "N/A"),
        "User Positive Count": item.get("positiveCount", "N/A"),
        "User Neutral Count": item.get("neutralCount", "N/A"),
        "User Negative Count": item.get("negativeCount", "N/A"),
        "User Sentiment": item.get("sentiment", "N/A"),
        "User Review URL": item.get("url", "N/A")
    }


# Function to handle null/empty cells in the output
def handle_null_values(row):
    for key, value in row.items():
        if value in [None, "", "null"]:  # Null/empty check
            row[key] = "N/A"
    return row


# Function to save extracted data to a CSV file
def save_to_csv(data, filename):
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"Data saved to {filename}")
    except PermissionError as e:
        print(f"Permission error: {e}")


# Main function
def main():
    input_file = r'C:\Users\astap\OneDrive\Documents\Sprendimai\Baigiamasis\Metacritic_Games_with_Platforms.csv'
    output_file = r'C:\Users\astap\OneDrive\Documents\Sprendimai\Baigiamasis\updated_game_reviews.csv'

    updated_data = []  # To store updated data with user review info

    try:
        with open(input_file, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            rows = list(reader)  # Process all rows in the dataset

        for row in rows:
            game_slug = row.get("Slug", "").strip()
            platform_slug = row.get("Platform Slug", "").strip()

            if game_slug and platform_slug:
                print(f"Processing: Game Slug: {game_slug}, Platform Slug: {platform_slug}")
                user_review_data = fetch_user_reviews(game_slug, platform_slug, api_key)

                # Merge original row data with user review data
                merged_data = {**row, **user_review_data}
                updated_data.append(handle_null_values(merged_data))  # Ensure null values are replaced
            else:
                print(f"Skipping row with missing game/platform slug: {row}")

        if updated_data:
            save_to_csv(updated_data, output_file)
        else:
            print("No valid data to save.")

    except FileNotFoundError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()