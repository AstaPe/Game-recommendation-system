import requests
import csv

# Define the base URL and API key
base_url = "https://backend.metacritic.com/composer/metacritic/pages/games-critic-reviews/{game_slug}/platform/{platform_slug}/web"
api_key = "1MOZgmNFxvmljaQR1X9KAij9Mo4xAY3u"


# Function to generate the URL dynamically based on game title and platform
def generate_url(game_title, platform, api_key):
    game_slug = game_title.replace(" ", "-").lower()  # Transform spaces to hyphens and lowercase
    platform_slug = platform.replace(" ", "-").lower()  # Transform spaces to hyphens and lowercase
    url = base_url.format(game_slug=game_slug, platform_slug=platform_slug)
    return f"{url}?filter=all&sort=score&apiKey={api_key}"


# Function to fetch and save game data
def fetch_and_save_game_data(game_title, api_key, csv_writer):
    # Get the game details URL
    game_slug = game_title.replace(" ", "-").lower()  # Transform spaces to hyphens and lowercase
    game_url = f"https://backend.metacritic.com/composer/metacritic/pages/games-critic-reviews/{game_slug}/web?apiKey={api_key}"

    response = requests.get(game_url)

    if response.status_code == 200:
        data = response.json()
        game_info = data.get('data', {})

        title = game_info.get('title', 'N/A')
        game_id = game_info.get('id', 'N/A')
        slug = game_info.get('slug', 'N/A')
        platforms = game_info.get('platforms', [])

        # Loop through each platform and extract the necessary details
        for platform_data in platforms:
            platform_name = platform_data.get('name', 'N/A')
            platform_slug = platform_name.replace(" ", "-").lower()  # Create slug from platform name
            critic_score_summary = platform_data.get('criticScoreSummary', {})
            score = critic_score_summary.get('score', 'N/A')
            review_count = critic_score_summary.get('reviewCount', 'N/A')
            positive_count = critic_score_summary.get('positiveCount', 'N/A')
            neutral_count = critic_score_summary.get('neutralCount', 'N/A')
            negative_count = critic_score_summary.get('negativeCount', 'N/A')
            url_path = critic_score_summary.get('url', 'N/A')

            # Write the data to the CSV file, now including the Platform column
            csv_writer.writerow(
                [title, game_id, platform_name, platform_slug, score, review_count, positive_count, neutral_count,
                 negative_count, url_path])
    else:
        print(f"Failed to fetch data for {game_title}, status code: {response.status_code}")


# Read game titles from the existing CSV file (Metacritic_Games1.csv)
existing_csv_file_path = r"C:\Users\astap\OneDrive\Documents\Sprendimai\Baigiamasis\Metacritic_Games1.csv"
game_titles = []

with open(existing_csv_file_path, mode='r', newline='', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header row
    for row in csv_reader:
        game_titles.append(row[0])  # Assuming the game title is in the first column

# Specify the file path for the new CSV where the data with platforms will be saved
new_csv_file_path = r"C:\Users\astap\OneDrive\Documents\Sprendimai\Baigiamasis\Metacritic_Games_with_Platforms.csv"

# Open the new CSV file in write mode
with open(new_csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)

    # Write the header row with the new 'Platform' column and 'platform_slug'
    csv_writer.writerow(
        ["Title", "Game ID", "Platform", "Platform Slug", "Score", "Review Count", "Positive Count", "Neutral Count",
         "Negative Count", "URL"])

    # Fetch data for each game from the existing CSV list
    for game_title in game_titles:
        print(f"Fetching data for {game_title}...")
        fetch_and_save_game_data(game_title, api_key, csv_writer)

print(f"Data saved to {new_csv_file_path}")
