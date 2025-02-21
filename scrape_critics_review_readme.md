
# Scrape Critics Review README

This project allows you to fetch and scrape critic reviews for various games from the Metacritic API. It takes game titles from an existing CSV file, fetches platform-specific reviews (such as critic scores and review counts), and saves the extracted data into a new CSV file.

## Features

- **Fetch Game Data**: Fetches game details from the Metacritic API based on the game title.
- **Platform Data Extraction**: Extracts platform-specific data, including critic scores, review counts, and the number of positive, neutral, and negative reviews.
- **CSV Output**: Saves the fetched data, including platform details, to a new CSV file for further analysis or reporting.

## Prerequisites

- **Python 3.x** (preferably the latest version).
- **Libraries**:
  - `requests`: To make API requests to the Metacritic API.
  - `csv`: To read and write data to CSV files.

You can install the required libraries using `pip`:
```bash
pip install requests
```

## Files and Directories

- **Metacritic_Games1.csv**: The existing CSV file containing a list of game titles.
- **Metacritic_Games_with_Platforms.csv**: The new CSV file where the game data (with platform details) will be saved.
- **scrape_critics_review.py**: The Python script to scrape data and save it to the new CSV file.

## How to Use

### 1. Prepare the Input CSV File

Ensure the input CSV file (`Metacritic_Games1.csv`) has the following structure:

```csv
Title, Game ID, Slug, Platform Name, Platform Slug, Score, Review Count, Positive Count, Neutral Count, Negative Count, URL
The Witcher 3, 12345, the-witcher-3, PC, pc, 85, 1200, 800, 300, 100, https://www.metacritic.com/game/pc/the-witcher-3
Cyberpunk 2077, 67890, cyberpunk-2077, PS4, ps4, 80, 500, 300, 150, 50, https://www.metacritic.com/game/playstation-4/cyberpunk-2077
```

The game titles should be in the first column, and other data such as `Platform`, `Platform Slug`, etc., will be populated dynamically.

### 2. Configure the Script

- **Set the API Key**: Set your Metacritic API key by updating the `api_key` variable in the script:
  ```python
  api_key = "your_api_key_here"
  ```

- **Set File Paths**: Ensure that the file paths for `Metacritic_Games1.csv` (input) and `Metacritic_Games_with_Platforms.csv` (output) are correct. You can adjust these file paths according to your directory structure.

  ```python
  existing_csv_file_path = "path_to_input_file/Metacritic_Games1.csv"
  new_csv_file_path = "path_to_output_file/Metacritic_Games_with_Platforms.csv"
  ```

### 3. Run the Script

Run the script to fetch and save the data:
```bash
python scrape_critics_review.py
```

The script will:
- Read the game titles from `Metacritic_Games1.csv`.
- Fetch platform-specific review data for each game using the Metacritic API.
- Write the game title, platform name, platform slug, critic score, review counts, and other relevant data into `Metacritic_Games_with_Platforms.csv`.

### 4. Output File Structure

After running the script, your output file (`Metacritic_Games_with_Platforms.csv`) will contain the following columns:

```csv
Title, Game ID, Platform, Platform Slug, Score, Review Count, Positive Count, Neutral Count, Negative Count, URL
The Witcher 3, 12345, PC, pc, 85, 1200, 800, 300, 100, https://www.metacritic.com/game/pc/the-witcher-3
Cyberpunk 2077, 67890, PS4, ps4, 80, 500, 300, 150, 50, https://www.metacritic.com/game/playstation-4/cyberpunk-2077
Horizon Zero Dawn, 54321, PS4, ps4, 90, 900, 700, 150, 50, https://www.metacritic.com/game/ps4/horizon-zero-dawn
```

## Code Overview

### 1. `scrape_critics_review.py`

This Python script does the following:

- **Reads Game Titles**: Extracts game titles from an existing CSV file (`Metacritic_Games1.csv`).
- **Fetches Game Data**: Uses the Metacritic API to fetch the game’s critic reviews data for each platform.
- **Writes to New CSV**: Writes the game title, platform details, scores, and review counts to a new CSV file (`Metacritic_Games_with_Platforms.csv`).

### 2. Key Functions

#### `generate_url(game_title, platform, api_key)`

Generates a dynamic URL for fetching game data from the Metacritic API.

#### `fetch_and_save_game_data(game_title, api_key, csv_writer)`

Fetches game data and extracts platform information such as:
- **Platform Name**: The platform for the game (e.g., PC, PS4).
- **Platform Slug**: Slugified platform name (e.g., pc, ps4).
- **Score**: Critic score for that platform.
- **Review Count**: The number of reviews.
- **Positive/Neutral/Negative Counts**: Breakdown of reviews by type.
- **URL**: URL to the game’s critic review page.

The function then writes the data to the CSV file.

### 3. Example of CSV Writing

The script writes data in this structure to the output CSV file:

```csv
Title, Game ID, Platform, Platform Slug, Score, Review Count, Positive Count, Neutral Count, Negative Count, URL
The Witcher 3, 12345, PC, pc, 85, 1200, 800, 300, 100, https://www.metacritic.com/game/pc/the-witcher-3
Cyberpunk 2077, 67890, PS4, ps4, 80, 500, 300, 150, 50, https://www.metacritic.com/game/ps4/cyberpunk-2077
```

## Troubleshooting

- **API Errors**: If the API request fails (non-200 status code), the script will print a failure message for that specific game title.
- **CSV File Errors**: Ensure the input CSV file is properly formatted with game titles in the first column.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
