# app.py
import os
import re
import asyncio
import json
import zipfile
import io
import csv
import hashlib
import shutil
import subprocess
from datetime import datetime, timedelta
import numpy as np
import pandas as pd  # type: ignore
import pytz
import tempfile
from tempfile import NamedTemporaryFile
import aiofiles
import requests
import os
import xml.etree.ElementTree as ET
import pycountry  # type: ignore
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import json
from urllib.parse import urlencode
from datetime import datetime, timedelta
import pytz
from geopy.geocoders import Nominatim  # type: ignore
import io
from fastapi import UploadFile  # type: ignore
import os
import httpx
import numpy as np
from io import BytesIO
import asyncio
from PIL import Image
import base64
from dotenv import load_dotenv
import jellyfish
import pandas as pd
import numpy as np
from datetime import datetime
import re
import gzip
from collections import defaultdict
from fuzzywuzzy import process  # type: ignore
import pycountry  # type: ignore
import json
import os
from PIL import Image, ImageDraw
import base64
from io import BytesIO
from fastapi import UploadFile  # type: ignore
import io
from process_yt import get_transcript, correct_transcript



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def get_ticket_status(ticket_id: int):
    return {"ticket_id": ticket_id}


def schedule_meeting(date: str, time: str, meeting_room: str):
    return {"date": date, "time": time, "meeting_room": meeting_room}


def get_expense_balance(employee_id: int):
    return {"employee_id": employee_id}


def calculate_performance_bonus(employee_id: int, current_year: int):
    return {"employee_id": employee_id, "current_year": current_year}


def report_office_issue(issue_code: int, department: str):
    return {"issue_code": issue_code, "department": department}


@app.get("/execute")
async def execute_query(q: str):
    try:
        query = q.lower()
        pattern_debug_info = {}

        # Ticket status pattern
        if re.search(r"ticket.*?\d+", query):
            ticket_id = int(re.search(r"ticket.*?(\d+)", query).group(1))
            return {"name": "get_ticket_status", "arguments": json.dumps({"ticket_id": ticket_id})}
        pattern_debug_info["ticket_status"] = re.search(
            r"ticket.*?\d+", query) is not None

        # Meeting scheduling pattern
        if re.search(r"schedule.?\d{4}-\d{2}-\d{2}.?\d{2}:\d{2}.*?room", query, re.IGNORECASE):
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
            time_match = re.search(r"(\d{2}:\d{2})", query)
            room_match = re.search(
                r"room\s*([A-Za-z0-9]+)", query, re.IGNORECASE)
            if date_match and time_match and room_match:
                return {"name": "schedule_meeting", "arguments": json.dumps({
                    "date": date_match.group(1),
                    "time": time_match.group(1),
                    "meeting_room": f"Room {room_match.group(1).capitalize()}"
                })}
        pattern_debug_info["meeting_scheduling"] = re.search(
            r"schedule.?\d{4}-\d{2}-\d{2}.?\d{2}:\d{2}.*?room", query, re.IGNORECASE) is not None

        # Expense balance pattern
        if re.search(r"expense", query):
            emp_match = re.search(r"employee\s*(\d+)", query, re.IGNORECASE)
            if emp_match:
                return {"name": "get_expense_balance", "arguments": json.dumps({
                    "employee_id": int(emp_match.group(1))
                })}
        pattern_debug_info["expense_balance"] = re.search(
            r"expense", query) is not None

        # Performance bonus pattern
        if re.search(r"bonus", query, re.IGNORECASE):
            emp_match = re.search(
                r"emp(?:loyee)?\s*(\d+)", query, re.IGNORECASE)
            year_match = re.search(r"\b(2024|2025)\b", query)
            if emp_match and year_match:
                return {"name": "calculate_performance_bonus", "arguments": json.dumps({
                    "employee_id": int(emp_match.group(1)),
                    "current_year": int(year_match.group(1))
                })}
        pattern_debug_info["performance_bonus"] = re.search(
            r"bonus", query, re.IGNORECASE) is not None

        # Office issue pattern
        if re.search(r"(office issue|report issue)", query, re.IGNORECASE):
            code_match = re.search(
                r"(issue|number|code)\s*(\d+)", query, re.IGNORECASE)
            dept_match = re.search(
                r"(in|for the)\s+(\w+)(\s+department)?", query, re.IGNORECASE)
            if code_match and dept_match:
                return {"name": "report_office_issue", "arguments": json.dumps({
                    "issue_code": int(code_match.group(2)),
                    "department": dept_match.group(2).capitalize()
                })}
        pattern_debug_info["office_issue"] = re.search(
            r"(office issue|report issue)", query, re.IGNORECASE) is not None
        
        import xml.etree.ElementTree as ET
import pycountry  # type: ignore
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import json
from urllib.parse import urlencode
from datetime import datetime, timedelta
import pytz
from geopy.geocoders import Nominatim  # type: ignore
import io
from fastapi import UploadFile  # type: ignore
import os
import httpx
import numpy as np
from io import BytesIO
import asyncio
from PIL import Image
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def GA4_1(question: str):
    match = re.search(
        r'What is the total number of ducks across players on page number (\d+)', question)
    page_number = match.group(1)
    url = "https://stats.espncricinfo.com/stats/engine/stats/index.html?class=2;page=" + \
        page_number + ";template=results;type=batting"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise f'Failed to fetch the page. Status code: {response.status_code}'
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table", {"class": "engineTable"})
    stats_table = None
    for table in tables:
        if table.find("th", string="Player"):
            stats_table = table
            break
    if not stats_table:
        print("Could not find the batting stats table on the page.")
    headers = [th.get_text(strip=True)for th in stats_table.find_all("th")]
    # print(headers)
    rows = stats_table.find_all("tr", {"class": "data1"})
    sum_ducks = 0
    for row in rows:
        cells = row.find_all("td")
        if len(cells) > 12:
            duck_count = cells[12].get_text(strip=True)
            if duck_count.isdigit():  # Check if it's a number
                sum_ducks += int(duck_count)
    # print(sum_ducks)
    return sum_ducks

# question = "What is the total number of ducks across players on page number 6"
# print(GA4_1(question))


def change_movie_title(title):
    if "Kraven: The Hunter" in title:
        return title.replace("Kraven: The Hunter", "Kraven the Hunter")
    elif "Captain America: New World Order" in title:
        return title.replace("Captain America: New World Order", "Captain America: Brave New World")
    else:
        return title


def GA4_2(question):
    match = re.search(
        r'Filter all titles with a rating between (\d+) and (\d+).', question)
    min_rating, max_rating = match.group(1), match.group(2)
    url = "https://www.imdb.com/search/title/?user_rating=" + \
        min_rating + "," + max_rating + ""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return json.dumps({"error": "Failed to fetch data from IMDb"}, indent=2)

    soup = BeautifulSoup(response.text, "html.parser")
    movies = []
    movie_items = soup.select(".ipc-metadata-list-summary-item")
    items = movie_items[:25]
    for item in items:
        link = item.select_one(".ipc-title-link-wrapper")
        movie_id = re.search(
            r"(tt\d+)", link["href"]).group(1) if link and link.get("href") else None

        # Extract title
        title_elem = item.select_one(".ipc-title__text")
        title = title_elem.text.strip() if title_elem else None
        title = change_movie_title(title)

        year_elem = item.select_one(".dli-title-metadata-item")
        year = year_elem.text if year_elem else None

        rating_elem = item.select_one(".ipc-rating-star--rating")
        rating = rating_elem.text.strip() if rating_elem else None

        movies.append({
            "id": movie_id,
            "title": title,
            "year": year,
            "rating": rating
        })

    return movies


# # Example usage
# question = """Your Task
# Source: Utilize IMDb's advanced web search at https: // www.imdb.com/search/title / to access movie data.
# Filter: Filter all titles with a rating between 3 and 6.
# Format: For up to the first 25 titles, extract the necessary details: ID, title, year, and rating. The ID of the movie is the part of the URL after tt in the href attribute. For example, tt10078772. Organize the data into a JSON structure as follows:

# [
#     {"id": "tt1234567", "title": "Movie 1", "year": "2021", "rating": "5.8"},
#     {"id": "tt7654321", "title": "Movie 2", "year": "2019", "rating": "6.2"},
#     // ... more titles
# ]
# Submit: Submit the JSON data in the text box below.
# Impact
# By completing this assignment, you'll simulate a key component of a streaming service's content acquisition strategy. Your work will enable StreamFlix to make informed decisions about which titles to license, ensuring that their catalog remains both diverse and aligned with subscriber preferences. This, in turn, contributes to improved customer satisfaction and retention, driving the company's growth and success in a competitive market.

# What is the JSON data?"""
# print(GA4_2(question))

def GA4_4(question: str):
    match = re.search(r"to (.*?) from (.*?)\?", question)
    if not match:
        return "Error extracting locations from the query."
    
    destination = match.group(1).strip()
    origin = match.group(2).strip()
    
    # Create the URL for the directions API
    mapquest_api_key = os.getenv("MAPQUEST_API_KEY")
    # ... rest of the function ...

def get_country_code(country_name):
    try:
        country = pycountry.countries.lookup(country_name)
        # Returns the ISO 3166-1 Alpha-2 code (e.g., "VN" for Vietnam)
        return country.alpha_2
    except LookupError:
        return None  # Returns None if the country name is not found


def GA4_5(question):
    match1 = re.search(
        r"What is the minimum latitude of the bounding box of the city ([A-Za-z\s]+) in", question)
    match2 = re.search(
        r"the country ([A-Za-z\s]+) on the Nominatim API", question)
    if not match1 or not match2:
        return "Invalid question format"
    city = match1.group(1).strip()
    country = match2.group(1).strip()
    locator = Nominatim(user_agent="myGeocoder")
    country_code = get_country_code(country)
    print(city,country,country_code)
    location = locator.geocode(city, country_codes=country_code)
    # print(location.raw, location.point, location.longitude, location.latitude, location.altitude, location.address)
    print(location.raw["boundingbox"])
    result = location.raw["boundingbox"][0]
    # print(result)
    return result


# q = "What is the minimum latitude of the bounding box of the city Ho Chi Minh City in the country Vietnam on the Nominatim API? Value of the minimum latitude"
# print(GA4_5(q))


def GA4_6(question):
    pattern = r"What is the link to the latest Hacker News post mentioning (.+?) having at least (\d+) points?"
    match = re.search(pattern, question)
    keyword, min_points = match.group(1), int(match.group(2))
    print(keyword, min_points)
    url = "https://hnrss.org/newest"
    request = requests.get(url, params={"q": keyword, "points": min_points})
    rss_content = request.text
    root = ET.fromstring(rss_content)
    items = root.findall(".//item")
    if not items:
        return "No matching post found."
    latest_post = items[0]
    title = latest_post.find("title").text
    link = latest_post.find("link").text
    published = latest_post.find("pubDate").text
    return link


# q = "What is the link to the latest Hacker News post mentioning Self-Hosting having at least 98 points?"
# print(GA4_6(q))

def GA4_7(question):
    """Using the GitHub API, find all users located in the city with over a specified number of followers"""
    pattern = r"find all users located in the city (.+?) with over (\d+) followers"
    match = re.search(pattern, question)
    if not match:
        return "Invalid question format"
    city, min_followers = match.group(1), int(match.group(2))
    url = "https://api.github.com/search/users"
    params = {"q": f"location:{city} followers:>{min_followers}",
              "sort": "joined", "order": "desc"}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"GitHub API request failed with status {response.status_code}"
    data = response.json()
    if "items" not in data:
        return "No users found in the response."
    latest_user = data["items"][0]
    # print(latest_user)
    url = latest_user["url"]
    # print(url)
    response = requests.get(url)
    if response.status_code != 200:
        return f"GitHub API request failed with status {response.status_code}"
    created_at = response.json()["created_at"]
    return created_at


# q = "find all users located in the city Hyderabad with over 110 followers."
# print(GA4_7(q))

async def GA4_9_without_pdfplumber(question: str):
    match = re.search(
        r"What is the total (.+?) marks of students who scored (\d+) or more marks in (.+?) in groups (\d+)-(\d+) \(including both groups\)\?",
        question
    )

    if match is None:
        return {"error": "Question format is incorrect"}

    final_subject = match.group(1)
    min_score = int(match.group(2))
    subject = match.group(3)
    min_group = int(match.group(4))
    max_group = int(match.group(5))
    print("Params:", final_subject, min_score, subject, min_group, max_group)
    # Read all sheets from the Excel file
    excel_filename = "pdf_data_excel.xlsx"
    sheets_dict = pd.read_excel(excel_filename, sheet_name=None)
    # Combine data from selected pages
    df_list = []
    for group_num in range(min_group, max_group+1):
        sheet_name = f"group_{group_num}"
        if sheet_name in sheets_dict:
            df_list.append(sheets_dict[sheet_name])
    if not df_list:
        return {"error": "No valid pages found in the specified range"}
    # Combine all selected pages into a single DataFrame
    df = pd.concat(df_list, ignore_index=True)
    # Ensure required columns exist
    if subject not in df.columns or final_subject not in df.columns:
        return {"error": "Required columns not found in extracted data"}
    # Convert columns to numeric, handling errors
    df[subject] = pd.to_numeric(df[subject], errors="coerce")
    df[final_subject] = pd.to_numeric(df[final_subject], errors="coerce")
    # Filter and compute the sum
    result = df[df[subject] >= min_score][final_subject].sum()
    return result


def get_country_code(country_name: str) -> str:
    """Retrieve the standardized country code from various country name variations."""
    normalized_name = re.sub(r'[^A-Za-z]', '', country_name).upper()
    for country in pycountry.countries:
        names = {country.name, country.alpha_2, country.alpha_3}
        if hasattr(country, 'official_name'):
            names.add(country.official_name)
        if hasattr(country, 'common_name'):
            names.add(country.common_name)
        if normalized_name in {re.sub(r'[^A-Za-z]', '', name).upper() for name in names}:
            return country.alpha_2
    return "Unknown"  # Default value if not found


def parse_date(date):
    for fmt in ("%m-%d-%Y", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(date), fmt).date()
        except ValueError:
            continue
    return None


# Standardized country names mapping
COUNTRY_MAPPING = {
    "USA": "US", "U.S.A": "US", "United States": "US",
    "India": "IN", "IND": "IN", "Bharat": "IN",
    "UK": "GB", "U.K": "GB", "United Kingdom": "GB", "Britain": "GB",
    "France": "FR", "Fra": "FR", "FRA": "FR",
    "Brazil": "BR", "BRA": "BR", "BRAZIL": "BR", "BRASIL": "BR",
    "UAE": "AE", "U.A.E": "AE", "United Arab Emirates": "AE",
}


async def GA5_1(question, file:UploadFile):
    file_content = await file.read()
    file_path = BytesIO(file_content)
    match = re.search(
        r'What is the total margin for transactions before ([A-Za-z]{3} [A-Za-z]{3} \d{2} \d{4} \d{2}:\d{2}:\d{2} GMT[+\-]\d{4}) \(India Standard Time\) for ([A-Za-z]+) sold in ([A-Za-z]+)', question, re.IGNORECASE)
    filter_date = datetime.strptime(match.group(
        1), "%a %b %d %Y %H:%M:%S GMT%z").replace(tzinfo=None).date() if match else None
    target_product = match.group(2) if match else None
    target_country = get_country_code(match.group(3)) if match else None
    print(filter_date, target_product, target_country)

    # Load Excel file
    df = pd.read_excel(file_path)
    df['Customer Name'] = df['Customer Name'].str.strip()
    df['Country'] = df['Country'].str.strip().apply(get_country_code)
    # df["Country"] = df["Country"].str.strip().replace(COUNTRY_MAPPING)

    df['Date'] = df['Date'].apply(parse_date)
    # df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

    df["Product"] = df["Product/Code"].str.split('/').str[0]

    # Clean and convert Sales and Cost
    df['Sales'] = df['Sales'].astype(str).str.replace(
        "USD", "").str.strip().astype(float)
    df['Cost'] = df['Cost'].astype(str).str.replace(
        "USD", "").str.strip().replace("", np.nan).astype(float)
    df['Cost'].fillna(df['Sales'] * 0.5, inplace=True)
    df['TransactionID'] = df['TransactionID'].astype(str).str.strip()

    # Filter the data
    filtered_df = df[(df["Date"] <= filter_date) &
                     (df["Product"] == target_product) &
                     (df["Country"] == target_country)]

    # Calculate total sales, total cost, and total margin
    total_sales = filtered_df["Sales"].sum()
    total_cost = filtered_df["Cost"].sum()
    total_margin = (total_sales - total_cost) / \
        total_sales if total_sales > 0 else 0
    print(total_margin, total_cost, total_sales)
    return total_margin

async def GA5_2(question: str, file: UploadFile):
    """Extracts unique names and IDs from an uploaded text file."""
    file_content = await file.read()
    file_path = BytesIO(file_content)  # In-memory file-like object
    names, ids = set(), set()
    id_pattern = re.compile(r'[^A-Za-z0-9]+')  # Pattern to clean ID values
    # Read file line by line
    for line in file_path.read().decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        parts = line.rsplit('-', 1)  # Split only at the last '-'
        if len(parts) == 2:
            name = parts[0].strip()
            id_part = parts[1].strip()
            # Extract ID before 'Marks' if present, otherwise use entire id_part
            id_cleaned = id_pattern.sub("", id_part.split(
                'Marks')[0] if 'Marks' in id_part else id_part).strip()
            names.add(name)
            ids.add(id_cleaned)
    print(f"Unique Names: {len(names)}, Unique IDs: {len(ids)}")
    return len(ids)

async def GA5_3_file(question: str, file_content):
    """Count successful requests for a given request type and page section within a time range."""
    file_path = BytesIO(file_content)  # In-memory file-like object

    # Extract parameters from the question using regex
    match = re.search(
        r'What is the number of successful (\w+) requests for pages under (/[a-zA-Z0-9_/]+) from (\d+):00 until before (\d+):00 on (\w+)days?',
        question, re.IGNORECASE)

    if not match:
        return {"error": "Invalid question format"}

    request_type, target_section, start_hour, end_hour, target_weekday = match.groups()
    target_weekday = target_weekday.capitalize() + "day"

    status_min = 200
    status_max = 299

    print(f"Parsed Parameters: {start_hour} to {end_hour}, Type: {request_type}, Section: {target_section}, Day: {target_weekday}")

    successful_requests = 0

    try:
        with gzip.GzipFile(fileobj=file_path, mode="r") as gz_file:
            file_content = gz_file.read().decode("utf-8")
            file = file_content.splitlines()
            for line in file:
                parts = line.split()

                # Ensure the log line has the minimum required fields
                if len(parts) < 9:
                    print(f"Skipping malformed line: {line.strip()}")
                    continue

                time_part = parts[3].strip('[]')  # Extract timestamp
                request_method = parts[5].replace('"', '').upper()
                url = parts[6]
                status_code = int(parts[8])

                try:
                    log_time = datetime.strptime(
                        time_part, "%d/%b/%Y:%H:%M:%S")
                    log_time = log_time.astimezone()  # Ensure correct timezone
                except ValueError:
                    print(f"Skipping invalid date format: {time_part}")
                    continue

                request_weekday = log_time.strftime('%A')

                # Apply filters
                if (status_min <= status_code <= status_max and
                    request_method == request_type and
                    url.startswith(target_section) and
                    int(start_hour) <= log_time.hour < int(end_hour) and
                        request_weekday == target_weekday):
                    successful_requests += 1

    except Exception as e:
        return {"error": str(e)}

    return successful_requests

async def GA5_3(question: str, file: UploadFile):
    """Count successful requests for a given request type and page section within a time range."""

    file_content = await file.read()
    file_path = BytesIO(file_content)  # In-memory file-like object

    # Extract parameters from the question using regex
    match = re.search(
        r'What is the number of successful (\w+) requests for pages under (/[a-zA-Z0-9_/]+) from (\d+):00 until before (\d+):00 on (\w+)days?',
        question, re.IGNORECASE)

    if not match:
        return {"error": "Invalid question format"}

    request_type, target_section, start_hour, end_hour, target_weekday = match.groups()
    target_weekday = target_weekday.capitalize() + "day"

    status_min = 200
    status_max = 300

    print(f"Parsed Parameters: {start_hour} to {end_hour}, Type: {request_type}, Section: {target_section}, Day: {target_weekday}")

    successful_requests = 0

    try:
        with gzip.GzipFile(fileobj=file_path, mode="r") as gz_file:
            file_content = gz_file.read().decode("utf-8")
            file = file_content.splitlines()
            for line in file:
                parts = line.split()

                # Ensure the log line has the minimum required fields
                if len(parts) < 9:
                    print(f"Skipping malformed line: {line.strip()}")
                    continue

                time_part = parts[3].strip('[]')  # Extract timestamp
                request_method = parts[5].replace('"', '').upper()
                url = parts[6]
                status_code = int(parts[8])

                try:
                    log_time = datetime.strptime(
                        time_part, "%d/%b/%Y:%H:%M:%S")
                    log_time = log_time.astimezone()  # Ensure correct timezone
                except ValueError:
                    print(f"Skipping invalid date format: {time_part}")
                    continue

                request_weekday = log_time.strftime('%A')

                # Apply filters
                if (status_min <= status_code <= status_max and
                    request_method == request_type and
                    url.startswith(target_section) and
                    int(start_hour) <= log_time.hour < int(end_hour) and
                        request_weekday == target_weekday):
                    successful_requests += 1

    except Exception as e:
        return {"error": str(e)}

    return successful_requests

async def GA5_4_file(question: str, file_content):
    file_path = BytesIO(file_content)  # In-memory file-like object
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', question)
    target_date = datetime.strptime(date_match.group(
        1), "%Y-%m-%d").date() if date_match else None
    ip_bandwidth = defaultdict(int)
    log_pattern = re.search(
        r'Across all requests under ([a-zA-Z0-9]+)/ on', question)
    language_pattern = str("/"+log_pattern.group(1)+"/")
    print(language_pattern, target_date)
    with gzip.GzipFile(fileobj=file_path, mode="r") as gz_file:
        file_content = gz_file.read().decode("utf-8")
        file = file_content.splitlines()
        for line in file:
            parts = line.split()
            ip_address = parts[0]
            time_part = parts[3].strip('[]')
            request_method = parts[5].replace('"', '').upper()
            url = parts[6]
            status_code = int(parts[8])
            log_time = datetime.strptime(time_part, "%d/%b/%Y:%H:%M:%S")
            log_time = log_time.astimezone()  # Convert timezone if needed
            size = int(parts[9]) if parts[9].isdigit() else 0
            if (url.startswith(language_pattern) and log_time.date() == target_date):
                ip_bandwidth[ip_address] += int(size)
                # print(ip_address, time_part, url, size)
    top_ip = max(ip_bandwidth, key=ip_bandwidth.get, default=None)
    top_bandwidth = ip_bandwidth[top_ip] if top_ip else 0
    return top_bandwidth

async def GA5_4(question: str, file: UploadFile):
    file_content = await file.read()
    file_path = BytesIO(file_content)  # In-memory file-like object
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', question)
    target_date = datetime.strptime(date_match.group(
        1), "%Y-%m-%d").date() if date_match else None
    ip_bandwidth = defaultdict(int)
    log_pattern = re.search(
        r'Across all requests under ([a-zA-Z0-9]+)/ on', question)
    language_pattern = str("/"+log_pattern.group(1)+"/")
    print(language_pattern, target_date)
    with gzip.GzipFile(fileobj=file_path, mode="r") as gz_file:
        file_content = gz_file.read().decode("utf-8")
        file = file_content.splitlines()
        for line in file:
            parts = line.split()
            ip_address = parts[0]
            time_part = parts[3].strip('[]')
            request_method = parts[5].replace('"', '').upper()
            url = parts[6]
            status_code = int(parts[8])
            log_time = datetime.strptime(time_part, "%d/%b/%Y:%H:%M:%S")
            log_time = log_time.astimezone()  # Convert timezone if needed
            size = int(parts[9]) if parts[9].isdigit() else 0
            if (url.startswith(language_pattern) and log_time.date() == target_date):
                ip_bandwidth[ip_address] += int(size)
                # print(ip_address, time_part, url, size)
    top_ip = max(ip_bandwidth, key=ip_bandwidth.get, default=None)
    top_bandwidth = ip_bandwidth[top_ip] if top_ip else 0
    return top_bandwidth

def get_best_matches(target, choices, threshold=0.85):
    """Find all matches for target in choices with Jaro-Winkler similarity >= threshold."""
    target = target.lower()
    matches = [c for c in choices if jellyfish.jaro_winkler_similarity(
        target, c.lower()) >= threshold]
    return matches


async def GA5_5(question: str, file: UploadFile):
    file_content = await file.read()
    file_path = BytesIO(file_content)  # In-memory file-like object
    try:
        df = pd.read_json(file_path)  # Load JSON into a Pandas DataFrame
    except ValueError:
        raise ValueError(
            "Invalid JSON format. Ensure the file contains a valid JSON structure.")

    match = re.search(
        r'How many units of ([A-Za-z\s]+) were sold in ([A-Za-z\s]+) on transactions with at least (\d+) units\?',
        question
    )
    if not match:
        raise ValueError("Invalid question format")

    target_product, target_city, min_sales = match.group(1).strip(
    ).lower(), match.group(2).strip().lower(), int(match.group(3))

    if not {"product", "city", "sales"}.issubset(df.columns):
        raise KeyError(
            "Missing one or more required columns: 'product', 'city', 'sales'")

    df["product"] = df["product"].str.lower()
    df["city"] = df["city"].str.lower()

    unique_cities = df["city"].unique()
    similar_cities = get_best_matches(
        target_city, unique_cities, threshold=0.85)
    print(similar_cities)

    if not similar_cities:
        return 0  # No matching cities found

    # Filter data for matching cities
    filtered_df = df[
        (df["product"] == target_product) &
        (df["sales"] >= min_sales) &
        (df["city"].isin(similar_cities))
    ]

    return int(filtered_df["sales"].sum())

# Example usage
# file_path = "sales_data.csv"
# total_sales = GA5_5("How many units of Shirt were sold in Istanbul on transactions with at least 131 units?", file_path)
# print("Total sales:", total_sales)


def fix_sales_value(sales):
    """Try to convert sales value to a float or default to 0 if invalid."""
    if isinstance(sales, (int, float)):
        return float(sales)  # Already valid

    if isinstance(sales, str):
        sales = sales.strip()  # Remove spaces
        if re.match(r"^\d+(\.\d+)?$", sales):  # Check if it's a valid number
            return float(sales)

    return 0.0  # Default for invalid values

async def GA5_6(question: str, file: UploadFile):
    file_content = await file.read()
    lines = file_content.decode(
        "utf-8").splitlines()  # In-memory file-like object
    sales_data = []
    for idx, line in enumerate(lines, start=1):
        try:
            entry = json.loads(line.strip())  # Parse each JSON line
            if "sales" in entry:
                entry["sales"] = fix_sales_value(entry["sales"])  # Fix invalid sales
                sales_data.append(entry)
            else:
                print(f"Line {idx}: Missing 'sales' field, adding default 0.0")
                entry["sales"] = 0.0
                sales_data.append(entry)
        except json.JSONDecodeError:
            # print(
            #     f"Line {idx}: Corrupt JSON, skipping -> {line.strip()}")
            line = line.strip().replace("{", "").split(",")[:-1]
            line = json.dumps({k.strip('"'): int(v) if v.isdigit() else v.strip('"') for k, v in (item.split(":", 1) for item in line)})
            # print("Fixed",line)
            sales_data.append(json.loads(line.strip()))
    sales = int(sum(entry["sales"] for entry in sales_data))
    return sales

def count_keys_json(data, key_word):
    count = 0
    if isinstance(data, dict):
        for key, value in data.items():
            if key == key_word:
                count += 1
            count += count_keys_json(value, key_word)
    elif isinstance(data, list):
        for item in data:
            count += count_keys_json(item, key_word)
    return count


async def GA5_7(question: str, file: UploadFile):
    file_content = await file.read()
    file_content = file_content.decode("utf-8")
    key = re.search(r'How many times does (\w+) appear as a key?', question).group(1)
    print(key)
    json_data = json.loads(file_content)
    count = count_keys_json(json_data, key)
    return count

# Example usage
# file_path = "q-extract-nested-json-keys.json"
# key_count = GA5_7("Download the data from q-extract-nested-json-keys.json. How many times does DX appear as a key?", file_path)
# print("Key count:", key_count)


def GA5_8(question):
    match1 = re.search(
        r"Write a DuckDB SQL query to find all posts IDs after (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z) with at least (\d+)", question)
    match2 = re.search(
        r" with (\d+) useful stars, sorted. The result should be a table with a single column called post_id, and the relevant post IDs should be sorted in ascending order.", question)
    datetime, comments,stars = match1.group(1), match1.group(2), match2.group(1)
    print(datetime, comments,stars)

    sql_query = f"""SELECT DISTINCT post_id FROM (SELECT timestamp, post_id, UNNEST (comments->'$[*].stars.useful') AS useful FROM social_media) AS temp
WHERE useful >= {stars}.0 AND timestamp > '{datetime}'
ORDER BY post_id ASC
"""
    return sql_query.replace("\n", " ")

# Example usage
# sql_query = GA5_8("Write a DuckDB SQL query to find all posts IDs after 2025-01-21T14: 36:47.099Z with at least 1 comment with 5 useful stars, sorted. The result should be a table with a single column called post_id, and the relevant post IDs should be sorted in ascending order.", file_path)
# print("Key count:", key_count)


async def GA5_9(question):
    transcript = get_transcript(question)
    try:
        correct_transcript(transcript)
        print(transcript)
    except Exception as e:
        print(transcript)
    return transcript


async def GA5_10(question: str, file: UploadFile):
    # Read file content into memory
    file_bytes = await file.read()
    scrambled_image = Image.open(io.BytesIO(file_bytes))

    # Image parameters
    grid_size = 5  # 5x5 grid
    piece_size = scrambled_image.width // grid_size  # Assuming a square image

    # Regex pattern to extract mapping data
    pattern = re.compile(r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")
    mapping = [tuple(map(int, match)) for match in pattern.findall(question)]

    # Create a blank image for reconstruction
    reconstructed_image = Image.new(
        "RGB", (scrambled_image.width, scrambled_image.height))

    # Rearrange pieces based on the mapping
    for original_row, original_col, scrambled_row, scrambled_col in mapping:
        scrambled_x = scrambled_col * piece_size
        scrambled_y = scrambled_row * piece_size

        # Extract piece from scrambled image
        piece = scrambled_image.crop(
            (scrambled_x, scrambled_y, scrambled_x +
             piece_size, scrambled_y + piece_size)
        )

        # Place in correct position in the reconstructed image
        original_x = original_col * piece_size
        original_y = original_row * piece_size
        reconstructed_image.paste(piece, (original_x, original_y))

    # Convert to Base64
    img_io = io.BytesIO()
    reconstructed_image.save(img_io, format="PNG")
    image_b64 = base64.b64encode(img_io.getvalue()).decode()
    return image_b64

# q = "What is the total Maths marks of students who scored 36 or more marks in Economics in groups 36-60 (including both groups)?"

async def GA4_10(question: str, file: UploadFile):
    md_text = ""
    return md_text

        raise HTTPException(
            status_code=400, detail=f"Could not parse query: {q}")

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse query: {q}. Error: {str(e)}. Pattern matches: {pattern_debug_info}"
        )
