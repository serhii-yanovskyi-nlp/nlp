import base64
import random
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


df = pd.DataFrame()

url = "https://api.github.com/repos/realpython/fake-jobs/contents/jobs"
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    file_names = [file["name"] for file in data]
    for file_name in file_names:
        file_url = f"https://api.github.com/repos/realpython/fake-jobs/contents/jobs/{file_name}"
        response = requests.get(file_url)
        if response.status_code == 200:
            content_bytes = base64.b64decode(response.json()["content"])
            html_content = content_bytes.decode('utf-8')
            soup = BeautifulSoup(html_content, "html.parser")
            job_title = soup.find("h1", class_="title is-2").text.strip()
            company = soup.find("h2", class_="subtitle is-4 company").text.strip()
            location = soup.find("p", id="location").text.split(":")[-1].strip()
            date_posted = soup.find("p", id="date").text.split(":")[-1].strip()
            df = df.append({
                "File Name": file_name,
                "Job Title": job_title,
                "Company": company,
                "Location": location,
                "Date Posted": date_posted
            }, ignore_index=True)
            time.sleep(random.randint(1, 5))
        else:
            print(f"error: {file_name}")
    df.to_excel("data.xlsx", index=False)
else:
    print(f"error: {url}")


