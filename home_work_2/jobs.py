import random
import time

import requests
from bs4 import BeautifulSoup
import pandas as pd


url = "https://raw.githubusercontent.com/realpython/fake-jobs/main/index.html"
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    job_cards = soup.find_all("div", class_="card-content")
    jobs_data = []
    df = pd.DataFrame()
    for card in job_cards:
        time.sleep(random.randint(1, 5))
        job_title = card.find("h2", class_="title is-5").text.strip()
        learn_link = card.find("a", string="Learn")["href"]
        apply_link = card.find("a", string="Apply")["href"]
        response = requests.get(apply_link)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            job_title = soup.find("h1", class_="title is-2").text.strip()
            company = soup.find("h2", class_="subtitle is-4 company").text.strip()
            location = soup.find("p", id="location").text.split(":")[-1].strip()
            date_posted = soup.find("p", id="date").text.split(":")[-1].strip()
            job_description = soup.find("div", class_="content").text.strip()
            jobs_data.append({
            "Job Title": job_title,
            "Company": company,
            "Location": location,
            "Date Posted": date_posted,
            "Learn Link": learn_link,
            "Apply Link": apply_link,
            "job_description":job_description

        })

        else:
            print(f"error: {apply_link}")
            continue
    df.to_excel("data.xlsx", index=False)
else:
    print(f"error: {url}")