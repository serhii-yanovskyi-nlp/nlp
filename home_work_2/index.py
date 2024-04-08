import requests
from bs4 import BeautifulSoup
import pandas as pd


url = "https://raw.githubusercontent.com/realpython/fake-jobs/main/index.html"
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    job_cards = soup.find_all("div", class_="card-content")
    jobs_data = []

    for card in job_cards:
        job_title = card.find("h2", class_="title is-5").text.strip()
        company_name = card.find("h3", class_="subtitle is-6 company").text.strip()
        location = card.find("p", class_="location").text.strip()
        date_posted = card.find("time").text.strip()
        learn_link = card.find("a", string="Learn")["href"]
        apply_link = card.find("a", string="Apply")["href"]

        jobs_data.append({
            "Job Title": job_title,
            "Company": company_name,
            "Location": location,
            "Date Posted": date_posted,
            "Learn Link": learn_link,
            "Apply Link": apply_link
        })

    df = pd.DataFrame(jobs_data)
    df.to_excel("data.xlsx", index=False)
    print("ok")

else:
    print(f"error {url}")