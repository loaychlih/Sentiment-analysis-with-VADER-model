from bs4 import BeautifulSoup
import requests
import csv

base_url = 'https://www.trustpilot.com/review/www.zalando.co.uk?page='
reviews = []

for page_number in range(1, 50):
    url = base_url + str(page_number)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    review_elements = soup.find_all(class_='typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn')
    

    for review_element in review_elements:
        review_text = review_element.get_text(strip=True)
        reviews.append(review_text)


with open('reviewsnew.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Review'])
    csvwriter.writerows([[review] for review in reviews])



