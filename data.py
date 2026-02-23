from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Setup Chrome driver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # run without browser window
driver = webdriver.Chrome(options=options)

driver.get("https://enam.gov.in/web/dashboard/trade-data")
time.sleep(5)  # wait for JavaScript to load data

# Get all rows in the trade data table
rows = driver.find_elements(By.XPATH, "//table//tr")

# Extract column headers
headers = [th.text.strip() for th in rows[0].find_elements(By.TAG_NAME, "th")]
print(" | ".join(headers))   # print header row once

# Loop through table rows
for row in rows[1:]:  # skip header
    cols = [c.text.strip() for c in row.find_elements(By.TAG_NAME, "td")]
    if cols:
        print(" | ".join(cols))

driver.quit()
