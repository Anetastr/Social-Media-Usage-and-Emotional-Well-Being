import requests
import pandas as pd

API_KEY = "de882d30"  # Tvůj API klíč z Mockaroo

# Inline schéma v URL
url = (
    f"https://api.mockaroo.com/api/generate.json?key={API_KEY}&count=100"
    "&fields=["
    '{"name":"User_ID","type":"Row Number"},'
    '{"name":"Age","type":"Number","min":18,"max":50},'
    '{"name":"Gender","type":"Gender (Binary)"},'
    '{"name":"Platform","type":"Custom List","values":["Facebook","Instagram","LinkedIn","Snapchat","Twitter","Whatsapp"]},'
    '{"name":"Daily_Usage_Time","type":"Number","min":10,"max":200},'
    '{"name":"Post_Per_Day","type":"Number","min":1,"max":20},'
    '{"name":"Likes_Received_Per_Day","type":"Number","min":1,"max":70},'
    '{"name":"Comments_Received_Per_Day","type":"Number","min":1,"max":70},'
    '{"name":"Messages_Sent_Per_Day","type":"Number","min":1,"max":50}'
    "]"
)

try:
    response = requests.get(url)
    response.raise_for_status()

    # Načtení JSON dat do DataFrame
    df = pd.DataFrame(response.json())
    print(df.head())

    # Uložení do CSV
    df.to_csv("mockaroo_data.json", index=False)
    print("Data byla uložena do souboru 'mockaroo_data.json'")

except requests.exceptions.RequestException as e:
    print("Chyba při stahování dat:", e)


