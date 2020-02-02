import requests

url = "http://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?api_key=fDFOk0vwiLzSJePAtLKg0oV8A8mex3yA9qYpdftK"

payload = "names=2018&leap_day=false&interval=30&utc=false&email=srinivasdeva3%40gmail.com&attributes=dhi%2Cdni%2Cwind_speed%2Cair_temperature&wkt=POINT(-84.4571435620623%2033.96894305)"

headers = {
    'content-type': "application/x-www-form-urlencoded",
    'cache-control': "no-cache"
}

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text.splitlines()[6229].split(',')[6])