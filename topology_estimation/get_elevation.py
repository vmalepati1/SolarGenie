import urllib

import requests
import urllib3
import json


def make_remote_request(url: str, params: dict) -> json:
    """
    Makes the remote request
    Continues making attempts until it succeeds
    """

    count = 1
    while True:
        try:
            response = requests.get((url + urllib.parse.urlencode(params)))
        except (OSError, urllib3.exceptions.ProtocolError) as error:
            print('\n')
            print('*' * 20, 'Error Occured', '*' * 20)
            print(f'Number of tries: {count}')
            print(f'URL: {url}')
            print(error)
            print('\n')
            count += 1
            continue
        break

    return response


def elevation_function(x):
    url = 'https://nationalmap.gov/epqs/pqs.php?'
    params = {'x': x[1],
              'y': x[0],
              'units': 'Meters',
              'output': 'json'}
    result = make_remote_request(url, params)
    return result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']