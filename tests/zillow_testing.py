from pyzillow.pyzillow import ZillowWrapper, GetDeepSearchResults, GetUpdatedPropertyDetails
address = '901 Highbury Ln, Marietta, GA'
zipcode = '30068'
zillow_data = ZillowWrapper('X1-ZWz1fjckjdd8gb_a2eph')

deep_search_response = zillow_data.get_deep_search_results(address,zipcode)
deep_search_result = GetDeepSearchResults(deep_search_response)

property_details_response = zillow_data.get_updated_property_details(deep_search_result.zillow_id)
property_details_result = GetUpdatedPropertyDetails(property_details_response)

print(deep_search_result.area_unit)
print(deep_search_result.home_size)
print(property_details_result.roof)