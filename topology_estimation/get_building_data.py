from pyzillow.pyzillow import ZillowWrapper, GetDeepSearchResults, GetUpdatedPropertyDetails, ZillowError
from topology_estimation.roof_pitch_estimation import roof_type_to_rad


def get_building_data(address, zipcode):
    sq_feet_to_meters = 0.092903

    zillow_data = ZillowWrapper('X1-ZWz1fjckjdd8gb_a2eph')

    try:
        deep_search_response = zillow_data.get_deep_search_results(address, zipcode)
        deep_search_result = GetDeepSearchResults(deep_search_response)

        sq_meters = float(deep_search_result.home_size) * sq_feet_to_meters
    except ZillowError:
        sq_meters = float(input("Please enter your home's square footage: ")) * sq_feet_to_meters

    try:
        property_details_response = zillow_data.get_updated_property_details(deep_search_result.zillow_id)
        property_details_result = GetUpdatedPropertyDetails(property_details_response)

        roof_pitch_rad = roof_type_to_rad(property_details_result.roof)
    except:
        # TODO: Revert (this was for testing purposes)
        # roof_pitch_rad = math.atan(float(input("Please enter the rise of your roof's pitch: ")) / 12)
        roof_pitch_rad = roof_type_to_rad('composite')

    return sq_meters, roof_pitch_rad