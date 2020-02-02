import math

roof_type_pitch_dict = {
    # Tuple of keywords, (minimum pitch, maximum pitch)
    ('built-up', 'bur'): (0.25/12, 3/12),
    ('torch-down', 'torch'): (0.25/12, 3/12),
    ('rubber membrane', 'rubber'): (0.25/12, 3/12),
    ('standing-seam metal', 'metal'): (1/12, 19/12),
    ('clay', 'cement'): (2.5/12, 19/12),
    ('asphalt', 'composition', 'composite'): (4/12, 20/12),
    ('shingles', 'wood', 'slate', 'shake'): (5/12, 12/12)
}

def roof_type_to_rad(roof_type):
    for key, value in roof_type_pitch_dict.items():
        for keyword in key:
            if keyword in roof_type.lower():
                lower_rad = math.atan(value[0])
                upper_rad = math.atan(value[1])
                lower_deg = math.degrees(lower_rad)
                upper_deg = math.degrees(upper_rad)
                print('Averaging roof type {} with pitch between {} degrees and {} degrees.'.format(roof_type, lower_deg, upper_deg))
                return (lower_rad + upper_rad) / 2