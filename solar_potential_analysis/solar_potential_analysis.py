import math

def get_pixelwise_solar_irradiance(beam_irr, diff_irr, planar_orr, roof_pitch, solar_elevation, solar_azimuth):
    return beam_irr * r_beam(roof_pitch, planar_orr, solar_elevation, solar_azimuth) + diff_irr * r_diff(roof_pitch)

def r_beam(beta, psi, alpha, theta):
    return math.cos(alpha) * math.sin(beta) * math.cos(psi - theta) + math.sin(alpha) * math.cos(beta)

def r_diff(beta):
    return (1 + math.cos(beta)) / 2
