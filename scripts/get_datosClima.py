import requests

import pandas as pd
import numpy as np
import refet


def get_power_data(lat, lon, params, start, end):
  """
  Extrae datos climáticos de NASA POWER
  """
  url = f"https://power.larc.nasa.gov/api/temporal/daily/point"
  query = {
      "parameters": ",".join(params),
      "community": "AG",
      "longitude": lon,
      "latitude": lat,
      "start": start,
      "end": end,
      "format": "JSON"
    }
  r = requests.get(url, params=query)
  return r.json()


def ET0_calculator(datos_clima, lat, elevacion):
  """
  Calcula ET0 diaria usando la librería refet (ASCE Standardized).
  
  Parámetros:
      datos_clima : dict   → JSON de get_power_data()
      lat         : float  → Latitud en grados decimales (negativo = sur)
      elevacion   : float  → Altitud en m.s.n.m.
  
  Retorna:
  DataFrame con ET0 en mm/día
  """
  props = datos_clima['properties']['parameter']

  tmax = np.array(list(props['T2M_MAX'].values()), dtype=float)            # °C  ✓
  tmin = np.array(list(props['T2M_MIN'].values()), dtype=float)            # °C  ✓
  rh   = np.array(list(props['RH2M'].values()),    dtype=float)            #  %  ✓
  ws   = np.array(list(props['WS2M'].values()),    dtype=float)            # m/s ✓ (a 2 m)
  rs   = np.array(list(props['ALLSKY_SFC_SW_DWN'].values()), dtype=float)  # MJ/m²/día ✓
  prec = np.array(list(props['PRECTOTCORR'].values()), dtype=float)        # mm/día ✓

  fechas = pd.to_datetime(list(props['T2M_MAX'].keys()), format='%Y%m%d')
  doy    = fechas.dayofyear.values

  def e_sat(T):
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))  # Buscar referencias (Pendiente)
  
  ea = (rh / 100) * (e_sat(tmax) + e_sat(tmin)) / 2  # kPa (Buscar referencias)

  ET0 = refet.Daily(tmin = tmin, tmax = tmax,
                    ea   = ea,
                    rs   = rs,
                    uz   = ws,
                    zw   = 2.0,               # altura del anemómetro: 2 m - NASA POWER
                    elev = elevacion,
                    lat  = np.deg2rad(lat),   # refet espera radianes
                    doy  = doy,
                    method = 'asce').eto()    # 'asce' = hierba corta ≈ FAO-56

  return pd.DataFrame({
    'MinTemp':       tmin,
    'MaxTemp':       tmax,
    'Precipitation': prec,
    'ReferenceET':    ET0,
    'Date':        fechas
  })