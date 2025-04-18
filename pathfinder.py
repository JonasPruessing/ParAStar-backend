import math
import heapq
import json
import rasterio
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# Für den KML-Export:
import simplekml
from pyproj import Transformer, Geod
import rasterio.transform

def lade_hoehenmodell(geotiff_datei):
    with rasterio.open(geotiff_datei) as src:
        hoehenmodell = src.read(1)
        transform = src.transform
        crs = src.crs
    return hoehenmodell, transform, crs

def lade_aufgabe(aufgabe_json_datei):
    with open(aufgabe_json_datei, 'r') as f:
        return json.load(f)

def lade_thermikdaten(thermik_json_datei):
    with open(thermik_json_datei, 'r') as f:
        thermals = json.load(f)
    thermik_dict = {}
    for t in thermals:
        key = (t["row"], t["col"])
        strength = t["strength"]
        alt = t["altitude"]
        if key in thermik_dict:
            if strength > thermik_dict[key]["strength"]:
                thermik_dict[key] = {"strength": strength, "altitude": alt}
        else:
            thermik_dict[key] = {"strength": strength, "altitude": alt}
    return thermik_dict

class AStar3D:
    def __init__(self, hoehenmodell, thermik_dict, start_hoehe, gleitgeschwindigkeit=38/3.6):
        self.terrain = hoehenmodell
        self.thermal_dict = thermik_dict
        self.glide_speed = gleitgeschwindigkeit
        self.horiz_step = 100.0
        self.vert_step = 12.5
        self.base_alt = float(np.min(hoehenmodell))
        self.max_alt = 2950.0
        self.max_level = int((self.max_alt - self.base_alt) / self.vert_step)
        self.n_rows, self.n_cols = hoehenmodell.shape
        self._initial_altitude = start_hoehe

    def altitude_of(self, level):
        return self.base_alt + level * self.vert_step

    def is_valid_state(self, r, c, level):
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols):
            return False
        if not (0 <= level <= self.max_level):
            return False
        alt = self.altitude_of(level)
        return alt > self.terrain[r, c]

    def heuristic(self, state, ziel_rc):
        r, c, level = state
        x = c * self.horiz_step
        y = r * self.horiz_step
        z = self.altitude_of(level)
        
        ziel_r, ziel_c = ziel_rc
        xz = ziel_c * self.horiz_step
        yz = ziel_r * self.horiz_step
        ziel_terrain = self.terrain[ziel_r, ziel_c]
        min_level_ziel = int(math.floor((ziel_terrain - self.base_alt) / self.vert_step)) + 1
        zz = self.altitude_of(min_level_ziel)
        dist = math.sqrt((x - xz)**2 + (y - yz)**2 + (z - zz)**2)
        return dist / self.glide_speed

    def get_neighbors(self, state):
        nachbarn = []
        r, c, level = state
        # Gleitbewegung: horizontale Bewegung mit Abstieg um ein Level
        if level - 1 >= 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    nlevel = level - 1
                    if not self.is_valid_state(nr, nc, nlevel):
                        continue
                    if abs(dr) + abs(dc) == 1:
                        horiz = self.horiz_step
                    else:
                        horiz = math.sqrt(self.horiz_step**2 + self.horiz_step**2)
                    d = math.sqrt(horiz**2 + self.vert_step**2)
                    zeit = d / self.glide_speed
                    nachbarn.append(((nr, nc, nlevel), zeit))
        # Thermikaufstieg
        if (r, c) in self.thermal_dict:
            nlevel = level + 1
            if self.is_valid_state(r, c, nlevel):
                staerke = self.thermal_dict[(r, c)]["strength"]
                thermik_steiggeschw = staerke
                zeit = self.vert_step / thermik_steiggeschw
                nachbarn.append(((r, c, nlevel), zeit))
        return nachbarn

    def reconstruct_path(self, came_from, current):
        pfad = [current]
        while current in came_from:
            current = came_from[current]
            pfad.append(current)
        pfad.reverse()
        return pfad

    @property
    def initial_altitude(self):
        return self._initial_altitude

    @initial_altitude.setter
    def initial_altitude(self, value):
        self._initial_altitude = value

def berechne_min_höhe_wp(aufgabe, hoehenmodell, thermik_dict, horiz_step=100.0, vert_step=12.5, sicherheitsabstand=50):
    schluessel_geordnet = ["start"]
    tps = sorted([k for k in aufgabe.keys() if k.startswith("tp")],
                 key=lambda k: int(k[2:]) if k[2:].isdigit() else 0)
    schluessel_geordnet.extend(tps)
    schluessel_geordnet.append("goal")
    ergebnis = []
    for key in schluessel_geordnet:
        if key in ["start", "goal"]:
            if isinstance(aufgabe[key], (list, tuple)):
                zeile_spalte = tuple(int(x) for x in aufgabe[key])
            else:
                zeile_spalte = tuple(int(x) for x in aufgabe[key]["grid"])
            ergebnis.append((zeile_spalte, None))
        else:
            tp = tuple(int(x) for x in aufgabe[key]["grid"])
            min_d = float('inf')
            beste_thermik_hoehe = None
            for (r_th, c_th), daten in thermik_dict.items():
                d = math.sqrt((tp[0]-r_th)**2 + (tp[1]-c_th)**2) * horiz_step
                if d < min_d:
                    min_d = d
                    beste_thermik_hoehe = daten["altitude"]
            if beste_thermik_hoehe is None:
                erforderl_hoehe = hoehenmodell[tp[0], tp[1]] + sicherheitsabstand
            else:
                zusatz = (min_d / 100.0) * vert_step + sicherheitsabstand
                erforderl_hoehe = max(hoehenmodell[tp[0], tp[1]] + sicherheitsabstand,
                                      beste_thermik_hoehe + zusatz)
            ergebnis.append((tp, erforderl_hoehe))
    return ergebnis

def zusammengesetztes_heuristik(state, wegpunkte, astar):
    r, c, level, ziel_index = state
    if ziel_index < len(wegpunkte):
        wp, radius, erforderl_hoehe = wegpunkte[ziel_index]
        d = max(0, math.sqrt((r - wp[0])**2 + (c - wp[1])**2) * astar.horiz_step - radius)
        h_aktuell = d / astar.glide_speed
        alt = astar.altitude_of(level)
        strafe = 0
        if erforderl_hoehe is not None and alt < erforderl_hoehe:
            strafe = (erforderl_hoehe - alt) / astar.glide_speed
        rest = 0
        for j in range(ziel_index, len(wegpunkte)-1):
            wp_akt = wegpunkte[j][0]
            wp_next = wegpunkte[j+1][0]
            d_leg = math.sqrt((wp_akt[0]-wp_next[0])**2 + (wp_akt[1]-wp_next[1])**2) * astar.horiz_step
            rest += d_leg / astar.glide_speed
        return h_aktuell + strafe + rest
    else:
        return 0

def rekonstruiere_pfad(kam_von, zustand):
    pfad = [zustand]
    while zustand in kam_von:
        zustand = kam_von[zustand]
        pfad.append(zustand)
    pfad.reverse()
    return pfad

def finde_mehrteiligen_pfad(astar, aufgabe):
    schluessel_geordnet = ["start"]
    tps = sorted([k for k in aufgabe.keys() if k.startswith("tp")],
                 key=lambda k: int(k[2:]) if k[2:].isdigit() else 0)
    schluessel_geordnet.extend(tps)
    schluessel_geordnet.append("goal")
    minimal_info = berechne_min_höhe_wp(aufgabe, astar.terrain, astar.thermal_dict,
                                        astar.horiz_step, astar.vert_step, sicherheitsabstand=50)
    wegpunkte = []
    for key, (wp, erforderl_hoehe) in zip(schluessel_geordnet, minimal_info):
        if key in ["start", "goal"]:
            wegpunkte.append((wp, 0, None))
        else:
            wegpunkte.append((wp, float(aufgabe[key]["radius"]), erforderl_hoehe))
    n_wegpunkte = len(wegpunkte)
    start_rc = wegpunkte[0][0]
    start_alt = astar.terrain[start_rc[0], start_rc[1]] + astar.vert_step
    start_level = int(round((start_alt - astar.base_alt) / astar.vert_step))
    start_zustand = (start_rc[0], start_rc[1], start_level, 1)
    offener_haufen = []
    h_start = zusammengesetztes_heuristik(start_zustand, wegpunkte, astar)
    heapq.heappush(offener_haufen, (h_start, 0.0, start_zustand))
    beste_kosten = {start_zustand: 0.0}
    kam_von = {}
    while offener_haufen:
        f_kosten, g_kosten, zustand = heapq.heappop(offener_haufen)
        if zustand[3] == n_wegpunkte:
            pfad = rekonstruiere_pfad(kam_von, zustand)
            return pfad, g_kosten
        r, c, level, ziel_index = zustand
        wp, radius, erforderl_hoehe = wegpunkte[ziel_index]
        d_grid = math.sqrt((r - wp[0])**2 + (c - wp[1])**2) * astar.horiz_step
        alt = astar.altitude_of(level)
        if d_grid <= radius and alt > astar.terrain[r, c]:
            if erforderl_hoehe is None or alt >= erforderl_hoehe:
                neuer_zustand = (r, c, level, ziel_index + 1)
                neue_kosten = g_kosten
                if neuer_zustand not in beste_kosten or neue_kosten < beste_kosten[neuer_zustand]:
                    beste_kosten[neuer_zustand] = neue_kosten
                    h_neu = zusammengesetztes_heuristik(neuer_zustand, wegpunkte, astar)
                    heapq.heappush(offener_haufen, (neue_kosten + h_neu, neue_kosten, neuer_zustand))
                    kam_von[neuer_zustand] = zustand
        for nachbar, bewegungskosten in astar.get_neighbors((r, c, level)):
            nr, nc, nlevel = nachbar
            neuer_zustand = (nr, nc, nlevel, ziel_index)
            neue_kosten = g_kosten + bewegungskosten
            if neuer_zustand not in beste_kosten or neue_kosten < beste_kosten[neuer_zustand]:
                beste_kosten[neuer_zustand] = neue_kosten
                h_neu = zusammengesetztes_heuristik(neuer_zustand, wegpunkte, astar)
                heapq.heappush(offener_haufen, (neue_kosten + h_neu, neue_kosten, neuer_zustand))
                kam_von[neuer_zustand] = zustand
    return None, math.inf

def kreis_koords(lon_zentrum, lat_zentrum, alt_zentrum, radius, anzahl_punkte=36):
    geod = Geod(ellps='WGS84')
    coords = []
    for i in range(anzahl_punkte + 1):
        azimut = 360.0 * i / anzahl_punkte
        lon, lat, _ = geod.fwd(lon_zentrum, lat_zentrum, azimut, radius)
        coords.append((lon, lat, alt_zentrum))
    return coords

def exportiere_kml(pfad, transform, crs, astar, aufgabe, thermik_dict, kml_datei="predicted_full_t4.kml"):
    kml = simplekml.Kml()
    
    if crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    else:
        transformer = None
    
    linien_koords = []
    for zustand in pfad:
        r, c, level, _ = zustand
        alt = astar.altitude_of(level)
        x, y = rasterio.transform.xy(transform, r, c, offset='center')
        if transformer:
            lon, lat = transformer.transform(x, y)
        else:
            lon, lat = x, y
        linien_koords.append((lon, lat, alt))
    linie = kml.newlinestring(name="Gleitschirmpfad", description="Mehrteiliger Pfad mit Lookahead")
    linie.coords = linien_koords
    linie.altitudemode = simplekml.AltitudeMode.absolute
    linie.extrude = 0
    linie.style.linestyle.width = 6
    linie.style.linestyle.color = simplekml.Color.red
    
    for key in ["start", "goal"]:
        if key in aufgabe:
            if isinstance(aufgabe[key], (list, tuple)):
                grid = tuple(int(x) for x in aufgabe[key])
            else:
                grid = tuple(int(x) for x in aufgabe[key]["grid"])
            r, c = grid
            x, y = rasterio.transform.xy(transform, r, c, offset='center')
            if transformer:
                lon, lat = transformer.transform(x, y)
            else:
                lon, lat = x, y
            alt = astar.terrain[r, c] + 20
            pnt = kml.newpoint(name=key.capitalize(), coords=[(lon, lat, alt)])
            if key == "start":
                pnt.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/grn-circle.png"
            else:
                pnt.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/red-circle.png"
            pnt.altitudemode = simplekml.AltitudeMode.absolute
    
    for key in aufgabe:
        if key.startswith("tp"):
            tp = tuple(int(x) for x in aufgabe[key]["grid"])
            radius = float(aufgabe[key]["radius"])
            r, c = tp
            x, y = rasterio.transform.xy(transform, r, c, offset='center')
            if transformer:
                lon, lat = transformer.transform(x, y)
            else:
                lon, lat = x, y
            base_alt = astar.terrain[r, c] + 20
            top_alt = 2800
            kreis = kreis_koords(lon, lat, top_alt, radius, anzahl_punkte=36)
            poly = kml.newpolygon(name=f"Wendepunkt {key}", extrude=1)
            poly.outerboundaryis = kreis
            poly.altitudemode = simplekml.AltitudeMode.absolute
            poly.style.polystyle.color = "7F00ff00"
            poly.style.polystyle.fill = 1
            poly.style.polystyle.outline = 1
            poly.style.linestyle.width = 2
    
    for (r, c), daten in thermik_dict.items():
        staerke = daten["strength"]
        thermik_hoehe = daten["altitude"]
        x, y = rasterio.transform.xy(transform, r, c, offset='center')
        if transformer:
            lon, lat = transformer.transform(x, y)
        else:
            lon, lat = x, y
        base_alt = thermik_hoehe
        top_alt = 2500
        kreis = kreis_koords(lon, lat, top_alt, 50, anzahl_punkte=12)
        poly = kml.newpolygon(name=f"Thermik (Stärke: {staerke})", extrude=1)
        poly.outerboundaryis = kreis
        poly.altitudemode = simplekml.AltitudeMode.absolute
        poly.style.polystyle.color = "7F00ffff"
        poly.style.polystyle.fill = 1
        poly.style.polystyle.outline = 1
        poly.style.linestyle.width = 1
    
    kml.save(kml_datei)
    print(f"KML-Datei gespeichert als {kml_datei}")


def solve_task():
    base = os.path.dirname(__file__)
    terrain_file = os.path.join(base, 'data', 'Terrain', 'resampled_100m_grid_exact.tif')
    task_file    = os.path.join(base, 'data', 'Task',    'HM2022T4_v2_grid_short.json')
    thermal_file = os.path.join(base, 'data', 'Thermals','thermal_data_predicted.json')

    terrain, transform, crs = lade_hoehenmodell(terrain_file)
    aufgabe = lade_aufgabe(task_file)
    thermik_dict = lade_thermikdaten(thermal_file)

    # determine start cell
    if isinstance(aufgabe['start'], (list, tuple)):
        start_rc = tuple(int(x) for x in aufgabe['start'])
    else:
        start_rc = tuple(int(x) for x in aufgabe['start']['grid'])

    start_height = terrain[start_rc[0], start_rc[1]] + 12.5
    astar = AStar3D(terrain, thermik_dict, start_height)

    path_states, _ = finde_mehrteiligen_pfad(astar, aufgabe)
    if path_states is None:
        return []

    # convert to lon/lat
    transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    coords = []
    for r, c, level, _ in path_states:
        x, y = rasterio.transform.xy(transform, r, c, offset='center')
        lon, lat = transformer.transform(x, y)
        coords.append([lon, lat])
    return coords


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(__file__)

    terrain_file = os.path.join(BASE_DIR, 'data', 'Terrain', 'resampled_100m_grid_exact.tif')
    task_file    = os.path.join(BASE_DIR, 'data', 'Task',    'HM2022T4_v2_grid_short.json')
    thermal_file = os.path.join(BASE_DIR, 'data', 'Thermals','thermal_data_predicted.json')
        
    print(f"Verwende Höhenmodell:  {terrain_file}")
    print(f"Verwende Aufgaben-Datei: {task_file}")
    print(f"Verwende Thermik-Datei: {thermal_file}")
    
    hoehenmodell, transform, crs = lade_hoehenmodell(terrain_file)
    aufgabe = lade_aufgabe(task_file)
    thermik_dict = lade_thermikdaten(thermal_file)
    
    if isinstance(aufgabe["start"], (list, tuple)):
        start_rc = tuple(int(x) for x in aufgabe["start"])
    else:
        start_rc = tuple(int(x) for x in aufgabe["start"]["grid"])
    
    if isinstance(aufgabe["goal"], (list, tuple)):
        goal_rc = tuple(int(x) for x in aufgabe["goal"])
    else:
        goal_rc = tuple(int(x) for x in aufgabe["goal"]["grid"])
    
    start_hoehe = hoehenmodell[start_rc[0], start_rc[1]] + 12.5
    astar = AStar3D(hoehenmodell, thermik_dict, start_hoehe)
    
    pfad, gesamt_zeit = finde_mehrteiligen_pfad(astar, aufgabe)
    
    if pfad is None:
        print("Kein gültiger Mehrteil-Pfad gefunden.")
    else:
        print(f"Mehrteiliger Pfad gefunden. Gesamtzeit: {gesamt_zeit:.2f} Sekunden.")
        for zustand in pfad:
            r, c, level, ziel_index = zustand
            alt = astar.altitude_of(level)
            print(f"   Zustand: ({r}, {c}), Höhe: {alt:.2f} m, Ziel-Index: {ziel_index}")
        
        exportiere_kml(pfad, transform, crs, astar, aufgabe, thermik_dict,
                       kml_datei="HMT4_kurz_vorhergesagt.kml")
        
        zell_groesse = 100
        n_rows, n_cols = hoehenmodell.shape
        ausdehnung = (0, n_cols, n_rows, 0)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(hoehenmodell, cmap='gray', extent=ausdehnung, origin='upper')
        plt.colorbar(label='Gelände-Höhe (m)')
        
        pfad_x = []
        pfad_y = []
        for zustand in pfad:
            r, c, _, _ = zustand
            pfad_x.append(c)
            pfad_y.append(r)
        plt.plot(pfad_x, pfad_y, color='red', linewidth=3, label='Bester Pfad')
        
        start_r, start_c = start_rc
        goal_r, goal_c = goal_rc
        plt.scatter(start_c, start_r, color='blue', s=120, label='Start')
        plt.scatter(goal_c, goal_r, color='purple', s=120, label='Ziel')
        
        for key in aufgabe:
            if key.startswith("tp"):
                tp = tuple(int(x) for x in aufgabe[key]["grid"])
                radius_m = float(aufgabe[key]["radius"])
                r, c = tp
                radius_plot = radius_m / zell_groesse
                kreis = plt.Circle((c, r), radius_plot, color='blue', fill=False, linewidth=2, label='Wendepunkt')
                plt.gca().add_patch(kreis)
                plt.text(c, r, key, color='blue', fontsize=10, ha='center', va='bottom')
        
        thermik_x = []
        thermik_y = []
        for (r, c) in thermik_dict.keys():
            thermik_x.append(c)
            thermik_y.append(r)
        if thermik_x:
            plt.scatter(thermik_x, thermik_y, marker='^', color='orange', s=100, label='Thermik')
        
        zeit_str = str(timedelta(seconds=int(gesamt_zeit)))
        plt.text(0.05, 0.95, f"Zeit bis Ziel: {zeit_str}", transform=plt.gca().transAxes,
                 fontsize=16, color='red', verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.7))
        
        plt.xlabel("Spalte (Gittereinheiten)")
        plt.ylabel("Zeile (Gittereinheiten)")
        plt.title("2D-Ansicht des besten Gleitschirm-Pfads (Gitterkoordinaten)")
        plt.legend(loc='lower right')
        
        # Plot speichern
        plt.savefig("HMT4_kurz_vorhergesagt.png", dpi=300)
        print("Plot als 'HMT4_kurz_vorhergesagt.png' gespeichert.")
        
        plt.show()


# at the bottom of pathfinder.py, after solve_task()



def load_task_waypoints():
    BASE_DIR = os.path.dirname(__file__)
    task_file = os.path.join(BASE_DIR, 'data', 'Task', 'HM2022T4_v2_grid_short.json')
    aufgabe   = lade_aufgabe(task_file)

    terrain_tif = os.path.join(BASE_DIR, 'data', 'Terrain', 'resampled_100m_grid_exact.tif')
    _, transform, crs = lade_hoehenmodell(terrain_tif)
    transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)

    waypoints = []
    keys = ['start'] + sorted([k for k in aufgabe if k.startswith('tp')]) + ['goal']
    for key in keys:
        entry = aufgabe[key]
        # unpack grid coords
        if isinstance(entry, (list, tuple)):
            r, c = entry
            radius = 0.0
        else:
            r, c = entry['grid']
            # only dicts have a radius field
            radius = float(entry.get('radius', 0))
        # compute lon/lat
        x, y = rasterio.transform.xy(transform, int(r), int(c), offset='center')
        lon, lat = transformer.transform(x, y)
        waypoints.append({'key': key, 'lon': lon, 'lat': lat, 'radius': radius})
    return waypoints

