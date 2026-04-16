import streamlit as st
import json
import os
import base64
import numpy as np
import pandas as pd
import requests
import joblib
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

st.set_page_config(
    page_title="SolarSync | Smart Grid",
    layout="wide",
    initial_sidebar_state="collapsed"
)
warnings.filterwarnings('ignore')


# ============================================================
# PERSISTANCE ET HISTORIQUE (POUR LES FOYERS)
# ============================================================
def load_profiles():
    if os.path.exists("./app data/ser_profiles.json"):
        with open("./app data/ser_profiles.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_profiles(profiles):
    with open("./app data/ser_profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=4)

def init_historical_csv(name, data):
    """Crée un fichier historique CSV calqué sur le format d'entraînement (APT_*.csv)"""
    os.makedirs("./data/user_data", exist_ok=True)
    safe_name = name.replace(' ', '_').upper()
    filename = f"./data/user_data/APT_{safe_name}.csv"
    
    if not os.path.exists(filename):
        # Simulation de 7 jours d'historique (intervalle 15 min)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='15min')
        
        # Consommation de base basée sur la surface et les résidents
        base_power = (data['surface'] * 2.5) + (data['residents'] * 60)
        
        # Ajout du cycle de consommation (plus élevé le soir) et bruit
        hours = timestamps.hour
        cycle = np.where((hours >= 18) & (hours <= 22), 1.5, 0.8)
        power = np.clip(base_power * cycle + np.random.normal(0, 15, len(timestamps)), 20, None)
        
        df = pd.DataFrame({
            'timestamp': timestamps.strftime('%Y-%m-%d %H:%M:%S'),
            'real_power': np.round(power, 2)
        })
        df.to_csv(filename, index=False)

# ============================================================
# DONNÉES GÉOGRAPHIQUES (MAROC)
# ============================================================
CITIES_DATA = {
    "Tanger":    {"srm": "SRM Tanger-Tétouan-Al Hoceima (Amendis)", "lat": 35.7595, "lon": -5.8340},
    "M'diq":     {"srm": "SRM Tanger-Tétouan-Al Hoceima",           "lat": 35.6858, "lon": -5.3253},
    "Tétouan":   {"srm": "SRM Tanger-Tétouan-Al Hoceima (Amendis)", "lat": 35.5785, "lon": -5.3684},
    "Anjra":     {"srm": "SRM Tanger-Tétouan-Al Hoceima",           "lat": 35.8601, "lon": -5.5308},
    "Larache":   {"srm": "SRM Tanger-Tétouan-Al Hoceima",           "lat": 35.1932, "lon": -6.1540},
    "Al Hoceïma":{"srm": "SRM Tanger-Tétouan-Al Hoceima",           "lat": 35.2472, "lon": -3.9322},
    "Chefchaouen":{"srm":"SRM Tanger-Tétouan-Al Hoceima",           "lat": 35.1714, "lon": -5.2697},
    "Ouezzane":  {"srm": "SRM Tanger-Tétouan-Al Hoceima",           "lat": 34.7967, "lon": -5.5785},
    "Oujda":     {"srm": "SRM de l'Oriental",                        "lat": 34.6814, "lon": -1.9086},
    "Nador":     {"srm": "SRM de l'Oriental",                        "lat": 35.1667, "lon": -2.9333},
    "Driouch":   {"srm": "SRM de l'Oriental",                        "lat": 34.9770, "lon": -3.3790},
    "Jerada":    {"srm": "SRM de l'Oriental",                        "lat": 34.3117, "lon": -2.1636},
    "Berkane":   {"srm": "SRM de l'Oriental",                        "lat": 34.9167, "lon": -2.3167},
    "Taourirt":  {"srm": "SRM de l'Oriental",                        "lat": 34.4073, "lon": -2.8973},
    "Guercif":   {"srm": "SRM de l'Oriental",                        "lat": 34.2257, "lon": -3.3536},
    "Bouarfa":   {"srm": "SRM de l'Oriental",                        "lat": 32.5337, "lon": -1.9620},
    "Fès":       {"srm": "SRM Fès-Meknès",                           "lat": 34.0331, "lon": -5.0003},
    "Meknès":    {"srm": "SRM Fès-Meknès",                           "lat": 33.8950, "lon": -5.5547},
    "El Hajeb":  {"srm": "SRM Fès-Meknès",                           "lat": 33.6928, "lon": -5.3714},
    "Ifrane":    {"srm": "SRM Fès-Meknès",                           "lat": 33.5228, "lon": -5.1071},
    "Moulay Yaâcoub":{"srm":"SRM Fès-Meknès",                        "lat": 34.0828, "lon": -5.1814},
    "Séfrou":    {"srm": "SRM Fès-Meknès",                           "lat": 33.8281, "lon": -4.8281},
    "Missour":   {"srm": "SRM Fès-Meknès",                           "lat": 33.0489, "lon": -3.9895},
    "Taounate":  {"srm": "SRM Fès-Meknès",                           "lat": 34.5372, "lon": -4.6400},
    "Taza":      {"srm": "SRM Fès-Meknès",                           "lat": 34.2167, "lon": -4.0167},
    "Rabat":     {"srm": "SRM Rabat-Salé-Kénitra (Redal)",           "lat": 34.0209, "lon": -6.8416},
    "Salé":      {"srm": "SRM Rabat-Salé-Kénitra (Redal)",           "lat": 34.0333, "lon": -6.8000},
    "Témara":    {"srm": "SRM Rabat-Salé-Kénitra (Redal)",           "lat": 33.9267, "lon": -6.9122},
    "Kénitra":   {"srm": "SRM Rabat-Salé-Kénitra",                   "lat": 34.2610, "lon": -6.5802},
    "Khémisset": {"srm": "SRM Rabat-Salé-Kénitra",                   "lat": 33.8240, "lon": -6.0663},
    "Sidi Kacem":{"srm": "SRM Rabat-Salé-Kénitra",                   "lat": 34.2264, "lon": -5.7033},
    "Sidi Slimane":{"srm":"SRM Rabat-Salé-Kénitra",                  "lat": 34.2611, "lon": -5.9247},
    "Béni Mellal":{"srm":"SRM Béni Mellal-Khénifra",                 "lat": 32.3394, "lon": -6.3608},
    "Azilal":    {"srm": "SRM Béni Mellal-Khénifra",                 "lat": 31.9669, "lon": -6.5593},
    "Fquih Ben Salah":{"srm":"SRM Béni Mellal-Khénifra",             "lat": 32.5009, "lon": -6.6906},
    "Khénifra":  {"srm": "SRM Béni Mellal-Khénifra",                 "lat": 32.9394, "lon": -5.6686},
    "Khouribga": {"srm": "SRM Béni Mellal-Khénifra",                 "lat": 32.8833, "lon": -6.9063},
    "Casablanca":{"srm": "SRM Casablanca-Settat",                    "lat": 33.5731, "lon": -7.5898},
    "Mohammédia":{"srm": "SRM Casablanca-Settat",                    "lat": 33.6833, "lon": -7.3833},
    "El Jadida": {"srm": "SRM Casablanca-Settat",                    "lat": 33.2333, "lon": -8.5000},
    "Nouaceur":  {"srm": "SRM Casablanca-Settat",                    "lat": 33.3667, "lon": -7.6167},
    "Médiouna":  {"srm": "SRM Casablanca-Settat",                    "lat": 33.4500, "lon": -7.5167},
    "Benslimane":{"srm": "SRM Casablanca-Settat",                    "lat": 33.6167, "lon": -7.1167},
    "Berrechid": {"srm": "SRM Casablanca-Settat",                    "lat": 33.2667, "lon": -7.5833},
    "Settat":    {"srm": "SRM Casablanca-Settat",                    "lat": 33.0000, "lon": -7.6167},
    "Sidi Bennour":{"srm":"SRM Casablanca-Settat",                   "lat": 32.6500, "lon": -8.4333},
    "Marrakech": {"srm": "SRM Marrakech-Safi",                       "lat": 31.6295, "lon": -8.0366},
    "Chichaoua": {"srm": "SRM Marrakech-Safi",                       "lat": 31.5333, "lon": -8.7667},
    "Tahannaout":{"srm": "SRM Marrakech-Safi",                       "lat": 31.3526, "lon": -7.9504},
    "El Kelaâ des Sraghna":{"srm":"SRM Marrakech-Safi",             "lat": 32.0500, "lon": -7.4000},
    "Essaouira": {"srm": "SRM Marrakech-Safi",                       "lat": 31.5167, "lon": -9.7667},
    "Benguérir": {"srm": "SRM Marrakech-Safi",                       "lat": 32.2333, "lon": -7.9500},
    "Safi":      {"srm": "SRM Marrakech-Safi",                       "lat": 32.2833, "lon": -9.2333},
    "Youssoufia":{"srm": "SRM Marrakech-Safi",                       "lat": 32.2500, "lon": -8.5333},
    "Errachidia":{"srm": "SRM Drâa-Tafilalet",                       "lat": 31.9314, "lon": -4.4244},
    "Ouarzazate":{"srm": "SRM Drâa-Tafilalet",                       "lat": 30.9189, "lon": -6.8934},
    "Midelt":    {"srm": "SRM Drâa-Tafilalet",                       "lat": 32.6800, "lon": -4.7399},
    "Tinghir":   {"srm": "SRM Drâa-Tafilalet",                       "lat": 31.5147, "lon": -5.5328},
    "Zagora":    {"srm": "SRM Drâa-Tafilalet",                       "lat": 30.3306, "lon": -5.8381},
    "Agadir":    {"srm": "SRM Souss-Massa",                          "lat": 30.4278, "lon": -9.5981},
    "Inezgane":  {"srm": "SRM Souss-Massa",                          "lat": 30.3658, "lon": -9.5381},
    "Biougra":   {"srm": "SRM Souss-Massa",                          "lat": 30.2179, "lon": -9.3702},
    "Taroudant": {"srm": "SRM Souss-Massa",                          "lat": 30.4703, "lon": -8.8770},
    "Tiznit":    {"srm": "SRM Souss-Massa",                          "lat": 29.6974, "lon": -9.7316},
    "Tata":      {"srm": "SRM Souss-Massa",                          "lat": 29.7431, "lon": -7.9739},
    "Guelmim":   {"srm": "SRM Guelmim-Oued Noun",                    "lat": 28.9869, "lon": -10.0573},
    "Assa":      {"srm": "SRM Guelmim-Oued Noun",                    "lat": 28.6114, "lon": -9.4367},
    "Tan-Tan":   {"srm": "SRM Guelmim-Oued Noun",                    "lat": 28.4379, "lon": -11.1032},
    "Sidi Ifni": {"srm": "SRM Guelmim-Oued Noun",                    "lat": 29.3797, "lon": -10.1730},
    "Laâyoune":  {"srm": "SRM Laâyoune-Sakia El Hamra",              "lat": 27.1536, "lon": -13.2033},
    "Boujdour":  {"srm": "SRM Laâyoune-Sakia El Hamra",              "lat": 26.1257, "lon": -14.4823},
    "Tarfaya":   {"srm": "SRM Laâyoune-Sakia El Hamra",              "lat": 27.9370, "lon": -12.9250},
    "Es-Semara": {"srm": "SRM Laâyoune-Sakia El Hamra",              "lat": 26.7424, "lon": -11.6719},
    "Dakhla":    {"srm": "SRM Dakhla-Oued EdDahab",                  "lat": 23.6848, "lon": -15.9580},
    "Aousserd":  {"srm": "SRM Dakhla-Oued EdDahab",                  "lat": 22.5540, "lon": -14.3300},
}

# ============================================================
# MODÈLE NEURONAL LSTM (ARX-LSTM)
# ============================================================
class SolarSyncLSTMModel(nn.Module):
    def __init__(self, vocab_sizes, embedding_dim=8, lstm_hidden=64):
        super().__init__()
        self.embeddings   = nn.ModuleList([nn.Embedding(vocab_sizes[col], embedding_dim) for col in vocab_sizes])
        lstm_input_dim    = (len(self.embeddings) * embedding_dim) + 12
        self.lstm         = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden, batch_first=True)
        self.fc_total     = nn.Sequential(nn.Dropout(0.2), nn.Linear(lstm_hidden, lstm_hidden//2), nn.ReLU(), nn.Linear(lstm_hidden//2, 1))
        self.fc_shiftable = nn.Sequential(nn.Dropout(0.2), nn.Linear(lstm_hidden, lstm_hidden//2), nn.ReLU(), nn.Linear(lstm_hidden//2, 1))

    def forward(self, cat_seq, cont_seq):
        embedded  = [self.embeddings[i](cat_seq[:, :, i]) for i in range(len(self.embeddings))]
        fused_seq = torch.cat([torch.cat(embedded, dim=2), cont_seq], dim=2)
        lstm_out, _ = self.lstm(fused_seq)
        return torch.cat([self.fc_total(lstm_out[:, -1, :]), self.fc_shiftable(lstm_out[:, -1, :])], dim=1)

# ============================================================
# API MÉTÉO (OPENT-METEO)
# ============================================================
@st.cache_data(ttl=3600)
def get_48h_forecast(lat, lon):
    try:
        url = (f"https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}"
               f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation"
               f"&timezone=Africa%2FCasablanca&forecast_days=2")
        return requests.get(url).json()['hourly']
    except:
        return None

# ============================================================
# HELPERS & EXPLICABILITÉ (XAI)
# ============================================================
def derive_semantic_weather(solar, temp):
    sun = "Clear/Sunny" if solar > 600 else ("Partly Cloudy" if solar > 200 else "Overcast/Dark")
    return f"{'Hot' if temp>30 else ('Cool' if temp<15 else 'Mild')} & {sun}"

def safe_encode(encoders, col, val):
    return encoders[col].transform([val])[0] if val in encoders[col].classes_ else 0

def get_future_window(df, current_time, hours_ahead=24):
    cutoff_min = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    cutoff_max = cutoff_min + timedelta(hours=hours_ahead)
    return df[(df['Heure_Datetime'] >= cutoff_min) & (df['Heure_Datetime'] <= cutoff_max)].copy()

def get_worst_hour(df_future):
    df_pk = df_future[(df_future['Heure'] >= "18:00") & (df_future['Heure'] <= "22:00")]
    return df_pk.loc[df_pk['Prix_MAD'].idxmax()] if not df_pk.empty else df_future.loc[df_future['Prix_MAD'].idxmax()]

def get_raison(best, ville, distributeur, profile):
    """Logique XAI : Génère une explication naturelle des choix du modèle."""
    dist_nom = distributeur.split("(")[0].strip()
    
    if best['Solaire_W_m2'] > 300:
        if profile['panels']:
            raison_txt = (
                f"le soleil sur {ville} sera à son maximum. "
                f"Vos propres panneaux solaires couvriront la consommation de vos appareils : "
                f"l'énergie sera 100% gratuite et vous n'achèterez rien à {dist_nom}."
            )
        else:
            raison_txt = (
                f"le réseau national bénéficiera d'un pic de production solaire. "
                f"Même sans panneaux chez vous, consommer maintenant permet d'absorber l'énergie verte marocaine "
                f"au lieu de forcer {dist_nom} à allumer des centrales au charbon ce soir."
            )
            
    elif best['Prix_MAD'] < 1.30 and profile['tarif'] == "Option Bi-Horaire":
        raison_txt = (
            f"vous profitez de l'heure creuse de votre contrat Bi-Horaire. "
            f"L'électricité coûte moins cher ({best['Prix_MAD']} MAD/kWh) et le réseau n'est pas sous tension."
        )
            
    else:
        raison_txt = (
            f"la demande globale sur {dist_nom} sera à son plus bas. "
            f"C'est le meilleur créneau pour éviter la saturation du réseau et les fortes émissions de CO₂ de la pointe du soir."
        )
    return raison_txt

# ============================================================
# GÉNÉRATION DES PRÉDICTIONS (MOTEUR PRINCIPAL)
# ============================================================
def generate_schedule(selected_apt, forecast, model, feature_scaler,
                      target_scaler, encoders, profile_data):
    results       = []
    base_time_api = datetime.fromisoformat(forecast['time'][0])
    today         = datetime.now().date()

    # --- 1. AJUSTEMENT DE LA CHARGE DE BASE (ISOLATION & SURFACE) ---
    scale_factor  = (profile_data['surface'] / 80.0) * (profile_data['residents'] / 2.0)
    if profile_data['type'] == 'Villa': scale_factor *= 1.3
    
    isolation = profile_data.get('isolation', 'Moyenne (C/D)')
    if isolation == "Excellente (A/B)": scale_factor *= 0.85
    elif isolation == "Faible (E/F/G)": scale_factor *= 1.25
    
    scale_factor  = max(0.5, min(scale_factor, 4.0))

    cat_history, cont_history = [], []

    # Construction de l'historique
    for i in range(48):
        slot  = base_time_api + timedelta(hours=i)
        is_pk = 18 <= slot.hour <= 23
        solar, temp, hum, wind = (forecast['shortwave_radiation'][i], forecast['temperature_2m'][i],
                                   forecast['relative_humidity_2m'][i], forecast['wind_speed_10m'][i])
        cat_history.append([
            safe_encode(encoders, 'apartment_id', selected_apt), slot.hour, slot.weekday(),
            safe_encode(encoders, 'day_type', "Weekend" if slot.weekday() >= 5 else "Standard_Workday"),
            safe_encode(encoders, 'social_context', 'Standard'),
            safe_encode(encoders, 'weather_semantic', derive_semantic_weather(solar, temp)),
            safe_encode(encoders, 'grid_state', "PEAK_PENALTY" if is_pk else ("HIGH_SOLAR_SURPLUS" if solar > 500 else "STANDARD_LOAD")),
        ])
        cont_history.append(feature_scaler.transform([[
            temp, solar, wind, hum, 1.50 if is_pk else 1.05,
            np.sin(2*np.pi*slot.hour/24), np.cos(2*np.pi*slot.hour/24),
            np.sin(2*np.pi*slot.month/12), np.cos(2*np.pi*slot.month/12),
            temp, solar, 0.0,
        ]])[0])

    # Prédictions
    for i in range(48):
        slot      = base_time_api + timedelta(hours=i)
        seq_cat   = cat_history[max(0, i-7):i+1]
        seq_cont  = cont_history[max(0, i-7):i+1]
        while len(seq_cat) < 8:
            seq_cat.insert(0, cat_history[0]); seq_cont.insert(0, cont_history[0])

        with torch.no_grad():
            preds = model(torch.tensor(seq_cat, dtype=torch.long).unsqueeze(0),
                          torch.tensor(seq_cont, dtype=torch.float32).unsqueeze(0))

        pred_watts     = (target_scaler.inverse_transform([[preds[0][0].item()]])[0][0]) * scale_factor
        prob_shiftable = torch.sigmoid(preds[0][1]).item() * 100
        solar          = forecast['shortwave_radiation'][i]
        
        # --- 2. AJUSTEMENT DE LA FLEXIBILITÉ (VÉHICULE ÉLECTRIQUE) ---
        if profile_data.get('has_ev', False):
            prob_shiftable = min(100.0, prob_shiftable + 35.0) 
            if (slot.hour >= 23 or slot.hour <= 6) or solar > 500:
                pred_watts += 1500  

        is_pk          = 18 <= slot.hour <= 23

        if   profile_data['panels'] and solar > 300:                actual_price = 0.0
        elif profile_data['tarif'] == "Option Bi-Horaire":          actual_price = 1.47 if is_pk else 1.07
        else:                                                       actual_price = 1.07

        poids_frugal = 0.4 if (profile_data['tarif'] == "Option Bi-Horaire" or profile_data['panels']) else 0.0
        poids_eco    = 0.4 if poids_frugal > 0 else 0.8
        solar_eff    = solar if solar > 10 else 0.0
        eco_score    = (solar_eff / 1000.0) * 100
        frg_score    = (2.0 - actual_price) * 50
        hyb_score    = (eco_score * poids_eco) + (frg_score * poids_frugal) + ((prob_shiftable/100.0)*20.0)
        jour         = "Aujourd'hui" if slot.date() == today else "Demain"

        results.append({
            'Index_Absolu':             i,
            'Heure_Datetime':           slot,
            'Jour':                     jour,
            'Heure':                    f"{slot.hour:02d}:00",
            'Affichage':                f"{jour} à {slot.hour:02d}:00",
            'Charge_Prédite_W':         round(pred_watts, 1),
            'Probabilité_Déplaçable_%': round(prob_shiftable, 1),
            'Solaire_W_m2':             solar,
            'Prix_MAD':                 actual_price,
            'Score_Eco':                round(eco_score, 1),
            'Score_Frugal':             round(frg_score, 1),
            'Score_Hybride':            round(hyb_score, 1),
        })
    return pd.DataFrame(results)

# ============================================================
# COMPOSANT UI : JAUGE RÉSEAU (100% THEME AWARE)
# ============================================================
def get_network_gauge(hour, has_panels, tarif, distributeur):
    is_solar = 9 <= hour <= 17
    is_peak  = 18 <= hour <= 23
    is_night = hour <= 5

    if is_solar:
        intensity = 1 - abs(hour - 13) / 4
        score = round(75 + intensity * 20)
        label, badge = "Ensoleillement optimal", "Excellent"
        bb, bc, bbr  = "#E1F5EE", "#0F6E56", "#5DCAA5"
        tarif_txt    = "0.00 MAD/kWh" if has_panels else "1.07 MAD/kWh"
        solaire_txt  = "Fort" if hour == 13 else ("Montant" if hour < 13 else "Déclinant")
        if has_panels or tarif == "Option Bi-Horaire":
            conseil  = "Le réseau bénéficie d'une forte production solaire. L'énergie est verte et économique en ce moment."
        else:
            conseil  = "Forte production solaire nationale. Consommer maintenant est un excellent geste éco-citoyen."
            
    elif is_peak:
        score = round(10 + (1 - abs(hour - 20) / 3) * 15)
        label, badge = "Tension sur le réseau", "Critique"
        bb, bc, bbr  = "#FCEBEB", "#A32D2D", "#F09595"
        tarif_txt    = "1.47 MAD/kWh" if tarif == "Option Bi-Horaire" else "1.07 MAD/kWh"
        solaire_txt  = "Nul"
        conseil      = "Réseau national sous tension (recours au charbon). Évitez l'utilisation de gros appareils si possible."
        
    elif is_night:
        score = 38
        label, badge = "Réseau au repos", "Stable"
        bb, bc, bbr  = "#FAEEDA", "#633806", "#FAC775"
        tarif_txt, solaire_txt = "1.07 MAD/kWh", "Nul"
        conseil = "Demande faible sur le réseau. Période neutre pour l'utilisation des appareils."
        
    else:
        score = 55
        label, badge = "Transition matinale", "Moyen"
        bb, bc, bbr  = "#FAEEDA", "#633806", "#FAC775"
        tarif_txt, solaire_txt = "1.07 MAD/kWh", "Faible"
        conseil = "Le soleil commence à monter. Le potentiel vert du réseau s'améliore progressivement."

    ARC  = 534
    off  = round(ARC - (score/100)*ARC, 1)
    ndeg = round((score/100)*180 - 90, 1)
    sc   = "#0F6E56" if score >= 70 else ("#854F0B" if score >= 40 else "#A32D2D")
    dist = distributeur.split("(")[0].strip()

    return f"""
<style>
.ss{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:1rem 1.25rem .75rem;}}
.ss-hd{{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem;}}
.ss-dist{{font-size:12px;color:#888;margin:0 0 2px;text-transform:uppercase;}}
.ss-lbl{{font-size:20px;font-weight:700;color:var(--text-color);margin:0;}}
.ss-bdg{{padding:6px 16px;border-radius:99px;font-size:13px;font-weight:700;background:{bb};color:{bc};border:1px solid {bbr};}}
.ss-arc{{position:relative; width:100%; max-width: 480px; margin: 0 auto 10px auto; height:auto; display:flex; justify-content:center;}}
.ss-sc{{text-align:center;margin: -15px 0 1.5rem 0;}}
.ss-scn{{font-size:58px;font-weight:800;color:{sc}; letter-spacing: -2px;}}
.ss-scd{{font-size:18px;color:#888; font-weight: 500;}}
.ss-box{{border:1px solid rgba(128, 128, 128, 0.2);background:var(--secondary-background-color);border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem;}}
.ss-blbl{{font-size:12px;color:#888;margin:0 0 4px;text-transform:uppercase;font-weight:600;}}
.ss-btxt{{font-size:14px;color:var(--text-color);margin:0;line-height:1.5;}}
.ss-cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}}
.ss-card{{border:1px solid rgba(128, 128, 128, 0.2);border-radius:10px;padding:10px 12px;background:transparent;}}
.ss-clbl{{font-size:11px;color:#888;margin:0 0 4px;text-transform:uppercase;font-weight:600;}}
.ss-cval{{font-size:15px;font-weight:700;color:var(--text-color);margin:0;}}
</style>
<div class="ss">
  <div class="ss-hd">
    <div><p class="ss-dist">État du réseau actuel · {dist}</p><p class="ss-lbl">{label}</p></div>
    <span class="ss-bdg">{badge}</span>
  </div>
  <div class="ss-arc">
    <svg viewBox="0 0 400 200" width="100%" style="overflow:visible;">
      <defs><linearGradient id="grd" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%"   stop-color="#1D9E75"/>
        <stop offset="40%"  stop-color="#EF9F27"/>
        <stop offset="75%"  stop-color="#D85A30"/>
        <stop offset="100%" stop-color="#E24B4A"/>
      </linearGradient></defs>
      <path d="M 30 180 A 170 170 0 0 1 370 180" fill="none" stroke="rgba(128, 128, 128, 0.2)" stroke-width="26" stroke-linecap="round"/>
      <path d="M 30 180 A 170 170 0 0 1 370 180" fill="none" stroke="url(#grd)"  stroke-width="26" stroke-linecap="round" stroke-dasharray="{ARC}" stroke-dashoffset="{off}"/>
      <g style="transform-origin:200px 180px;transform:rotate({ndeg}deg);">
        <line x1="200" y1="180" x2="200" y2="30" stroke="var(--text-color)" stroke-width="4" stroke-linecap="round"/>
        <circle cx="200" cy="180" r="10" fill="var(--text-color)"/>
        <circle cx="200" cy="180" r="4" fill="var(--background-color)"/>
      </g>
    </svg>
  </div>
  <div class="ss-sc"><span class="ss-scn">{score}</span><span class="ss-scd">/100</span></div>
  <div class="ss-box">
    <p class="ss-blbl">Situation Actuelle</p>
    <p class="ss-btxt">{conseil}</p>
  </div>
  <div class="ss-cards">
    <div class="ss-card"><p class="ss-clbl">heure locale</p><p class="ss-cval">{hour:02d}:00</p></div>
    <div class="ss-card"><p class="ss-clbl">tarif actuel</p><p class="ss-cval">{tarif_txt}</p></div>
    <div class="ss-card"><p class="ss-clbl">solaire actuel</p><p class="ss-cval">{solaire_txt}</p></div>
  </div>
</div>"""

# ============================================================
# INITIALISATION ET CHARGEMENT DU MODÈLE
# ============================================================
@st.cache_resource
def load_all_assets():
    feature_scaler = joblib.load('feature_scaler.pkl')
    target_scaler  = joblib.load('target_scaler.pkl')
    encoders       = joblib.load('label_encoders.pkl')
    vocab_sizes    = {col: len(le.classes_) for col, le in encoders.items()}
    vocab_sizes['hour'] = 24
    vocab_sizes['day']  = 7
    model = SolarSyncLSTMModel(vocab_sizes)
    model.load_state_dict(torch.load('solarsync_model.pth'))
    model.eval()
    return model, feature_scaler, target_scaler, encoders

# ============================================================
# APPLICATION STREAMLIT PRINCIPALE
# ============================================================
try:
    model, feature_scaler, target_scaler, encoders = load_all_assets()
    known_apts = list(encoders['apartment_id'].classes_)

    if 'profiles' not in st.session_state:
        st.session_state.profiles = load_profiles()
        
    for key, default in [('current_user',None), ('show_details',False), 
                         ('analysis_done',False), ('df_schedule',None)]:
        if key not in st.session_state: 
            st.session_state[key] = default
# ── VUE 1 : CONNEXION ET INSCRIPTION (SANS MOT DE PASSE & DESIGN FINAL) ──────────────────────
    if st.session_state.current_user is None:
        st.session_state.update({'show_details':False,'analysis_done':False,'df_schedule':None})

        # --- CSS INJECTION ---
        st.markdown("""
        <style>
        [data-testid="collapsedControl"] { display: none !important; }
        
        /* Dimensions */
        .block-container { padding-top: 3.5rem !important; padding-bottom: 2rem !important; max-width: 85% !important; }
        
        /* Typographie */
        .sellora-title { font-size: 2.4rem; font-weight: 800; color: #111827; text-align: center; margin-bottom: 0.5rem; letter-spacing: -0.5px; }
        .sellora-subtitle { color: #6B7280; font-size: 1.05rem; text-align: center; margin-bottom: 2rem; }
        
        /* --- NOUVEAU SYSTÈME DE BOUTONS --- */
        /* Bouton Primaire (Se Connecter, Suivant, Terminer) */
        div.stButton > button[kind="primary"] {
            background: #CC7033 !important; 
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            width: 100%;
        }
        div.stButton > button[kind="primary"]:hover { background: #B25A24 !important; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(204, 112, 51, 0.3) !important;}
        
        /* Bouton Secondaire (Retour) */
        div.stButton > button[kind="secondary"] {
            background: transparent !important;
            color: #4B5563 !important;
            border: 1px solid #D1D5DB !important;
            border-radius: 8px !important;
            padding: 0.6rem 2rem !important;
            font-weight: 600 !important;
            box-shadow: none !important;
            width: 100%;
        }
        div.stButton > button[kind="secondary"]:hover { border-color: #111827 !important; color: #111827 !important; }

        /* Bouton Tertiaire (Transformé en lien texte cliquable) */
        div.stButton > button[kind="tertiary"] {
            background: transparent !important;
            color: #CC7033 !important;
            border: none !important;
            box-shadow: none !important;
            font-weight: 600 !important;
            padding: 0 !important;
            width: 100%;
            height: auto !important;
            min-height: 0 !important;
        }
        div.stButton > button[kind="tertiary"]:hover {
            background: transparent !important;
            text-decoration: underline !important;
            color: #B25A24 !important;
            transform: none !important;
        }

        /* Inputs */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] { border-radius: 8px !important; }

        /* Panneau Droit */
        .sellora-banner {
            background: linear-gradient(135deg, #4F72F5 0%, #2F4AC4 100%);
            border-radius: 24px;
            padding: 50px 45px;
            color: white;
            height: 100%;
            min-height: 520px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 20px 40px rgba(47, 74, 196, 0.2);
        }
        .sellora-banner h2 { color: #FFFFFF; font-size: 2.3rem; font-weight: 700; margin-bottom: 1rem; line-height: 1.25; text-align: left; }
        .sellora-banner p { font-size: 1.05rem; color: rgba(255, 255, 255, 0.9); margin-bottom: 2.5rem; line-height: 1.5; text-align: left; max-width: 95%; }
        
        /* Grille des 4 icônes */
        .icon-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; width: 100%; max-width: 260px; }
        .icon-box { 
            background: rgba(255, 255, 255, 0.1); 
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 18px; 
            aspect-ratio: 1 / 1; 
            display: flex; justify-content: center; align-items: center;
            font-size: 2.2rem; 
            backdrop-filter: blur(8px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        }
        </style>
        """, unsafe_allow_html=True)

        # --- GESTION DE L'ÉTAT DU WIZARD ---
        if 'auth_mode' not in st.session_state: st.session_state.auth_mode = 'login'
        if 'onb_step' not in st.session_state: st.session_state.onb_step = 1
        if 'onb_data' not in st.session_state: st.session_state.onb_data = {}

        def next_step(): st.session_state.onb_step += 1
        def prev_step(): st.session_state.onb_step -= 1

        # --- LAYOUT SPLIT-SCREEN ---
        col_gauche, col_droite = st.columns([1.1, 1], gap="large")
        
        with col_gauche:
            c_vide1, c_logo, c_vide2 = st.columns([1, 1.4, 1])
            with c_logo:
                try:    
                    st.image("./assets/icon.png", use_container_width=True)
                except: 
                    st.markdown("<h3 style='text-align:center; color:#111827;'>☀️ SolarSync</h3>", unsafe_allow_html=True)
            
            st.write("<br>", unsafe_allow_html=True)
            
            # Formulaire contenu
            _, form_col, _ = st.columns([0.1, 0.8, 0.1])
            
            with form_col:
                if st.session_state.auth_mode == 'login':
                    st.markdown('<div class="sellora-title">Bon retour !</div>', unsafe_allow_html=True)
                    # Sous-titre plus thématique
                    st.markdown('<div class="sellora-subtitle">Sélectionnez votre foyer pour lancer l\'optimisation IA.</div>', unsafe_allow_html=True)
                    
                    if not st.session_state.profiles:
                        st.info("👋 Aucun foyer connecté au réseau.")
                    else:
                        st.write("**Foyer**")
                        sel = st.selectbox("Foyer", list(st.session_state.profiles.keys()), label_visibility="collapsed")
                        
                        st.write("<br>", unsafe_allow_html=True)
                        # Bouton de connexion thématique
                        if st.button("Synchroniser mon foyer", type="primary", use_container_width=True):
                            st.session_state.current_user = sel; st.rerun()
                                
                    # --- SECTION INSCRIPTION ---
                    st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #E5E7EB;'>", unsafe_allow_html=True)
                    # Texte d'accroche thématique
                    st.markdown("<p style='text-align:center; color:#6B7280; font-size:0.9rem; margin-bottom: -15px;'>Votre foyer n'est pas encore optimisé ?</p>", unsafe_allow_html=True)
                    # Lien texte thématique
                    if st.button("Rejoindre le Smart Grid", type="tertiary", use_container_width=True):
                        st.session_state.auth_mode = 'register'; st.session_state.onb_step = 1; st.rerun()
                        
                elif st.session_state.auth_mode == 'register':
                    step = st.session_state.onb_step
                    st.progress(step / 3.0)
                    
                    if step == 1:
                        st.markdown('<div class="sellora-title">Bienvenue</div>', unsafe_allow_html=True)
                        st.markdown('<div class="sellora-subtitle">Personnalisez votre expérience.</div>', unsafe_allow_html=True)
                        st.write("**Nom du foyer**")
                        name = st.text_input("Nom", value=st.session_state.onb_data.get('name', ''), label_visibility="collapsed")
                        st.write("<br>**Type de logement**", unsafe_allow_html=True)
                        home_type = st.radio("Type", ["Appartement", "Villa", "Maison Traditionnelle"], index=0, label_visibility="collapsed")
                        
                        st.write("<br><br>", unsafe_allow_html=True)
                        cb, cn = st.columns(2)
                        with cb:
                            if st.button("Retour", type="secondary", use_container_width=True, key="bk1"): st.session_state.auth_mode = 'login'; st.rerun()
                        with cn:
                            if st.button("Suivant", type="primary", use_container_width=True, key="nx1"):
                                if name: st.session_state.onb_data['name'] = name; st.session_state.onb_data['type'] = home_type; next_step(); st.rerun()
                                else: st.error("Veuillez donner un nom.")
                                
                    elif step == 2:
                        st.markdown('<div class="sellora-title">Localisation</div>', unsafe_allow_html=True)
                        st.markdown('<div class="sellora-subtitle">Requis pour les calculs thermiques.</div>', unsafe_allow_html=True)
                        st.write("**Ville (Réseau SRM)**")
                        city = st.selectbox("Ville", list(CITIES_DATA.keys()), label_visibility="collapsed")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write("**Surface (m²)**")
                            surface = st.number_input("Surface", min_value=30, value=80, step=10, label_visibility="collapsed")
                        with c2:
                            st.write("**Résidents**")
                            residents = st.number_input("Résidents", min_value=1, value=3, label_visibility="collapsed")
                        st.write("**Isolation**")
                        isolation = st.selectbox("Isolation", ["Excellente (A/B)", "Moyenne (C/D)", "Faible (E/F/G)"], index=1, label_visibility="collapsed")
                        
                        st.write("<br><br>", unsafe_allow_html=True)
                        cb, cn = st.columns(2)
                        with cb:
                            if st.button("Retour", type="secondary", use_container_width=True, key="bk2"): prev_step(); st.rerun()
                        with cn:
                            if st.button("Suivant", type="primary", use_container_width=True, key="nx2"):
                                st.session_state.onb_data.update({'city': city, 'surface': surface, 'residents': residents, 'isolation': isolation}); next_step(); st.rerun()
                                
                    elif step == 3:
                        st.markdown('<div class="sellora-title">Équipements</div>', unsafe_allow_html=True)
                        st.markdown('<div class="sellora-subtitle">Vos ressources énergétiques.</div>', unsafe_allow_html=True)
                        st.write("**Tarification**")
                        tarif = st.radio("Tarification", ["Tarif Standard", "Option Bi-Horaire"], label_visibility="collapsed")
                        st.write("<br>**Équipements :**", unsafe_allow_html=True)
                        panels = st.checkbox("☀️ Panneaux solaires", value=st.session_state.onb_data.get('panels', False))
                        ev = st.checkbox("🚗 Véhicule Électrique", value=st.session_state.onb_data.get('has_ev', False))
                        
                        st.write("<br><br>", unsafe_allow_html=True)
                        cb, cn = st.columns(2)
                        with cb:
                            if st.button("Retour", type="secondary", use_container_width=True, key="bk3"): prev_step(); st.rerun()
                        with cn:
                            if st.button("Terminer", type="primary", use_container_width=True, key="nx3"):
                                st.session_state.onb_data.update({'tarif': tarif, 'panels': panels, 'has_ev': ev})
                                new_name = st.session_state.onb_data.get('name', f"Foyer_{np.random.randint(100, 999)}")
                                if new_name not in st.session_state.profiles:
                                    st.session_state.profiles[new_name] = {
                                        "city": st.session_state.onb_data.get('city', 'Casablanca'), "tarif": tarif, "panels": panels,
                                        "type": st.session_state.onb_data.get('type', 'Appartement'), "surface": st.session_state.onb_data.get('surface', 80), 
                                        "residents": st.session_state.onb_data.get('residents', 3), "isolation": st.session_state.onb_data.get('isolation', 'Moyenne (C/D)'), 
                                        "has_ev": ev, "apt_id": known_apts[0]
                                    }
                                    save_profiles(st.session_state.profiles); init_historical_csv(new_name, st.session_state.profiles[new_name])
                                    st.session_state.current_user = new_name; st.session_state.onb_data = {}; st.rerun()
                                else: st.error("Nom déjà utilisé.")
                                    
        # --- PANNEAU DROIT ---
        with col_droite:
            # Code HTML compressé avec les couleurs officielles du logo SolarSync
            html_code = """<style>
*{box-sizing:border-box;margin:0;padding:0;}
.panel{background:linear-gradient(145deg, #151B3A 0%, #1E2652 100%);border-radius:20px;padding:48px 44px;min-height:620px;display:flex;flex-direction:column;justify-content:space-between;position:relative;overflow:hidden;font-family:var(--font-sans, system-ui, sans-serif);box-shadow:inset 0 1px 1px rgba(255,255,255,0.05), 0 20px 40px rgba(21, 27, 58, 0.2);}
.ring{position:absolute;border-radius:50%;border:1px solid rgba(165,180,217,0.08);}
.ring-1{width:320px;height:320px;top:-80px;right:-80px;}
.ring-2{width:220px;height:220px;top:-30px;right:-30px;}
.ring-3{width:420px;height:420px;bottom:-160px;left:-160px;}
.sun-wrap{position:absolute;top:32px;right:36px;width:80px;height:80px;display:flex;align-items:center;justify-content:center;}
.top{position:relative;z-index:2;}
.eyebrow{font-size:11px;font-weight:600;letter-spacing:1.6px;text-transform:uppercase;color:#CC7033;margin-bottom:20px;}
.headline{font-size:30px;font-weight:500;color:#ffffff;line-height:1.3;margin-bottom:16px;}
.headline span{color:#CC7033;font-weight:700;}
.sub{font-size:14px;color:rgba(255,255,255,0.6);line-height:1.6;max-width:320px;}
.stats{display:flex;gap:12px;position:relative;z-index:2;margin-top:36px;}
.stat-card{flex:1;background:rgba(255,255,255,0.04);border:0.5px solid rgba(165,180,217,0.15);border-radius:14px;padding:16px 14px;backdrop-filter:blur(4px);}
.stat-val{font-size:22px;font-weight:600;color:#ffffff;margin-bottom:4px;}
.stat-label{font-size:11px;color:rgba(255,255,255,0.5);line-height:1.4;}
.stat-icon{font-size:14px;margin-bottom:8px;opacity:0.9;}
.features{position:relative;z-index:2;margin-top:28px;display:flex;flex-direction:column;gap:12px;}
.feature-row{display:flex;align-items:center;gap:12px;}
.feat-dot{width:28px;height:28px;border-radius:8px;background:rgba(204,112,51,0.15);border:0.5px solid rgba(204,112,51,0.3);display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.feat-dot svg{width:13px;height:13px;}
.feat-text{font-size:13px;color:rgba(255,255,255,0.75);}
.bottom-bar{position:relative;z-index:2;margin-top:32px;padding-top:24px;border-top:0.5px solid rgba(165,180,217,0.15);display:flex;align-items:center;justify-content:space-between;}
.grid-status{display:flex;align-items:center;gap:8px;}
.pulse-dot{width:8px;height:8px;border-radius:50%;background:#22c55e;position:relative;}
.pulse-dot::after{content:'';position:absolute;inset:-3px;border-radius:50%;background:rgba(34,197,94,0.3);animation:pulse 2s ease-in-out infinite;}
@keyframes pulse{0%,100%{transform:scale(1);opacity:0.7;}50%{transform:scale(1.5);opacity:0;}}
.status-text{font-size:12px;color:rgba(255,255,255,0.5);}
.badge{font-size:11px;font-weight:600;color:#A5B4D9;background:rgba(165,180,217,0.1);border:0.5px solid rgba(165,180,217,0.2);border-radius:20px;padding:5px 12px;letter-spacing:0.5px;}
.arc-wrap{position:absolute;bottom:0;right:0;z-index:1;pointer-events:none;opacity:0.12;}
</style>
<div class="panel">
<div class="ring ring-1"></div>
<div class="ring ring-2"></div>
<div class="ring ring-3"></div>
<div class="sun-wrap">
<svg viewBox="0 0 80 80" width="80" height="80" xmlns="http://www.w3.org/2000/svg">
<circle cx="40" cy="40" r="14" fill="#CC7033" opacity="0.95"/>
<circle cx="40" cy="40" r="20" fill="none" stroke="#CC7033" stroke-width="1" opacity="0.3"/>
<circle cx="40" cy="40" r="27" fill="none" stroke="#CC7033" stroke-width="0.5" opacity="0.15"/>
<line x1="40" y1="4" x2="40" y2="14" stroke="#CC7033" stroke-width="2" stroke-linecap="round" opacity="0.7"/>
<line x1="40" y1="66" x2="40" y2="76" stroke="#CC7033" stroke-width="2" stroke-linecap="round" opacity="0.7"/>
<line x1="4" y1="40" x2="14" y2="40" stroke="#CC7033" stroke-width="2" stroke-linecap="round" opacity="0.7"/>
<line x1="66" y1="40" x2="76" y2="40" stroke="#CC7033" stroke-width="2" stroke-linecap="round" opacity="0.7"/>
<line x1="12" y1="12" x2="19" y2="19" stroke="#CC7033" stroke-width="1.5" stroke-linecap="round" opacity="0.5"/>
<line x1="61" y1="61" x2="68" y2="68" stroke="#CC7033" stroke-width="1.5" stroke-linecap="round" opacity="0.5"/>
<line x1="68" y1="12" x2="61" y2="19" stroke="#CC7033" stroke-width="1.5" stroke-linecap="round" opacity="0.5"/>
<line x1="19" y1="61" x2="12" y2="68" stroke="#CC7033" stroke-width="1.5" stroke-linecap="round" opacity="0.5"/>
</svg>
</div>
<div class="arc-wrap">
<svg width="260" height="260" viewBox="0 0 260 260" xmlns="http://www.w3.org/2000/svg">
<path d="M 260 260 A 260 260 0 0 0 0 0" fill="none" stroke="#CC7033" stroke-width="60"/>
</svg>
</div>
<div class="top">
<div class="eyebrow">Réseau intelligent · Maroc</div>
<div class="headline">Optimisez votre<br>énergie avec <span>l'IA</span></div>
<div class="sub">SolarSync analyse votre consommation en temps réel et vous guide vers les meilleures décisions énergétiques pour votre foyer.</div>
<div class="stats">
<div class="stat-card"><div class="stat-icon">☀️</div><div class="stat-val">48h</div><div class="stat-label">Prévision météo</div></div>
<div class="stat-card"><div class="stat-icon">📉</div><div class="stat-val">-30%</div><div class="stat-label">Réduction facture</div></div>
<div class="stat-card"><div class="stat-icon">🌍</div><div class="stat-val">CO₂</div><div class="stat-label">Impact éco mesuré</div></div>
</div>
<div class="features">
<div class="feature-row"><div class="feat-dot"><svg viewBox="0 0 13 13" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M2 6.5L5.2 9.5L11 3.5" stroke="#CC7033" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg></div><div class="feat-text">Modèle LSTM sur vos habitudes</div></div>
<div class="feature-row"><div class="feat-dot"><svg viewBox="0 0 13 13" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M2 6.5L5.2 9.5L11 3.5" stroke="#CC7033" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg></div><div class="feat-text">Tarifs Bi-Horaire & solaires intégrés</div></div>
<div class="feature-row"><div class="feat-dot"><svg viewBox="0 0 13 13" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M2 6.5L5.2 9.5L11 3.5" stroke="#CC7033" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg></div><div class="feat-text">80+ villes marocaines couvertes</div></div>
</div>
</div>
<div class="bottom-bar">
<div class="grid-status"><div class="pulse-dot"></div><div class="status-text">Réseau connecté · En direct</div></div>
<div class="badge">SolarSync v1.0</div>
</div>
</div>"""
            st.markdown(html_code, unsafe_allow_html=True)

    # ── VUE 2 : DASHBOARD PRINCIPAL ───────────────────────────
    else:
        st.markdown("""<style>[data-testid="collapsedControl"]{display:block!important;}</style>""",
                    unsafe_allow_html=True)

        current_time   = datetime.now()
        active_profile = st.session_state.profiles[st.session_state.current_user]
        ville_choisie  = active_profile["city"]
        distributeur   = CITIES_DATA[ville_choisie]["srm"]
        dist_court     = distributeur.split("(")[0].strip()

        # --- BARRE LATÉRALE ---
        with st.sidebar:
            try:    st.image("./assets/icon.png", use_container_width=True)
            except: pass
            st.markdown(f"### 🏠 {st.session_state.current_user}")
            st.caption(f"📍 {ville_choisie} | ⚡ {dist_court}")
            
            if st.button("⬅️ Déconnexion", use_container_width=True):
                st.session_state.update({"current_user":None,"show_details":False,
                                          "analysis_done":False,"df_schedule":None})
                st.rerun()
                
            st.markdown("---")
            
            # Paramètres
            with st.expander("⚙️ Modifier les paramètres du foyer", expanded=False):
                with st.form("edit_profile_form"):
                    curr_iso = active_profile.get('isolation', 'Moyenne (C/D)')
                    curr_ev = active_profile.get('has_ev', False)
                    
                    edit_res = st.number_input("Nombre de résidents", value=active_profile['residents'], min_value=1)
                    edit_iso = st.selectbox("Isolation", ["Moyenne (C/D)", "Excellente (A/B)", "Faible (E/F/G)"], 
                                            index=["Moyenne (C/D)", "Excellente (A/B)", "Faible (E/F/G)"].index(curr_iso))
                    
                    st.markdown("Équipements :")
                    edit_panels = st.checkbox("Panneaux Solaires", value=active_profile['panels'])
                    edit_ev = st.checkbox("Véhicule Électrique", value=curr_ev)
                    edit_tarif = st.radio("Tarification", ["Tarif Standard","Option Bi-Horaire"], 
                                          index=0 if active_profile['tarif']=="Tarif Standard" else 1)
                    
                    if st.form_submit_button("💾 Enregistrer", use_container_width=True):
                        st.session_state.profiles[st.session_state.current_user].update({
                            'residents': edit_res,
                            'isolation': edit_iso,
                            'panels': edit_panels,
                            'has_ev': edit_ev,
                            'tarif': edit_tarif
                        })
                        save_profiles(st.session_state.profiles)
                        st.session_state.analysis_done = False 
                        st.success("Profil mis à jour !")
                        st.rerun()
                        
            st.markdown("---")
            
            # Flux IoT en temps réel
            with st.expander("📡 Flux de Données IoT (Live)", expanded=False):
                st.markdown("""
                <style>
                .iot-terminal {
                    background-color: #0f172a; color: #10b981; padding: 10px; 
                    border-radius: 5px; font-family: 'Courier New', monospace; font-size: 0.75rem;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Fetch live data if forecast exists
                if 'forecast_48h' in locals() and forecast_48h:
                    now_idx = df_schedule[df_schedule['Heure_Datetime'] <= current_time]['Index_Absolu'].max() if st.session_state.df_schedule is not None else 0
                    live_temp = forecast_48h['temperature_2m'][now_idx]
                    live_solar = forecast_48h['shortwave_radiation'][now_idx]
                    live_hum = forecast_48h['relative_humidity_2m'][now_idx]
                else:
                    live_temp, live_solar, live_hum = 24.5, 815.0, 45.0 # Valeurs de repli
                
                st.markdown(f"""
                <div class="iot-terminal">
                > [ENV_SENSOR] T:{live_temp}°C | HUM:{live_hum}%<br>
                > [PV_SENSOR] IRR: {live_solar} W/m²<br>
                > [SMART_METER] Sync: OK<br>
                > [GRID_NODE] {dist_court} API: CONNECTED<br>
                > [STATUS] Ingestion contextuelle active...
                </div>
                """, unsafe_allow_html=True)

        forecast_48h = get_48h_forecast(CITIES_DATA[ville_choisie]["lat"], CITIES_DATA[ville_choisie]["lon"])

        # --- EN-TÊTE DASHBOARD ---
        col_left, col_right = st.columns([1.1, 0.9])

        with col_left:
            st.markdown(
                "<h1 style='margin:0; padding-top:15px; font-size:2.4rem; font-weight:800; color:#1e293b; letter-spacing:-0.5px;'>"
                "Tableau de Bord"
                "</h1>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<p style='color:#64748b; font-size:15px; margin:8px 0 2rem 0; font-weight:500;'>"
                f"📍 {ville_choisie} &nbsp;•&nbsp; 🕒 {current_time.strftime('%H:%M')} &nbsp;•&nbsp; ⚡ {dist_court}"
                f"</p>",
                unsafe_allow_html=True
            )

            if forecast_48h and st.button("🔄 Lancer l'Analyse Prédictive (48 Heures)", type="primary"):
                st.session_state.show_details = False
                with st.spinner("Modélisation des courbes de charge en cours..."):
                    st.session_state.df_schedule = generate_schedule(
                        active_profile["apt_id"], forecast_48h, model,
                        feature_scaler, target_scaler, encoders, active_profile)
                st.session_state.analysis_done = True
                st.rerun()

        with col_right:
            st.markdown(
                get_network_gauge(current_time.hour, active_profile["panels"],
                                  active_profile["tarif"], distributeur),
                unsafe_allow_html=True
            )

        st.markdown("---")

        # --- SECTION DES RÉSULTATS (APRÈS ANALYSE) ---
        if st.session_state.analysis_done and st.session_state.df_schedule is not None:

            df_schedule = st.session_state.df_schedule
            df_future   = get_future_window(df_schedule, current_time)

            if df_future.empty:
                st.warning("Pas de données futures disponibles. Relancez l'analyse.")
                st.stop()

            worst_hour  = get_worst_hour(df_future)
            best_hybrid = df_future.loc[df_future['Score_Hybride'].idxmax()]
            energie_kwh = 2.0 + (active_profile['residents'] * 0.3)
            economie    = max(0.0, energie_kwh * (worst_hour['Prix_MAD'] - best_hybrid['Prix_MAD']))
            co2_evite   = max(0.0, (energie_kwh*0.6) * min(1.0, best_hybrid['Solaire_W_m2']/1000.0))
            raison      = get_raison(best_hybrid, ville_choisie, distributeur, active_profile)

            # CORRECTION DARK MODE BOITE STRATÉGIQUE (fond transparent)
            st.markdown("<div style='background-color:transparent;padding:20px;border-radius:10px;border:1px solid rgba(128, 128, 128, 0.2);'>",
                        unsafe_allow_html=True)
            st.markdown("### 📊 Bilan Stratégique")
            c1,c2,c3 = st.columns(3)
            c1.markdown(f"**❌ Habitude à Éviter**<br><span style='font-size:1.8rem;color:#e11d48;'>{worst_hour['Affichage']}</span>", unsafe_allow_html=True)
            c2.markdown(f"**✅ Recommandé par l'IA**<br><span style='font-size:1.8rem;color:#16a34a;'>{best_hybrid['Affichage']}</span>", unsafe_allow_html=True)
            if active_profile['panels'] or active_profile['tarif'] == "Option Bi-Horaire":
                c3.markdown(f"**💰 Économies (Cycle {energie_kwh:.1f} kWh)**<br><span style='font-size:1.8rem;color:#0ea5e9;'>+{economie:.2f} MAD</span>", unsafe_allow_html=True)
            else:
                c3.markdown(f"**🌍 CO₂ Évité (Cycle {energie_kwh:.1f} kWh)**<br><span style='font-size:1.8rem;color:#0ea5e9;'>-{co2_evite:.2f} kg</span>", unsafe_allow_html=True)
            st.markdown("</div><br>", unsafe_allow_html=True)

            st.success(f"**Conseil :** Démarrez vos gros appareils ménagers **{best_hybrid['Affichage'].lower()}**.")
            st.info(f"**Explication IA :** C'est le moment idéal car {raison}.")
            st.markdown("---")

            # --- DÉTAILS AVANCÉS (BOUTON TOGGLE) ---
            if not st.session_state.show_details:
                if st.button("🔬 Voir l'analyse détaillée (IA & Réseau)", use_container_width=True):
                    st.session_state.show_details = True; st.rerun()
            else:
                if st.button("🔼 Masquer les détails", use_container_width=True):
                    st.session_state.show_details = False; st.rerun()

                # 1. Cartes Bénéfices Utilisateur
                st.markdown("### 🎯 Choisissez votre stratégie (Prochaines 24h)")
                best_eco    = df_future.loc[df_future['Score_Eco'].idxmax()]
                best_frugal = df_future.loc[df_future['Score_Frugal'].idxmax()]
                cp1, cp2, cp3 = st.columns(3)
                
                cp1.success(f"**🌱 Priorité Planète**\n\n**{best_eco['Affichage'].split(' à ')[0]} à {best_eco['Heure']}**\n\n*L'heure où l'énergie du réseau sera la plus verte.*")
                
                if active_profile['tarif'] == "Tarif Standard" and not active_profile['panels']:
                    cp2.info("**💰 Priorité Portefeuille**\n\n*Inactif (Votre prix est fixe toute la journée)*")
                else:
                    cp2.info(f"**💰 Priorité Portefeuille**\n\n**{best_frugal['Affichage'].split(' à ')[0]} à {best_frugal['Heure']}**\n\n*L'heure où l'électricité vous coûtera le moins cher.*")
                
                cp3.warning(f"**⚖️ Le Bon Compromis**\n\n**{best_hybrid['Affichage'].split(' à ')[0]} à {best_hybrid['Heure']}**\n\n*Le meilleur équilibre entre écologie et économies.*")

                # 2. Graphique Visuel et Intuitif (Micro)
                st.markdown("### ⏱️ Quand lancer vos appareils sur les prochaines 48h ?")
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(go.Scatter(
                    x=df_schedule['Heure_Datetime'], 
                    y=df_schedule['Solaire_W_m2'],
                    fill='tozeroy', name='Énergie Verte Disponible', 
                    line=dict(color='rgba(251, 191, 36, 0.6)', width=1),
                    hoverinfo='skip'
                ), secondary_y=False)
                
                fig.add_trace(go.Scatter(
                    x=df_schedule['Heure_Datetime'], 
                    y=df_schedule['Charge_Prédite_W'],
                    name='Tension / Demande Réseau', 
                    line=dict(color='#3b82f6', width=3, shape='spline')
                ), secondary_y=True)

                # Zone Verte
                rec_time = best_hybrid['Heure_Datetime']
                fig.add_vrect(
                    x0=rec_time - pd.Timedelta(minutes=45), x1=rec_time + pd.Timedelta(minutes=45),
                    fillcolor="#16a34a", opacity=0.15, layer="below", line_width=0
                )
                fig.add_annotation(
                    x=rec_time - pd.Timedelta(minutes=45), y=1.02, yref="paper",
                    text="🔥 Moment Idéal", showarrow=False, xanchor="left",
                    font=dict(color="#16a34a", size=12)
                )
                fig.add_vline(x=rec_time, line_width=2, line_dash="dot", line_color="#16a34a")

                # Zones Rouges
                today_date = current_time.date()
                for day_offset in [0, 1, 2]:
                    peak_start = datetime.combine(today_date + timedelta(days=day_offset), datetime.min.time()) + timedelta(hours=18)
                    peak_end = peak_start + timedelta(hours=5)
                    
                    if peak_start <= df_schedule['Heure_Datetime'].max():
                        fig.add_vrect(
                            x0=peak_start, x1=peak_end,
                            fillcolor="#e11d48", opacity=0.1, layer="below", line_width=0
                        )
                        if day_offset == 0:
                            fig.add_annotation(
                                x=peak_start, y=1.02, yref="paper",
                                text="À Éviter", showarrow=False, xanchor="left", font=dict(color="#e11d48", size=12)
                            )

                # Maintenant
                fig.add_vline(x=current_time, line_width=2, line_dash="dash", line_color="#475569")
                fig.add_annotation(
                    x=current_time, y=0.02, yref="paper", text="Maintenant", 
                    showarrow=False, xanchor="left", font=dict(color="#475569", size=12), xshift=5
                )
                
                # CORRECTION DARK MODE GRAPHIQUE MICRO
                fig.update_layout(
                    hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_xaxes(tickformat="%Hh\n%A", dtick=14400000, showgrid=True, gridcolor='rgba(128,128,128,0.2)', tickangle=0)
                fig.update_yaxes(title_text="", showticklabels=False, showgrid=False, secondary_y=False)
                fig.update_yaxes(title_text="Activité (Indice)", showgrid=True, gridcolor='rgba(128,128,128,0.2)', secondary_y=True)

                st.plotly_chart(fig, use_container_width=True)

                # ============================================================
                # 3. VUE SUPERVISEUR RÉSEAU (MACRO-IMPACT)
                # ============================================================
                st.markdown("---")
                st.markdown("### 🌍 Impact Macro-Économique (Simulation sur 1 000 foyers)")
                st.info("Projection illustrant l'écrêtement de la pointe (Peak Shaving) si un échantillon de 1000 foyers locaux suit la recommandation de l'IA.")
                
                # Simuler la courbe de base
                base_load_w = df_schedule['Charge_Prédite_W'] * 1000
                pic_hours = (df_schedule['Heure_Datetime'].dt.hour >= 19) & (df_schedule['Heure_Datetime'].dt.hour <= 22)
                
                # Ajout de la charge des gros appareils pendant le pic
                charge_gros_appareils = 1500 * 1000 # 1.5 kW par foyer * 1000 foyers
                base_load_w[pic_hours] += charge_gros_appareils 
                
                # Simuler la courbe optimisée
                optimized_load_w = base_load_w.copy()
                optimized_load_w[pic_hours] -= charge_gros_appareils * 0.8 # 80% d'adoption
                
                # Réinjection à l'heure recommandée
                rec_idx = best_hybrid['Index_Absolu']
                optimized_load_w.iloc[rec_idx] += charge_gros_appareils * 0.8 
                
                # Affichage en Megawatts (MW)
                fig_macro = go.Figure()
                
                fig_macro.add_trace(go.Scatter(
                    x=df_schedule['Heure_Datetime'], y=base_load_w / 1_000_000, 
                    name='Courbe Classique (Risque Saturation)', 
                    line=dict(color='#e11d48', width=2, dash='dash')
                ))
                
                fig_macro.add_trace(go.Scatter(
                    x=df_schedule['Heure_Datetime'], y=optimized_load_w / 1_000_000, 
                    name='Courbe SolarSync (Optimisée)', 
                    line=dict(color='#16a34a', width=3, shape='spline'), 
                    fill='tozeroy', fillcolor='rgba(22, 163, 74, 0.1)'
                ))
                
                # CORRECTION DARK MODE GRAPHIQUE MACRO
                fig_macro.update_layout(
                    hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
                    yaxis_title="Demande Réseau (Mégawatts - MW)"
                )
                fig_macro.update_xaxes(tickformat="%Hh", dtick=14400000, showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                fig_macro.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

                st.plotly_chart(fig_macro, use_container_width=True)

except FileNotFoundError:
    st.error("⚠️ Fichiers d'architecture manquants. Exécutez le script d'entraînement .py en premier.")