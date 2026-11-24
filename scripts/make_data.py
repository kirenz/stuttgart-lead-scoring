# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas"]
# ///
"""Synthetic CRM leads of a fictional B2B software company in Stuttgart (demo data, no real companies or people).

Built in on purpose:
- deal_stage almost reveals the outcome (leakage).
- The Ulm region has only 9 leads, with a high conversion rate by chance.

Run from the repository root: uv run scripts/make_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2611)

# region: city (lat, lon), spread in degrees, share of leads
REGIONS = {
    "Stuttgart":   (48.7758, 9.1829, .05, .22),
    "Böblingen":   (48.6833, 9.0167, .06, .10),
    "Esslingen":   (48.7406, 9.3108, .06, .10),
    "Ludwigsburg": (48.8975, 9.1919, .06, .10),
    "Rems-Murr":   (48.8300, 9.4200, .08, .08),
    "Göppingen":   (48.7031, 9.6521, .07, .06),
    "Reutlingen":  (48.4914, 9.2043, .07, .07),
    "Tübingen":    (48.5216, 9.0576, .06, .06),
    "Heilbronn":   (49.1427, 9.2109, .08, .08),
    "Karlsruhe":   (49.0069, 8.4037, .08, .08),
    "Pforzheim":   (48.8922, 8.6946, .06, .05),
}
N = 2000
N_ULM = 9

INDUSTRIES = ["Mechanical engineering", "Automotive supplier", "Electrical engineering", "Logistics", "Retail", "Services", "Medical technology"]
SOURCES = ["Website", "Trade fair", "Referral", "Partner", "Cold outreach", "Webinar"]
SIZES = ["10-49", "50-249", "250-999", "1000+"]

shares = np.array([v[3] for v in REGIONS.values()]); shares /= shares.sum()
names = list(REGIONS)
region = list(rng.choice(names, size=N - N_ULM, p=shares)) + ["Ulm"] * N_ULM
REGIONS["Ulm"] = (48.4011, 9.9876, .05, 0)

lat = np.array([REGIONS[r][0] + rng.normal(0, REGIONS[r][2]) for r in region])
lon = np.array([REGIONS[r][1] + rng.normal(0, REGIONS[r][2] * 1.4) for r in region])

# existing customers: 60 sites, denser around Stuttgart
cust_reg = rng.choice(names, size=60, p=shares)
cust = np.array([(REGIONS[r][0] + rng.normal(0, REGIONS[r][2]), REGIONS[r][1] + rng.normal(0, REGIONS[r][2] * 1.4)) for r in cust_reg])

def km(a_lat, a_lon, b_lat, b_lon):
    p = np.pi / 180
    d = .5 - np.cos((b_lat - a_lat) * p) / 2 + np.cos(a_lat * p) * np.cos(b_lat * p) * (1 - np.cos((b_lon - a_lon) * p)) / 2
    return 12742 * np.arcsin(np.sqrt(d))

dist = np.array([km(la, lo, cust[:, 0], cust[:, 1]).min() for la, lo in zip(lat, lon)]).round(1)

size = rng.choice(SIZES, size=N, p=[.35, .35, .2, .1])
industry = rng.choice(INDUSTRIES, size=N)
source = rng.choice(SOURCES, size=N, p=[.3, .15, .15, .12, .18, .1])
visits = rng.poisson(3, N) + (source == "Webinar") * rng.poisson(3, N)
demo = rng.random(N) < .12 + .04 * np.minimum(visits, 8)

# true conversion probability
size_eff = {"10-49": -.6, "50-249": .2, "250-999": .5, "1000+": .1}
src_eff = {"Referral": .9, "Trade fair": .4, "Partner": .5, "Webinar": .3, "Website": 0, "Cold outreach": -.8}
ind_eff = {"Mechanical engineering": .4, "Automotive supplier": .3, "Medical technology": .2}
logit = (-2.2 + np.array([size_eff[s] for s in size]) + np.array([src_eff[s] for s in source])
         + np.array([ind_eff.get(i, 0) for i in industry]) + .12 * np.minimum(visits, 10)
         + 1.1 * demo - .025 * dist)
converted = rng.random(N) < 1 / (1 + np.exp(-logit))
# Ulm: small sample with a high rate by chance
ulm = np.array(region) == "Ulm"
converted[ulm] = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0], bool)

# leakage: deal_stage is set late in the sales process and almost reveals the outcome
stage = np.where(converted,
                 rng.choice(["Contract", "Negotiation", "Proposal"], size=N, p=[.75, .2, .05]),
                 rng.choice(["First contact", "Qualified", "Proposal", "Lost"], size=N, p=[.3, .25, .1, .35]))

company = [f"Fictional Co. #{n:04d}" for n in range(1, N + 1)]
cities = {r: r if r not in ("Rems-Murr",) else "Waiblingen" for r in REGIONS}
df = pd.DataFrame({
    "lead_id": [f"L{n:05d}" for n in range(1, N + 1)], "company": company,
    "company_size": size, "industry": industry, "source": source,
    "region": region, "city": [cities[r] for r in region],
    "lat": lat.round(5), "lon": lon.round(5),
    "distance_to_nearest_customer_km": dist,
    "website_visits": visits, "demo_requested": demo.astype(int),
    "deal_stage": stage, "converted": converted.astype(int),
}).sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv("data/leads.csv", index=False)
pd.DataFrame({"customer_id": [f"C{n:03d}" for n in range(1, 61)], "region": cust_reg,
              "lat": cust[:, 0].round(5), "lon": cust[:, 1].round(5)}).to_csv("data/customers.csv", index=False)
print(df.region.value_counts().to_string()); print("conversion rate:", round(df.converted.mean(), 3))
