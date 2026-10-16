"""SkyBite prototype: dietary constraint engine, substitution ladder, dynamic buffer.
Run: python3 aaharsetu.py   (runs self-tests and the buffer simulation; synthetic data only)"""
import math, random
from dataclasses import dataclass, field

# ---------- 1. Dish ingredient flags and passenger profile ----------
@dataclass
class Dish:
    name: str
    flags: set            # ingredient/process flags, e.g. {"meat","dairy","onion_garlic"}
    tier: str = "economy" # economy | premium
    halal_cert: bool = False
    hot: bool = True

@dataclass
class Profile:
    base: str = "veg"                 # veg | egg | nonveg
    jain: bool = False                # no root vegetables, onion, garlic
    sattvic: bool = False             # no onion/garlic (Swaminarayan, sattvic)
    vrat: bool = False                # fasting: no grains/pulses, no onion/garlic, rock salt
    halal: bool = False
    no_beef: bool = False
    no_pork: bool = False
    diabetic: bool = False
    low_sodium: bool = False
    gluten_free: bool = False
    allergens: set = field(default_factory=set)   # e.g. {"nuts","dairy"}

def check(profile: Profile, dish: Dish):
    """Return (ok, reasons). Unknown/missing ingredient data must be flagged 'unverified' by caller."""
    f, why = dish.flags, []
    if profile.base == "veg" and f & {"meat", "fish", "egg"}: why.append("not vegetarian")
    if profile.base == "egg" and f & {"meat", "fish"}: why.append("contains meat/fish")
    if profile.jain and f & {"meat", "fish", "egg", "root_veg", "onion_garlic"}: why.append("not Jain")
    if (profile.sattvic or profile.vrat) and "onion_garlic" in f: why.append("contains onion/garlic")
    if profile.vrat and f & {"grain", "pulse", "regular_salt", "meat", "fish", "egg"}: why.append("not vrat-compliant")
    if profile.halal and (f & {"pork", "alcohol"} or ("meat" in f and not dish.halal_cert)): why.append("not halal")
    if profile.no_beef and "beef" in f: why.append("contains beef")
    if profile.no_pork and "pork" in f: why.append("contains pork")
    if profile.diabetic and "added_sugar" in f: why.append("added sugar")
    if profile.low_sodium and "high_sodium" in f: why.append("high sodium")
    if profile.gluten_free and "gluten" in f: why.append("contains gluten")
    hit = profile.allergens & f
    if hit: why.append("allergen: " + ",".join(sorted(hit)))
    return (not why), why

# ---------- 2. Substitution ladder for late requests ----------
def fulfil(profile, menu, surplus, airport_packs, cutoff_ok):
    """Walk the ladder; return (tier, dish). menu = planned meals, surplus = loaded buffer dishes,
    airport_packs = sealed packs sourceable at the airport (only if cutoff_ok)."""
    for label, pool in (("1 planned meal", menu), ("2 buffer stock", surplus)):
        for d in pool:
            if check(profile, d)[0]: return label, d
    for d in sorted(menu + surplus, key=lambda x: x.tier != "premium"):   # 3 premium alternative
        if d.tier == "premium" and check(profile, d)[0]: return "3 premium alternative", d
    if cutoff_ok:
        for d in airport_packs:
            if check(profile, d)[0]: return "4 airport sealed pack", d
    return "5 recovery (safe snack + amenity + apology)", None

# ---------- 3. Predictive buffer ----------
def poisson_quantile(lam, q):
    k, p, c = 0, math.exp(-lam), 0.0
    c = p
    while c < q:
        k += 1; p *= lam / k; c += p
    return k

def negbin_quantile(mean, q, r=3.0):
    """Over-dispersed demand (gamma-Poisson). r = dispersion; smaller r = burstier."""
    p = r / (r + mean); pmf = p ** r; c = pmf; k = 0
    while c < q:
        pmf *= (k + r) / (k + 1) * (1 - p); k += 1; c += pmf
    return k

def dynamic_buffer(prebooked, pax, base_rate, festival_mult, q=0.90):
    """Buffer = q-quantile of expected late special-meal demand (negative binomial)."""
    return negbin_quantile(pax * base_rate * festival_mult, q)

def simulate(n=20000, seed=7):
    rnd = random.Random(seed)
    def gamma_poisson(mean, disp=3.0):          # over-dispersed demand
        lam = rnd.gammavariate(disp, mean / disp)
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= rnd.random()
            if p <= L: return k
            k += 1
    out = {"static_2": [0,0,0,0], "static_6": [0,0,0,0], "dynamic": [0,0,0,0]}   # short flights, unmet, buffer, unused buffer
    for _ in range(n):
        pax = rnd.choice([150, 180, 220])
        fest = rnd.choice([1.0, 1.0, 1.0, 1.6])         # festival / fasting-period uplift
        base = 0.012                                    # late special-meal rate per passenger (assumed)
        demand = gamma_poisson(pax * base * fest)
        for name in out:
            buf = {"static_2": 2, "static_6": 6}.get(name) or dynamic_buffer(0, pax, base, fest)
            unmet = max(0, demand - buf)
            out[name][0] += unmet > 0; out[name][1] += unmet; out[name][2] += buf; out[name][3] += max(0, buf - demand)
    return {k: {"flights_with_shortfall_%": round(100 * v[0] / n, 1),
                "avg_unmet_per_flight": round(v[1] / n, 2),
                "avg_buffer_meals": round(v[2] / n, 2), "avg_unused_buffer": round(v[3] / n, 2)} for k, v in out.items()}

if __name__ == "__main__":
    paneer = Dish("Paneer butter masala + rice", {"dairy", "grain", "onion_garlic"})
    jain_meal = Dish("Jain dal + jeera rice", {"grain", "pulse"})
    chicken = Dish("Chicken curry", {"meat", "onion_garlic"}, halal_cert=False)
    sabu = Dish("Sabudana khichdi (vrat)", {"nuts"}, tier="premium")      # peanuts used
    fruit = Dish("Fruit + makhana bowl", {"dairy"}, tier="premium")
    assert not check(Profile(jain=True), paneer)[0]
    assert check(Profile(jain=True), jain_meal)[0]
    assert not check(Profile(base="nonveg", halal=True), chicken)[0]          # not halal-certified
    assert not check(Profile(vrat=True, allergens={"nuts"}), sabu)[0]         # nut allergy beats vrat
    tier, d = fulfil(Profile(vrat=True, allergens={"nuts"}), [paneer, chicken], [], [fruit], True)
    assert tier.startswith("4") and d is fruit, (tier, d)
    tier, d = fulfil(Profile(jain=True), [paneer], [], [], False)
    assert tier.startswith("5"), tier
    print("self-tests passed\n")
    for k, v in simulate().items(): print(k, v)
