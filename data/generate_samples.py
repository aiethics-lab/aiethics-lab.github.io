"""
Generate the three synthetic audit datasets used by the Dataset Bias Auditor.

These are SIMULATED datasets, not samples of any real registry. They are
generated from explicit, documented processes so that every disparity a
student finds in the toolkit can be traced back to a line in this file.

Each dataset ships two distinct label columns, because fairness metrics need
both and conflating them is the most common error in intro audits:

  * a DECISION column  - what the system output    (the prediction)
  * an OUTCOME column  - what actually happened    (the ground truth)

Without both, only selection-rate metrics (demographic parity, disparate
impact) are computable. Error-rate metrics (equal opportunity, predictive
equality, equalized odds) require the confusion matrix, which requires both.

--------------------------------------------------------------------------
recidivism-sample.csv  - reproduces the COMPAS impossibility result
--------------------------------------------------------------------------
RiskScore is computed from criminal-history features ONLY. Race is not an
input to the scoring model, and no race coefficient appears anywhere in the
generating process. Race correlates with priors and juvenile history instead,
which is the documented real-world pattern produced by differential policing
exposure rather than differential criminality.

The result is the Chouldechova (2017) / Kleinberg et al. (2017) impossibility:
the score is PERFECTLY CALIBRATED across race - within any risk decile, the
observed reoffence rate is the same for every group - and yet its FALSE
POSITIVE RATE is roughly twice as high for Black defendants. Both properties
cannot hold at once when base rates differ. Calibration is enforced here by
construction (the score is a deterministic decile of each person's true
probability) so that the false positive gap cannot be dismissed as model
error. It is produced by unequal base rates alone. This is exactly the disagreement
between ProPublica's analysis and Northpointe's rebuttal, and it is the whole
reason the case is taught.

--------------------------------------------------------------------------
hiring-sample.csv  - explicit disparate treatment
--------------------------------------------------------------------------
The hiring rule applies a direct penalty to female applicants. JobPerformance
is generated from skill and experience only and is independent of gender, so
equally-performing women are hired less often. This shows up as an equal
opportunity violation, not just a selection-rate gap - the distinction
students are asked to make.

JobPerformance is recorded for every applicant, including those not hired.
In reality it would only be observable for people actually hired (the
"selective labels" problem). It is included here as a simulation affordance
so the confusion matrix is computable; that limitation is worth stating in
class.

--------------------------------------------------------------------------
credit-risk-sample.csv  - disparate treatment plus a proxy
--------------------------------------------------------------------------
Two mechanisms, deliberately different:
  * Gender  - a direct penalty in the approval rule (disparate treatment)
  * Race    - no term in the rule at all. Race correlates with
              NeighborhoodIncome, and the rule uses NeighborhoodIncome.
              (disparate impact via a proxy variable)

Dropping Gender from the model removes the gender gap. Dropping Race does
nothing to the race gap, because Race was never in the model. That contrast
is the point of the Proxy Variable Detector.

Usage:  python3 generate_samples.py
"""

import csv
import math
import os
import random

SEED = 20260301
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def bounded(value, low, high):
    return max(low, min(high, value))


# ---------------------------------------------------------------- recidivism

def generate_recidivism(n=600):
    """Risk scoring with no race input; unequal FPR emerges from base rates."""
    rng = random.Random(SEED)
    races = ["Black", "White", "Hispanic", "Other"]
    race_weights = [0.51, 0.34, 0.10, 0.05]

    # Structural exposure differences. These drive the criminal-history
    # features, which is how race enters the data WITHOUT entering the model.
    priors_mean = {"Black": 3.6, "White": 2.4, "Hispanic": 2.9, "Other": 2.2}
    juv_mean = {"Black": 1.3, "White": 0.7, "Hispanic": 0.9, "Other": 0.6}
    age_mean = {"Black": 31.0, "White": 36.0, "Hispanic": 33.0, "Other": 35.0}

    rows = []
    for _ in range(n):
        race = rng.choices(races, weights=race_weights)[0]
        gender = "Male" if rng.random() < 0.79 else "Female"

        age = int(bounded(rng.gauss(age_mean[race], 9.5), 18, 72))
        priors = max(0, int(rng.gauss(priors_mean[race], 2.6)))
        juv_fel = max(0, int(rng.gauss(juv_mean[race] * 0.4, 0.7)))
        juv_mis = max(0, int(rng.gauss(juv_mean[race], 1.2)))
        charge = "Felony" if rng.random() < 0.42 else "Misdemeanor"

        # Each defendant's true probability of reoffending within two years.
        # Criminal history and age only. No race term.
        risk_logit = (
            -0.90
            + 0.150 * priors
            + 0.18 * juv_fel
            + 0.080 * juv_mis
            - 0.030 * (age - 30)
            + (0.20 if charge == "Felony" else 0.0)
            + rng.gauss(0, 0.45)          # unmodelled individual variation
        )
        risk_p = sigmoid(risk_logit)
        recidivated = rng.random() < risk_p

        # The risk score is a deterministic decile of that true probability.
        # This makes the model PERFECTLY CALIBRATED by construction, which is
        # deliberate: it removes model error from the picture so the false
        # positive gap below cannot be blamed on a bad model. It is produced
        # by unequal base rates alone.
        decile = int(bounded(math.floor(risk_p * 10) + 1, 1, 10))

        rows.append({
            "Race": race,
            "Gender": gender,
            "Age": age,
            "Priors": priors,
            "ChargeGrade": charge,
            "JuvenileFelonies": juv_fel,
            "JuvenileMisdemeanors": juv_mis,
            "RiskScore": decile,
            "PredictedRisk": "High" if decile >= 5 else "Low",
            "Recidivated": "Yes" if recidivated else "No",
        })
    return rows


# -------------------------------------------------------------------- hiring

def generate_hiring(n=400):
    """Direct gender penalty in the hiring rule; performance independent of it."""
    rng = random.Random(SEED + 1)
    races = ["White", "Black", "Hispanic", "Asian", "Other"]
    race_weights = [0.42, 0.20, 0.22, 0.11, 0.05]
    educations = ["High School", "Bachelors", "Masters", "PhD"]
    edu_weights = [0.22, 0.46, 0.24, 0.08]
    edu_value = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}

    rows = []
    for _ in range(n):
        gender = "Female" if rng.random() < 0.44 else "Male"
        race = rng.choices(races, weights=race_weights)[0]
        age = int(bounded(rng.gauss(34, 8.5), 21, 62))
        education = rng.choices(educations, weights=edu_weights)[0]
        experience = int(bounded(rng.gauss(2 + edu_value[education] * 1.6, 4.0), 0, 25))
        skill = int(bounded(round(rng.gauss(5.4 + edu_value[education] * 0.5, 2.0)), 1, 10))
        interview = int(bounded(round(skill * 0.55 + rng.gauss(2.4, 1.9)), 1, 10))

        # Ground truth: how well they would perform. Skill and experience only.
        # Independent of gender and race by construction.
        perf_logit = (
            -2.35
            + 0.36 * skill
            + 0.075 * experience
            + 0.20 * edu_value[education]
        )
        performs = rng.random() < sigmoid(perf_logit)

        # The hiring rule. Note the explicit gender term: disparate treatment.
        hire_logit = (
            -3.45
            + 0.30 * skill
            + 0.20 * interview
            + 0.055 * experience
            + 0.16 * edu_value[education]
            - (1.05 if gender == "Female" else 0.0)
            + rng.gauss(0, 0.45)
        )
        hired = rng.random() < sigmoid(hire_logit)

        rows.append({
            "Gender": gender,
            "Age": age,
            "Race": race,
            "Education": education,
            "Experience": experience,
            "SkillScore": skill,
            "InterviewScore": interview,
            "Hired": "Hired" if hired else "Rejected",
            "JobPerformance": "Strong" if performs else "Weak",
        })
    return rows


# --------------------------------------------------------------------- credit

def generate_credit(n=500):
    """Gender enters the rule directly; race enters only through a proxy."""
    rng = random.Random(SEED + 2)
    races = ["White", "Black", "Hispanic", "Asian"]
    race_weights = [0.47, 0.23, 0.20, 0.10]

    # The proxy. Residential segregation makes neighborhood income track race.
    # The approval rule below reads NeighborhoodIncome and never reads Race.
    nbhd_mean = {"White": 74000, "Black": 46000, "Hispanic": 52000, "Asian": 78000}

    rows = []
    for _ in range(n):
        race = rng.choices(races, weights=race_weights)[0]
        gender = "Female" if rng.random() < 0.48 else "Male"
        age = int(bounded(rng.gauss(41, 12), 21, 74))

        nbhd = int(bounded(rng.gauss(nbhd_mean[race], 14000), 18000, 140000))
        income = int(bounded(rng.gauss(nbhd * 0.78 + 14000, 15000), 16000, 190000))
        credit_score = int(bounded(rng.gauss(600 + (income - 55000) / 1400.0, 62), 300, 850))
        debt_ratio = round(bounded(rng.gauss(0.40 - (income - 55000) / 900000.0, 0.13), 0.03, 0.85), 2)
        loan_amount = rng.choice([5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000])
        employment_years = int(bounded(rng.gauss(7.5, 5.2), 0, 30))

        # Ground truth: who actually defaults. Financial factors only.
        default_logit = (
            -1.95
            - 0.0062 * (credit_score - 600)
            + 2.5 * (debt_ratio - 0.40)
            - 0.045 * employment_years
            + 0.0000115 * loan_amount
        )
        defaulted = rng.random() < sigmoid(default_logit)

        # The approval rule.
        #   Gender      -> explicit penalty            (disparate treatment)
        #   Neighborhood-> legitimate-looking feature   (proxy for race)
        #   Race        -> absent entirely
        approve_logit = (
            -0.25
            + 0.0090 * (credit_score - 600)
            - 3.1 * (debt_ratio - 0.40)
            + 0.0000175 * (nbhd - 55000)
            + 0.040 * employment_years
            - (0.95 if gender == "Female" else 0.0)
            + rng.gauss(0, 0.40)
        )
        approved = rng.random() < sigmoid(approve_logit)

        rows.append({
            "Gender": gender,
            "Age": age,
            "Race": race,
            "Income": income,
            "NeighborhoodIncome": nbhd,
            "CreditScore": credit_score,
            "DebtRatio": debt_ratio,
            "LoanAmount": loan_amount,
            "EmploymentYears": employment_years,
            "Decision": "Approved" if approved else "Denied",
            "Defaulted": "Yes" if defaulted else "No",
        })
    return rows


# ----------------------------------------------------------------------- io

def write_csv(filename, rows):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


def report(rows, protected, decision, favorable, outcome, good_outcome):
    """Print the audit a student should be able to reproduce in the toolkit."""
    groups = {}
    for r in rows:
        g = groups.setdefault(r[protected], {"n": 0, "sel": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0})
        g["n"] += 1
        predicted_favorable = r[decision] == favorable
        actually_good = r[outcome] == good_outcome
        if predicted_favorable:
            g["sel"] += 1
        if actually_good and predicted_favorable:
            g["tp"] += 1
        elif not actually_good and predicted_favorable:
            g["fp"] += 1
        elif not actually_good and not predicted_favorable:
            g["tn"] += 1
        else:
            g["fn"] += 1

    print(f"    {'group':10s} {'n':>4s} {'sel':>7s} {'TPR':>7s} {'FPR':>7s} {'PPV':>7s} {'base':>7s}")
    for name in sorted(groups, key=lambda k: -groups[k]["sel"] / groups[k]["n"]):
        g = groups[name]
        pos = g["tp"] + g["fn"]
        neg = g["fp"] + g["tn"]
        tpr = g["tp"] / pos if pos else float("nan")
        fpr = g["fp"] / neg if neg else float("nan")
        ppv = g["tp"] / (g["tp"] + g["fp"]) if (g["tp"] + g["fp"]) else float("nan")
        print(f"    {name:10s} {g['n']:4d} {g['sel']/g['n']:7.3f} "
              f"{tpr:7.3f} {fpr:7.3f} {ppv:7.3f} {pos/g['n']:7.3f}")


if __name__ == "__main__":
    rec = generate_recidivism()
    hire = generate_hiring()
    credit = generate_credit()

    write_csv("recidivism-sample.csv", rec)
    write_csv("hiring-sample.csv", hire)
    write_csv("credit-risk-sample.csv", credit)

    print("\nrecidivism - Race, PredictedRisk=High vs Recidivated=Yes")
    print("  (favorable outcome for the defendant is Low risk; here we score the")
    print("   High-risk flag so TPR/FPR read the ProPublica way)")
    report(rec, "Race", "PredictedRisk", "High", "Recidivated", "Yes")

    print("\nhiring - Gender, Hired vs JobPerformance=Strong")
    report(hire, "Gender", "Hired", "Hired", "JobPerformance", "Strong")

    print("\ncredit - Gender, Decision=Approved vs Defaulted=No")
    report(credit, "Gender", "Decision", "Approved", "Defaulted", "No")

    print("\ncredit - Race, Decision=Approved vs Defaulted=No")
    report(credit, "Race", "Decision", "Approved", "Defaulted", "No")
