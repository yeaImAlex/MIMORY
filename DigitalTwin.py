# ==========================================
# DIGITAL TWIN – FINAL PROFESSIONAL VERSION
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("SRAM_Digital_Twin_Dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================================
# 2. DEFINE INPUTS & TARGET
# ==========================================

X = df[["VDD (V)", "Frequency (Hz)"]]

# IMPORTANT: LOG TRANSFORM (stabilizes exponential data)
y = np.log10(df["Dynamic Power (W)"])

# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 4. POLYNOMIAL REGRESSION
# ==========================================

poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("scaler", StandardScaler()),
    ("linear", LinearRegression())
])

poly_model.fit(X_train, y_train)
poly_pred = poly_model.predict(X_test)

# ==========================================
# 5. RANDOM FOREST
# ==========================================

rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=15,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ==========================================
# 6. SUPPORT VECTOR REGRESSION
# ==========================================

svr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel='rbf', C=100, gamma='scale'))
])

svr_model.fit(X_train, y_train)
svr_pred = svr_model.predict(X_test)

# ==========================================
# 7. EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Performance:")
    print(f"RMSE (log scale): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return rmse, r2

print("\n========== DIGITAL TWIN MODEL PERFORMANCE ==========")

poly_rmse, poly_r2 = evaluate_model("Polynomial Regression", y_test, poly_pred)
rf_rmse, rf_r2 = evaluate_model("Random Forest", y_test, rf_pred)
svr_rmse, svr_r2 = evaluate_model("SVR", y_test, svr_pred)

# ==========================================
# 8. SELECT BEST MODEL
# ==========================================

r2_scores = {
    "Polynomial Regression": poly_r2,
    "Random Forest": rf_r2,
    "SVR": svr_r2
}

best_model = max(r2_scores, key=r2_scores.get)

print("\nBest Model:", best_model)
print("Best R² Score:", r2_scores[best_model])
import matplotlib.pyplot as plt

# Convert back from log scale for visualization
y_test_real = 10**y_test
rf_pred_real = 10**rf_pred

plt.figure()
plt.scatter(y_test_real, rf_pred_real)
plt.xlabel("Actual Dynamic Power (W)")
plt.ylabel("Predicted Dynamic Power (W)")
plt.title("Digital Twin Prediction Accuracy (Random Forest)")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# ==============================
# Convert Back from Log Scale
# ==============================

# Convert actual values back
y_test_real = 10 ** y_test

# Predict (log scale)
rf_pred_log = rf_model.predict(X_test)

# Convert predictions back
rf_pred_real = 10 ** rf_pred_log

# ==============================
# Compute Metrics (Real Scale)
# ==============================

r2_real = r2_score(y_test_real, rf_pred_real)
rmse_real = np.sqrt(mean_squared_error(y_test_real, rf_pred_real))

print("===== DIGITAL TWIN VALIDATION (REAL SCALE) =====")
print("R² Score:", r2_real)
print("RMSE:", rmse_real)

# ==============================
# Visualization
# ==============================

plt.figure(figsize=(7,7))

plt.scatter(y_test_real, rf_pred_real)

# Perfect prediction line
min_val = min(y_test_real.min(), rf_pred_real.min())
max_val = max(y_test_real.max(), rf_pred_real.max())

plt.plot([min_val, max_val],
         [min_val, max_val])

plt.xlabel("Actual Dynamic Power (W)")
plt.ylabel("Predicted Dynamic Power (W)")
plt.title("Digital Twin Prediction Accuracy (Random Forest)")
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================
# 0️⃣ IMPORT LIBRARIES
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from collections import Counter

# ============================================
# 1️⃣ LOAD AND CLEAN DATASET
# ============================================

df = pd.read_csv("SRAM_Digital_Twin_Dataset.csv")   # <<< CHANGE FILE NAME HERE

print("Original Columns:", df.columns)

# Clean column names
df.columns = df.columns.str.strip()

df.rename(columns={
    "VDD (V)": "VDD",
    "Frequency (Hz)": "Frequency",
    "Dynamic Power (W)": "Pdyn",
    "Leakage Power (W)": "Pleak",
    "Energy per Operation (J)": "Eop"
}, inplace=True)

print("Cleaned Columns:", df.columns)

# Remove non-functional region
df = df[df["VDD"] >= 0.5]

# ============================================
# 2️⃣ TRAIN DIGITAL TWIN (Random Forest)
# ============================================

X = df[["VDD", "Frequency"]]
y = np.log10(df["Pdyn"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Digital Twin trained successfully.")

# Prediction function
def digital_twin_predict(vdd, freq):
    input_data = np.array([[vdd, freq]])
    pred_log = model.predict(input_data)
    return 10**pred_log[0]

# ============================================
# 3️⃣ GENERATE 24-HOUR WORKLOAD
# ============================================

np.random.seed(42)

timesteps = 24 * 60   # 1-minute resolution
dt = 60               # seconds per timestep

workload = []

for t in range(timesteps):
    if np.random.rand() < 0.2:  # 20% burst
        workload.append(np.random.choice([150e6, 200e6]))
    else:  # 80% idle
        workload.append(np.random.choice([10e6, 50e6]))

workload = np.array(workload)

# Plot workload
plt.figure()
plt.plot(workload / 1e6)
plt.title("24-Hour Synthetic Workload Trace")
plt.xlabel("Time (minutes)")
plt.ylabel("Frequency (MHz)")
plt.show()

# ============================================
# 4️⃣ DEFINE OPERATING MODES
# ============================================

modes = [
    {"name": "Turbo", "vdd": 1.0, "fmax": 200e6},
    {"name": "Normal", "vdd": 0.8, "fmax": 150e6},
    {"name": "Eco", "vdd": 0.6, "fmax": 100e6},
    {"name": "Retention", "vdd": 0.5, "fmax": 10e6},
]

# ============================================
# 5️⃣ STATIC TURBO SIMULATION
# ============================================

total_energy_nominal = 0

for f in workload:
    power = digital_twin_predict(0.8, f)
    total_energy_nominal += power * dt

# ============================================
# 6️⃣ CI-AWARE ADAPTIVE CONTROLLER
# ============================================

def run_adaptive_controller(CI_value):

    total_energy = 0
    sla_violation = 0
    mode_log = []

    for f in workload:
        valid_modes = []

        for mode in modes:
            if f <= mode["fmax"]:
                valid_modes.append(mode)

        if len(valid_modes) == 0:
            sla_violation += 1
            selected_mode = modes[0]  # fallback
        else:
            # Select mode minimizing carbon cost
            selected_mode = min(
                valid_modes,
                key=lambda x: digital_twin_predict(x["vdd"], f) * CI_value
            )

        power = digital_twin_predict(selected_mode["vdd"], f)
        total_energy += power * dt
        mode_log.append(selected_mode["name"])

    return total_energy, sla_violation, mode_log

# ============================================
# 7️⃣ ADAPTIVE DIGITAL TWIN CONTROLLER
# ============================================

total_energy_adaptive = 0
sla_violation_adaptive = 0
mode_log = []

for f in workload:
    valid_modes = []

    for mode in modes:
        if f <= mode["fmax"]:
            valid_modes.append(mode)

    if len(valid_modes) == 0:
        sla_violation_adaptive += 1
        selected_mode = modes[0]
    else:
        selected_mode = min(valid_modes, key=lambda x: x["vdd"])

    power = digital_twin_predict(selected_mode["vdd"], f)
    total_energy_adaptive += power * dt
    mode_log.append(selected_mode["name"])

# ============================================
# 8️⃣ ENERGY REDUCTION
# ============================================

energy_reduction = (
    (total_energy_turbo - total_energy_adaptive)
    / total_energy_turbo
    * 100
)

# ============================================
# 9️⃣ CARBON CALCULATION
# ============================================

PUE = 1.5
CI_values = [300, 500, 700]

carbon_results = {}

for CI in CI_values:
    carbon_turbo = total_energy_turbo * PUE * CI
    carbon_adaptive = total_energy_adaptive * PUE * CI
    reduction = (
        (carbon_turbo - carbon_adaptive)
        / carbon_turbo
        * 100
    )
    carbon_results[CI] = reduction

# ============================================
# 🔟 MODE DISTRIBUTION
# ============================================

mode_counts = Counter(mode_log)
mode_percentage = {
    mode: count / len(workload) * 100
    for mode, count in mode_counts.items()
}

# ============================================
# 1️⃣1️⃣ PRINT RESULTS
# ============================================

print("\n========== ENERGY ==========")
print("Turbo:", total_energy_turbo)
print("Eco:", total_energy_eco)
print("Adaptive:", total_energy_adaptive)
print("Energy Reduction (%):", round(energy_reduction, 2))

print("\n========== SLA ==========")
print("Eco Violations:", sla_violation_eco)
print("Adaptive Violations:", sla_violation_adaptive)

print("\n========== CARBON REDUCTION ==========")
for CI, reduction in carbon_results.items():
    print(f"CI={CI} gCO2/kWh -> {reduction:.2f}%")

print("\n========== MODE DISTRIBUTION (%) ==========")
for mode, percent in mode_percentage.items():
    print(f"{mode}: {percent:.2f}%")

# ============================================
# 1️⃣2️⃣ PLOTS
# ============================================

# Energy comparison
plt.figure()
plt.bar(["Turbo", "Eco", "Adaptive"],
        [total_energy_turbo,
         total_energy_eco,
         total_energy_adaptive])
plt.title("Total Energy Comparison")
plt.ylabel("Energy (J)")
plt.show()

# Carbon comparison
plt.figure()
plt.bar([str(ci) for ci in carbon_results.keys()],
        list(carbon_results.values()))
plt.title("Carbon Reduction vs CI")
plt.xlabel("Carbon Intensity (gCO2/kWh)")
plt.ylabel("Reduction (%)")
plt.show()

# SLA comparison
plt.figure()
plt.bar(["Eco", "Adaptive"],
        [sla_violation_eco, sla_violation_adaptive])
plt.title("SLA Violations")
plt.ylabel("Violation Count")
plt.show()

# Mode distribution
plt.figure()
plt.pie(mode_percentage.values(),
        labels=mode_percentage.keys(),
        autopct='%1.1f%%')
plt.title("Adaptive Mode Distribution")
plt.show()

PUE = 1.5
CI_values = [300, 500, 700]

results = {}

for CI in CI_values:
    energy_adaptive, sla_adaptive, mode_log = run_adaptive_controller(CI)

    energy_reduction = (
        (total_energy_nominal - energy_adaptive)
        / total_energy_nominal
        * 100
    )

    carbon_nominal = total_energy_nominal * PUE * CI
    carbon_adaptive = energy_adaptive * PUE * CI

    carbon_reduction = (
        (carbon_nominal - carbon_adaptive)
        / carbon_nominal
        * 100
    )

    results[CI] = {
        "energy": energy_adaptive,
        "energy_reduction": energy_reduction,
        "carbon_reduction": carbon_reduction,
        "sla": sla_adaptive,
        "mode_log": mode_log
    }
    print("\n========== BASELINE (Nominal 0.8V) ==========")
print("Total Energy Nominal:", total_energy_nominal)

for CI in results:
    print(f"\n--- CI = {CI} gCO2/kWh ---")
    print("Energy Adaptive:", results[CI]["energy"])
    print("Energy Reduction (%):", round(results[CI]["energy_reduction"], 2))
    print("Carbon Reduction (%):", round(results[CI]["carbon_reduction"], 2))
    print("SLA Violations:", results[CI]["sla"])

# ============================================
# 0️⃣ IMPORTS
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from collections import Counter

# ============================================
# 1️⃣ LOAD & CLEAN DATA
# ============================================

df = pd.read_csv("SRAM_Digital_Twin_Dataset.csv")  # <<< CHANGE FILE NAME

df.columns = df.columns.str.strip()

df.rename(columns={
    "VDD (V)": "VDD",
    "Frequency (Hz)": "Frequency",
    "Dynamic Power (W)": "Pdyn",
    "Leakage Power (W)": "Pleak",
    "Energy per Operation (J)": "Eop"
}, inplace=True)

df = df[df["VDD"] >= 0.5]

# ============================================
# 2️⃣ TRAIN DIGITAL TWIN
# ============================================

X = df[["VDD", "Frequency"]]
y = np.log10(df["Pdyn"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

def digital_twin_predict(vdd, freq):
    input_df = pd.DataFrame([[vdd, freq]], columns=["VDD", "Frequency"])
    pred_log = model.predict(input_df)
    return 10**pred_log[0]

print("Digital Twin trained successfully.")

# ============================================
# 3️⃣ GENERATE 24-HOUR WORKLOAD
# ============================================

np.random.seed(42)

timesteps = 24 * 60
dt = 60  # seconds

workload = []

for t in range(timesteps):
    if np.random.rand() < 0.2:
        workload.append(np.random.choice([150e6, 200e6]))
    else:
        workload.append(np.random.choice([10e6, 50e6]))

workload = np.array(workload)

plt.figure()
plt.plot(workload / 1e6)
plt.title("24-Hour Synthetic Workload Trace")
plt.xlabel("Time (minutes)")
plt.ylabel("Frequency (MHz)")
plt.show()

# ============================================
# 4️⃣ DEFINE OPERATING MODES
# ============================================

modes = [
    {"name": "Turbo", "vdd": 1.0, "fmax": 200e6},
    {"name": "Normal", "vdd": 0.8, "fmax": 150e6},
    {"name": "Eco", "vdd": 0.6, "fmax": 100e6},
    {"name": "Retention", "vdd": 0.5, "fmax": 10e6},
]

# ============================================
# 5️⃣ STATIC NOMINAL BASELINE (0.8V)
# ============================================

total_energy_nominal = 0

for f in workload:
    power = digital_twin_predict(0.8, f)
    total_energy_nominal += power * dt

# ============================================
# 6️⃣ CI-AWARE ADAPTIVE CONTROLLER
# ============================================

def run_adaptive_controller(CI_value):
    total_energy = 0
    sla_violation = 0
    mode_log = []

    for f in workload:
        valid_modes = [m for m in modes if f <= m["fmax"]]

        if not valid_modes:
            sla_violation += 1
            selected = modes[0]
        else:
            # Precompute cost once
            best_mode = None
            best_cost = float("inf")

            for mode in valid_modes:
                power = digital_twin_predict(mode["vdd"], f)
                cost = power * CI_value
                if cost < best_cost:
                    best_cost = cost
                    best_mode = mode

            selected = best_mode

        power = digital_twin_predict(selected["vdd"], f)
        total_energy += power * dt
        mode_log.append(selected["name"])

    return total_energy, sla_violation, mode_log

# ============================================
# 7️⃣ RUN FOR CI SCENARIOS
# ============================================

PUE = 1.5
CI_values = [300, 500, 700]

results = {}

for CI in CI_values:
    energy_adaptive, sla_adaptive, mode_log = run_adaptive_controller(CI)

    energy_reduction = (
        (total_energy_nominal - energy_adaptive)
        / total_energy_nominal * 100
    )

    carbon_nominal = total_energy_nominal * PUE * CI
    carbon_adaptive = energy_adaptive * PUE * CI

    carbon_reduction = (
        (carbon_nominal - carbon_adaptive)
        / carbon_nominal * 100
    )

    results[CI] = {
        "energy": energy_adaptive,
        "energy_reduction": energy_reduction,
        "carbon_nominal": carbon_nominal,
        "carbon_adaptive": carbon_adaptive,
        "carbon_reduction": carbon_reduction,
        "sla": sla_adaptive,
        "mode_log": mode_log
    }

# ============================================
# 8️⃣ PRINT RESULTS
# ============================================

print("\n========== BASELINE (0.8V Nominal) ==========")
print("Total Energy Nominal:", total_energy_nominal)

for CI in CI_values:
    print(f"\n--- CI = {CI} gCO2/kWh ---")
    print("Adaptive Energy:", results[CI]["energy"])
    print("Energy Reduction (%):", round(results[CI]["energy_reduction"], 2))
    print("Carbon Reduction (%):", round(results[CI]["carbon_reduction"], 2))
    print("SLA Violations:", results[CI]["sla"])

# ============================================
# 9️⃣ ENERGY COMPARISON PLOT
# ============================================

plt.figure()
plt.bar(
    ["Nominal 0.8V"] + [f"Adaptive CI={ci}" for ci in CI_values],
    [total_energy_nominal] + [results[ci]["energy"] for ci in CI_values]
)
plt.title("Total Energy Comparison")
plt.ylabel("Energy (J)")
plt.show()

# ============================================
# 🔟 CARBON COMPARISON (ABSOLUTE)
# ============================================

carbon_nominal_vals = []
carbon_adaptive_vals = []

for CI in CI_values:
    carbon_nominal_vals.append(results[CI]["carbon_nominal"])
    carbon_adaptive_vals.append(results[CI]["carbon_adaptive"])

x = np.arange(len(CI_values))

plt.figure()
plt.bar(x - 0.2, carbon_nominal_vals, width=0.4, label="Nominal")
plt.bar(x + 0.2, carbon_adaptive_vals, width=0.4, label="Adaptive")

plt.xticks(x, CI_values)
plt.xlabel("Carbon Intensity (gCO2/kWh)")
plt.ylabel("Carbon Emission")
plt.title("Carbon Emission Comparison")
plt.legend()
plt.show()

# ============================================
# 1️⃣1️⃣ MODE DISTRIBUTION (CI=500 Example)
# ============================================

mode_counts = Counter(results[500]["mode_log"])
mode_percentage = {
    k: v / timesteps * 100 for k, v in mode_counts.items()
}

plt.figure()
plt.pie(
    mode_percentage.values(),
    labels=mode_percentage.keys(),
    autopct='%1.1f%%'
)
plt.title("Operating Mode Utilization (CI=500)")
plt.show()

# ============================================
# 1️⃣2️⃣ SLA PLOT
# ============================================

plt.figure()
plt.bar(
    ["Adaptive"],
    [results[500]["sla"]]
)
plt.title("SLA Violations (Adaptive)")
plt.ylabel("Violation Count")
plt.show()

# =========================================
# 0. Import Libraries
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error


# =========================================
# 1. Load Dataset
# =========================================

df = pd.read_csv("SRAM_Digital_Twin_Dataset.csv")   # <-- replace with your file name

# Clean column names
df.columns = df.columns.str.strip()

# Rename columns to simplified format
df.rename(columns={
    "VDD (V)": "VDD",
    "Frequency (Hz)": "Frequency",
    "TPD (s)": "TPD",
    "Dynamic Power (W)": "Pdyn",
    "Leakage Power (W)": "Pleak",
    "Energy per Operation (J)": "Eop"
}, inplace=True)

print("Columns after renaming:")
print(df.columns)


# =========================================
# 2. Remove Non-Functional Points
# =========================================

df = df[df["VDD"] >= 0.5]

# Log transform dynamic power
df["log_Pdyn"] = np.log10(df["Pdyn"])


# =========================================
# 3. Define Features and Target
# =========================================

X = df[["VDD", "Frequency"]]
y = df["log_Pdyn"]


# =========================================
# 4. Train/Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# =========================================
# 5. Basic Accuracy Metrics
# =========================================

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance:")
print("R²:", r2)
print("RMSE (log scale):", rmse)


# =========================================
# 6. Residual Analysis
# =========================================

residuals = y_test - y_pred

plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=20, kde=True)
plt.xlabel("Residual Error (log scale)")
plt.ylabel("Frequency")
plt.title("Residual Error Distribution")
plt.axvline(0, color='red')
plt.tight_layout()
plt.show()


# =========================================
# 7. Error vs Voltage Plot
# =========================================

plt.figure(figsize=(6,4))
plt.scatter(X_test["VDD"], residuals)
plt.xlabel("Supply Voltage (VDD)")
plt.ylabel("Residual Error (log scale)")
plt.title("Prediction Error vs Supply Voltage")
plt.axhline(0, color='red')
plt.tight_layout()
plt.show()


# =========================================
# 8. 5-Fold Cross Validation
# =========================================

cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print("\n5-Fold Cross Validation:")
print("Mean R²:", np.mean(cv_scores))
print("Std Dev R²:", np.std(cv_scores))


# =========================================
# 9. Feature Importance
# =========================================

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(6,4))
plt.bar(features, importances)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance Ranking")
plt.tight_layout()
plt.show()

print("\nFeature Importance Values:")
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance}")


# =========================================
# 10. Extrapolation Stress Test
# =========================================

# Choose a point NOT explicitly used
test_voltage = 0.85
test_frequency = 150e6

new_point = pd.DataFrame(
    [[test_voltage, test_frequency]],
    columns=["VDD", "Frequency"]
)

pred_log = model.predict(new_point)
pred_power = 10**pred_log

print("\nExtrapolation Test Prediction:")
print("Predicted Dynamic Power:", pred_power[0])


# ---- INSERT YOUR SPICE GROUND TRUTH VALUE BELOW ----
spice_ground_truth = 3.51050021374e-12   # <-- replace with real SPICE value

if spice_ground_truth > 0:
    error_percent = abs(pred_power[0] - spice_ground_truth) / spice_ground_truth * 100
    print("SPICE Ground Truth:", spice_ground_truth)
    print("Prediction Error (%):", error_percent)
else:
    print("Insert SPICE ground truth value to compute error.")


# =========================================
# 11. Prediction Time Benchmark
# =========================================

start = time.time()

for _ in range(1000):
    model.predict(new_point)

end = time.time()

prediction_time = (end - start) / 1000
print("Average Prediction Time (ms):", prediction_time * 1000)

print("\nPrediction Time:")
print("Digital Twin Time (seconds):", prediction_time)
print("Digital Twin Time (milliseconds):", prediction_time * 1000)

# SPICE time from your log
spice_time = 3.704   # seconds (from LTspice log)

acceleration = spice_time / prediction_time

print("Acceleration Factor:", acceleration)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Load dataset
df = pd.read_csv("SRAM_Digital_Twin_Dataset.csv")

df.columns = df.columns.str.strip()

df.rename(columns={
    "VDD (V)": "VDD",
    "Frequency (Hz)": "Frequency",
    "Dynamic Power (W)": "Pdyn"
}, inplace=True)

# Remove non-functional region
df = df[df["VDD"] >= 0.5]

# Convert frequency to MHz for cleaner axis
df["Frequency_MHz"] = df["Frequency"] / 1e6

# Create grid for smooth surface
vdd_range = np.linspace(df["VDD"].min(), df["VDD"].max(), 50)
freq_range = np.linspace(df["Frequency_MHz"].min(), df["Frequency_MHz"].max(), 50)

VDD_grid, FREQ_grid = np.meshgrid(vdd_range, freq_range)

# Interpolate surface
Pdyn_grid = griddata(
    (df["VDD"], df["Frequency_MHz"]),
    df["Pdyn"],
    (VDD_grid, FREQ_grid),
    method='cubic'
)

# Plot 3D surface
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(
    VDD_grid,
    FREQ_grid,
    Pdyn_grid,
    cmap='viridis',
    edgecolor='none'
)

ax.set_xlabel("Supply Voltage VDD (V)")
ax.set_ylabel("Frequency (MHz)")
ax.set_zlabel("Dynamic Power (W)")
ax.set_title("Voltage–Frequency–Dynamic Power Surface")

fig.colorbar(surface, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()
plt.figure(figsize=(6,5))
contour = plt.contourf(
    VDD_grid,
    FREQ_grid,
    Pdyn_grid,
    levels=20,
    cmap='viridis'
)

plt.xlabel("Supply Voltage VDD (V)")
plt.ylabel("Frequency (MHz)")
plt.title("Dynamic Power Contour Map")

plt.colorbar(contour, label="Dynamic Power (W)")
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ===============================
# 1️⃣ Generate 24-Hour Workload
# ===============================

np.random.seed(42)

timesteps = 24 * 60   # 1-minute resolution
dt = 60               # seconds per timestep

workload = []

for t in range(timesteps):
    if np.random.rand() < 0.2:  # 20% burst
        workload.append(np.random.choice([150e6, 200e6]))
    else:  # 80% idle/low
        workload.append(np.random.choice([10e6, 50e6]))

workload = np.array(workload)

# ===============================
# 2️⃣ Digital Twin Prediction Function
# ===============================

def digital_twin_predict(vdd, freq):
    input_data = np.array([[vdd, freq]])
    pred_log = model.predict(input_data)
    return 10**pred_log[0]

# ===============================
# 3️⃣ Define Operating Modes
# ===============================

modes = [
    {"name": "Turbo", "vdd": 1.0, "fmax": 200e6},
    {"name": "Normal", "vdd": 0.8, "fmax": 150e6},
    {"name": "Eco", "vdd": 0.6, "fmax": 100e6},
    {"name": "Retention", "vdd": 0.5, "fmax": 10e6},
]

# ===============================
# 4️⃣ Static Turbo Simulation
# ===============================

total_energy_turbo = 0

for f in workload:
    power = digital_twin_predict(1.0, f)
    total_energy_turbo += power * dt

# ===============================
# 5️⃣ Static Eco Simulation
# ===============================

total_energy_eco = 0
sla_violation_eco = 0

for f in workload:
    if f > 100e6:
        sla_violation_eco += 1

    power = digital_twin_predict(0.6, f)
    total_energy_eco += power * dt

# ===============================
# 6️⃣ Adaptive Digital Twin Controller
# ===============================

total_energy_adaptive = 0
sla_violation_adaptive = 0
mode_log = []

for f in workload:
    valid_modes = []

    for mode in modes:
        if f <= mode["fmax"]:
            valid_modes.append(mode)

    if len(valid_modes) == 0:
        sla_violation_adaptive += 1
        selected_mode = modes[0]  # fallback Turbo
    else:
        selected_mode = min(valid_modes, key=lambda x: x["vdd"])

    power = digital_twin_predict(selected_mode["vdd"], f)
    total_energy_adaptive += power * dt
    mode_log.append(selected_mode["name"])

# ===============================
# 7️⃣ Energy Reduction %
# ===============================

energy_reduction = (total_energy_turbo - total_energy_adaptive) / total_energy_turbo * 100

# ===============================
# 8️⃣ Carbon Calculation
# ===============================

PUE = 1.5
CI_values = [300, 500, 700]

carbon_results = {}

for CI in CI_values:
    carbon_turbo = total_energy_turbo * PUE * CI
    carbon_adaptive = total_energy_adaptive * PUE * CI
    reduction = (carbon_turbo - carbon_adaptive) / carbon_turbo * 100
    carbon_results[CI] = reduction

# ===============================
# 9️⃣ Mode Distribution
# ===============================

mode_counts = Counter(mode_log)
mode_percentage = {}

for mode in mode_counts:
    mode_percentage[mode] = mode_counts[mode] / len(workload) * 100

# ===============================
# 🔟 PRINT FINAL RESULTS
# ===============================

print("========== ENERGY RESULTS ==========")
print("Total Energy (Turbo):", total_energy_turbo)
print("Total Energy (Eco):", total_energy_eco)
print("Total Energy (Adaptive):", total_energy_adaptive)
print("Energy Reduction (%):", round(energy_reduction, 2))

print("\n========== SLA VIOLATIONS ==========")
print("Eco Violations:", sla_violation_eco)
print("Adaptive Violations:", sla_violation_adaptive)

print("\n========== CARBON REDUCTION ==========")
for CI in carbon_results:
    print(f"Carbon Reduction at CI={CI} gCO2/kWh: {carbon_results[CI]:.2f}%")

print("\n========== MODE DISTRIBUTION (%) ==========")
for mode in mode_percentage:
    print(f"{mode}: {mode_percentage[mode]:.2f}%")

# ===============================
# 📊 11️⃣ PLOTS FOR SECTION 5
# ===============================

# Energy Comparison
plt.figure()
plt.bar(["Turbo", "Eco", "Adaptive"],
        [total_energy_turbo, total_energy_eco, total_energy_adaptive])
plt.title("Total Energy Comparison")
plt.ylabel("Energy (J)")
plt.show()

# Carbon Reduction Plot
plt.figure()
plt.bar([str(ci) for ci in carbon_results.keys()],
        list(carbon_results.values()))
plt.title("Carbon Reduction vs Grid Carbon Intensity")
plt.xlabel("CI (gCO2/kWh)")
plt.ylabel("Carbon Reduction (%)")
plt.show()

# SLA Violations
plt.figure()
plt.bar(["Eco", "Adaptive"],
        [sla_violation_eco, sla_violation_adaptive])
plt.title("SLA Violations Comparison")
plt.ylabel("Violation Count")
plt.show()

# Mode Distribution
plt.figure()
plt.pie(mode_percentage.values(),
        labels=mode_percentage.keys(),
        autopct='%1.1f%%')
plt.title("Adaptive Mode Distribution")
plt.show()

# ==========================================================
# SECTION 5 – INDUSTRY-LEVEL VALIDATION SCRIPT
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from collections import Counter
import time

# ==========================================================
# 1️⃣ LOAD DATA
# ==========================================================

df = pd.read_csv("SRAM_Digital_Twin_Dataset.csv")
df.columns = df.columns.str.strip()

df.rename(columns={
    "VDD (V)": "VDD",
    "Frequency (Hz)": "Frequency",
    "Dynamic Power (W)": "Pdyn",
    "Leakage Power (W)": "Pleak",
    "Energy per Operation (J)": "Eop"
}, inplace=True)

df = df[df["VDD"] >= 0.5]

# ==========================================================
# 2️⃣ TRAIN DIGITAL TWIN
# ==========================================================

X = df[["VDD", "Frequency"]]
y = np.log10(df["Pdyn"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Prediction on test set
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

cv_scores = cross_val_score(model, X, y, cv=5)
cv_mean = np.mean(cv_scores)

print("\n===== DIGITAL TWIN ACCURACY =====")
print("R²:", r2)
print("RMSE (log):", rmse)
print("5-Fold CV Mean R²:", cv_mean)

# ==========================================================
# 3️⃣ PREDICTED VS SPICE PLOT
# ==========================================================

plt.figure()
plt.scatter(10**y_test, 10**y_pred)
plt.plot([min(10**y_test), max(10**y_test)],
         [min(10**y_test), max(10**y_test)],
         'r--')
plt.xlabel("SPICE Dynamic Power (W)")
plt.ylabel("Predicted Dynamic Power (W)")
plt.title("Predicted vs SPICE")
plt.show()

# ==========================================================
# 4️⃣ RESIDUAL DISTRIBUTION
# ==========================================================

residuals = y_test - y_pred
plt.figure()
plt.hist(residuals, bins=15)
plt.title("Residual Distribution (Log Scale)")
plt.xlabel("Residual Error")
plt.show()

# ==========================================================
# 5️⃣ WORKLOAD GENERATION
# ==========================================================

def generate_workload(burst_prob=0.2):
    np.random.seed(42)
    workload = []
    for _ in range(24 * 60):
        if np.random.rand() < burst_prob:
            workload.append(np.random.choice([150e6, 200e6]))
        else:
            workload.append(np.random.choice([10e6, 50e6]))
    return np.array(workload)

workload = generate_workload(0.2)

# ==========================================================
# 6️⃣ OPERATING MODES
# ==========================================================

modes = [
    {"name": "Turbo", "vdd": 1.0, "fmax": 200e6},
    {"name": "Normal", "vdd": 0.8, "fmax": 150e6},
    {"name": "Eco", "vdd": 0.6, "fmax": 100e6},
    {"name": "Retention", "vdd": 0.5, "fmax": 10e6},
]

def predict_power(vdd, freq):
    return 10**model.predict(pd.DataFrame([[vdd, freq]],
           columns=["VDD","Frequency"]))[0]

# ==========================================================
# 7️⃣ ENERGY & SLA COMPUTATION
# ==========================================================

def compute_strategy(vdd_fixed=None):
    total_energy = 0
    sla = 0
    mode_log = []
    min_margin = float("inf")

    for f in workload:

        if vdd_fixed is not None:
            selected = next(m for m in modes if m["vdd"]==vdd_fixed)
        else:
            valid_modes = [m for m in modes if f <= m["fmax"]]
            if not valid_modes:
                sla += 1
                selected = modes[0]
            else:
                selected = min(valid_modes,
                    key=lambda m: predict_power(m["vdd"],f))

        if f > selected["fmax"]:
            sla += 1

        margin = selected["fmax"] - f
        min_margin = min(min_margin, margin)

        total_energy += predict_power(selected["vdd"], f)*60
        mode_log.append(selected["name"])

    return total_energy, sla, mode_log, min_margin

# Static strategies
energy_turbo, sla_turbo, _, _ = compute_strategy(1.0)
energy_eco, sla_eco, _, _ = compute_strategy(0.6)

# Adaptive
energy_adapt, sla_adapt, mode_log, min_margin = compute_strategy(None)

print("\n===== SLA BENCHMARK =====")
print("Turbo SLA:", sla_turbo)
print("Eco SLA:", sla_eco)
print("Adaptive SLA:", sla_adapt)

# ==========================================================
# 8️⃣ ENERGY COMPARISON PLOT
# ==========================================================

plt.figure()
plt.bar(["Turbo","Eco","Adaptive"],
        [energy_turbo, energy_eco, energy_adapt])
plt.ylabel("Total Energy (J)")
plt.title("Energy Comparison")
plt.show()

energy_reduction = (energy_turbo-energy_adapt)/energy_turbo*100
print("Energy Reduction (%):", energy_reduction)

# ==========================================================
# 9️⃣ MODE DISTRIBUTION
# ==========================================================

counts = Counter(mode_log)
percent = {k:v/len(workload)*100 for k,v in counts.items()}

plt.figure()
plt.pie(percent.values(), labels=percent.keys(), autopct='%1.1f%%')
plt.title("Adaptive Mode Distribution")
plt.show()

# ==========================================================
# 🔟 EDCP (ENERGY-DELAY-CARBON PRODUCT)
# ==========================================================

avg_delay = df["TPD (s)"].mean()
PUE = 1.5
CI = 500

carbon_turbo = energy_turbo*PUE*CI
carbon_adapt = energy_adapt*PUE*CI

EDCP_turbo = energy_turbo*avg_delay*carbon_turbo
EDCP_adapt = energy_adapt*avg_delay*carbon_adapt

print("\n===== EDCP =====")
print("EDCP Reduction (%):",
      (EDCP_turbo-EDCP_adapt)/EDCP_turbo*100)

# ==========================================================
# 1️⃣1️⃣ CARBON VS CI
# ==========================================================

CI_values=[300,500,700]
carbon_nom=[]
carbon_adp=[]

for c in CI_values:
    carbon_nom.append(energy_turbo*PUE*c)
    carbon_adp.append(energy_adapt*PUE*c)

plt.figure()
x=np.arange(len(CI_values))
plt.bar(x-0.2,carbon_nom,0.4,label="Turbo")
plt.bar(x+0.2,carbon_adp,0.4,label="Adaptive")
plt.xticks(x,CI_values)
plt.title("Carbon Emission vs CI")
plt.legend()
plt.show()

# ==========================================================
# 1️⃣2️⃣ GUARD BAND VALIDATION
# ==========================================================

print("\nMinimum Frequency Guardband (MHz):",
      min_margin/1e6)

# ==========================================================
# 1️⃣3️⃣ INFERENCE LATENCY
# ==========================================================

start=time.time()
for _ in range(1000):
    predict_power(0.8,100e6)
end=time.time()

latency=(end-start)/1000
print("Inference Time (ms):",latency*1000)
print("Acceleration vs 3.7s SPICE:",
      3.7/latency)
carbon_savings = []

for ci in CI_values:
    carbon_turbo = energy_turbo * PUE * ci
    carbon_adapt = energy_adapt * PUE * ci
    savings = carbon_turbo - carbon_adapt
    carbon_savings.append(savings)

plt.figure()
plt.plot(CI_values, carbon_savings, marker='o')
plt.xlabel("Carbon Intensity (gCO2/kWh)")
plt.ylabel("Absolute Carbon Saved (gCO2)")
plt.title("Absolute Carbon Savings vs Carbon Intensity")
plt.grid(True)
plt.show()
