# DVA_EAXM

# 🚕 Data Visualization and Analysis (DVA) Lab Exam Solution

## 1. Introduction
The objective of this task is to analyze the behavior of a system using a mathematical function derived from the roll number and apply it to real-world data (NYC Taxi dataset) to identify zones of stability and inefficiency.

---

## 2. Mathematical Derivation

### 2.1 Given Function
f(x) = (x^3)/3 - r x^2 + (r^2 - 1)x

The parameter **r** is calculated as the digital root of the sum of the last four digits of the roll number.

For this case:
r = 8

---

### 2.2 First Derivative
f'(x) = x^2 - 2rx + (r^2 - 1)

Substituting r = 8:
f'(x) = x^2 - 16x + 63

---

### 2.3 Critical Points
f'(x) = 0  
x^2 - 16x + 63 = 0  
(x - 7)(x - 9) = 0  

**Critical Points:**
x = 7, 9

---

### 2.4 Interpretation of Critical Points
- These values represent transition points in system behavior  
- They divide the system into:
  - Stable region  
  - Stress region  
  - Unstable region  

---

## 3. Data Analysis

### 3.1 Dataset Used
- NYC Yellow Taxi Trip Data (January 2023)  
- Contains trip-level information such as time, distance, and fare  

---

### 3.2 Data Preprocessing
- Selected relevant columns:
  - tpep_pickup_datetime  
  - trip_distance  
  - fare_amount  
- Removed null values  
- Removed rows where trip_distance = 0  

---

### 3.3 Feature Engineering
efficiency = fare_amount / trip_distance  

This represents revenue generated per unit distance.

---

## 4. Zone Classification

- **Stable Zone:** efficiency < 7  
- **Stress Zone:** 7 ≤ efficiency ≤ 9  
- **Unstable Zone:** efficiency > 9  

Each trip was classified into one of these zones.

---

## 5. Visualization
A bar chart was created to show the distribution of trips across:
- Stable  
- Stress  
- Unstable  

---

## 6. Interpretation of Results
- No drastic drop in trips or revenue  
- Significant number of trips fall in the stress zone (7–9)  

This indicates:
- Hidden inefficiencies  
- Behavioral changes  
- System instability not visible in averages  

---

## 7. Conclusion
The system is not failing outright but is gradually drifting toward inefficiency, as indicated by the concentration of trips in the stress region.

---

## 8. Summary
- Critical points: 7 and 9  
- Efficiency metric: fare / distance  
- Zones created and visualized  
- System shows signs of inefficiency drift  

---

## ✅ Final Outcome
Mathematical insights were successfully applied to real-world data to detect hidden system behavior changes.
