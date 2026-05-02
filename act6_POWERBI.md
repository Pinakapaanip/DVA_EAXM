1. Efficiency Zone Distribution (Before Policy)
👉 https://app.powerbi.com/groups/me/reports/67e7831e-bf37-4564-9c04-649870f85eb6

2. Trips Analysis (Hour/Day Patterns)
👉 https://app.powerbi.com/groups/me/reports/3d187d7b-ba1a-49ac-b71d-b95a079dfe25

3. Policy Simulation (After Fare Increase)
👉 https://app.powerbi.com/groups/me/reports/c555f819-2872-46d9-9eaa-21e3d7889af0


I know it cant be acessed ..but as reffernece i have pasted links here!!





ACT 6: THE DECISION ROOM
1. What Can Be Trusted (Data Integrity)

The dataset was first evaluated for reliability. Raw taxi records contained inconsistencies such as zero trip distances, zero fares, and missing values. These entries were removed to ensure that only valid trips were considered.

A comparison between the original and cleaned dataset shows that a portion of data was discarded due to invalid conditions. This cleaning step is essential, as conclusions drawn from unfiltered data would be misleading.

Therefore, the analysis is based only on trips that satisfy realistic conditions, making the results more trustworthy.





2. What Is Actually Happening (Observed Reality)

The system behavior was analyzed using an efficiency metric defined as:

efficiency = fare_amount / trip_distance

Based on this, trips were classified into:

Stable zone (efficiency < 7)
Stress zone (7–9)
Unstable zone (> 9)

The distribution shows that most trips fall within the stable zone, while a noticeable portion lies in the stress region.

Additionally, time-based analysis reveals:

Clear hourly patterns, indicating peak demand periods
Daily variations, suggesting non-random system behavior

These observations reflect actual data patterns without assumptions.






3. Where the System Shows Strain or Stability

The presence of a significant number of trips in the stress zone indicates that the system is not uniformly efficient.

Temporal patterns further show:

Peaks during certain hours (high demand)
Lower activity during off-peak periods

This suggests that system pressure is not constant but varies with time.
The system appears stable overall, but with localized inefficiencies under certain conditions.






4. What Happens If a Decision Is Made (Policy Simulation)

A simulated policy change was introduced by increasing fares by 8%.

After recalculating efficiency:

More trips shifted toward the stress and unstable zones
The number of stable trips decreased

This indicates that the system is sensitive to fare changes.
Instead of stabilizing the system, the policy amplifies pressure, suggesting that even small interventions can 
significantly impact system behavior.





5. Risk of Misinterpretation

If the analysis were performed on uncleaned data:

Invalid trips would distort efficiency values
False patterns might appear
Incorrect conclusions could be drawn

For example, zero-distance trips would artificially inflate efficiency, misleading the analysis.

By comparing cleaned vs uncleaned data, it becomes clear that careless analysis can produce convincing but incorrect results.

Conclusion

The system demonstrates structured behavior rather than randomness. While it remains largely stable, the presence of stress zones and sensitivity to policy changes indicate underlying inefficiencies.

However, conclusions are limited by assumptions such as fixed demand and chosen thresholds. Therefore, results should be interpreted as directional insights rather than absolute truths.



