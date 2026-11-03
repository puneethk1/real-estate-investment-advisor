# EDA findings (auto-generated)

### 1. Distribution of property prices

Mean 254.6L, median 253.9L, skew 0.01 (positive = a tail of expensive properties).

### 2. Distribution of property sizes

Mean 2750 sq ft, median 2747, range 545-4956.

### 3. Price per sq ft by property type

Highest median: Independent House (Rs 9,305); lowest: Villa (Rs 9,190).

### 4. Size vs price

Correlation = -0.00.

### 5. Outliers

IQR outliers - price/sqft: 19,702; size: 0. Price and size were winsorised at 1%/99% during cleaning (raw flags kept in Outlier_* columns).

### 6. Average price/sqft by state

Highest: Karnataka (Rs 13,233); lowest: Rajasthan (Rs 12,913).

### 7. Average price by city

Most expensive city: Bangalore (258.4L); cheapest: Cuttack (250.8L).

### 8. Median age by locality

Overall median 19 yrs; oldest locality Mysore - Locality_444 (36 yrs). Full table in q08 CSV.

### 9. BHK distribution across cities

Most common BHK overall: 1. See heatmap for city-level mix.

### 10. Price trend, top-5 priciest localities

Priciest: Mumbai - Locality_350, Mysore - Locality_42, Pune - Locality_129, Mangalore - Locality_77, Surat - Locality_138. The dataset has no sale dates, so the trend is price/sqft by construction year.

### 11. Numeric correlations

Strongest pair: Size_in_SqFt & Price_per_SqFt (r = -0.62).

### 12. Nearby_Schools vs price/sqft

Correlation = -0.00.

### 13. Nearby_Hospitals vs price/sqft

Correlation = -0.00.

### 14. Price by furnished status

Median price (lakhs): Furnished 253.6, Semi-furnished 253.8, Unfurnished 254.2.

### 15. Price/sqft by facing

Highest: West (Rs 13,118); lowest: East (Rs 13,004); spread 0.9%.

### 16. Count by Owner_Type

Broker: 83,479, Owner: 83,268, Builder: 83,253.

### 17. Count by Availability_Status

Under_Construction: 125,035, Ready_to_Move: 124,965.

### 18. Parking and price

Correlation = 0.00. Avg price by parking, 0=No/1=Yes (lakhs): 0: 254.4, 1: 254.7.

### 19. Amenities and price/sqft

Correlation of amenity count with price/sqft = 0.00. Amenity with highest avg price/sqft: Gym.

### 20. Public transport vs price/sqft and investment potential

Correlation with price/sqft = -0.00; with Good_Investment = 0.11.

### Extra: Good-investment share by city

Highest: Mumbai (41%); lowest: Kolkata (39%).

### Extra: Area vs return

Correlation of size with implied 5-yr growth rate = -0.01.
