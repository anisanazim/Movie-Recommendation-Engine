# Travel Recommendation Engine

The Travel Recommendation System leverages hotel, restaurant, and weather datasets from the US and Europe to provide insightful recommendations based on user preferences, location, and environmental conditions.

## Project Overview

Our recommendation engine:
- Integrates data from multiple sources and regions (US and Europe)
- Implements extensive feature engineering to extract meaningful attributes
- Applies robust feature selection to identify the most predictive features
- Uses K-Nearest Neighbors (KNN) for content-based recommendations
- Employs Random Forest models for ratings and price prediction
- Supports different traveler profiles with personalized recommendations

## Requirements

The project requires Python 3.7+ and the following packages:
```
matplotlib==3.4.3
numpy==1.19.2
pandas==1.2.3
scikit_learn==0.24.1
seaborn==0.11.1
jupyter==1.0.0
ipykernel==6.29.0
plotly==5.18.0
joblib==1.3.2
streamlit==1.27.0
pillow==9.0.1
scipy==1.7.1
```

## Project Structure

```
cap5771sp25-project/
│
├── Data/
│   ├── Europe_Hotel_Reviews.csv
│   ├── Europe_Restaurant_Reviews.csv
│   ├── US_Hotel_Reviews.csv
│   └── US_Restaurant_Review.csv
|   |__ Weather_Data.csv
├── Report/
│   └── Milestone1.pdf
│   └── Milestone2.pdf
│   └── Milestone3.pdf
├── Visualizations/
|   │── Europe_Hotel_Rating_Dist.png
|   │── Europe_Hotel_Pricg.png
|   │── EUrope_Hotel_Top_Naitonalities.png
|   │── European_restaurant_consolidated.png
|   │── European_restaurants_heatmap.png
|   │── US_Hotel_price_levels.png
|   │── US_Hotel_rating_distribution.png
|   │── US_Hotel_top_provinces.png
|   │── US_Restaurant1.png
|   │── US_Restaurant2.png
|   │── US_Restaurant3.png
|   │── weather_event_duration.png
|   │── weather_events_by_type_severity.png
|   │── weather_seasonal_events.png
|── Scripts/
│   ├── analysis.py
│   ├── visualizations.py
│   ├── requirements.txt
│   ├── model_analysis.txt
│   ├── recommendation_example.py
│   ├── recommendation_model.py
│   ├── feature_selection.py
│   ├── feature_engineering.py
│   ├── run_travel_recommendation.py
│   ├── main.py
│   ├── dashboard.py
├── Processed_Data/
├── Engineered_Features/
├── Selected_Features/
├── Models/                   
```

## Key Features Implemented

### Feature Engineering
We've developed several novel features to enhance recommendation quality:
- **Price categorization**: Unified pricing across regions (Budget, Standard, Luxury)
- **Rating standardization**: Normalized ratings using z-scores for cross-platform comparison
- **Amenity scores**: Quantified available amenities based on review mentions
- **Family-friendliness score**: Derived from family-related review terms
- **Business traveler score**: Generated from business facilities mentions
- **Special diet accommodation score**: Based on vegetarian, vegan, and gluten-free options
- **Sentiment analysis features**: Extracted positive-to-negative word ratios from reviews
- **Geographical clustering**: Created location-based clusters for regional recommendations
- **Quality-price ratio**: Combined rating and price to identify value-for-money options

### Feature Selection
We implemented multiple techniques to identify the most predictive features:
- **Tree-based feature importance** via Random Forest
- **Lasso regression** (L1 regularization)
- **Correlation analysis** to remove redundant features
- **Principal Component Analysis** for dimensionality exploration

### Model Development
- **KNN-based recommendation models**: Separate models for hotels and restaurants (optimized with k=5)
- **Random Forest Regression**: For hotel rating prediction
- **Random Forest Classification**: For restaurant price category prediction

## Dashboard Features (Milestone 3)

Our interactive Streamlit dashboard provides:

- **Data Exploration**: Visualize hotel and restaurant distributions, pricing trends, and feature relationships
- **Model Performance Analysis**: Compare different algorithms with comprehensive metrics and visualizations
- **Feature Analysis**: Explore important features driving recommendations
- **Personalized Recommendations**: Get tailored hotel and restaurant suggestions based on user preferences
- **Region Filtering**: Filter recommendations by geographical region (US or Europe)
- **Preference Customization**: Adjust preference sliders for features like price, rating, and amenities
- **Result Visualization**: View recommendation results with similarity scores and feature highlights

The dashboard allows non-technical users to interact with complex recommendation models through an intuitive interface.

## Setup & Installation

1. Clone the repository:
```bash
git clone <repository-url>

## Setup & Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r Scripts/requirements.txt
```

4. Run main.py:
```bash
python Scripts/main.py
```

5. Run run_travel_recommendation.py:
```bash
python Scripts/run_travel_recommendation.py
```

6. Run recommendation_example.py:
```bash
python Scripts/recommendation_example.py
```

7. Run recommendation_model.py:
```bash
python Scripts/recommendation_model.py
```

8. Run model_analysis.py:
```bash
python Scripts/model_analysis.py
```

9. Run the Streamlit dashboard:
```bash
streamlit run Scripts/dashboard.py
```
### drive link for all the data sets: https://drive.google.com/drive/folders/1aZja7RZ31SjId81FlcZAlLGKXazrF3o1?usp=sharing

### drive link for Demo Video: https://drive.google.com/drive/folders/1oz5kjSz2r6m9iAGINGtUl3xAMu5ChEKY?usp=drive_link

### drive link for PPT: https://drive.google.com/drive/folders/1bgyleZW3yXAnx4EAqTAVyn_FMf2OkYX2?usp=sharing