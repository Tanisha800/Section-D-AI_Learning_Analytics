
import joblib
import pandas as pd
import numpy as np

try:
    lin_model = joblib.load("linear_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    
    print("Linear Model Intercept:", lin_model.intercept_)
    print("\nCoefficients:")
    coeffs = pd.DataFrame({
        "Feature": feature_columns,
        "Coefficient": lin_model.coef_
    })
    print(coeffs.sort_values(by="Coefficient", ascending=False))
    
    # Test effect of WklyStudyHours
    # Get index of WklyStudyHours
    idx = feature_columns.index("WklyStudyHours")
    coef = lin_model.coef_[idx]
    print(f"\nWklyStudyHours Coefficient: {coef}")
    
    # Calculate effect of moving from <5 (3) to >10 (12)
    # Note: The model behaves on SCALED data!
    scaler = joblib.load("scaler.pkl")
    # We need to see the range of WklyStudyHours in the training data to know what 3 and 12 map to.
    
    print("\nScaler Mean:", scaler.mean_)
    print("Scaler Scale:", scaler.scale_)
    
    # Find WklyStudyHours index in scaler (it corresponds to feature_columns structure?)
    # train_models calls scaler.fit_transform(X_train). X_train columns are feature_columns.
    # So scaler mean/scale aligns with feature_columns.
    
    mean_hours = scaler.mean_[idx]
    scale_hours = scaler.scale_[idx]
    
    print(f"\nWklyStudyHours Mean: {mean_hours}, Scale: {scale_hours}")
    
    val_low = 3
    val_high = 12
    
    scaled_low = (val_low - mean_hours) / scale_hours
    scaled_high = (val_high - mean_hours) / scale_hours
    
    effect_low = scaled_low * coef
    effect_high = scaled_high * coef
    
    print(f"Effect of 3 hours (scaled {scaled_low:.2f}): {effect_low:.2f}")
    print(f"Effect of 12 hours (scaled {scaled_high:.2f}): {effect_high:.2f}")
    print(f"Total Difference in Score: {effect_high - effect_low:.2f}")

except Exception as e:
    print(f"Error: {e}")
