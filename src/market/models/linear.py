from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def linear_regression(X_train, X_test, y_train, **kwargs):
    # Standardize data:
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    #  Train model:
    model = LinearRegression(fit_intercept=True).fit(X_train, y_train)
    # Compute forecasts:
    preds = model.predict(X_test)
    return preds
