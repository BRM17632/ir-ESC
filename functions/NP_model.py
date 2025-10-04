import pickle
import pandas as pd
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
from neuralprophet import set_log_level
from sklearn.metrics import mean_absolute_error
import gc
import torch  
import functions.settings


def save_model_attributes(forecast, lr, banco=""):
    # Define the model attributes
    model_attributes = {
        "seasonality_mode": 'additive',
        "learning_rate": lr,
        "yearly_seasonality": 'auto',
        "n_forecasts": forecast,
        }
    # Save the model attributes using pickle
    with open(f'{functions.settings.current_wd}/Proyectos/{banco}{functions.settings.project_name}/modelo_{functions.settings.project_name}.pkl', 'wb') as f:
        pickle.dump(model_attributes, f)

    print("El modelo se ha guardado correctamente")


def save_regressors(name, future_regressor, banco=""):
    # Save the future regressors using pickle
    with open(f'{functions.settings.current_wd}/Proyectos/{banco}{functions.settings.project_name}/{name}.pkl', 'wb') as f:
        pickle.dump(future_regressor, f)

    print("Los regresores seleccionados se han guardado correctamente")


def save_events(name, events, banco=""):
    # Save the future regressors using pickle
    with open(f'{functions.settings.current_wd}/Proyectos/{banco}{functions.settings.project_name}/{name}.pkl', 'wb') as f:
        pickle.dump(events, f)

    print("Los eventos ingresados se han guardado correctamente")



def find_best_lr(basic, test_start_date, forecast, regressors, events):
    # Disable logging messages unless there is an error
    set_log_level("ERROR")
    set_random_seed(20)

    # Define the learning rates to try
    learning_rates = [0.08,0.06,0.04,0.02,0.008,0.006,0.004,0.002]
    mae_values_all = []  # List to store MAE values for all variables
    error_values_all = []  # List to store error values for each learning rate

    # Check if the CSV file already exists
    last_processed_lr = None
    try:
        progress_df = pd.read_csv(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/Progress/learning_rate_progress.csv')
        last_processed_lr = progress_df['lr_rate'].iloc[-1]  # Get the last processed variable
        print(f"Resuming from: {last_processed_lr}")
    except FileNotFoundError:
        print("No previous progress found. Starting from the first learning_rate.")

    # Find the index of the last processed variable
    start_index = 0
    if last_processed_lr:
        start_index = learning_rates.index(last_processed_lr) + 1

    # Split the dataset into train and test sets
    df_train = basic[basic['ds'] < test_start_date].copy()
    df_test = basic[basic['ds'] >= test_start_date].copy()

    # Function to fit the model, make predictions, and compute MAE
    def evaluate_model(learning_rate):
        set_random_seed(20)
        m = NeuralProphet(
            seasonality_mode='additive',
            learning_rate=learning_rate,
            yearly_seasonality='auto',
            n_forecasts=forecast
        )
        
        for e in events:
            m.add_events(e)
        for var in regressors:
            m.add_future_regressor(var)

        # Fit the model
        metrics = m.fit(df_train, validation_df=df_test, freq='MS')

        # Prepare the next month's data (future data)
        next_months_df = basic[basic['ds'] >= test_start_date]
        next_months_df = next_months_df[['ds'] + regressors + events]
        next_months_df['y'] = None
        next_months_df = next_months_df[['ds', 'y'] + regressors + events]

        # Concatenate the training data and the future data
        df_test_forecast_all = pd.concat([df_train, next_months_df], ignore_index=True)

        # Get the forecast for all variables included
        forecast_all = m.predict(df_test_forecast_all)

        # Calculate Mean Absolute Error (MAE) for the forecast period
        error_all = mean_absolute_error(df_test['y'].iloc[-forecast:], forecast_all['yhat1'].iloc[-forecast:])
        # Get the last MAE_val from metrics
        mae_val = metrics['MAE_val'].iloc[-1]
        
        # Clean up
        del m
        gc.collect()
        
        return mae_val, error_all  # Return both values

    # Try each learning rate and store the corresponding MAE and error values
    for lr in learning_rates:
        mae_val, error_all = evaluate_model(lr)  # Unpack the returned values
        mae_values_all.append((lr, mae_val))
        error_values_all.append((lr, error_all))  # Store the error for later comparison

        # Save the progress (errors and the last processed variable)
        progress_df = pd.DataFrame(mae_values_all, columns=['lr_rate', 'MAE'])
        progress_df.to_csv(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/Progress/learning_rate_progress.csv', index=False)
        print(f"Progress saved. Last processed lr: {lr}.")

        
    # Find the minimum MAE value and its corresponding learning rate
    best_mae_all = min(mae_values_all, key=lambda x: x[1])
    best_learning_rate_all = best_mae_all[0]
    best_mae_all_value = best_mae_all[1]

    # Find the best error all (min error across all learning rates)
    best_mae_error_all = min(error_values_all, key=lambda x: x[1])
    best_learning_rate_error_all = best_mae_error_all[0]
    best_mae_error_all_value = best_mae_error_all[1]

    print("Best MAE with all variables:", best_mae_all_value)
    print("Best Learning Rate for all variables:", best_learning_rate_all)
    print("Best MAE Error:", best_mae_error_all_value)
    print("Best Learning Rate for MAE Error:", best_learning_rate_error_all)

    # Create a DataFrame for metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Best MAE', 'Best Learning Rate for MAE', 'Best MAE Error', 'Best Learning Rate for MAE Error'],
        'Value': [best_mae_all_value, best_learning_rate_all, best_mae_error_all_value, best_learning_rate_error_all]
    })

    # Save the metrics to a CSV file
    metrics_df.to_csv(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/Progress/metrics_lr.csv', index=False)

    print("Metrics have been saved to metrics_lr.csv")


    return best_learning_rate_all




def regressor_error(basic, df_train, df_test, lr, test_start_date, forecast, regressors, events):
    set_random_seed(20)

    errors = []  # To store the errors for each regressor

    # Check if there was a previous progress file
    last_processed_var = None
    try:
        progress_df = pd.read_csv(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/Progress/regressor_errors_progress.csv')
        last_processed_var = progress_df['Variable'].iloc[-1]  # Get the last processed variable
        print(f"Resuming from: {last_processed_var}")
    except FileNotFoundError:
        print("No previous progress found. Starting from the first regressor.")

    # Find the index of the last processed variable
    start_index = 0
    if last_processed_var:
        start_index = regressors.index(last_processed_var) + 1

    # Loop through the regressors starting from the last processed variable
    for excluded_var in regressors[start_index:]:
        print(f"Processing regressor: {excluded_var}")

        # Remove the excluded variable from the datasets
        df_train_excluded = df_train.drop(columns=[excluded_var])
        df_test_excluded = df_test.drop(columns=[excluded_var])

        # Initialize the NeuralProphet model
        set_random_seed(20)
        m = NeuralProphet(
            seasonality_mode='additive',
            learning_rate=lr,
            yearly_seasonality='auto',
            n_forecasts=forecast  # Define the `forecast` variable appropriately
        )

        for e in events:
            m.add_events(e)

        # Add regressors excluding the current variable
        for var in regressors:
            if var != excluded_var and var in df_train_excluded.columns:
                m.add_future_regressor(var)

        # Fit the model on the adjusted datasets (without the excluded variable)
        m.fit(df_train_excluded, validation_df=df_test_excluded, freq='MS')

        # Prepare the next month's data (future data)
        next_months_df = basic[basic['ds'] >= test_start_date]
        next_months_df = next_months_df[['ds'] + [var for var in regressors if var != excluded_var] + events]

        # Set 'y' as None for future data (since we are predicting it)
        next_months_df['y'] = None

        # Reorder columns as needed: ['ds', 'y', future regressors, 'covid']
        next_months_df = next_months_df[['ds', 'y'] + [var for var in regressors if var != excluded_var] + events]

        # Concatenate the training data and the future data
        df_test_forecast = pd.concat([df_train_excluded, next_months_df], ignore_index=True)
        forecast_all = m.predict(df_test_forecast)

        # Calculate Mean Absolute Error (MAE) for the forecast period (the last `forecast` periods)
        error = mean_absolute_error(df_test_excluded['y'].iloc[-forecast:], forecast_all['yhat1'].iloc[-forecast:])
        errors.append((excluded_var, error))  # Track the error for this regressor

        # Save the progress (errors and the last processed variable)
        progress_df = pd.DataFrame(errors, columns=['Variable', 'MAE'])
        progress_df.to_csv(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/Progress/regressor_errors_progress.csv', index=False)
        print(f"Progress saved. Last processed regressor: {excluded_var}.")

        # Release memory after each iteration (optional but recommended)
        del m  # Delete the model to free up memory
        del df_train_excluded, df_test_excluded, df_test_forecast  # Delete dataframes
        gc.collect()  # Explicitly call garbage collection

        # Clear PyTorch GPU memory (only if using PyTorch)
        torch.cuda.empty_cache()  # Only if using PyTorch backend

    # After all regressors are processed, sort errors by the error value (lower MAE = better)
    errors = sorted(errors, key=lambda x: x[1], reverse=True)

    # Optionally, save all errors to a single CSV file for later analysis
    errors_df = pd.DataFrame(errors, columns=['Variable', 'MAE'])
    errors_df.to_csv(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/Progress/all_regressor_errors.csv', index=False)

    # After finishing everything, explicitly call garbage collection one last time
    gc.collect()



def get_importance_scores(basic, df_train, df_test, lr, test_start_date, forecast, regressors, events):
    set_random_seed(20)

    # Store the errors for comparison
    errors_with_all = []
    mae_val_all=[]

    # Initialize the model with all regressors
    set_random_seed(20)
    m_all = NeuralProphet(
        seasonality_mode='additive',
        learning_rate=lr,
        yearly_seasonality='auto',
        n_forecasts=forecast  # Define the `forecast` variable appropriately
    )

    for e in events:
        m_all.add_events(e)

    # Add all regressors to the model
    for var in regressors:
        print(f' variable: {var}')
        m_all.add_future_regressor(var)

    # Fit the model on the full dataset
    metrics = m_all.fit(df_train, validation_df=df_test, freq='MS')

    # Prepare the next month's data (future data)
    next_months_df = basic[basic['ds'] >= test_start_date]
    next_months_df = next_months_df[['ds'] + regressors + events]
    next_months_df['y'] = None
    next_months_df = next_months_df[['ds', 'y'] + regressors + events]

    # Concatenate the training data and the future data
    df_test_forecast_all = pd.concat([df_train, next_months_df], ignore_index=True)

    # Get the forecast for all variables included
    forecast_all = m_all.predict(df_test_forecast_all)

    # Calculate Mean Absolute Error (MAE) for the forecast period with all variables
    error_all = mean_absolute_error(df_test['y'].iloc[-forecast:], forecast_all['yhat1'].iloc[-forecast:])
    # Get the last MAE_val from metrics
    mae_val = metrics['MAE_val'].iloc[-1]
    errors_with_all.append(('All variables', error_all))
    mae_val_all.append(('All variables', mae_val))


    errors_without_variable = pd.read_csv(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/Progress/all_regressor_errors.csv')
    # Now, evaluate the importance of each variable by comparing errors
    importance_scores = []
    # Iterate over each row in the DataFrame
    for _, row in errors_without_variable.iterrows():
        var = row['Variable']
        error_excluded = row['MAE']
        # The importance score is the difference in error between all variables and the current variable excluded
        importance_score = error_excluded - error_all
        importance_scores.append((var, importance_score))

    # Sort by the largest difference (which means the largest impact on the model's performance)
    importance_scores = sorted(importance_scores, key=lambda x: x[1], reverse=True)

    importance_table=pd.DataFrame(importance_scores, columns=['Varible','Score'])
    # Print the sorted importance scores
    #for var, score in importance_scores:
    #    print(f"Variable: {var}, Importance Score: {score}")

    return importance_scores



def train_model(df_train, df_test, lr, forecast, regressors, events):
    set_log_level("ERROR")
    set_random_seed(20)
    
    m = NeuralProphet(
            seasonality_mode='additive',
            learning_rate=lr,
            yearly_seasonality='auto',
            n_forecasts=forecast,        
    )

    for e in events:
        m.add_events(e)
    for var in regressors:
        print(f' variable: {var}')
        m.add_future_regressor(var)


    # Use static plotly in notebooks
    m.set_plotting_backend("plotly-static")
    # Fit the model
    metrics = m.fit(df_train, validation_df=df_test, freq='MS');
    return m, metrics



def run_model(model_attributes, future_regressor, basic, df_base, df_adv, events):
    set_random_seed(20)
    m = NeuralProphet(**model_attributes)


    for e in events:
        m.add_events(e)

    # Add future regressors
    for regressor in future_regressor:
        m.add_future_regressor(regressor)

        # Use static plotly in notebooks
    m.set_plotting_backend("plotly-static")
    # Fit the model
    metrics = m.fit(basic,freq='MS');

    df_base_forecast=m.predict(df_base)
    #m.plot(df_base_forecast)

    df_adv_forecast=m.predict(df_adv)
    #m.plot(df_adv_forecast)

    return df_base_forecast, df_adv_forecast