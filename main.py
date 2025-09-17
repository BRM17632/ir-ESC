import os
import sys
import kaleido
import pandas as pd
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas._libs.tslibs.parsing import DateParseError
import matplotlib.pyplot as plt
import seaborn as sns
import functions.gui
from dateutil.relativedelta import relativedelta
import functions.GrangerCausality
import functions.NP_model
import functions.settings
functions.settings.init() 
import matplotlib
matplotlib.use('agg')

functions.settings.current_wd = os.getcwd()


def apply_granger_causality(df_inicial, df_train):
    
    stationary_info = functions.GrangerCausality.make_series_stationary(df_train)

    # Create a new DataFrame with differenced variables
    differenced_df = functions.GrangerCausality.make_differenced_dataframe(df_train, stationary_info)


    # Perform Granger causality test with automatic lag selection based on SSR Chi-Square test
    granger_results = functions.GrangerCausality.granger_causality_test(differenced_df, 'y', test='ssr_chi2test')

    #print("Granger Causality Test Results using SSR Chi-Square Test:")
    #for variable, result in granger_results.items():
    #    print(f"{variable}: p-value = {result['p_value']}, Best Lag = {result['best_lag']}")

    # Extract significant variables and their p-values
    significant_variables = [(variable, result['p_value']) for variable, result in granger_results.items() if result['p_value'] <= 0.05]

    # Create a DataFrame from the list
    tabla1=pd.DataFrame(significant_variables, columns=['Variable', 'p-value'])

    # Print the results
    #print("Significant Variables (p-value <= 0.05):")
    #for variable, p_value in significant_variables_list:
    #    print(f"{variable}: p-value = {p_value}")

    # Extract significant variables as a list
    significant_variables_list = [variable for variable, result in granger_results.items() if result['p_value'] <= 0.05]
    future_regressor=significant_variables_list
    #future_regressor
    #Guardamos las variables con significancia
    functions.NP_model.save_regressors('granger_results', future_regressor)

    df_red=df_inicial[['ds','y'] + future_regressor]

    return df_red, future_regressor, granger_results


def entrenamiento(basic, train_end_date, test_start_date, forecast, future_regressor, granger_results):

    events = []
    added_events = []

    rerun = functions.gui.ask_to_rerun_opt(window_title="Reintentar", 
                     prompt_message="Desea agregar un evento?")
    
    while rerun:
        event_name, event_startdate, event_enddate = functions.gui.event_info()

        df_event = pd.DataFrame(
            {
                "ds": pd.date_range(start=event_startdate, end=event_enddate, freq='MS'),
                event_name: 1  # Initialize with 1 for all dates within the range
            }
        )


        # Merge the 'covid' variable with your existing DataFrame
        basic = pd.merge(basic, df_event, on='ds', how='left')
        # Set 'covid' to 0 for dates outside the range
        basic[event_name] = basic[event_name].fillna(0)
        basic[basic[event_name]==1.0]

        events.append(df_event)
        added_events.append(event_name)

        rerun = functions.gui.ask_to_rerun_opt(window_title="Reintentar", 
                     prompt_message="Desea agregar otro evento?")

    # Filter the data based on the cutoff date
    df_train = basic[basic['ds'] < test_start_date]
    df_test = basic[basic['ds'] >= test_start_date]

    next_months_df= basic[basic['ds'] >= test_start_date]
    next_months_df=next_months_df[['ds']+ future_regressor + added_events]
    next_months_df['y']=None
    next_months_df=next_months_df[['ds','y']+ future_regressor + added_events]
    df_train_f=pd.concat([df_train,next_months_df],ignore_index=True)

    lr = functions.NP_model.find_best_lr(basic, test_start_date, forecast, future_regressor, added_events)


    # Split the dataset into train and test sets
    df_train = basic[basic['ds'] < test_start_date].copy()
    df_test = basic[basic['ds'] >= test_start_date].copy()

    #Evaluates and saves all errors to a single CSV file for later analysis
    functions.NP_model.regressor_error(basic, df_train, df_test, lr, test_start_date, forecast, future_regressor, added_events)

    #Model training
    importance_scores = functions.NP_model.get_importance_scores(basic, df_train, df_test, lr, test_start_date, forecast, future_regressor, added_events)

    # Prepare data for plotting
    variables = [var for var, score in importance_scores]
    scores = [score for var, score in importance_scores]


    #################################################################################################
    # Create a DataFrame for seaborn
    df_importance = pd.DataFrame({'Variable': variables, 'Importance Score': scores})

    # Create a horizontal bar plot
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance Score', y='Variable', data=df_importance, palette='viridis')
    plt.xlabel('Importance Score')
    plt.ylabel('Variable')
    plt.title('Variable Importance Scores')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()
    fig.savefig(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/Output/Importance_Scores.png', dpi=fig.dpi)
    #################################################################################################

    #Model training
    m, metrics = functions.NP_model.train_model(df_train, df_test, lr, forecast, future_regressor, added_events)


    #################################################################################################
    # Create a larger figure
    fig = plt.figure(figsize=(12, 6))
    # Plot training loss curve
    plt.plot(metrics['Loss'], label='Training Loss', color='blue')
    # Plot validation loss curve
    plt.plot(metrics['Loss_val'], label='Validation Loss', color='orange')
    # Add labels and legend
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    # Show the plot
    plt.show()
    fig.savefig(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/Output/Model_Loss.png', dpi=fig.dpi)
    #################################################################################################


    #m.plot_parameters().savefig('Model_Parameters.png')

    for variable, result in granger_results.items():
        if variable in future_regressor:
            print(f"{variable}: Best Lag = {result['best_lag']}")

    df_train_forecast=m.predict(df_train_f)
    # Visualize the forecast
    m.plot(df_train_forecast)


    #################################################################################################
    dd= df_train_forecast.merge(basic[['ds','y']], left_on='ds', right_on='ds', how='outer')
    dd[['ds','y_y','yhat1']].tail(forecast+1)
    # Extract 'ds', 'yhat1', and 'y_y' columns from the DataFrame
    ds = dd['ds']
    yhat1 = dd['yhat1']
    y_y = dd['y_y']

    # Plot the data
    fig = plt.figure(figsize=(10, 6))
    plt.plot(ds, yhat1, label='yhat1', color='blue')
    plt.plot(ds, y_y, label='y_y', color='green')
    plt.axvline(x=train_end_date, color='red', linestyle='--', label='train_end_date')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Comparison of yhat1 and y_y')
    plt.legend()
    plt.show()
    fig.savefig(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/Output/Comparacion_training.png', dpi=fig.dpi)
    #################################################################################################


    #Guardamos el modelo entrenado
    functions.NP_model.save_model_attributes(forecast, lr)

    #Guardamos los regresores
    functions.NP_model.save_regressors('future_regressors', future_regressor)

    #Guardamos los eventos
    functions.NP_model.save_events('time_events', events)





def pasos_iniciales(Bs_Hist, fecha_real):
    #########Formatos y variables adicionales########################################################
    # Additional variables to include in all DataFrames
    additional_vars = ['Fondeo1dia', 'Cetes28', 'Cetes91', 'Cetes182', 'Cetes364', 
                    'BonoM3', 'BonoM5', 'BonoM10', 'TasaFedEUA', 'Tbill1m', 'Tbill3m', 
                    'Tbill6m', 'Tbill12m', 'Tnote3A', 'Tnote5A', 'Tnote10A', 'InflacionAn', 
                    'MXNaUSD', 'USDaEUR', 'CambioPIB_anual', 'Desempleo', 'IPC', 
                    'SyP', 'ExpNoPetro', 'VIX_BMV', 'VIX_USA']

    df_inicial = Bs_Hist[['ds', 'y'] + additional_vars].reset_index(drop=True).copy()


    fecha_train = functions.gui.get_user_input(prompt="Ingrese el añomes (AAAAMM) para hacer el corte de train/test:", window_title="Fecha de Corte")
    if fecha_train is not None:
        print(f"You entered: {fecha_train}")
    else:
        print("No value entered.")
        sys.exit()

    #Formateo de fecha de entrenamiento añomes --> AAAA-MM-DD
    date_train = datetime.strptime(str(fecha_train),"%Y%m")
    date_train = datetime.strftime(date_train,"%Y-%m-01")
    date_train = pd.to_datetime(date_train)

    df_train = df_inicial[df_inicial['ds']<= date_train]
    #df_train.tail()
    #################################################################################################

    
    #########Granger Causality#######################################################################
    df_red, future_regressor, granger_results = apply_granger_causality(df_inicial, df_train)



    #########Entrenamiento###########################################################################
    train_end_date = date_train
    test_start_date = train_end_date + pd.DateOffset(months=1)

    fecha_real_formato = datetime.strptime(str(fecha_real),"%Y%m")
    fecha_real_formato = datetime.strftime(fecha_real_formato,"%Y-%m-01")
    fecha_real_formato = pd.to_datetime(fecha_real_formato)

    rel = relativedelta(train_end_date, fecha_real_formato)
    forecast = int(abs(rel.years * 12 + rel.months))

    basic=df_red[['ds','y']+ future_regressor]

    entrenamiento(basic, train_end_date, test_start_date, forecast, future_regressor, granger_results)
    #################################################################################################



def inferencia(model_attributes, future_regressor, Bs_Hist, VarsEco_Base, VarsEco_Adv, significant_variables, events):
    
    
    future_regressor = functions.gui.select_multiple_items(significant_variables, preselected_items=future_regressor, window_title="Seleccione los regresores para usar en el modelo")
    print("Regresores seleccionados:", future_regressor)



    basic=Bs_Hist[['ds','y']+ future_regressor]

    added_events = []

    for df_event in events:
        event_name = df_event.columns[1]
        added_events.append(event_name)
        # Merge the 'covid' variable with your existing DataFrame
        basic = pd.merge(basic, df_event, on='ds', how='left')
        # Set 'covid' to 0 for dates outside the range
        basic[event_name] = basic[event_name].fillna(0)
        basic[basic[event_name]==1.0]


    Esc_Base=VarsEco_Base.loc[VarsEco_Base['aniomes']>fecha_real,['aniomes']+future_regressor]
    # Rename the selected columns
    Esc_Base.rename(columns={'aniomes':'ds'},inplace=True)
    Esc_Base['ds']=(Esc_Base['ds'].astype(str).str[0:4])+'-'+(Esc_Base['ds'].astype(str).str[4:6])+'-'+'01'
    Esc_Base['ds']=pd.to_datetime(Esc_Base['ds'])
    Esc_Base['y']=None
    Esc_Base=Esc_Base[['ds','y']+ future_regressor]
    df_base=pd.concat([basic,Esc_Base],ignore_index=True)
    for event_name in added_events:
        df_base[event_name]=df_base[event_name].fillna(0)
    #df_base.tail()


    #print(basic.loc[basic['y']==basic['y'].max(),'ds'])
    #print(basic.loc[basic['y']==basic['y'].min(),'ds'])

    Esc_Adv=VarsEco_Adv.loc[VarsEco_Adv['aniomes']>fecha_real,['aniomes']+future_regressor]
    # Rename the selected columns
    Esc_Adv.rename(columns={'aniomes':'ds'},inplace=True)
    Esc_Adv['ds']=(Esc_Adv['ds'].astype(str).str[0:4])+'-'+(Esc_Adv['ds'].astype(str).str[4:6])+'-'+'01'
    Esc_Adv['ds']=pd.to_datetime(Esc_Adv['ds'])
    Esc_Adv['y']=None
    Esc_Adv=Esc_Adv[['ds','y']+ future_regressor]
    df_adv=pd.concat([basic,Esc_Adv],ignore_index=True)
    for event_name in added_events:
        df_adv[event_name]=df_adv[event_name].fillna(0)
    #df_adv.tail()


    forecast = len(Esc_Base)
    print("Numero de predicciones: " + str(forecast))

    #################################################################################################

    df_base_forecast, df_adv_forecast = functions.NP_model.run_model(model_attributes, future_regressor, basic, df_base, df_adv, added_events)

    df_base_forecast.rename(columns={'yhat1':'y_Base'},inplace=True)
    df_adv_forecast.rename(columns={'yhat1':'y_Adv'},inplace=True)

    df_Estres= df_base_forecast.merge(df_adv_forecast[['ds','y_Adv']], left_on='ds', right_on='ds', how='outer')

    # Extract 'ds', 'yhat1', and 'y_y' columns from the DataFrame
    ds = df_Estres['ds']
    yBase = df_Estres['y_Base']
    yAdv = df_Estres['y_Adv']
    yReal = df_Estres['y']

    # Plot the data
    fig = plt.figure(figsize=(10, 6))
    plt.plot(ds, yBase, label='yBase', color='blue')
    plt.plot(ds, yAdv, label='yAdv', color='green')
    plt.plot(ds, yReal, label='yReal', color='black')
    plt.axvline(x=basic.ds.max(), color='red', linestyle='--', label='real')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Results')
    plt.legend()
    plt.show()
    fig.savefig(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/Output/Forecast.png', dpi=fig.dpi)

    # Show it in wx GUI
    functions.gui.show_plot_in_wx_gui(fig, window_title="Forecast")

    #################################################################################################

    resultados = df_Estres[['ds','y_Base','y_Adv']].iloc[-forecast:]
    resultados.to_csv(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/Output/resultados.csv', index=False)


    functions.gui.ask_to_rerun(function_to_call=lambda: functions.NP_model.save_regressors('future_regressors', future_regressor), 
                     window_title="Resultados", 
                     prompt_message="El modelo corrio exitosamente.\nLos resultados se guardaron en el archivo 'resultados.csv'.\n\nGuardar los nuevos regresores?")
    rerun = functions.gui.ask_to_rerun_opt(window_title="Reintentar", 
                     prompt_message="Desea reevaluar el modelo?")
    
    if rerun:
        inferencia(model_attributes, future_regressor, Bs_Hist, VarsEco_Base, VarsEco_Adv, significant_variables)
    





if __name__ == "__main__":

    functions.settings.project_name = functions.gui.get_user_input(prompt="Introduzca el nombre del proyecto:", window_title="Titulo")
    if functions.settings.project_name is not None:
        functions.settings.create_dir()
        print(functions.settings.project_name)
    else:
        print("No input received.")
        sys.exit()

    #########Carga de datos##########################################################################
    df = functions.gui.select_file_via_gui(window_title="Seleccione el archivo con datos historicos")
    if df is not None:
        col_fecha = functions.gui.select_column_from_dataframe(df, window_title="Seleccione la columna de fecha")
        if col_fecha:
            print(f"Selected column: {col_fecha}")
        else:
            print("No column selected.")
            sys.exit()

        col_pred = functions.gui.select_column_from_dataframe(df, window_title="Seleccione la columna para la predicción")
        if col_pred:
            print(f"Selected column: {col_pred}")
        else:
            print("No column selected.")
            sys.exit()
    else:
        print("No file selected or failed to load.")
        sys.exit()


    VarsEco_Base = functions.gui.select_file_via_gui(window_title="Seleccione el archivo con las variables economicas (Escenario Base)")
    if VarsEco_Base is not None:
        VarsEco_Base.head()
    else:
        print("No file selected or failed to load.")
        sys.exit()

    VarsEco_Adv = functions.gui.select_file_via_gui(window_title="Seleccione el archivo con las variables economicas (Escenario Adverso)")
    if VarsEco_Adv is not None:
        VarsEco_Adv.head()
    else:
        print("No file selected or failed to load.")
        sys.exit()

    #Definimos columnas de añomes como strings
    df[col_fecha] = df[col_fecha].astype(str)
    if len(df[col_fecha].max()) != 6:
        try:
            df['aniomes'] = pd.to_datetime(df[col_fecha]).dt.strftime('%Y%m')
        except DateParseError as e:
            print("Error: Revisar el formato de la columna de fecha. Asegurar que venga en formato de añomes(AAAAMM) o de fecha-hora.")
            sys.exit()
            
    VarsEco_Base['aniomes'] = VarsEco_Base['aniomes'].astype(str)
    VarsEco_Adv['aniomes'] = VarsEco_Adv['aniomes'].astype(str)
    fecha_real = df['aniomes'].max()

    Eco_Base = VarsEco_Base[VarsEco_Base['aniomes']<= fecha_real]
    Eco_Base.head()

    Eco_Adv = VarsEco_Adv[VarsEco_Adv['aniomes']<= fecha_real]
    Eco_Adv.head()

    same = (Eco_Base.columns==Eco_Adv.columns).all()
    if same:
        print("The column names have the same order.")
    else:
        print("The column names do not have the same order.")
        sys.exit()

    Bs_Hist = df.merge(Eco_Base, how='left', on='aniomes')


    #################################################################################################

    # Rename the selected columns
    Bs_Hist.rename(columns={'aniomes':'ds'},inplace=True)
    Bs_Hist.rename(columns={col_pred:'y'},inplace=True)
    Bs_Hist['ds']=(Bs_Hist['ds'].astype(str).str[0:4])+'-'+(Bs_Hist['ds'].astype(str).str[4:6])+'-'+'01'
    Bs_Hist['ds']=pd.to_datetime(Bs_Hist['ds'])
    #df_inicial.info()



    #########Entrenamiento###########################################################################

    train_choice = functions.gui.ask_to_rerun_opt(window_title="Entrenamiento", 
                    prompt_message="Desea entrenar el modelo?")
    
    if train_choice:
        pasos_iniciales(Bs_Hist, fecha_real)


    #########Inferencia##############################################################################

    #Leemos los archivos guardados previamente para el proyecto
    with open(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/modelo_NP.pkl', 'rb') as f:
        model_attributes = pickle.load(f)

    with open(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/future_regressors.pkl', 'rb') as f:
        future_regressor = pickle.load(f)

    with open(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/granger_results.pkl', 'rb') as f:
        significant_variables = pickle.load(f)

    with open(f'{functions.settings.current_wd}/Proyectos/{functions.settings.project_name}/time_events.pkl', 'rb') as f:
        events = pickle.load(f)


    inferencia(model_attributes, future_regressor, Bs_Hist, VarsEco_Base, VarsEco_Adv, significant_variables, events)




        