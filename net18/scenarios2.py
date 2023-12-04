import numpy as np
import pandas as pd
import os


def get_data_by_scenario_and_case(scenario, case):
    assert case > 0
    assert scenario > 0

    file_excel = f'./validation_data/Scenario{scenario}.xlsx'
    xls = pd.ExcelFile(file_excel)
    scheda = xls.sheet_names[case-1]
    dati_scheda = pd.read_excel(file_excel, sheet_name=scheda)
    valori_veri_misure = dati_scheda['Valori veri misure'].tolist()
    misure = dati_scheda['Misure'].tolist()
    valori_veri_stima = dati_scheda['Valori veri stima'].tolist()
    valori_stimati = dati_scheda['Valori stimati'].tolist()
    x = np.hstack((
        [valori_veri_misure[5:22]], [valori_veri_misure[22:39]], [valori_veri_misure[:5]], [valori_veri_misure[39:46]], [valori_veri_misure[46:]]
    ))

    x_hat = np.hstack((
        [misure[5:22]], [misure[22:39]], [misure[:5]], [misure[39:46]], [misure[46:]]
    ))
    y = [valori_veri_stima[:18]]
    y_hat = [valori_stimati[:18]]

    sens_tab = pd.read_excel(file_excel, sheet_name='Sensitivity')
    sens = []


    # sens = np.hstack((s1_sens_pi, s1_sens_qi, s1_sens_vmpu, s1_sens_pf, s1_sens_qf))
    return x, x_hat, y, y_hat, sens

get_data_by_scenario_and_case(1,1)

