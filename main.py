from datetime import datetime
from GenerateSalesData import GenerateSalesData
from SkLearnCls import SkLearnCls
from LightGBM import LightGBM
from XGBoost import XGBoost


if __name__ == '__main__':
    salesData = GenerateSalesData()
    df = salesData.generate_sales_data(100000, 10)

    print("==========SKlearn prediction==========")
    sklearn = SkLearnCls()
    sk_starttime = datetime.now()
    sklearn.machine_learning(df)
    sklearn.identify_products_to_purchase(df)
    sk_endtime = datetime.now()
    time_required = sk_endtime - sk_starttime

    print("==========LightGBM Prediction==========")
    lgbm = LightGBM()
    lgbm_starttime = datetime.now()
    lgbm.machine_learning(df)
    lgbm.identify_products_to_purchase(df)
    lgbm_endtime = datetime.now()
    lgbm_time_required = lgbm_endtime - lgbm_starttime

    print("==========XGBoost Prediction==========")
    xgboost = XGBoost()
    xgboost_starttime = datetime.now()
    xgboost.identify_products_to_purchase(df)
    xgboost_endtime = datetime.now()
    xgboost_time_required = xgboost_endtime - xgboost_starttime

    print("==========Time Comparison==========")
    print("SKLearn Time required >> Start:" + str(sk_starttime) + " End: " + str(sk_endtime) + " Difference: " + str(time_required))
    print("LightGBM Time required >> Start: " + str(lgbm_starttime) + " End: " + str(lgbm_endtime) + " Difference: " + str(lgbm_time_required))
    print("XGBoost Time required >> Start: " + str(xgboost_starttime) + " End: " + str(xgboost_endtime) + " Difference: " + str(xgboost_time_required))
