from pathlib import Path
from fastapi import HTTPException

async def getAll():
    base_dir = Path(__file__).resolve().parent
    return dict({"RF" : {"model" : {"CaMau" : (base_dir / "../../../../Drafts/Temp Prediction/models/trained_models/ML/Random Forest/RF_trained_CaMau.pkl").resolve(),
                                    "DH"    : (base_dir / "../../../../Drafts/Temp Prediction/models/trained_models/ML/Random Forest/RF_trained_DH.pkl").resolve(),
                                    "QN"    : (base_dir / "../../../../Drafts/Temp Prediction/models/trained_models/ML/Random Forest/RF_trained_QN.pkl").resolve(),
                                    "TH"    : (base_dir / "../../../../Drafts/Temp Prediction/models/trained_models/ML/Random Forest/RF_trained_TH.pkl").resolve(),
                                    "TSN"   : (base_dir / "../../../../Drafts/Temp Prediction/models/trained_models/ML/Random Forest/RF_trained_TSN.pkl").resolve(),
                                    "NB"    : (base_dir / "../../../../Drafts/Temp Prediction/models/trained_models/ML/Random Forest/RF_trained_NB.pkl").resolve()},
                         "scaler" : {"CaMau" : {"x" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/Random Forest/scaler_x_CaMau.pkl").resolve(),
                                                "y" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/Random Forest/scaler_y_CaMau.pkl").resolve()},
                                     "DH"    : {"x" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/Random Forest/scaler_x_DH.pkl").resolve(),
                                                "y" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/Random Forest/scaler_y_DH.pkl").resolve()},
                                     "QN"    : {"x" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/Random Forest/scaler_x_QN.pkl").resolve(),
                                                "y" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/Random Forest/scaler_y_QN.pkl").resolve()},
                                     "TH"    : {"x" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/Random Forest/scaler_x_TH.pkl").resolve(),
                                                "y" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/Random Forest/scaler_y_TH.pkl").resolve()},
                                     "TSN"   : {"x" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/Random Forest/scaler_x_TSN.pkl").resolve(),
                                                "y" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/Random Forest/scaler_y_TSN.pkl").resolve()},
                                     "NB"    : {"x" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/Random Forest/scaler_x_NB.pkl").resolve(),
                                                "y" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/Random Forest/scaler_y_NB.pkl").resolve()}}},
                 "XGB" : {"model" : {"CaMau" : (base_dir / "../../../../Drafts/Temp Prediction/models/trained_models/ML/XGBOOST/XGB_trained_CaMau.pkl").resolve(),
                                     "DH"    : (base_dir / "../../../../Drafts/Temp Prediction/models/trained_models/ML/XGBOOST/XGB_trained_DH.pkl").resolve(),
                                     "QN"    : (base_dir / "../../../../Drafts/Temp Prediction/models/trained_models/ML/XGBOOST/XGB_trained_QN.pkl").resolve(),
                                     "TH"    : (base_dir / "../../../../Drafts/Temp Prediction/models/trained_models/ML/XGBOOST/XGB_trained_TH.pkl").resolve(),
                                     "TSN"   : (base_dir / "../../../../Drafts/Temp Prediction/models/trained_models/ML/XGBOOST/XGB_trained_TSN.pkl").resolve(),
                                     "NB"    : (base_dir / "../../../../Drafts/Temp Prediction/models/trained_models/ML/XGBOOST/XGB_trained_NB.pkl").resolve()},
                         "scaler" : {"CaMau" : {"x" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/XgBoost/scaler_x_CaMau.pkl").resolve(),
                                                "y" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/XgBoost/scaler_y_CaMau.pkl").resolve()},
                                     "DH"    : {"x" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/XgBoost/scaler_x_DH.pkl").resolve(),
                                                "y" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/XgBoost/scaler_y_DH.pkl").resolve()},
                                     "QN"    : {"x" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/XgBoost/scaler_x_QN.pkl").resolve(),
                                                "y" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/XgBoost/scaler_y_QN.pkl").resolve()},
                                     "TH"    : {"x" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/XgBoost/scaler_x_TH.pkl").resolve(),
                                                "y" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/XgBoost/scaler_y_TH.pkl").resolve()},
                                     "TSN"   : {"x" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/XgBoost/scaler_x_TSN.pkl").resolve(),
                                                "y" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/XgBoost/scaler_y_TSN.pkl").resolve()},
                                     "NB"    : {"x" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/XgBoost/scaler_x_NB.pkl").resolve(),
                                                "y" : (base_dir / "../../../../Drafts/Temp Prediction/models/checkpoints/ML/XgBoost/scaler_y_NB.pkl").resolve()}}}})
async def findByCountryName(name):
    pass
async def findByStoreID(id):
    pass
async def findAll():
    pass