import os
import pickle
from typing import Any, Type, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

DEFAULT_FILENAME = "CustomerSatisfactionEnvironment"

class cs_materializer(BaseMaterializer):
    """
    Using Custom Materializer for the Customer Satisfaction model
    """

    ASSOCIATED_TYPES = (
        str,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
        CatBoostRegressor,
        RandomForestRegressor,
        LGBMRegressor,
        XGBRegressor,
    )

    def handle_input(self, data_type: Type[Any]) -> Union[
        str,
        np.ndarray,
        pd.Series,
        pd.Dataframe,
        CatBoostRegressor,
        RandomForestRegressor,
        LGBMRegressor,
        XGBRegressor,
    ]:
        """
        this function will load saved model "model.pickle" from artifacts
        and return it
        Args:
            data_type : type of model to be loaded
        """
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            obj = pickle.load(fid)
        return obj

    def handle_return(self,obj: Union[
        str,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
        CatBoostRegressor,
        RandomForestRegressor,
        LGBMRegressor,
        XGBRegressor,
    ],
                      ) -> None:
        """
        saves model to artifact store

        Args:
            model: model to be saved
        """

        super().handle_return(obj)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(obj, fid)