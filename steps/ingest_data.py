import logging

import panda as pd
from zenml import step

class IngestData:
    """
    data ingestion class which ingests data from source and returns a DataFrame
    """

    def __init__(self) -> None:
        """ Initializes data ingestion class """
        pass

    def get_data(self) -> pd.DataFrame:
        df = pd.read_csv("./data/olist_customers_dataset.csv")
        return df


@step
def ingest_data() -> pd.DataFrame:
    """
    returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e