import os
import urllib.request as request
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.utils.common import get_size

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config
    
    def download_file(self)->str:
      try:
        dataset_url=self.config.source_URL
        zip_download_dir=self.config.local_data_file
        os.makedirs("artifacts/data_ingestion",exist_ok=True)
        logger.info(f"Downloading data from {dataset_url}into file{zip_download_dir}")
        
        gdown.download(dataset_url,zip_download_dir)
        
        logger.info(f"Downlaoded data from {dataset_url} into file{zip_download_dir}")       
      except Exception as e:     
         raise e
    def extract_zip_file(self):    
        unzip_path=self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_path)