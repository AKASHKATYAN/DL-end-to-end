from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import config, logger 


STAGE_NAME='Prepare base model'

class PrepareBaseModelPipeline:
    def __init__(self):
        pass 
    
    def main(self):
        config=ConfigurationManager()
        prepare_base_model_config=config.get_prepare_base_model_config()
        prepare_base_model=PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
        
if __name__=="main":
    try:
        logger.info("*********")    
        logger.info(f">>>>> stage {STAGE_NAME} started")
        obj=PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed! ")
        logger.info("*********")
    except Exception as e:
        logger.exception(e)
        raise e