from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training import Training
from cnnClassifier import config, logger

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
     pass
    
    def main(self):
        config=ConfigurationManager()
        training_config=config.get_training_config()   
        training=Training(config=training_config)
        training.get_base_model()
        training.train_validator_generator()
        training.train() 


if __name__ =='main':
    try:
        logger.info(f"**************")
        logger.info(f">>>>>>>stage{STAGE_NAME}started <<<<<<<")
        obj=ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>stage{STAGE_NAME}completed<<<<<" )
    except Exception as e:
        logger.exception(e)
        raise e    