from src.Translate import logger
from Translate.pipeline.stage_01_data_injestion import DataIngestionTrainingPipeline
from Translate.pipeline.stage_02_prepare_model import PrepareModelPipeline
from Translate.pipeline.stage_03_train_model import TrainingPipeline
from Translate.pipeline.stage_04_model_eval import EvaluationPipeline
# logger.info("Logging Trail")
import warnings

warnings.filterwarnings("ignore")

STAGE_NAME = "Data Ingestion stage"
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_obj = DataIngestionTrainingPipeline()
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = data_obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Prepare model stage"
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_model_obj = PrepareModelPipeline()
        model = prepare_model_obj.main(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Training Pipeline"
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline(train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt,model)
        obj.main()
        # print(model)

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Evaluation Pipeline"
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline(val_dataloader, tokenizer_src, tokenizer_tgt,model)
        obj.main()
        # print(model)

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
       