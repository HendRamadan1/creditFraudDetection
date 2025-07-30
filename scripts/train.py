# scripts/train.py
from src.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()