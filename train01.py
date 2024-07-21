from ragatouille import RAGTrainer
from transformers import pipeline
from datasets import load_dataset
import logging

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load the HRMIS_MASTER dataset
    logging.info("Loading dataset...")
    dataset = load_dataset("Kaballas/HRMIS_MASTER")

    augmented_pairs = []

    # Generate augmented pairs from the dataset
    logging.info("Generating augmented pairs...")
    for item in dataset['train']:  # Use the 'train' split or any other desired split
        questions = item['questions']
        answer = item['answers']
        augmented_pairs.append((questions, answer))

    logging.info("Augmented pairs generated.")

    # Initialize the RAGTrainer
    logging.info("Initializing RAGTrainer...")
    trainer = RAGTrainer(
        model_name="HRMIS_QuestionGeneratedColBERT",
        pretrained_model_name="mixedbread-ai/mxbai-colbert-v1",
        language_code="en"
    )

    # Prepare the training data
    logging.info("Preparing training data...")
    trainer.prepare_training_data(
        raw_data=augmented_pairs,
        data_out_path="./data/",
        all_documents=augmented_pairs,
        num_new_negatives=10,
        mine_hard_negatives=True
    )
    logging.info("Training data prepared.")

if __name__ == "__main__":
    main()
