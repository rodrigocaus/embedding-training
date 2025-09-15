# Embedding Training 🚀

Welcome to the Embedding Training repository! This framework is designed to help you train and evaluate sentence embedding models with ease, using the powerful `sentence-transformers` library. It's built to be highly configurable and extensible, so you can experiment with different model architectures, loss functions, and training objectives. Let's get started! 🎉

## 🏗️ Training Architecture

The training process is orchestrated by a configuration file that specifies the model, datasets, and training objectives. Here are the key components of the training architecture:

*   **🤖 Base Model**: A `SentenceTransformer` model that serves as the foundation for your training.
*   **🎯 Training Objectives**: You can configure one or more objectives for your training process. Each objective includes:
    *   A **loss function**: The loss function to guide the learning process. We support a variety of built-in and custom loss functions, such as:
        *   `CoSENTLoss`: A loss function based on cosine similarity.
        *   `MultipleNegativesSymmetricRankingLoss`: A ranking loss ideal for information retrieval tasks.
        *   **Knowledge Distillation**: You can distill knowledge from a teacher model to a student model.
        *   **Variance Penalty**: A penalty term to encourage your model to produce embeddings with a specific variance.
    *   A **dataset**: The dataset for the objective. You can easily load datasets from the Hugging Face Hub.
    *   A **preprocessor**: A preprocessor to apply to your dataset before training. We provide preprocessors for ranking, retrieval, and similarity tasks.
*   **📊 Evaluator**: The framework uses an `InformationRetrievalEvaluator` to monitor your model's performance during training. This evaluator measures how well your model performs on an information retrieval task.

## ⚙️ Configuration File

The training process is configured using a YAML or JSON file. Here is an example of the configuration file structure:

```yaml
run_name: "my-experiment"
base_model:
  name: "bert-base-uncased"
  args: {}
output_dir: "models"
objectives:
  - type: "contrastive"
    datasets:
      train:
        name: "squad"
        args: {}
        preprocess_args: {}
      validation:
        name: "squad"
        args: {}
        preprocess_args: {}
    loss: "default"
    loss_args: {}
    matryoshka: {}
    distillation:
      alpha: 0.5
      teacher:
        name: "bert-large-uncased"
        args: {}
      loss_args: {}
    margin: 0.5
    variation_penalty:
      alpha: 0.1
      loss_args: {}
evaluator:
  dataset:
    name: "squad"
    args: {}
    preprocess_args: {}
  args: {}
early_stopping: {}
args:
  num_train_epochs: 3
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  learning_rate: 2.0e-5
```

### Configuration Options

*   `run_name`: The name of your training run.
*   `base_model`: The base model you want to train.
    *   `name`: The name or path of the model.
    *   `args`: Additional arguments for loading the model.
*   `output_dir`: The directory where your trained model will be saved.
*   `objectives`: A list of your training objectives.
    *   `type`: The type of objective, which can be `contrastive` or `similarity`.
    *   `datasets`: The datasets for the objective.
        *   `train`: The training dataset.
        *   `validation`: The validation dataset.
    *   `loss`: The loss function for the objective.
    *   `loss_args`: Additional arguments for the loss function.
    *   `matryoshka`: Configuration for `MatryoshkaLoss`.
    *   `distillation`: Configuration for knowledge distillation.
    *   `margin`: The margin for the similarity function.
    *   `variation_penalty`: Configuration for the variance penalty.
*   `evaluator`: The evaluator to use during training.
    *   `dataset`: The dataset for evaluation.
    *   `args`: Additional arguments for the evaluator.
*   `early_stopping`: Configuration for early stopping.
*   `args`: Additional arguments for the `SentenceTransformerTrainer`.

## 📈 Evaluation

The framework uses an `InformationRetrievalEvaluator` to evaluate your model during training. This evaluator measures the model's performance on an information retrieval task and computes the following metrics:

*   **Mean Reciprocal Rank (MRR)**
*   **Mean Average Precision (MAP)**
*   **Normalized Discounted Cumulative Gain (NDCG)**

## 🚀 How to Run

To start the training process, simply use the following command:

```bash
python embedding train --config <path/to/config.yaml>
```

Happy training! 😃
