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
  name: sentence-transformers/all-MiniLM-L6-v2
output_dir: models/
objectives:
  - type: contrastive
    datasets:
      train:
        name: "GoBotsAI/e-faq"
        args:
          split: train
        preprocess_args:
          negative_samples: 3
      validation:
        name: "GoBotsAI/e-faq"
        args:
          split: validation
        preprocess_args:
          negative_samples: 1
    matryoshka:
      matryoshka_dims: [64, 128, 256, 384]
  - type: similarity
    datasets:
      train:
        name: "GoBotsAI/e-faq"
        args:
          split: train
    matryoshka:
      matryoshka_dims: [64, 128, 256, 384]
args:
  max_steps: 100
  batch_sampler: "no_duplicates"
  per_device_train_batch_size: 512
  per_device_eval_batch_size: 2048
  gradient_checkpointing: true
  warmup_ratio: 0.1
  learning_rate: 0.00002
  eval_strategy: "steps"
  eval_steps: 25
  save_strategy: "steps"
  save_steps: 25
  save_total_limit: 1
  save_only_model: true
  metric_for_best_model: "cosine_map@1"
  load_best_model_at_end: true
  logging_first_step: true
  logging_strategy: "steps"
  logging_steps: 10
  push_to_hub: false
evaluator:
  dataset:
    name: "GoBotsAI/e-faq"
    args:
      split: validation
    preprocess_args:
      negative_samples: 5
  args:
    mrr_at_k: [10]
    ndcg_at_k: [10]
    accuracy_at_k: [1, 10]
    precision_recall_at_k: [1, 10]
    map_at_k: [1, 10]
early_stopping:
  patience: 5
  threshold: 0.001
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



## 🔬 Parametrized Evaluation

In addition to evaluation during training, the framework provides a way to run standalone, parametrized evaluations on your models. This is useful for comparing different models or for evaluating a model on multiple datasets. The evaluation process is configured using a YAML or JSON file.

### Evaluation Configuration

Here is an example of an evaluation configuration file:

```yaml
model:
  name: Qwen/Qwen3-Embedding-0.6B
dimensions: [768, 1024]
tasks:
  - type: ir
    args:
      mrr_at_k: [10]
      ndcg_at_k: [10]
      accuracy_at_k: [1, 3, 10]
      precision_recall_at_k: [1, 3, 10]
      map_at_k: [1, 3, 10]
    dataset:
      name: GoBotsAI/e-faq
      args:
        split: pt
  - type: ir
    args:
      mrr_at_k: [10]
      ndcg_at_k: [10]
      accuracy_at_k: [1, 3, 10]
      precision_recall_at_k: [1, 3, 10]
      map_at_k: [1, 3, 10]
    dataset:
      name: GoBotsAI/e-faq
      args:
        split: es
  - type: sts
    dataset:
      name: GoBotsAI/GoSim-3
      args:
        split: test
```

### Evaluation Configuration Options

*   `model`: The model you want to evaluate.
    *   `name`: The name or path of the model.
    *   `args`: Additional arguments for loading the model.
*   `dimensions`: A list of embedding dimensions to evaluate.
*   `output_dir`: The directory where the evaluation results will be saved.
*   `tasks`: A list of evaluation tasks.
    *   `type`: The type of task, which can be `sts` or `ir`.
    *   `dataset`: The dataset for the task.
        *   `name`: The name of the dataset.
        *   `args`: Additional arguments for loading the dataset.
        *   `preprocess_args`: Arguments for preprocessing the dataset.
    *   `reranker`: An optional reranker to use for information retrieval tasks.
        *   `type`: The type of reranker, which can be `sentence_transformer` or `served`.
        *   `model`: The reranker model.

### How to Run Evaluation

To start the evaluation process, use the following command:

```bash
python embedding evaluate --config <path/to/eval-config.yaml>
```
