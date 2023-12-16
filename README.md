# Finetuining large language model
 Finetuning a large language model is the process of adapting a pretrained model to perform a specific task by trainging it with your own data.its like customizing it for a particular job.
 when you are finetuning a large language model you need to remember some important points.
 1. Fine-tuning requires more technical expertise than using an LLM out-of-the-box.
 2. It can be computationally expensive, especially for large models and datasets.
 3. Too little data can lead to overfitting, where the model performs well on the training data but poorly on unseen examples.
 4. Fine-tuning can sometimes introduce biases present in the training data.
 5. Finetining large language model need much more gpu power than usual and take a long time to traing because of the size of the model.

In this project i used Falcon-7b which is a quite famouse large language model with 7 billion parameters.At first you need know some key characteristics of falcon-7b
1. Causal Decoder-only architecture: This design allows for efficient generation of text and code, making it ideal for tasks like creative writing, translation, and code completion.
2. RefinedWeb and curated corpora training: Trained on a massive dataset of 1,500B tokens from RefinedWeb enhanced with curated corpora, it boasts a strong understanding of natural language and the real world.
3. Finetuning available: The model can be further fine-tuned for specific tasks with additional training on smaller, task-specific datasets.
4. Hugging Face integration: Available on Hugging Face, a popular platform for NLP research and development, facilitating easy access and experimentation.

However, fine-tuning LLMs is very much challenging. The first barrier is the hardware limitation. LLMs are large because of their large amount of parameters. They have billions of parameters that require substantial computational resources for training. This means that fine-tuning LLMs often requires access to high-end hardware, which may not be readily available to all researchers and developers. This hardware limitation often acts as a barrier, causing a general reluctance to fine-tune LLMs.

With the help of low rank addaption (LoRA) we can solve this problem and train our llm efficiently.But first we need to understand what is LoRA and how its work-

Low-Rank Adaptation (LoRA) is a technique designed to make the fine-tuning process more efficient and accessible. LoRA introduces a low-rank matrix that is added to the pre-existing weight matrix of the model during the fine-tuning process. This low-rank matrix is much smaller than the original weight matrix, making it easier and faster to update during training.

![Lora](https://images.ctfassets.net/xjan103pcp94/6fct47v2q8PU36X9A1TUzN/62bf8834293c1ec4a7e591f42ed1ffd1/pretrainined-weights-diagram-lora-blog.png)

##### Mathematical Concepts Related to LoRA
Understanding LoRA requires a grasp of certain mathematical concepts, particularly those related to matrix operations. LoRA involves the addition of a low-rank matrix to the weight matrix of the model. A low-rank matrix is a matrix in which the number of linearly independent rows or columns is less than the maximum possible. In other words, it’s a matrix that can be factored into the product of two smaller matrices.

The concept of rank is crucial here. In linear algebra, the rank of a matrix is the maximum number of linearly independent rows or columns in the matrix. A low-rank matrix, therefore, is one where this number is as small as possible. This property of low-rank matrices is what makes them computationally efficient to work with, and it’s the reason why they’re used in LoRA.

The beauty of LoRA, or Low-Rank Adaptation, lies in its elegant simplicity and its profound impact on the training of Large Language Models (LLMs). The core concept of LoRA revolves around an equation that succinctly summarizes how a neural network adapts to specific tasks during training. Let’s break down this equation:

![equation](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*9upxmNtY-azQzi9O0t1GjQ.png)

In this equation, the terms represent:

1. Output (h): This is the result of our operation. It represents the processed data after it has been run through a layer of the neural network.
2. Input (x): This represents the data being input into the layer of the neural network for processing.
3. PretrainedWeight (W₀): This represents the pre-existing weights of the neural network. These weights have been learned over time through extensive training and are kept static during the adaptation process, meaning they do not receive gradient updates.
4. LowRankUpdate(BA): This is where the true magic of LoRA comes in. LoRA assumes that the update to the weights during task-specific adaptation can be represented as a low-rank decomposition, where matrices B and A encapsulate these updates in a more compact, memory-efficient manner.

The equation thus expresses the operation as two components: the first being the multiplication of the input with the original, pre-trained weights of the neural network, and the second being the multiplication of the input with the low-rank approximation of the weight updates. The two results are then summed to obtain the final output.

This ingenious adaptation technique allows LoRA to deliver impressive memory and storage savings. For large transformers trained with Adam, the memory consumption during training can be reduced significantly, allowing the training process to be carried out on less powerful hardware.

LoRA also offers flexibility. It allows for easy task-switching during deployment by just swapping out the LoRA weights, instead of all the parameters. This adds a degree of portability and ease in deploying custom models.

However, it is important to note that LoRA also has limitations. For instance, when you have to batch inputs for different tasks in a single forward pass, handling different A and B matrices might not be straightforward. But these challenges are minor when considering the overarching benefits of LoRA in terms of training efficiency and resource utilization.

By leveraging the low-rank structure of weight adaptations, LoRA brings about a profound change in how we adapt pre-trained models to specific tasks. This innovation presents us with an efficient, flexible, and memory-friendly approach to training Large Language Models

##### 4-Bit NormalFloat (NF4)
4-Bit NormalFloat, or NF4, is a specialized quantization technique employed in machine learning to reduce the memory footprint of models. Quantization is a process where we approximate a continuous set of values (or a very large set of discrete values) with a finite set of discrete symbols or integer values. In the case of 4-bit NormalFloat, the weights of the model are compressed from 32-bit floating-point numbers to 4-bit integers.

A 4-bit integer can range from -8 to 7, which is a much smaller range than a 32-bit floating-point number. However, this is sufficient for many machine learning tasks, especially when combined with other techniques such as low-rank approximations (LoRAs).

One of the key advantages of 4-bit NormalFloat is that it significantly reduces the memory requirements of the model. This is particularly important for large language models, which can have billions of parameters. By reducing the precision of these parameters from 32 bits to 4 bits, we can reduce the memory footprint of the model by a factor of 8. This makes it possible to train larger models on hardware with limited memory.

However, 4-bit NormalFloat is not without its challenges. One of the main issues is that the reduced precision can lead to a loss of accuracy in the model’s predictions. To mitigate this, techniques such as double quantization and low-rank approximations are used.

##### QLoRA and Its Process
Quantization and Low-Rank Adapters (QLoRA) is an innovative method that enhances memory efficiency during the training of complex models with a considerable number of parameters, such as 13 billion. QLoRA combines the concepts of 4-bit quantization and Low-Rank Adapters (LoRAs) to create a more memory-efficient and computationally efficient training process.

In QLoRA, the original pre-trained weights of the model are quantized to 4-bit and kept fixed (frozen) during fine-tuning. Then, a small number of trainable parameters in the form of low-rank adapters are introduced during fine-tuning. These adapters are trained to adapt the pre-trained model to the specific task it is being fine-tuned for, in 32-bit floating-point format.

During any computation in the system, either the forward pass (making predictions) or the backward pass (updating the weights during training), the 4-bit quantized weights are automatically dequantized back to a 32-bit floating-point format. This is because computations with 32-bit floating-point can be faster than with lower precision due to hardware optimization.

After the fine-tuning process, the model consists of the original weights in 4-bit form, and the additional low-rank adapters in their higher precision format. This combined software-hardware approach allows for efficient fine-tuning of large language models on consumer-grade hardware.

QLoRA also leverages Nvidia’s unified memory to make sure that enough memory is free to prevent Out-Of-Memory issues during weight updates. This is particularly important when dealing with large models that have billions of parameters.

The QLoRA process can be summarized by the equation:

![equation](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*yVU2NO_SqEMs3_Aq2N6glw.png)

This equation captures a two-step operation. The first part involves the input X(BF16) being processed through a function that performs double dequantization using the quantization factors C1(FP32) and C2(FP8) and the weights W(NF4). The output of this operation is in Bfloat16 format, preserving the precision necessary for the training process.

The second part of the operation involves another manipulation of the input X(BF16), this time being multiplied by the low-rank approximations L1(BF16) and L2(BF16) of the weights. This part does not involve quantization or dequantization, as the LORAs remain unquantized and hold the rest of the network in a 4-bit representation.

The results of these two parts are added together to yield the final output Y(BF16). This equation, therefore, illustrates a balance between computational efficiency and precision, achieved by the combination of double quantization and low-rank approximations, reducing memory footprint significantly during the training process.

##### QLoRA vs LoRA: The Advantages of QLoRA

Quantization and Low-Rank Adapters (QLoRA) and Low-Rank Adapters (LoRA) are both innovative methods that enhance memory efficiency during the training of complex models. However, QLoRA brings several advantages over LoRA, such as the ability to attach LoRA adaptors at every layer and improved performance with fewer training samples.

According to the original QLoRA research paper, QLoRA can perform as well as full-model fine-tuning. The paper also highlights the impact of NormalFloat4 over standard Float4, showing that NF4 improves performance significantly over FP4 and Int4, and double quantization reduces the memory footprint without degrading performance.

The research findings suggest that 4-bit quantization for inference is possible and that the performance lost due to the imprecise quantization can be fully recovered through adapter fine-tuning after quantization.

The paper also indicates that 4-bit QLoRA with NF4 data type matches 16-bit full fine-tuning and 16-bit LoRA fine-tuning performance on academic benchmarks with well-established evaluation setups. This suggests that QLoRA offers a balance between computational efficiency and precision, making it a promising approach for training large language models.

![image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*gzdNbhaBsQUg-SaqOZzKjg.png)

## Implementing QLoRA for Fine-Tuning Falcon-7b LLM

In this section, we dive into the practical implementation of Quantization and Low-Rank Adapters (QLoRA) for fine-tuning the Falcon-7b Large Language Model (LLM) on a custom dataset. We’ll walk through the code that loads the necessary libraries, prepares the pre-trained model for QLoRA, sets up the LoRA configuration, loads and prepares the dataset, sets up the training arguments, and finally, trains the model. This hands-on approach provides a deeper understanding of how QLoRA works in practice, demonstrating its efficiency and effectiveness in fine-tuning LLMs. 

##### Loading the Required Libraries

     %%capture
     !pip install -Uqqq pip --progress-bar off
     !pip install -qqq bitsandbytes==0.41.3
     !pip install -qqq torch--2.0.1 --progress-bar off
     !pip install -qqq -U git+https://github.com/huggingface/transformers.git@e03a9cc --progress-bar off
     !pip install -qqq -U git+https://github.com/huggingface/peft.git@42a184f --progress-bar off
     !pip install -qqq -U git+https://github.com/huggingface/accelerate.git@c9fbb71 --progress-bar off
     !pip install -qqq datasets==2.12.0 --progress-bar off
     !pip install -qqq loralib==0.1.1 --progress-bar off
     !pip install einops


The first step in the process is to load the necessary libraries. These include `bitsandbytes`, `torch`,`transformers`,`peft`, `datasets`, `accelerate`, `loralib`, `einops`, and `xformers`. Each of these libraries serves a specific purpose in the fine-tuning process:

1. `bitsandbytes`: A lightweight wrapper by Hugging Face (🤗) around CUDA custom functions, particularly 8-bit optimizers and quantization functions. It’s used to handle the quantization process in QLoRA.
2. `peft`: A library by 🤗 that enables parameter efficient fine tuning.
3. `transformers`: A library by 🤗 that provides pre-trained models and training utilities for various natural language processing tasks.
4. `datasets`: Another library by 🤗 that provides easy access to a wide range of datasets.
5. `accelerate`: A library that by 🤗 abstracts the boilerplate code related to multi-GPUs/TPU/fp16, making it easier to write the training loop of PyTorch models.
6. `loralib`: A PyTorch implementation of Low-Rank Adaptation (LoRA), a parameter-efficient approach to adapt a large pre-trained deep learning model.
7. `einops`: A library that simplifies tensor operations.
8. `xformers`: A collection of composable Transformer building blocks.

### import the necessary library to use

    import os
    import bitsandbytes as bnb
    import pandas as pd
    import torch
    import torch.nn as nn
    import transformers
    from datasets import load_dataset
    from huggingface_hub import notebook_login
    from peft import (
        LoraConfig,
        PeftConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

after importing the necessary library i will use a hggingface api token witch is a free token to log in their notebook-

    notebook_login("your api token")

##### Loading the Pre-Trained Model
The next step is to load the pre-trained model. In this case, the Falcon 7b model is loaded using the `AutoModelForCausalLM.from_pretrained()` function from the 🤗 transformers library. The model is loaded in 4-bit using the `BitsAndBytesConfig` from the bitsandbytes library. This is part of the QLoRA process, which involves quantizing the pre-trained weights of the model to 4-bit and keeping them fixed during fine-tuning.

    MODEL_NAME = "OpenAssistant/falcon-7b-sft-mix-2000"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model =AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

for loaw rank addaptation for my model i am using `LoraConfig` from `peft` library.


    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)

 The LoRA configuration is set up using the `LoraConfig` class. The parameters in this configuration include:

`r`: The rank of the update matrices. Lower rank results in smaller update matrices with fewer trainable parameters.
`lora_alpha`: LoRA scaling factor.
`target_modules`: The modules (for example, attention blocks) to apply the LoRA update matrices.
`lora_dropout`: Dropout probability of the LoRA layers.
`bias`: Specifies if the bias parameters should be trained. Can be ‘none’, ‘all’ or ‘lora_only’.
The model is then updated with the LoRA configuration using the `get_peft_model()` function.

For this project i use a question answer dataset from amajon service. My goal is to make a amajon assistant who can tell me the various types of policy using in amajon.

After loading the model we need a efficient prompt to finetune our model. prompt plays an efficient role in finetuning llm. If your promt is perfect that means your large language model will understand how give result of the query. The importance of prompt are given below-

1. Context and Intent: Prompts provide context and specify the desired intent of the fine-tuning task. They tell the LLM what to focus on and what kind of output is expected. This helps the model learn the specific patterns and nuances required for the task.
2. Controlling Creativity: By crafting specific prompts, you can guide the model's creative direction and prevent it from generating irrelevant or off-topic outputs. This is particularly important for tasks like creative writing or code generation.
3. Targeted Learning: Prompts allow you to focus the LLM's learning on a specific subset of the training data. This can significantly reduce the amount of data needed for fine-tuning and improve training efficiency.
4. Mitigating Bias: By carefully crafting prompts, you can mitigate potential biases present in the training data. This helps ensure the LLM generates fair and unbiased outputs in the fine-tuned task.
5. Versatility: Prompts can be easily adapted to different fine-tuning tasks. This allows you to use the same LLM for various purposes without extensive retraining.
6. Fine-grained Control: You can fine-tune the model's behavior by adjusting the wording, style, and structure of the prompts. This gives you a high degree of control over the model's output and allows you to tailor it to your specific needs.

###### So now we will write a prompt for our project-

    def generate_prompt(data_point):
        return f"""
    <|prompter|> {data_point["Question"]}
    <|endoftext|><|assistant|> {data_point["Answer"]}
      """.strip()
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        print(full_prompt)
        tokenized_full_prompt = tokenizer(full_prompt,padding=True, truncation=True)
        return tokenized_full_prompt

the first function will generate prompt for our data input and the second function tokenize all the in put with the prompt.

Then a read the data from the directory with the help of pandas and convert the dataset into a huggingface dataset using `datasets` library. and then i use map function to map the `generate_and_tokenize` function to my dataset. This way of processing the input data will help my llm to understand the task very efficiently.

    import pandas as pd
    df=pd.read_csv("/content/Amazon_Policy_Self.csv")
    
    from datasets import Dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle().map(generate_and_tokenize_prompt)

#### The next task is to Setting Up the Training Arguments

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        learning_rate=2e-4,
        fp16=True,
        save_total_limit=4,
        logging_steps=2,
        output_dir=OUTPUT_DIR,
        max_steps=500,
        optim="paged_adamw_8bit",
        lr_scheduler_type = 'cosine',
        warmup_ratio = 0.05,
        report_to = 'tensorboard'
    )

The training arguments are set up using the `TrainingArguments` class from the transformers library. These arguments include:

`auto_find_batch_size`: If set to True, the trainer will automatically find the largest batch size that fits into memory.
`num_train_epochs`: The number of training epochs.
`learning_rate`: The learning rate for the optimizer.
`bf16`: If set to True, the trainer will use bf16 precision for training.
`save_total_limit`: The total number of checkpoints that can be saved.
`logging_steps`: The number of steps between each logging.
`output_dir`: The directory where the model checkpoints will be saved.
`save_strategy`: The strategy to use for saving checkpoints. In this case, a checkpoint is saved after each epoch.

#### Training the Model

    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()


Finally, the model is trained using the `Trainer` class from the transformers library. The trainer takes in the model, the dataset, the training arguments, and a data collator for language modeling. The training process is then started using the `train()` method of the trainer.
