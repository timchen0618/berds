import torch

def load_model(args, device, logger):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
    from peft import AutoPeftModelForCausalLM, PeftModel, PeftConfig
    from transformers import pipeline
    
    if args.model == 'google/t5_xxl_true_nli_mixture':
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        logger.info('finish loading model')
        model.to(device)
        return model, tokenizer
    elif args.model.find('Mistral') != -1:
        if args.model.find('saved') != -1 or args.model.find('timchen0618') != -1:
            model = AutoPeftModelForCausalLM.from_pretrained(args.model)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        logger.info('finish loading model')
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        print(tokenizer.pad_token)
        model.to(device)
        return model, tokenizer
    elif args.model.find('zephyr') != -1:
        pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")
        logger.info('finish loading model')
        return pipe, None
    
    elif args.model.find('gemma') != -1:        
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map="auto")
        logger.info('finish loading model')
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model.to(device)
        return model, tokenizer
    elif args.model.find('llama') != -1:
        from llama import Llama, Dialog
        import yaml
        with open(args.llama_config, "r") as stream:
            try:
                configs = yaml.safe_load(stream)[args.model]
            except yaml.YAMLError as exc:
                logger.error(exc)
        print(configs)
                
        model = Llama.build(
            ckpt_dir=configs['ckpt_dir'],
            tokenizer_path=configs['tokenizer_path'],
            max_seq_len=configs['max_seq_len'],
            max_batch_size=configs['max_batch_size']
        )
        configs = configs['generation']
        logger.info('finish loading model')
        return model, configs
    else:
        raise NotImplementedError
    

  