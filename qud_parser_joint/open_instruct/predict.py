import argparse
import json
import os
from utils import generate_completions, load_hf_lm_and_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None, help="huggingface model name or path.")
    parser.add_argument("--input_files", type=str, nargs="+")
    parser.add_argument("--output_file", type=str, default="data/eval_creative_tasks/model_outputs.jsonl")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for prediction.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_chat_format", action="store_true", help="whether to use chat format to encode the prompt.")
    parser.add_argument("--num_return_sequences", default=1, type=int)
    parser.add_argument("--stop_sequences", type=str, nargs="+")
    args = parser.parse_args()

    # check if output directory exists
    if args.output_file is not None:
        output_dir = os.path.dirname(args.output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # load the data
    instances = []
    for input_file in args.input_files:
        print(input_file)
        with open(input_file, "r") as f:
            data = [json.loads(x) for x in f.readlines()]
        instances = data 
    
    # filter out instances that the prompt is too long
    instances = [x for x in instances if len(x["prompt"].split(" ")) <= 2048]
    print(f"Total number of instances: {len(instances)}")

    if args.model_name_or_path is not None:
        if args.use_chat_format:
            prompts = [
                "<|user|>\n" + x["prompt"] + "\n<|assistant|>\n" for x in instances
            ]
        else:
            prompts = [x["prompt"] for x in instances]
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path, 
            load_in_8bit=args.load_in_8bit, 
            load_in_half=True,
            gptq_model=args.gptq
        )
        if args.stop_sequences:
            stop_sequences = [tokenizer.encode(" " + x, add_special_tokens=False)[1:] for x in args.stop_sequences]
            print(stop_sequences)
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                batch_size=args.batch_size,
                max_new_tokens=256,
                stop_id_sequences=stop_sequences,
                num_return_sequences = args.num_return_sequences,
                num_beams = args.num_return_sequences
            )
        else:
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                batch_size=args.batch_size,
                max_new_tokens=256,
                num_return_sequences = args.num_return_sequences,
                num_beams = args.num_return_sequences
            )
        with open(args.output_file, "w") as f:
            for instance, output in zip(instances, outputs):
                instance["output"] = output
                f.write(json.dumps(instance) + "\n")

    print("Done.")