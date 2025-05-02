import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
#script for 8-bit post-training quantization using bitsandbytes

def quantize_model(model_name, output_dir):
    # Load model with 8-bit quantization config
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        load_in_8bit_fp32_cpu_offload=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Quantized model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Pretrained model name or path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    quantize_model(args.model, args.output)
