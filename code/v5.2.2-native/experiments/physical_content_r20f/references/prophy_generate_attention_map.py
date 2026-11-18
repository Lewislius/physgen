import argparse
import hashlib
import json
import os

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import Qwen2_5_VLProcessor

from configs.attention_map import (
    APPEARANCE_DICT,
    APPEARANCE_QUESTION_TEMPLATE,
    CUSTOM_ATTRIBUTE_QUESTION_TEMPLATE,
    DESCRIBE_QUESTION,
    EOS_TOKEN_ID,
    PHENOMENON_DICT,
    PHENOMENON_QUESTION_TEMPLATE,
    SYSTEM_PROMPT,
    VIDEO_TOKEN_ID,
)
from prophy.models.vision_language import Qwen2_5_VLForConditionalGeneration

# ================== Helper functions ==================
def find_actual_generation_lengths(generated_sequences, original_seq_len, eos_token_id=EOS_TOKEN_ID):
    """Return actual generation length per sample in the batch."""
    batch_size = generated_sequences.shape[0]
    generated_tokens = generated_sequences[:, original_seq_len:]

    actual_lengths = []

    for b in range(batch_size):
        sample_tokens = generated_tokens[b]
        eos_positions = (sample_tokens == eos_token_id).nonzero(as_tuple=True)[0]

        if len(eos_positions) > 0:
            actual_length = eos_positions[0].item()
        else:
            actual_length = len(sample_tokens)

        actual_lengths.append(actual_length)

    return actual_lengths


@torch.no_grad()
def compute_attention_scores_memory_efficient(
    video_k_weights,
    generated_q_weights,
    actual_lengths,
    scale_factor=None,
    device="cuda",
):
    """Compute attention scores with three-stage averaging: layers -> heads -> q_len.
    Supports variable video token length per sample in the batch."""
    if not video_k_weights or not generated_q_weights:
        return None

    num_layers = min(len(video_k_weights), len(generated_q_weights))
    k_sample = video_k_weights[0]
    q_sample = generated_q_weights[0]

    B, num_heads_k, _, head_dim = k_sample.shape  # video_len may differ per sample
    _, num_heads_q, _, _ = q_sample.shape

    if scale_factor is None:
        scale_factor = 1.0 / (head_dim**0.5)

    num_groups = num_heads_q // num_heads_k
    batch_attention_scores = []

    for batch_idx in range(B):
        actual_len = actual_lengths[batch_idx]
        sample_layer_scores = []

        for layer_idx in range(num_layers):
            # Keep computation on GPU in original precision (typically bfloat16)
            k = video_k_weights[layer_idx][batch_idx : batch_idx + 1].to(device)
            q = generated_q_weights[layer_idx][batch_idx : batch_idx + 1, :, :actual_len, :].to(
                device
            )

            # Current sample's video token length
            video_len = k.shape[2]

            group_attention_scores = []

            for k_head_idx in range(num_heads_k):
                k_head = k[:, k_head_idx : k_head_idx + 1, :, :]  # [1, 1, video_len, head_dim]
                q_start = k_head_idx * num_groups
                q_end = q_start + num_groups
                q_group = q[:, q_start:q_end, :, :]  # [1, num_groups, actual_len, head_dim]

                k_head_t = k_head.transpose(-1, -2)  # [1, 1, head_dim, video_len]
                attention_scores = (
                    torch.matmul(q_group, k_head_t) * scale_factor
                )  # [1, num_groups, actual_len, video_len]
                group_attention_scores.append(attention_scores)

            layer_attention = torch.cat(
                group_attention_scores, dim=1
            )  # [1, num_heads_q, actual_len, video_len]
            sample_layer_scores.append(layer_attention)

        # Three-stage average on GPU
        all_layers_attention = torch.stack(
            sample_layer_scores, dim=0
        )  # [num_layers, 1, num_heads_q, actual_len, video_len]
        layer_averaged = torch.mean(
            all_layers_attention, dim=0
        )  # [1, num_heads_q, actual_len, video_len]
        head_averaged = torch.mean(layer_averaged, dim=1)  # [1, actual_len, video_len]
        final_averaged = torch.mean(head_averaged, dim=1)  # [1, video_len]

        # Move to CPU only at the end for saving
        batch_attention_scores.append(final_averaged.squeeze(0).cpu())  # [video_len]

    return batch_attention_scores


def generate_questions_for_video(activate_expert):
    """Generate questions to ask based on activate_expert.

    Each activate_expert entry can be:
    - int: look up the built-in phenomenon / appearance dictionaries.
    - str: use the string directly as a custom physical attribute prompt.
    """

    def build_question_type(label, prefix=""):
        safe_chars = []
        for char in label.strip().lower():
            if char.isalnum():
                safe_chars.append(char)
            else:
                safe_chars.append("_")

        normalized = "".join(safe_chars)
        normalized = "_".join(part for part in normalized.split("_") if part)

        if not normalized:
            normalized = hashlib.md5(label.encode("utf-8")).hexdigest()[:8]

        return f"{prefix}{normalized}"

    def add_question(questions, seen_question_types, label, template):
        question_type = build_question_type(label)
        if question_type in seen_question_types:
            return

        questions.append((question_type, template.format(label=label)))
        seen_question_types.add(question_type)

    questions = [("describe", DESCRIBE_QUESTION)]
    seen_question_types = {"describe"}

    for expert in activate_expert:
        if expert in PHENOMENON_DICT:
            add_question(
                questions,
                seen_question_types,
                PHENOMENON_DICT[expert],
                PHENOMENON_QUESTION_TEMPLATE,
            )
        elif expert in APPEARANCE_DICT:
            add_question(
                questions,
                seen_question_types,
                APPEARANCE_DICT[expert],
                APPEARANCE_QUESTION_TEMPLATE,
            )
        elif isinstance(expert, str):
            custom_attribute = expert.strip()
            if custom_attribute:
                add_question(
                    questions,
                    seen_question_types,
                    custom_attribute,
                    CUSTOM_ATTRIBUTE_QUESTION_TEMPLATE,
                )

    return questions


def create_batch_messages(video_paths, questions_batch):
    """Create batched messages for the model."""
    messages = []

    for video_path, question_text in zip(video_paths, questions_batch):
        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": question_text},
                ],
            },
        ]
        messages.append(message)

    return messages


@torch.no_grad()
def process_batch(
    model,
    processor,
    video_paths,
    questions_batch,
    question_types_batch,
    max_new_tokens=1024,
):
    """Process one batch of videos and questions."""

    # Build messages
    messages = create_batch_messages(video_paths, questions_batch)

    # Prepare inputs
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )

    # Find video token positions per sample
    batch_size = inputs["input_ids"].shape[0]
    batch_video_positions = []

    for batch_idx in range(batch_size):
        input_ids_sample = inputs["input_ids"][batch_idx]
        video_token_positions = (input_ids_sample == VIDEO_TOKEN_ID).nonzero(as_tuple=True)[0]

        if len(video_token_positions) == 0:
            return None  # no video token in this sample

        video_start = video_token_positions[0].item()
        video_end = video_token_positions[-1].item()
        batch_video_positions.append((video_start, video_end))

    # Move to GPU
    inputs = inputs.to("cuda")
    original_seq_len = inputs["input_ids"].shape[1]

    # Configure attention saving with per-sample video token positions
    model.configure_attention_saving_batch(
        batch_video_positions=batch_video_positions,
        original_seq_len=original_seq_len,
    )

    # Generate
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
    }

    with torch.no_grad():
        generation_output = model.generate(**inputs, **generation_kwargs)

    # Extract generation output
    if hasattr(generation_output, "sequences"):
        generated_sequences = generation_output.sequences
    else:
        generated_sequences = generation_output

    # Detect actual generation lengths
    actual_lengths = find_actual_generation_lengths(generated_sequences, original_seq_len)

    # Decode generated text
    generated_tokens = generated_sequences[:, original_seq_len:]
    generated_texts = []
    for i in range(generated_sequences.shape[0]):
        sample_generated = generated_tokens[i : i + 1, : actual_lengths[i]]
        sample_text = processor.batch_decode(
            sample_generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        generated_texts.append(sample_text)

    # Get attention weights
    saved_attention = model.get_saved_attention_weights()
    video_k_weights = saved_attention["video_k_weights"]
    generated_q_weights = saved_attention["generated_q_weights"]

    # Compute attention scores on GPU
    attention_scores = compute_attention_scores_memory_efficient(
        video_k_weights, generated_q_weights, actual_lengths, device="cuda"
    )

    # Clear saved attention weights
    model.clear_saved_attention_weights()

    # Assemble results
    results = []
    for i in range(len(video_paths)):
        result = {
            "type": question_types_batch[i],
            "answer": generated_texts[i],
            "attention_map": attention_scores[i] if attention_scores else None,
        }
        results.append(result)

    return results


def save_attention_result(video_name, question_type, result, output_dir):
    """Save attention result to output_dir."""
    # Video basename without extension
    video_basename = os.path.splitext(os.path.basename(video_name))[0]
    filename = f"{video_basename}_{question_type}.pth"
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(result, filepath)


def check_existing_results(video_name, question_type, output_dir):
    """Return True if result file for this video and question_type already exists."""
    video_basename = os.path.splitext(os.path.basename(video_name))[0]
    filename = f"{video_basename}_{question_type}.pth"
    filepath = os.path.join(output_dir, filename)
    return os.path.exists(filepath)


def parse_args():
    """Parse CLI. Run from the ProPhy repository root with PYTHONPATH=. so that prophy is importable."""
    parser = argparse.ArgumentParser(
        description="Generate per-video attention maps for physical phenomena (ProPhy tool)."
    )
    parser.add_argument(
        "--data_json_path",
        type=str,
        required=True,
        help=(
            "Path to dataset JSON (each item: video_name, activate_expert). "
            "activate_expert supports both built-in integer IDs and custom string attributes."
        ),
    )
    parser.add_argument(
        "--video_base_path",
        type=str,
        required=True,
        help="Base path for video files (joined with video_name from JSON).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for .pth attention results.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to Qwen2.5-VL checkpoint (used with ProPhy VL model with attention saving).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Max new tokens for generation (default: 1024).",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use Flash Attention 2.",
    )
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    with open(args.data_json_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} videos")

    print("Loading model...")
    model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        **model_kwargs,
    )

    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path)
    print("Model loaded successfully")

    print("Preparing tasks...")
    all_tasks = []
    skipped_existing = 0

    for item in data:
        video_name = item["video_name"]
        video_path = os.path.join(args.video_base_path, video_name)
        activate_expert = item["activate_expert"]

        if not os.path.exists(video_path):
            continue

        questions = generate_questions_for_video(activate_expert)

        for question_type, question_text in questions:
            if check_existing_results(video_name, question_type, args.output_dir):
                skipped_existing += 1
                continue

            all_tasks.append((video_path, video_name, question_type, question_text))

    print(f"Total tasks: {len(all_tasks)}")
    print(f"Skipped existing: {skipped_existing}")

    if len(all_tasks) == 0:
        print("No tasks to process!")
        return

    # Process in batches
    with tqdm(total=len(all_tasks), desc="Processing videos") as pbar:
        for i in range(0, len(all_tasks), args.batch_size):
            batch_tasks = all_tasks[i : i + args.batch_size]

            video_paths = [task[0] for task in batch_tasks]
            video_names = [task[1] for task in batch_tasks]
            question_types = [task[2] for task in batch_tasks]
            questions = [task[3] for task in batch_tasks]

            try:
                results = process_batch(
                    model,
                    processor,
                    video_paths,
                    questions,
                    question_types,
                    max_new_tokens=args.max_new_tokens,
                )

                if results:
                    for j, result in enumerate(results):
                        save_attention_result(
                            video_names[j], question_types[j], result, args.output_dir
                        )

                pbar.update(len(batch_tasks))

            except Exception as e:
                print(f"Error processing batch {i//args.batch_size}: {e}")
                pbar.update(len(batch_tasks))
                continue

    print("Processing completed!")


if __name__ == "__main__":
    main()

