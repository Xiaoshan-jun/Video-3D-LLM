# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict
import transformers
import tokenizers
import deepspeed
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

#---------------------------------------------------------Dataset code for LLM3D task----------------------------------------------------------------------------
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    embodiedscan_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    add_spatial_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)

    voxel_size: Optional[float] = field(default=0.1)
    min_xyz_range: List[float] = field(default_factory=lambda: [-15, -15, -5])
    max_xyz_range: List[float] = field(default_factory=lambda: [15, 15, 5])

    frame_sampling_strategy: str = 'uniform' # uniform, mc32, mc24

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.list_data_dict = []
        self.video_processor = VideoProcessor(
            data_args.video_folder,
            annotation_dir=data_args.embodiedscan_folder,
            voxel_size=getattr(data_args, "voxel_size", None),
            min_xyz_range=getattr(data_args, "min_xyz_range", None),
            max_xyz_range=getattr(data_args, "max_xyz_range", None),
            frame_sampling_strategy=getattr(data_args, "frame_sampling_strategy", "uniform"),
        )

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # SEGA
            if sample.get("metadata", {}).get("dataset", None) in ["scanrefer", "multi3drefer"]:
                length_list.append(1)
            elif sample.get("metadata", {}).get("dataset", None) in ["scanQA", "sqa3d"]:
                length_list.append(2)
            elif sample.get("metadata", {}).get("dataset", None) in ["scan2cap"]:
                length_list.append(3)
            else:
                raise ValueError("bad data")
        return length_list

    @property
    def task_lengths(self):
        task_mapping = {
            "scanqa": 0,
            "sqa3d": 0,
            "scan2cap": 1,
            "scanrefer": 2,
            "multi3drefer": 2,
        }
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            length_list.append(
                (task_mapping[sample['metadata']['dataset'].lower()], cur_len)
            )
        return length_list

    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad
                if len(image_file) > 1:
                    image = [self.process_image(f, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            else:
                image = [self.process_image(image_file)]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        elif "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]

            # if self.list_data_dict[i].get("metadata", {}).get("dataset", None) in ["scanrefer"]:
            #     # data_dict["box_label"] = torch.tensor(self.list_data_dict[i]["box"])
            #     box_label = self.list_data_dict[i]["box"]
            #     processed_box_label = self.video_processor.process_box_label(
            #         box_label,
            #     )
            #     sources[0]['conversations'][1]["value"] = processed_box_label

            if self.list_data_dict[i].get("metadata", {}).get("dataset", None) in ["scan2cap"]:
                box_input = self.list_data_dict[i]["box_input"][:3]
                # if 'discrete' in model.config.world_position_embedding_type:
                # box_input = self.video_processor.discrete_point(
                #     box_input[:3],
                # )
                # center_point = [str(i) for i in center_point]
                # sources[0]['conversations'][0]["value"] = sources[0]['conversations'][0]["value"].replace('<coord>', ','.join(center_point))
            else:
                box_input = None

            try:
                video_dict = self.video_processor.process_3d_video(
                    video_file,
                    image_processor=self.data_args.image_processor,
                    force_sample=self.data_args.force_sample,
                    frames_upbound=self.data_args.frames_upbound,
                )
                image = video_dict.pop("images")
                video_size = video_dict.pop("video_size")
                video_dict["box_input"] = box_input

                if self.data_args.add_time_instruction:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                    sources[0]["conversations"][0][
                        "value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
                if self.data_args.add_spatial_instruction:
                    spatial_instruction = f"The video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions."
                    sources[0]["conversations"][0][
                        "value"] = f'{DEFAULT_IMAGE_TOKEN}\n{spatial_instruction}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'

                image = [(image, video_size, "video")]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                # print(sources)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to read video file: {video_file}")
                raise e
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif "video" in self.list_data_dict[i]:
            data_dict["image"] = image
            data_dict["video_dict"] = video_dict
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]),
                 "text"),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.list_data_dict[i].get("id", i)

        if self.list_data_dict[i].get("metadata", {}).get("dataset", None) in ["scanrefer", "multi3drefer"]:
            # data_dict["box_label"] = self.list_data_dict[i]["metadata"]["object_id"]
            box_label = self.list_data_dict[i]["metadata"]["object_id"]
            if not isinstance(box_label, list):
                box_label = [box_label]
            box_label = [int(i) for i in box_label]
            data_dict["box_label"] = box_label

            # --------------------- GRPO required fields ---------------------
            dataset_name = self.list_data_dict[i].get("metadata", {}).get("dataset", "").lower()
            convs = self.list_data_dict[i]["conversations"]

            if dataset_name in ["sqa3d", "scanqa", "scan2cap"]:
                raw_answer = convs[1]["value"].strip()
                wrapped_answer = f"<answer>{raw_answer}</answer>"
            elif dataset_name in ["scanrefer", "multi3drefer"]:
                wrapped_answer = "<answer><ground></answer>"
            else:
                raise ValueError(f"Unsupported dataset '{dataset_name}' for GRPO training.")

            data_dict["solution"] = wrapped_answer
            data_dict["problem_type"] = self.list_data_dict[i]["metadata"]["question_type"]
            data_dict["data_type"] = "video"  # or "image" if applicable
            data_dict["path"] = self.list_data_dict[i].get("video") or self.list_data_dict[i].get("image")
            # ---------------------------------------------------------------

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "id"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0  # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels,
                     attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        # batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ids=ids)

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # if all(x is not None and x.shape == images[0].shape for x in images):
            # Image: (N, P, C, H, W)
            # Video: (N, F, C, H, W)
            #     batch["images"] = torch.stack(images)
            # else:
            batch["images"] = images
            if "video_dict" in instances[0]:
                batch["video_dict"] = merge_video_dict([instance["video_dict"] for instance in instances])

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        if "box_label" in instances[0]:
            batch["box_labels"] = [instance["box_label"] for instance in instances]
            batch["use_object_proposals"] = True

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
#---------------------------------------------------------Dataset code for LLM3D task----------------------------------------------------------------------------
#---------------------------------------------------------GRPO argument----------------------------------------------------------------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    # reward_funcs stores a list of reward function
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    #
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    # decide if we use the temporal
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
#---------------------------------------------------------GRPO argument----------------------------------------------------------------------------
#---------------------------------------------------------reward code for grpo rl----------------------------------------------------------------------------
def extract_answer(text):
    """Extract content inside <answer>...</answer>"""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def accuracy_reward(completions, solution, problem_type=None, **kwargs):
    """
    Generic accuracy-style reward for 'sqa3d', 'scanqa', and 'scan2cap'.
    Returns 1.0 if correct, 0.0 otherwise; or uses soft score for open-ended types.
    """
    rewards = []
    for pred, sol in zip(completions, solution):
        output = extract_answer(pred[0]["content"])
        target = extract_answer(sol)

        if problem_type == "multiple choice":
            reward = float(output.strip().upper() == target.strip().upper())

        elif problem_type == "numerical":
            try:
                reward = float(round(float(output), 2) == round(float(target), 2))
            except:
                reward = 0.0

        elif problem_type in ["free-form", "caption", "unknow"]:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            score = scorer.score(target, output)["rougeL"].fmeasure
            reward = max(0.0, min(1.0, score))

        elif problem_type == "OCR":
            reward = 1.0 if output.strip().lower() == target.strip().lower() else 0.0

        else:
            reward = 0.0

        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """
    Reward 1 if output is in proper <answer>...</answer> format
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return [1.0 if re.fullmatch(pattern, c[0]["content"], re.DOTALL) else 0.0 for c in completions]


def grounding_reward(completions, **kwargs):
    """
    Dummy reward for grounding-style tasks like scanrefer / multi3drefer.
    You can later plug in IoU metrics here if box prediction is enabled.
    """
    # Currently always 1.0 to encourage format; extend later with matching
    return [1.0 for _ in completions]


# registry for flexibility
REWARD_FUNCS = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "grounding": grounding_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
#---------------------------------------------------------reward code for grpo rl----------------------------------------------------------------------------

def main(script_args, training_args, model_args):
    # -----------------------------------------------Get reward functions--------------------------------------------------------
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # -----------------------------------------------Get reward functions--------------------------------------------------------
    # -----------------------------------------------load dataset and convert data--------------------------------------------------------
    parser = transformers.HfArgumentParser((DataArguments))
    data_args = parser.parse_args_into_dataclasses()
    dataset = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )
    # for making conversation, the user will ask the following text corresponding to the problem type
    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    def make_conversation_image(example):

        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }

    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }

    def make_conversation_image_and_video(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        msg = {
            "prompt":
                [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                            # example['data_type']: os.getcwd() + "/Video-R1-data" + example['path'][1:]
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                        }
                    ]
                }]
        }

        return msg
    def convert3dtoconversation(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        msg = {
            "prompt":
                [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                            # example['data_type']: os.getcwd() + "/Video-R1-data" + example['path'][1:]
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                        }
                    ]
                }]
        }

        return msg

    # convert the dataset into a chat prompt format
    dataset = dataset.map(make_conversation_image_and_video)
    # -----------------------------------------------load dataset and convert data--------------------------------------------------------

    # define trainer
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)