from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, DatasetDict, Dataset

import os

SPJ_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/slimpajama6B/")
SPJ_ENTIRE_6B_DATA_PATH = os.path.join(SPJ_DATA_PATH, "entire_6B_dataset_files")
BASE_DATASET_NAME_6B = "DKYoon/SlimPajama-6B"

tknzr = tiktoken.get_encoding("gpt2")

SOURCES_CURRICULUM = [
    "RedPajamaC4",
    "RedPajamaWikipedia",
    "RedPajamaGithub",
    "RedPajamaArXiv",
    "RedPajamaStackExchange",
    "RedPajamaBook"
]

def _tokenize_example(example):
    ids = tknzr.encode_ordinary(example["text"])
    ids.append(tknzr.eot_token)
    return {"ids": ids, "len": len(ids)}

def _preprocess_and_save_categorized_slimpajama(
    base_dataset_name, # e.g., "DKYoon/SlimPajama-6B"
    base_dataset_config=None, # Optional config for the base_dataset_name
    num_proc=40,
    trust_remote_code_for_base=True # Set to False if not needed for base_dataset_name
):
    """
    Loads the base dataset, filters by 'meta.redpajama_set_name', tokenizes,
    and saves each category into separate .bin files.
    """
    print(f"Starting categorization preprocessing using base dataset: {base_dataset_name}")
    dataset_args = {"path": base_dataset_name, "trust_remote_code": trust_remote_code_for_base}
    if base_dataset_config:
        dataset_args["name"] = base_dataset_config
    
    try:
        print(f"Loading base dataset '{base_dataset_name}' (config: {base_dataset_config}). This can take time and disk space.")
        try:
            full_dataset_or_split = load_dataset(**dataset_args, split="train")
        except ValueError:
            print(f"Could not load 'train' split directly for {base_dataset_name}. Loading full dataset and looking for 'train' split.")
            loaded_data = load_dataset(**dataset_args)
            if isinstance(loaded_data, DatasetDict) and "train" in loaded_data:
                full_dataset_or_split = loaded_data["train"]
            elif isinstance(loaded_data, Dataset):
                 full_dataset_or_split = loaded_data
            else:
                raise ValueError(f"Base dataset '{base_dataset_name}' loaded, but 'train' split not found or structure is not Dataset/DatasetDict.")
        print("Base dataset/split loaded.")
    except Exception as e:
        print(f"Fatal Error: Could not load or access 'train' split from base dataset '{base_dataset_name}': {e}")
        print("Ensure the dataset is correctly specified and accessible.")
        print("For categorization, it MUST contain 'text' and 'meta' fields, with 'meta.redpajama_set_name'.")
        raise

    for source_category in SOURCES_CURRICULUM:
        category_dir = os.path.join(SPJ_DATA_PATH, source_category)
        os.makedirs(category_dir, exist_ok=True)
        
        train_bin_path = os.path.join(category_dir, "train.bin")
        val_bin_path = os.path.join(category_dir, "val.bin")

        if os.path.exists(train_bin_path) and os.path.exists(val_bin_path):
            print(f"Data for category '{source_category}' already processed at {category_dir}. Skipping.")
            continue

        print(f"Processing category: {source_category}")
        
        filtered_category_dataset = full_dataset_or_split.filter(
            lambda x: x.get('meta', {}).get('redpajama_set_name') == source_category,
            num_proc=num_proc
        )

        if not filtered_category_dataset or len(filtered_category_dataset) == 0:
            print(f"No data found for category '{source_category}' after filtering. Creating empty .bin files.")
            for p in [train_bin_path, val_bin_path]:
                with open(p, 'wb') as f: pass
            continue
        
        current_test_size = 0.0005
        if len(filtered_category_dataset) <= 1 or int(len(filtered_category_dataset) * current_test_size) == 0:
            cat_splits = DatasetDict({
                "train": filtered_category_dataset, 
                "val": filtered_category_dataset.select([])
            })
            print(f"Warning: Category {source_category} has too few samples for validation split. Val set will be empty.")
        else:
            split_dict = filtered_category_dataset.train_test_split(
                test_size=current_test_size, seed=2357, shuffle=True
            )
            cat_splits = DatasetDict({"train": split_dict["train"], "val": split_dict["test"]})

        for split_name, dset_split in cat_splits.items():
            output_filename = os.path.join(category_dir, f"{split_name}.bin")
            
            if len(dset_split) == 0:
                print(f"Split '{split_name}' for '{source_category}' is empty. Creating empty .bin file.")
                with open(output_filename, 'wb') as f: pass
                continue

            columns_to_remove = [col for col in dset_split.column_names if col in ["text", "meta"]]
            
            tokenized_dset = dset_split.map(
                _tokenize_example,
                remove_columns=columns_to_remove,
                desc=f"Tokenizing {source_category} - {split_name}",
                num_proc=num_proc,
            )

            if len(tokenized_dset) == 0 or "len" not in tokenized_dset.column_names or sum(tokenized_dset["len"]) == 0:
                print(f"Tokenized split '{split_name}' for '{source_category}' is empty or has no token lengths. Creating empty .bin file.")
                with open(output_filename, 'wb') as f: pass
                continue
            
            arr_len = np.sum(tokenized_dset["len"])
            dtype = np.uint16
            arr = np.memmap(output_filename, dtype=dtype, mode="w+", shape=(arr_len,))
            
            idx = 0
            total_batches = min(1024, len(tokenized_dset)) if len(tokenized_dset) > 0 else 0
            if total_batches > 0:
                for batch_idx in tqdm(range(total_batches), desc=f"Writing {output_filename}"):
                    try:
                        batch_data = tokenized_dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
                        if "ids" in batch_data.column_names and len(batch_data["ids"]) > 0:
                            current_batch_ids_list = batch_data["ids"]
                            if isinstance(current_batch_ids_list[0], (list, np.ndarray)):
                                arr_batch = np.concatenate(current_batch_ids_list).astype(dtype)
                            else:
                                arr_batch = np.asarray(current_batch_ids_list, dtype=dtype)
                            
                            if arr_batch.size > 0:
                                arr[idx : idx + len(arr_batch)] = arr_batch
                                idx += len(arr_batch)
                    except Exception as e_batch:
                        print(f"Error writing batch {batch_idx} for {output_filename}: {e_batch}. File may be incomplete.")
                        break 
            arr.flush()
    print("Finished SlimPajama categorization preprocessing.")

def _ensure_entire_6b_dataset_processed(num_proc=40):
    """
    Ensures that the *entire* 'DKYoon/SlimPajama-6B' dataset is downloaded,
    processed, tokenized, and saved into .bin files in SPJ_ENTIRE_6B_DATA_PATH.
    Skips processing if .bin files already exist.
    """
    os.makedirs(SPJ_ENTIRE_6B_DATA_PATH, exist_ok=True)

    train_bin_path_entire_6b = os.path.join(SPJ_ENTIRE_6B_DATA_PATH, "train.bin")
    val_bin_path_entire_6b = os.path.join(SPJ_ENTIRE_6B_DATA_PATH, "val.bin")

    if os.path.exists(train_bin_path_entire_6b) and os.path.exists(val_bin_path_entire_6b):
        print(f"Entire 6B dataset already processed at {SPJ_ENTIRE_6B_DATA_PATH}. Skipping preprocessing.")
        return

    print(f"Processing entire 6B dataset ({BASE_DATASET_NAME_6B}), will save to {SPJ_ENTIRE_6B_DATA_PATH}")
    
    actual_dataset_to_split = None
    try:
        print(f"Loading base dataset '{BASE_DATASET_NAME_6B}' (train split).")
        actual_dataset_to_split = load_dataset(BASE_DATASET_NAME_6B, split="train", trust_remote_code=True)
    except Exception as e:
        raise ValueError(f"Fatal Error: Could not load 'train' split from '{BASE_DATASET_NAME_6B}': {e}")

    if not isinstance(actual_dataset_to_split, Dataset):
        if isinstance(actual_dataset_to_split, DatasetDict) and "train" in actual_dataset_to_split and len(actual_dataset_to_split.keys()) == 1:
            actual_dataset_to_split = actual_dataset_to_split["train"]
            print(f"Loaded '{BASE_DATASET_NAME_6B}' as a DatasetDict, selected 'train' key.")
        else:
            raise ValueError(f"Loaded data for '{BASE_DATASET_NAME_6B}' is not a datasets.Dataset object as expected. Type: {type(actual_dataset_to_split)}")

    print(f"Splitting entire {BASE_DATASET_NAME_6B} data into train/val...")
    current_test_size = 0.0005 
    if len(actual_dataset_to_split) <= 1 or int(len(actual_dataset_to_split) * current_test_size) == 0:
        final_splits_entire_6b = DatasetDict({
            "train": actual_dataset_to_split,
            "val": actual_dataset_to_split.select([])
        })
        print(f"Warning: Dataset {BASE_DATASET_NAME_6B} has too few samples for validation split. Val set will be empty.")
    else:
        split_dataset_entire_6b = actual_dataset_to_split.train_test_split(
            test_size=current_test_size, seed=2357, shuffle=True
        )
        final_splits_entire_6b = DatasetDict({
            "train": split_dataset_entire_6b["train"],
            "val": split_dataset_entire_6b["test"]
        })
    
    columns_to_remove_entire_6b = [col for col in actual_dataset_to_split.column_names if col in ["text", "meta"]]
    if "text" not in columns_to_remove_entire_6b and "text" in actual_dataset_to_split.column_names:
         columns_to_remove_entire_6b.append("text")
    if not columns_to_remove_entire_6b and "text" in actual_dataset_to_split.column_names:
        columns_to_remove_entire_6b = ["text"]

    print(f"Tokenizing splits for entire {BASE_DATASET_NAME_6B}...")
    tokenized_entire_6b = final_splits_entire_6b.map(
        _tokenize_example,
        remove_columns=columns_to_remove_entire_6b,
        desc=f"Tokenizing splits for {BASE_DATASET_NAME_6B}",
        num_proc=num_proc,
    )

    for split, dset in tokenized_entire_6b.items():
        if len(dset) == 0 or "len" not in dset.column_names or np.sum(dset["len"]) == 0:
            print(f"Tokenized split '{split}' for {BASE_DATASET_NAME_6B} is empty. Creating empty .bin file.")
            filename = os.path.join(SPJ_ENTIRE_6B_DATA_PATH, f"{split}.bin")
            with open(filename, 'wb') as f: pass
            continue

        arr_len = np.sum(dset["len"])
        filename = os.path.join(SPJ_ENTIRE_6B_DATA_PATH, f"{split}.bin")
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        
        total_batches = min(1024, len(dset)) if len(dset) > 0 else 0
        idx = 0
        if total_batches > 0:
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                try:
                    batch_data = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
                    if "ids" in batch_data.column_names and len(batch_data["ids"]) > 0:
                        current_batch_ids_list = batch_data["ids"]
                        arr_batch = np.concatenate(current_batch_ids_list).astype(dtype)
                        if arr_batch.size > 0:
                            arr[idx : idx + arr_batch.size] = arr_batch
                            idx += arr_batch.size
                except Exception as e_batch:
                    print(f"Error writing batch {batch_idx} for {filename}: {e_batch}. File may be incomplete.")
                    break 
        arr.flush()
    print(f"Finished processing and saving entire {BASE_DATASET_NAME_6B} data to {SPJ_ENTIRE_6B_DATA_PATH}")

def get_slimpajama_data(num_proc=None):
    """
    Ensures SlimPajama data (categorized from 6B and entire 6B) is processed and saved,
    then loads and concatenates it.
    Categories are loaded first, followed by the entire 'DKYoon/SlimPajama-6B' data.
    Returns a dictionary with 'train' and 'val' NumPy arrays.
    """
    effective_num_proc = num_proc if num_proc is not None else 40

    print(f"Starting SlimPajama data retrieval (using {BASE_DATASET_NAME_6B}) with num_proc = {effective_num_proc}.")

    print(f"Step 1: Ensuring categorized SlimPajama data (from {BASE_DATASET_NAME_6B}) is processed...")
    _preprocess_and_save_categorized_slimpajama(
        base_dataset_name=BASE_DATASET_NAME_6B,
        num_proc=effective_num_proc,
        trust_remote_code_for_base=True
    )
    print("Categorized data processing check complete.")

    print(f"Step 2: Ensuring entire {BASE_DATASET_NAME_6B} dataset is processed...")
    _ensure_entire_6b_dataset_processed(num_proc=effective_num_proc)
    print(f"Entire {BASE_DATASET_NAME_6B} dataset processing check complete.")

    print("Step 3: Loading all processed SlimPajama .bin files...")
    train_arrays = []
    val_arrays = []

    print("Loading categorized data...")
    for category in SOURCES_CURRICULUM:
        category_train_path = os.path.join(SPJ_DATA_PATH, category, "train.bin")
        category_val_path = os.path.join(SPJ_DATA_PATH, category, "val.bin")

        if os.path.exists(category_train_path) and os.path.getsize(category_train_path) > 0:
            try:
                train_arrays.append(np.memmap(category_train_path, dtype=np.uint16, mode='r'))
                print(f"Loaded train data for category: {category}")
            except Exception as e:
                print(f"Warning: Could not load train data for category '{category}' from {category_train_path}: {e}")
        elif not os.path.exists(category_train_path):
            print(f"Info: Train data file for category '{category}' not found at {category_train_path}.")
        
        if os.path.exists(category_val_path) and os.path.getsize(category_val_path) > 0:
            try:
                val_arrays.append(np.memmap(category_val_path, dtype=np.uint16, mode='r'))
                print(f"Loaded val data for category: {category}")
            except Exception as e:
                print(f"Warning: Could not load validation data for category '{category}' from {category_val_path}: {e}")
        elif not os.path.exists(category_val_path):
            print(f"Info: Validation data file for category '{category}' not found at {category_val_path}.")

    print(f"Loading entire {BASE_DATASET_NAME_6B} dataset files...")
    entire_6b_train_path = os.path.join(SPJ_ENTIRE_6B_DATA_PATH, "train.bin")
    entire_6b_val_path = os.path.join(SPJ_ENTIRE_6B_DATA_PATH, "val.bin")

    if os.path.exists(entire_6b_train_path) and os.path.getsize(entire_6b_train_path) > 0:
        try:
            train_arrays.append(np.memmap(entire_6b_train_path, dtype=np.uint16, mode='r'))
            print(f"Loaded main train data from {entire_6b_train_path}")
        except Exception as e:
            print(f"Warning: Could not load main train data ({BASE_DATASET_NAME_6B}) from {entire_6b_train_path}: {e}")
    elif os.path.exists(entire_6b_train_path):
         print(f"Info: Main train data file ({BASE_DATASET_NAME_6B}) at {entire_6b_train_path} is empty.")
    else:
        print(f"Error: Main train data file ({BASE_DATASET_NAME_6B}) not found at {entire_6b_train_path}. This indicates an issue with its processing.")

    if os.path.exists(entire_6b_val_path) and os.path.getsize(entire_6b_val_path) > 0:
        try:
            val_arrays.append(np.memmap(entire_6b_val_path, dtype=np.uint16, mode='r'))
            print(f"Loaded main val data from {entire_6b_val_path}")
        except Exception as e:
            print(f"Warning: Could not load main validation data ({BASE_DATASET_NAME_6B}) from {entire_6b_val_path}: {e}")
    elif os.path.exists(entire_6b_val_path):
        print(f"Info: Main validation data file ({BASE_DATASET_NAME_6B}) at {entire_6b_val_path} is empty.")
    else:
        print(f"Error: Main validation data file ({BASE_DATASET_NAME_6B}) not found at {entire_6b_val_path}. This indicates an issue with its processing.")

    final_train_data = np.concatenate(train_arrays) if train_arrays else np.array([], dtype=np.uint16)
    final_val_data = np.concatenate(val_arrays) if val_arrays else np.array([], dtype=np.uint16)
    
    if final_train_data.size == 0:
        print("Warning: Total loaded training data for SlimPajama is empty.")
    if final_val_data.size == 0:
        print("Warning: Total loaded validation data for SlimPajama is empty.")
        
    print(f"SlimPajama data loading complete. Total train tokens: {final_train_data.size}, Total val tokens: {final_val_data.size}")
    return {'train': final_train_data, 'val': final_val_data}
