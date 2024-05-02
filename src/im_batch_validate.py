import pandas as pd
import json
import os
from pandas import DataFrame
from pathlib import Path



def filter_boxes(batch_dict: dict, 
                 batch_span: tuple, 
                 confidence_det_threshold: float,
                 ) -> list:
    """
    Filters act_nr within a batch of images based on det confidence threshold and allowed span within batch.
    
    Args:
    - batch_dict: Dictionary containing json out from htrflow_core.
    - batch_span: Tuple specifying the start and end of the valid range for act_numbers.
    - confidence_threshold: Minimum score threshold for a segment to be considered.
    
    Returns:
    - List of filtered image dictionaries.
    """
    filtered_batch = []
    for img_dict in batch_dict.get('contains', []):
        filtered_img = {
            'classification': img_dict.get('classification'),
            'img_name': img_dict.get('image_name'),
            'act_nrs': [],
            'values': [],
            'crossed_over_act_nrs': [],
            'values_crossed_over_act_nrs': [],
            'act_nrs_out_of_bounds': [],
            'values_out_of_bounds': []
        }

        for segment in img_dict.get('contains', []):
            seg_info = segment.get('segment', {})
            seg_score = seg_info.get('score', 0)
            seg_label = seg_info.get('class_label')

            # Skip segments that do not meet the confidence thres
            if seg_score < confidence_det_threshold: #or seg_label == 'crossed_over_act_nr':
                continue

            # Extract text scores and check against the batch span
            text_score = segment.get('text_result', {}).get('scores', [0])[0]
            for text in segment.get('text_result', {}).get('texts', []):
                try:
                    act_nrs = [int(x) for x in text.split(',')]
                except ValueError:
                    continue  # skip texts that cannot be converted to integers

                for act_nr in act_nrs:
                    entry = {'seg_cat': seg_label, 'seg_score': seg_score, 'text_score': text_score}
                    if act_nr in range(batch_span[0], batch_span[1] + 1) and (seg_label == 'act_nr' or seg_label == 'act_nr_span'):
                        filtered_img['act_nrs'].append(act_nr)
                        filtered_img['values'].append(entry)
                    elif act_nr not in range(batch_span[0], batch_span[1] + 1) and (seg_label == 'act_nr' or seg_label == 'act_nr_span'):
                        filtered_img['act_nrs_out_of_bounds'].append(act_nr)
                        filtered_img['values_out_of_bounds'].append(entry)
                    elif act_nr in range(batch_span[0], batch_span[1] + 1) and seg_label == 'crossed_over_act_nr':
                        filtered_img['crossed_over_act_nrs'].append(act_nr)
                        filtered_img['values_crossed_over_act_nrs'].append(entry)

                break #only first beam
                        
        filtered_batch.append(filtered_img)

    return filtered_batch

def manual_check(batch: list, check_first_page_and_empty: bool = True,
                 check_span_and_len_1: bool = True, check_many_act_nrs: bool = True,
                 check_vertical: bool = True, check_low_text_score: bool = True,
                 check_jump: bool = True, htr_confidence_thr: float = 0.9,
                 max_num_of_act_nrs: int = 7, first_page_thr: float = 0.07,
                 jump_limit: int = 2) -> list:
    """
    Evaluates a batch of image dictionaries for conditions that require manual checking based on multiple criteria.
    
    Args:
    - batch: Dictionary of images, each represented by key-value pairs.
    - check_first_page_and_empty: Check for high first-page probability and no action numbers.
    - check_span_and_len_1: Check for a single act number when det label is span.
    - check_many_act_nrs: Check for a high count of action numbers.
    - check_vertical: Check for vertical alignment of action numbers.
    - check_low_text_score: Check for low text scores in values.
    - check_jump: Check for significant jumps in consecutive action numbers.
    - htr_confidence_thr: if any valid act_nr htr gets a confidence below this threshhold, then manual check
    - max_num_of_act_nrs: if number of act_numbers in htr exceeds this, then manual check
    - first_page_htr: If len(act_nr) == 0 and first_page confidence over this thr, then manual check
    - jump_limit: if there's a jump larger than this thr within the suggested act nrs for an image, then manual check
    
    Returns:
    - Updated dictionary with flags indicating if a manual check is needed and the reasons.
    """
    for img_dict in batch:
        img_dict['manual_check_reason'] = ''
        img_dict['manual_check'] = False

        labels = {x['seg_cat'] for x in img_dict.get('values', [])}
        act_nrs = sorted(img_dict.get('act_nrs', []))

        # Check for first page with no action numbers
        if (check_first_page_and_empty and
            img_dict.get('classification', {}).get('first_page', 0) > first_page_thr and
            not act_nrs):
            img_dict['manual_check'] = True
            img_dict['manual_check_reason'] = 'first_page_and_empty'

        # Check for a span and exactly one action number
        if (check_span_and_len_1 and 'act_nr_span' in labels and len(act_nrs) == 1):
            img_dict['manual_check'] = True
            img_dict['manual_check_reason'] = 'span_and_len_1'

        # Check for multiple action numbers
        if (check_many_act_nrs and len(act_nrs) >= max_num_of_act_nrs):
            img_dict['manual_check'] = True
            img_dict['manual_check_reason'] = 'many_act_nrs'

        # Check for "vartical" labels in detection model
        if (check_vertical and ('act_nr_vertical' in labels or 'act_nr_vertical_span' in labels)):
            img_dict['manual_check'] = True
            img_dict['manual_check_reason'] = 'vertical'

        # Check each value for low text scores
        if (check_low_text_score and
            any(value['text_score'] < htr_confidence_thr for value in img_dict.get('values', []))):
            img_dict['manual_check'] = True
            img_dict['manual_check_reason'] += (',' if img_dict['manual_check_reason'] else '') + 'low_text_score'

        # Check for significant jumps in act numbers
        if check_jump and len(act_nrs) > 1:
            jumps = [act_nrs[i] > act_nrs[i-1] + jump_limit for i in range(1, len(act_nrs))]
            if any(jumps):
                img_dict['manual_check'] = True
                img_dict['manual_check_reason'] += (',' if img_dict['manual_check_reason'] else '') + 'jump_in_act_numbers'

    return batch

def fill_in_non_first_pages(batch: list, first_page_confidence_thr: float = 0.20) -> list:
    """
    The function fills in empty non-first pages with the first of the previous pages to have act_nrs
    
    Args:
    - first_page_confidence_thr: if first_page_con below this threshhold and act_nrs is empty then copy value from above
    
    Returns:
    - Updated list of dictionaries where images with no act_nrs has been filled in.
    """
    
    for i, current_img in enumerate(batch):
        current_img['copied'] = False  # Mark current image as not copied by default
        
        # Conditions for copying attributes from previous images
        first_page_confidence = current_img['classification']['first_page']
        act_numbers_empty = len(current_img['act_nrs']) == 0
        
        if first_page_confidence < first_page_confidence_thr and act_numbers_empty:
            # Find a previous image with set 'act_nrs' or marked for manual check
            for offset in range(1, i + 1):
                prev_img = batch[i - offset]
                if prev_img['manual_check']:  # Stop if manual check is required
                    current_img['manual_check'] = True
                    current_img['act_nrs'] = prev_img['act_nrs']
                    current_img['copied'] = True
                    break
                elif len(prev_img['act_nrs']) != 0:  # Copy attributes if 'act_nrs' is not empty
                    current_img['act_nrs'] = prev_img['act_nrs']
                    current_img['copied'] = True
                    break

    return batch

def check_manual_ratio(batch: list) -> list:
    """returns ratio of manual check to total number of images"""
    num_of_manuals = 0
    for img_dict in batch:
        if img_dict['manual_check'] == True and img_dict['copied'] == False:
            num_of_manuals += 1
    return num_of_manuals / len(batch)

def get_manual_stats(batch: list) -> dict:
    """
    Calculates statistics for manual checks based on given reasons within a batch of images.
    
    Each image dictionary in the batch must have a 'manual_check_reason' key with comma-separated reasons.
    This function counts occurrences of each reason and calculates their ratios relative to the total number of images.

    Args:
    batch (list): A list of dictionaries, where each dictionary represents image data and includes
                  a 'manual_check_reason' key containing reasons for manual checks.

    Returns:
    dict: A dictionary with counts and ratios for each reason. Keys for counts are the reason names,
          and keys for ratios are the reason names followed by '_ratio'.
    """
    # Initialize counters for each reason and their ratios
    reasons = [
        "first_page_and_empty", "span_and_len_1", "many_act_nrs",
        "vertical", "low_text_score", "jump_in_act_numbers"
    ]
    batch_stat = {reason: 0 for reason in reasons}
    batch_stat.update({f"{reason}_ratio": 0 for reason in reasons})
    
    # Count the occurrences of each reason
    for img_dict in batch:
        encountered_reasons = img_dict['manual_check_reason'].split(',')
        for reason in encountered_reasons:
            if reason in batch_stat:
                batch_stat[reason] += 1
    
    # Calculate ratios
    num_of_images = len(batch)
    if num_of_images > 0:
        for reason in reasons:
            batch_stat[f"{reason}_ratio"] = batch_stat[reason] / num_of_images

    return batch_stat


def calculate_average_scores(batch: list) -> tuple[float, float]:
    """
    Calculates the average segmentation score and text score across a batch of images.

    Args:
    batch (list): A list of dictionaries, where each dictionary represents image data.
                  Each dictionary must contain a 'values' key, which is a list of dictionaries
                  representing individual values, where each value dictionary contains
                  'seg_score' and 'text_score' keys representing segmentation and text scores respectively.

    Returns:
    tuple[float, float]: A tuple containing the average segmentation score and average text score.
    """
    total_score_detect = 0
    total_score_tr = 0
    total_inst = 0
    
    for img_dict in batch:
        for value in img_dict['values']:
            total_score_detect += value['seg_score']
            total_score_tr += value['text_score']
            total_inst += 1

    average_score_detect = total_score_detect / total_inst
    average_score_tr = total_score_tr / total_inst

    return average_score_detect, average_score_tr


#just temporary, we will get an export from arkis with allowed span for each batch, then no need for this index file
def read_index_file(path_to_ind_file: str) -> DataFrame:

    columns = [
        'batch',
        'img_name',
        'ind_short',
        'ind_long',
        'page_nr'
    ]

    df = pd.read_csv(path_to_ind_file, sep='\t', header=None, usecols=[0,2,6,7,8], names=columns, index_col=1)
    return df

#change when we get the export from arkis
def extract_batch_span(batch_df: DataFrame) -> tuple:
    first_row_value = int(batch_df['ind_short'].iloc[0].split('-')[0])
    last_row_value = int(batch_df['ind_short'].iloc[-1].split('-')[0])
    return (first_row_value, last_row_value)

def write_json(out_dict: dict, out_file: str):
    with open(out_file, 'w') as file:
        json.dump(out_dict, file, indent=4)

import argparse
import json
from pathlib import Path

def main():
    # Setup argparse to handle command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('path_to_json', type=str, help='Path to the JSON file')
    parser.add_argument('path_to_ind_file', type=str, help='Path to the index file')

    # Parse arguments
    args = parser.parse_args()

    # Assign variables from arguments
    path_to_json = args.path_to_json
    path_to_ind_file = args.path_to_ind_file
    output_path_filtered_json = '/home/sneriko/projects/im_eval/output/test3.json'
    output_path_batch_log = '/home/sneriko/projects/im_eval/output/test_log3.json'

    # Load JSON data
    with open(path_to_json, 'r') as f:
        raw_batch = json.load(f)

    df_complete_index = read_index_file(path_to_ind_file=path_to_ind_file)
    batch_str = Path(path_to_json).stem
    df_batch_index = df_complete_index[df_complete_index['batch'] == int(batch_str)]
    batch_span = extract_batch_span(df_batch_index)

    filtered_batch = filter_boxes(batch_dict=raw_batch, batch_span=batch_span, confidence_det_threshold=0.3)
    manual_check_batch = manual_check(filtered_batch)
    filled_in_batch = fill_in_non_first_pages(manual_check_batch)

    manual_ratio = check_manual_ratio(filled_in_batch)
    manual_dict = get_manual_stats(filled_in_batch)
    average_detect, average_tr = calculate_average_scores(batch=filled_in_batch)

    batch_stats = dict()
    batch_stats['batch'] = batch_str
    batch_stats['num_of_indexed_images'] = len(filled_in_batch)
    batch_stats['average_detect'] = average_detect
    batch_stats['average_text_recog'] = average_tr
    batch_stats['manual_ratio'] = manual_ratio
    batch_stats['manual_stats'] = manual_dict

    # Function to write data to JSON
    write_json(out_dict=filled_in_batch, out_file=output_path_filtered_json)
    write_json(out_dict=batch_stats, out_file=output_path_batch_log)

if __name__ == '__main__':
    main()
