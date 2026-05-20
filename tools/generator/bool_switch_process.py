import argparse
import configparser
import yaml
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Tuple
import numpy as np

import itertools

# load bool switch config
def load_bool_switch_config(config_file):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_file)

    specific_values = {}
    for key, value in config['SpecificValues'].items():
        values = [v.strip() == 'True' for v in value.split(',')]
        specific_values[key] = values if len(values) > 1 else values[0]

    default_value_str = config['DefaultValues'].get('default_value', 'False')
    default_value = [v.strip() == 'True' for v in default_value_str.split(',')]

    return specific_values, default_value

def generate_bool_combinations(bool_keys, specific_values=None, default_value=[True,False]):

    # If no specific values are provided, default all keys to have possible values of [True, False]
    if specific_values is None:
        specific_values = {}

    # Special handling for split.
    specific_values["Split"] = [True, False]

    # Create a list of possible value lists for each key
    key_value_lists = []
    for key in bool_keys:
        # If the key has specific values provided, use those values
        if key in specific_values:
            value_list = specific_values[key]
            # If the provided value is not a list, convert it to a list
            if not isinstance(value_list, list):
                value_list = [value_list]
            key_value_lists.append(value_list)
        # Otherwise, use the default value of[True], [False] or [True, False]
        else:
            # Change to default value
            if not isinstance(default_value, list):
                default_value = [default_value]
            key_value_lists.append(default_value)

    # Generate all possible combinations using Cartesian product
    bool_combinations = list(itertools.product(*key_value_lists))

    return bool_combinations

def apply_constraints(bool_config, head_dim, api):
    if (api == 'fwd'):
        # 1. If Is_causal, set Is_local and Has_attn_mask to False
        if bool_config['Is_causal']:
            bool_config['Is_local'] = False
            bool_config['Has_attn_mask'] = False

        # 2. If Is_local, set Has_attn_mask and IsEvenMNConst to False
        if bool_config['Is_local']:
            bool_config['Has_attn_mask'] = False
            bool_config['Is_even_MN'] = False

        # 3. If Has_alibi, set Has_attn_mask to False
        if bool_config['Has_alibi']:
            bool_config['Has_attn_mask'] = False

        # 4. If not IsEvenKConst, set IsEvenMNConst to False
        if not bool_config['Is_even_K']:
            bool_config['Is_even_MN'] = False

        # 5. If not Is_dropout, set Return_softmax to False
        if not bool_config['Is_dropout']:
            bool_config['Return_softmax'] = False

        # 6. If Return_softmax, set IsEvenMNConst to False
        if bool_config['Return_softmax']:
            bool_config['Is_even_MN'] = False

        if not bool_config['Has_attn_mask']:
            bool_config['Merge_attn_mask_ldg'] = False

        # 7. If Is_dropout, set Is_softcap to false
        if bool_config['Is_dropout']:
            bool_config['Is_softcap'] = False

    if (api == 'bwd'):
        if bool_config['Is_causal']:
            bool_config['Is_local'] = False
            bool_config['Has_attn_mask'] = False

        if bool_config['Is_local']:
            bool_config['Has_attn_mask'] = False
            bool_config['Is_even_MN'] = False

        if bool_config['Has_alibi']:
            bool_config['Has_attn_mask'] = False

        if not bool_config['Is_causal'] or bool_config['Is_deterministic']:
            bool_config['Is_balance'] = False

        if not bool_config['Is_even_K']:
            bool_config['Is_even_MN'] = False

        if bool_config['Is_dropout']:
            bool_config['Is_softcap'] = False

    if (api == 'fwd_split'):
        if bool_config['Is_causal']:
            bool_config['Is_local'] = False

        if bool_config['AppendKV']:
            bool_config['Is_even_MN'] = False

        if not bool_config['Is_even_K']:
            bool_config['Is_even_MN'] = False

        if bool_config['Is_local']:
            bool_config['Is_even_MN'] = False

        if head_dim > 128:
            bool_config['Is_even_MN'] = False

    return bool_config

def construct_bool_config(api, head_dim, arch = 'xcore1000'):
    bool_keys_all = {
        'fwd':[
            'Is_dropout', 'Is_causal', 'Is_local', 'Has_alibi', 'Has_attn_mask',
            'Is_even_MN', 'Is_even_K', 'Is_softcap',
            'Return_softmax', 'Rowblock_Parallel_Num', 'Merge_attn_mask_ldg',
        ],
        'bwd':[
            'Is_dropout', 'Is_causal', 'Is_local', 'Has_alibi', 'Has_attn_mask',
            'Is_even_MN', 'Is_even_K', 'Is_softcap',
            'Is_deterministic', 'Is_balance',
        ],
        'fwd_split':[
            'Is_causal', 'Is_local', 'Has_alibi','Is_even_MN', 'Is_even_K', 'Is_softcap',
            'Split', 'AppendKV', 'Is_page_attn'
        ],
    }

    bool_keys = bool_keys_all.get(api)
    # bool_combinations = list(itertools.product([True, False], repeat=len(bool_keys)))

    specific_values = {
        'Is_dropout': True,
        'Is_causal': [True],
        'Is_local': False
    }
    default_value = [False]

    bool_switch_config = "bool_switch.ini"
    specific_values, default_value = load_bool_switch_config(bool_switch_config)
    # print(specific_values, default_value)

    bool_combinations = generate_bool_combinations(bool_keys, specific_values, default_value)
    # print(f"{api}, {bool_combinations}")

    # bool_combinations = generate_bool_combinations(bool_keys)

    valid_configs = []
    for combination in bool_combinations:
        bool_config = dict(zip(bool_keys, combination))

        bool_config = apply_constraints(bool_config, head_dim, api)
        if (api == 'fwd' or api == 'fwd_split' or api == 'bwd'):
            bool_config['Arch'] = arch
        str_config = {key: str(value).lower() for key, value in bool_config.items()}
        valid_configs.append(str_config)

    return valid_configs

def generate_bool_macros(bool_configs):
    bool_key_macros = {
        'Is_dropout': 'DROPOUT',
        'Is_causal' : 'CAUSAL',
        'Is_local'  : 'LOCAL',
        'Has_alibi' : 'ALIBI',
        'Has_attn_mask' : 'ATTN_MASK',
        'Is_even_MN': 'EVENMN',
        'Is_even_K': 'EVENK',
        'Is_softcap': 'SOFTCAP',
        'Return_softmax': 'RETURN_SOFTMAX',
        'Rowblock_Parallel_Num': 'ROWNUM',
        'Merge_attn_mask_ldg': 'MERGE_ATTN_MASK_LDG',
        'Is_deterministic': 'DETERMINISTIC',
        'Is_balance': 'BALANCE',
        'Split': 'SPLIT',
        'AppendKV': 'APPENDKV',
        'Is_page_attn': 'PAGE_ATTN',
    }

    unique_values = {}

    for bool_config in bool_configs:
        for key, value in bool_config.items():
            if key not in unique_values:
                unique_values[key] = set()
            unique_values[key].add(value)

    macros = []
    for key, values in unique_values.items():
        if len(values) == 1:
            value = values.pop()
            if value == 'true':
                macros.append(f"{bool_key_macros[key]}_TRUE")
            elif value == 'false':
                macros.append(f"{bool_key_macros[key]}_FALSE")

    return macros
