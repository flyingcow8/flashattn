
import argparse
import re
import ast
import yaml
import os

# Define the mapping for the configuration fields based on category names
field_mappings = {}
# Function to preprocess the list of configurations and fix non-standard syntax
def preprocess_config(config_str):
    # Find patterns like `32x64` and replace them with strings `'32x64'`
    return re.sub(r'(\d+x\d+)', r'"\1"', config_str)
# Function to parse the base config and generate kernel configs
def parse_base_config(file_path):
    kernel_configs = {}
    current_category = None
    with open(file_path, 'r') as file:
        for line in file:
            # Match a category (fwd, bwd, fwd_split)
            pattern = r'(\w+):\s*(\w+),\s*(\w+),\s*\[\(([\w\s,]+)\)\]'
            match = re.match(pattern, line.strip())
            if match:
                current_category = match.group(1)
                field_mappings[current_category] = [item.strip() for item in match.group(4).split(',')]
                kernel_configs[current_category] = []
                continue
            # Parse lines that define configurations under a category
            if current_category and line.strip():
                # Split the line into three parts: hdim_qk, hdim_v, and the list of tuples
                match = re.match(r'(\d+|\w+),\s*(\d+|\w+),\s*\[(.+)\](?:,\s*\((.+)\))?', line.strip())
                # match = re.match(r'(\d+\w+),\s*(\d+\w+),\s*\[(.+)\], \((.*?)\)', line.strip())
                if match:
                    hdim_qk = int(match.group(1))
                    hdim_v = match.group(2)
                    configs_list_str = match.group(3)
                    # Preprocess the string to handle non-standard syntax
                    configs_list_str = preprocess_config(configs_list_str)
                    # Convert the string of list of tuples into a Python object
                    configs_list = ast.literal_eval(f'[{configs_list_str}]')
                    # Rename the config fields based on the category's mapping
                    field_names = field_mappings.get(current_category, [])
                    for dtype in ['bfloat16','float16']:
                        for conf in configs_list:
                            renamed_config = dict(zip(field_names, conf))
                            if hdim_v == 'k64':
                                for v in range(64, 257, 64):
                                    base_config = {
                                        'hdim_qk': hdim_qk,
                                        'hdim_v': v,
                                        'dtype': dtype,
                                        **renamed_config
                                    }
                                    kernel_configs[current_category].append({**base_config})
                            else:
                                base_config = {
                                        'hdim_qk': hdim_qk,
                                        'hdim_v': int(hdim_v),
                                        'dtype': dtype,
                                        **renamed_config
                                }
                                kernel_configs[current_category].append({**base_config})
    return kernel_configs

# Function to write the kernel configs to a Yaml file
def write_kernel_configs_to_file(kernel_configs, output_file):
    formatted_configs = {}
    # Create the desired format for the YAML output
    for category, configs in kernel_configs.items():
        formatted_configs[category] = []
        for config in configs:
            # Specify the keys that should retain both key and value in the combined key
            selected_keys = ['hdim_qk', 'hdim_v', 'block_m', 'block_n']
            # Construct the combined key
            combined_key_parts = [category]
            # Add selected keys with both key and value in the combined key
            for k in selected_keys:
                if k in config:
                    combined_key_parts.append(f"{k.replace('_','')}_{config[k]}")
            # Add the values of other keys (without their names)
            for k, v in config.items():
                if k not in selected_keys:
                    combined_key_parts.append(str(v))  # Only append the value
            # Join all parts to form the combined key
            combined_key = "_".join(combined_key_parts)
            config['kernel_id']= combined_key
            formatted_configs[category].append(config)
    # Write the formatted configuration as YAML
    with open(output_file, 'w') as f:
        yaml.dump(formatted_configs, f, default_flow_style=False)


parser = argparse.ArgumentParser(description="Generate kernel traits candidate yaml.")
parser.add_argument("-a", "--arch", help="device arch",  default="xcore1000")
args = parser.parse_args()
# File path to the base config
file_path = 'xcore1000_kernel_traits_config.txt'
if args.arch == "xcore1000":
    file_path = 'xcore1000_kernel_traits_config.txt'
elif args.arch == "xcore1500":
    file_path = 'xcore1500_kernel_traits_config.txt'
else:
    assert False, f"the arch {args.arch} is not support"

output_file = '../../flash_attn/tuning/kernel_traits_candidates.yaml'
if not os.path.exists('out'):
    os.makedirs('out')
# Generate kernel configs
kernel_configs = parse_base_config(file_path)
# Write the kernel configs to kernel_configs.py
write_kernel_configs_to_file(kernel_configs, output_file)
