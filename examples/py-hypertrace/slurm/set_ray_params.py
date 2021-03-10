import argparse
import json

parser = argparse.ArgumentParser(description="Update config with ray parameters.")
parser.add_argument('config_path',  type=str)
parser.add_argument('redis_password',  type=str)
parser.add_argument('ip_head',  type=str)
parser.add_argument('n_cores',  type=int)
args = parser.parse_args()
print("")
print("")
print("Input arguments: ")
print(args)
print("")
print("")

with open(args.config_path, 'r') as fin:
    config = json.load(fin)

print(config)

print("")
print("")
print("Print config['implementation']")
print(config['isofit']['implementation'])

config['isofit']['implementation']['redis_password'] = args.redis_password
config['isofit']['implementation']['ip_head'] = args.ip_head
config['isofit']['implementation']['n_cores'] = args.n_cores
    
with open(args.config_path, 'w') as fout:
    fout.write(json.dumps(config, indent=4, sort_keys=True))
