import subprocess 
import os

# generate the raw USD code and camera views
for i in range(10):
    subprocess.run(['python', 'generate_train_data.py', '--asset_dir', 'assets', '--data_dir', 'usd_data', '--max_objects', '80', '--num_each_count', '120', '--count_start_index', f'{120*i}', '--init_random_seed', f'{i}'])

for i in range(10):
    subprocess.run(['python', 'generate_test_data.py', '--asset_dir', 'assets', '--data_dir', 'test_data', '--max_objects', '80', '--num_each_count', '12', '--count_start_index', f'{12*i}', '--init_random_seed', f'{i}'])

# render USD
isaac = os.path.expanduser('~/isaacsim/python.sh')
subprocess.run([isaac, 'render_train_images.py'])
subprocess.run([isaac, 'render_train_images.py', '--data_dir', 'test_data', '--end_sample', '119'])

# prepare training data 
subprocess.run(['python', 'prepare_data.py'])