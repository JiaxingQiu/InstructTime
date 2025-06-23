import os
os.chdir(vital_path)
model_name = ''.join([dataset_name, '_at', vital_suffix])

if dataset_name == "syn_gt":
    exec(open(os.path.join(vital_path, 'run/settings.py')).read())
    exec(open(os.path.join(vital_path, 'run/configs/synthetic_w_gt.py')).read())
    exec(open(os.path.join(vital_path, 'run/prepare_datasets/synthetic.py')).read())

elif dataset_name == "syn":
    exec(open(os.path.join(vital_path, 'run/settings.py')).read())
    exec(open(os.path.join(vital_path, 'run/configs/synthetic.py')).read())
    exec(open(os.path.join(vital_path, 'run/prepare_datasets/synthetic.py')).read())

elif dataset_name == "air":
    exec(open(os.path.join(vital_path, 'run/settings.py')).read())
    exec(open(os.path.join(vital_path, 'run/configs/air_quality.py')).read())
    exec(open(os.path.join(vital_path, 'run/prepare_datasets/air_quality.py')).read())

elif dataset_name == "nicu":
    exec(open(os.path.join(vital_path, 'run/settings.py')).read())
    exec(open(os.path.join(vital_path, 'run/configs/nicu.py')).read())
    exec(open(os.path.join(vital_path, 'run/prepare_datasets/nicu.py')).read())

# Load model
overwrite = False
exec(open(os.path.join(vital_path, 'run/inputs.py')).read())
exec(open(os.path.join(vital_path, 'run/model.py')).read())
vital_model = model

os.chdir(tedit_path)