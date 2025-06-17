import os
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "ospanbatyr/dmi_embs"
pkl_names = [
	'audiocaps.tar.gz',
	'candels.tar.gz',
	'chebi.tar.gz',
	'clothodetail.tar.gz',
	'coco.tar.gz',
	'openvid.tar.gz',
	'prefixes.tar.gz',
	'sharegpt4v.tar.gz',
	'sharegpt4video.tar.gz',
	'sydney.tar.gz'
]

for pkl_name in pkl_names:
	filepath = hf_hub_download(repo_id=repo_id, filename=pkl_name, repo_type='dataset', local_dir=".")
	filepath = Path(pkl_name)
	stem = filepath.with_suffix('').with_suffix('')
	os.makedirs(stem, exist_ok=True)
	new_filepath = stem / filepath
	
	try:
		print(filepath, stem, new_filepath)
		os.system(f"mv {filepath} {new_filepath}")
		os.chdir(stem)
		os.system(f"tar -xzf {filepath}")
		os.system(f"rm -f {filepath}")
		os.chdir("..")
	except Exception as e:
		print(e)
		pass
	
