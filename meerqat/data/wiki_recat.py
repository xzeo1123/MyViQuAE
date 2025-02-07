from datasets import load_dataset, concatenate_datasets

# Định nghĩa danh sách file của dataset
data_files = {
    'humans_with_faces': 'humans_with_faces.jsonl.gz',
    'humans_without_faces': 'humans_without_faces.jsonl.gz',
    'non_humans': 'non_humans.jsonl.gz'
}

# Tải dataset từ Hugging Face
kb = load_dataset('PaulLerner/viquae_wikipedia', data_files=data_files)

# Gán nhãn is_human
kb['humans_with_faces'] = kb['humans_with_faces'].map(lambda item: {'is_human': True})
kb['humans_without_faces'] = kb['humans_without_faces'].map(lambda item: {'is_human': True})
kb['non_humans'] = kb['non_humans'].map(lambda item: {'is_human': False})

# Kết hợp lại
kb_recat = concatenate_datasets([kb['non_humans'], kb['humans_with_faces'], kb['humans_without_faces']])

# Lưu ra file
kb_recat.save_to_disk('viquae_wikipedia_recat')
