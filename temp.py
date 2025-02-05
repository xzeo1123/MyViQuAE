from datasets import load_dataset, concatenate_datasets

# Load tập dữ liệu gốc
kb = load_dataset('PaulLerner/viquae_wikipedia')

# Chia nhỏ dữ liệu: lấy ngẫu nhiên 10% của mỗi phần
kb['humans_with_faces'] = kb['humans_with_faces'].train_test_split(test_size=0.1)['test']
kb['humans_without_faces'] = kb['humans_without_faces'].train_test_split(test_size=0.1)['test']
kb['non_humans'] = kb['non_humans'].train_test_split(test_size=0.1)['test']

# Thực hiện các thao tác như cũ
kb['humans_with_faces'] = kb['humans_with_faces'].map(lambda item: {'is_human': True})
kb['humans_without_faces'] = kb['humans_without_faces'].map(lambda item: {'is_human': True})
kb['non_humans'] = kb['non_humans'].map(lambda item: {'is_human': False})

# Kết hợp lại
kb_recat = concatenate_datasets([kb['non_humans'], kb['humans_with_faces'], kb['humans_without_faces']])

# Lưu vào đĩa
kb_recat.save_to_disk('data/viquae_wikipedia_recat_10percent')
