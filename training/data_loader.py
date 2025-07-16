import os
import re
import glob
import json
import hashlib
import torch
import torch.nn.functional as F


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()

def clean_text(text):
    return re.sub(r'\s', '', text).lower()


class WerewolfBeliefDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cache_dir):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.json_files = glob.glob(os.path.join(data_dir, '**/*.json'), recursive=True)
        self.player_names = ['player1', 'player2', 'player3', 'player4', 'player5']
        self.player_to_id = {'player1': 0, 'player2': 1, 'player3': 2, 'player4': 3, 'player5': 4}
        self.face_to_id = {'sad': 0, 'anger': 1, 'neutral': 2, 'happy': 3, 'surprise': 4, 'fear': 5, 'disgust': 6, 'other': 7}
        self.tone_to_id = {'sad': 0, 'anger': 1, 'neutral': 2, 'happy': 3, 'surprise': 4, 'fear': 5, 'disgust': 6, 'other': 7}
        self.action_to_id = {
            'point_as_werewolf': 0,
            'point_as_villager': 1,
            'point_as_seer': 2,
            'point_as_troublemaker': 3,
            'point_as_robber': 4,
            'point_as_insomniac': 5,
            'support': 6,
            'oppose': 7
        }
    
    def __len__(self):
        return len(self.json_files)
    
    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        cache_path = os.path.join(self.cache_dir, md5(json_path) + '.pt')

        if os.path.exists(cache_path):
            return torch.load(cache_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)

        subject_ids = []
        action_ids = []
        object_ids = []
        face_ids = []
        tone_ids = []
        labels = []

        for log_item in log_data['messages']:
            guesses = log_item['guesses']
            if guesses is None:
                continue

            face = clean_text(log_item['face'])
            face_id = self.face_to_id.get(face, 7)
            tone = clean_text(log_item['tone'])
            tone_id = self.tone_to_id.get(tone, 7)

            belief_label = [[0] * len(self.player_names) for _ in range(len(self.player_names))]
            for i_idx, subject in enumerate(self.player_names):
                guess_list = []
                for j_idx, object in enumerate(self.player_names):
                    if subject not in guesses or object not in guesses[subject]:
                        continue
                    guess = guesses[subject][object]
                    for role in guess.split('/'):
                        role = clean_text(role)
                        if role == 'werewolf':
                            guess_list.append(j_idx)
                if len(guess_list) > 0:
                    for j_idx in guess_list:
                        belief_label[i_idx][j_idx] = 1 / len(guess_list)
                else:
                    belief_label[i_idx] = [1 / len(self.player_names)] * len(self.player_names)

            for subject, action, object in log_item['sp_actions']:
                subject = clean_text(subject)
                action = clean_text(action)
                object = clean_text(object)
                if subject not in self.player_to_id or \
                    object not in self.player_to_id or \
                    action not in self.action_to_id:
                    continue
                subject_ids.append(self.player_to_id[subject])
                object_ids.append(self.player_to_id[object])
                action_ids.append(self.action_to_id[action])
                face_ids.append(face_id)
                tone_ids.append(tone_id)
                labels.append(belief_label)

        data_item = {
            'subject_ids': torch.tensor(subject_ids, dtype=torch.long),
            'action_ids': torch.tensor(action_ids, dtype=torch.long),
            'object_ids': torch.tensor(object_ids, dtype=torch.long),
            'face_ids': torch.tensor(face_ids, dtype=torch.long),
            'tone_ids': torch.tensor(tone_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

        torch.save(data_item, cache_path)

        return data_item

    @staticmethod
    def collate_fn(batch):
        max_len = max(item['subject_ids'].size(0) for item in batch)

        batch_size = len(batch)
        subject_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        action_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        object_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        face_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        tone_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        num_players = 5

        labels = torch.zeros(
            batch_size, max_len, num_players, num_players,
            dtype=torch.float
        )

        for i, item in enumerate(batch):
            seq_len = item['subject_ids'].size(0)

            subject_ids[i, :seq_len] = item['subject_ids']
            action_ids[i, :seq_len] = item['action_ids']
            object_ids[i, :seq_len] = item['object_ids']
            face_ids[i, :seq_len] = item['face_ids']
            tone_ids[i, :seq_len] = item['tone_ids']

            attention_mask[i, :seq_len] = 1

            if 'labels' in item and item['labels'] is not None:
                labels_len = min(seq_len, item['labels'].size(0))
                if labels_len == 0:
                    continue
                labels[i, :labels_len] = item['labels'][:labels_len]
        
        return {
            'subject_ids': subject_ids,
            'action_ids': action_ids,
            'object_ids': object_ids,
            'face_ids': face_ids,
            'tone_ids': tone_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
