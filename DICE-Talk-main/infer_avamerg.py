import os
import argparse
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

#os.environ['CUDA_VISIBLE_DEVICES']="2"

from dice_talk import DICE_Talk
pipe = DICE_Talk(0)

emotion_projection_map_empathe_mead = {
    "neutral": "neutral", 
    "happy": "happy", 
    "surprised": "surprised", 
    "angry": "angry", 
    "fear": "fear", 
    "sad": "sad", 
    "disgusted": "disgusted", 
    "contempt": "contempt",

    "afraid": "fear",
    "annoyed": "angry",
    "joyful": "happy",

    "anticipating": "happy",
    "anxious": "fear",
    "apprehensive": "fear",
    "ashamed": "sad",
    "caring": "happy",
    "confident": "happy",
    "content": "happy",
    "devastated": "surprised",
    "disappointed": "contempt",
    "embarrassed": "disgusted",
    "excited": "happy",
    "faithful": "happy",
    "furious": "angry",
    "grateful": "happy",
    "guilty": "sad",
    "hopeful": "happy",
    "impressed": "surprised",
    "jealous": "contempt",
    "lonely": "sad",
    "nostalgic": "neutral",
    "prepared": "neutral",
    "proud": "happy",
    "sentimental": "sad",
    "terrified": "fear",
    "trusting": "happy"
}

# INPUT_JSON_list = 'datasets/AvaMERG/test/test.json' #直接使用
INPUT_JSON_list = 'infer_outputs_avamerg/test_emo.json'
output_dir_path="infer_outputs_avamerg/DICE_Talk_outputs"
image_dir_path="infer_outputs_avamerg/avamerg_test_images"
audio_dir_path="infer_outputs_avamerg/Openvoice_v2_outputs"
emotion_dir_path="DICE-Talk-main/examples/emo"
DEFAULT_REF_IMAGE = 'infer_outputs_avamerg/avamerg_test_images/dia00002utt1_47.jpg'

parser = argparse.ArgumentParser()
parser.add_argument('--ref_scale', type=float, default=3.0)
parser.add_argument('--emo_scale', type=float, default=6.0)
parser.add_argument('--crop', action='store_true')
parser.add_argument('--seed', type=int, default=None)

args = parser.parse_args()

os.makedirs(os.path.dirname(output_dir_path), exist_ok=True)


# 读取 JSON 数据
with open(INPUT_JSON_list, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历所有对话
error_list = []
for conv in tqdm(data[:]): #[15:1000] [1000:2000] [2000:3000] [3000:]
    conv_id = conv['conversation_id']

    listener_profile_ID = conv['listener_profile']['ID'] # listener_profile

    for turn in conv['turns']:
        # dialogue_history = turn.get('dialogue_history', [])
        # for utt in dialogue_history:
        #     if utt['role'] != 'listener':
        #         continue

        #     index = utt['index']
        #     utterance = utt['utterance']

        turn_id = turn['turn_id']
        # text = turn['response']

        fname = f'dia{conv_id}turn{turn_id}_{listener_profile_ID}_gen'
        print(fname)
        out_path = os.path.join(output_dir_path, fname + '.mp4')

        audio_path = os.path.join(audio_dir_path, fname + '.wav')
        
        for root, dirs, files in os.walk(image_dir_path):
            for image_file in files:
                keyword_conv=f'dia{conv_id}'
                keyword_speaker_id=f'_{listener_profile_ID}.jpg'
                if keyword_conv in image_file and keyword_speaker_id in image_file:
                    ref_fname = os.path.join(image_dir_path, image_file)
                    break
                else:
                    ref_fname = DEFAULT_REF_IMAGE
        # image_path = os.path.join(image_dir_path, fname + '.jpg')

        predicted_emotion=emotion_projection_map_empathe_mead[conv['emotion']]
        emotion_path = os.path.join(emotion_dir_path, predicted_emotion+".npy")

        face_info = pipe.preprocess(ref_fname, expand_ratio=0.5)
        print(face_info)
        if face_info['face_num'] >= 0:
            if args.crop:
                crop_image_path = ref_fname + '_crop.jpg'
                pipe.crop_image(ref_fname, crop_image_path, face_info['crop_bbox'])
                ref_fname = crop_image_path

            pipe.process(ref_fname, audio_path, emotion_path, out_path, min_resolution=512, inference_steps=25, ref_scale=args.ref_scale, emo_scale=args.emo_scale, seed=args.seed)
        else:
            error_list.append(fname)

print(f"error_list: {error_list}")
print("Talking Head 视频生成完成")
