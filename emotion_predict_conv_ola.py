import os
import argparse
import json
from tqdm import tqdm

import random

# from LLM_needs.Ola_main import infer

os.environ['LOWRES_RESIZE'] = '384x32'
os.environ['HIGHRES_BASE'] = '0x32'
os.environ['VIDEO_RESIZE'] = "0x64"
os.environ['VIDEO_MAXRES'] = "480"
os.environ['VIDEO_MINRES'] = "288"
os.environ['MAXRES'] = '1536'
os.environ['MINRES'] = '0'
os.environ['FORCE_NO_DOWNSAMPLE'] = '1'
os.environ['LOAD_VISION_EARLY'] = '1'
os.environ['PAD2STRIDE'] = '1'

import torch
import re
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import transformers
import moviepy.editor as mp
from typing import Dict, Optional, Sequence, List
import librosa
import whisper
from LLM_needs.Ola_main.ola.conversation import conv_templates, SeparatorStyle
from LLM_needs.Ola_main.ola.model.builder import load_pretrained_model
from LLM_needs.Ola_main.ola.datasets.preprocess import tokenizer_image_token, tokenizer_speech_image_token, tokenizer_speech_question_image_token, tokenizer_speech_token
from LLM_needs.Ola_main.ola.mm_utils import KeywordsStoppingCriteria, process_anyres_video, process_anyres_highres_image
from LLM_needs.Ola_main.ola.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX

emotion_projection_map_empathe_mead = {
    "neutral": "neutral", 
    "happy": "happy", 
    "surprised": "surprised", 
    "angry": "angry", 
    "fear": "fear", 
    "sad": "sad", 
    "disgusted": "disgusted", 
    "contempt": "contempt",

    # happy
    "joyful": "happy",
    "prepared": "happy",
    "content": "happy",
    "caring": "happy",
    "trusting": "happy",
    "faithful": "happy",
    "confident": "happy",
    "hopeful": "happy",
    "grateful": "happy",
    "proud": "happy",
    "excited": "happy",
    "anticipating": "happy",

    # sad
    "lonely": "sad",
    "guilty": "sad",
    "anxious": "sad",
    "nostalgic": "sad",
    "embarrassed": "sad",
    "disappointed": "sad",
    "sentimental": "sad",
    "ashamed": "sad",
    "devastated": "sad",

    # surprised
    "impressed": "surprised",

    # angry
    "furious": "angry",
    "annoyed": "angry",

    # fear
    "afraid": "fear",
    "terrified": "fear",
    "apprehensive": "fear",

    # disgusted
    "embarrassed": "disgusted",

    # contempt
    "jealous": "contempt",
}

def load_audio(audio_file_name):
    speech_wav, samplerate = librosa.load(audio_file_name, sr=16000)
    if len(speech_wav.shape) > 1:
        speech_wav = speech_wav[:, 0]
    speech_wav = speech_wav.astype(np.float32)
    CHUNK_LIM = 480000
    SAMPLE_RATE = 16000
    speechs = []
    speech_wavs = []

    if len(speech_wav) <= CHUNK_LIM:
        speech = whisper.pad_or_trim(speech_wav)
        speech_wav = whisper.pad_or_trim(speech_wav)
        speechs.append(speech)
        speech_wavs.append(torch.from_numpy(speech_wav).unsqueeze(0))
    else:
        for i in range(0, len(speech_wav), CHUNK_LIM):
            chunk = speech_wav[i : i + CHUNK_LIM]
            if len(chunk) < CHUNK_LIM:
                chunk = whisper.pad_or_trim(chunk)
            speechs.append(chunk)
            speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
    mels = []
    for chunk in speechs:
        chunk = whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0)
        mels.append(chunk)

    mels = torch.cat(mels, dim=0)
    speech_wavs = torch.cat(speech_wavs, dim=0)
    if mels.shape[0] > 25:
        mels = mels[:25]
        speech_wavs = speech_wavs[:25]

    speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
    speech_chunks = torch.LongTensor([mels.shape[0]])
    return mels, speech_length, speech_chunks, speech_wavs

def extract_audio(videos_file_path):
    my_clip = mp.VideoFileClip(videos_file_path)
    return my_clip.audio

# https://github.com/Ola-Omni/Ola/tree/main
# cd LLM_needs/Ola_main && pip install -e ".[train]"
# pip install flash-attn==2.5.2 --no-build-isolation 
# pip install moviepy==1.0.3 numpy==1.26.4

# os.environ['CUDA_VISIBLE_DEVICES']="3"
video_data_path = 'datasets/AvaMERG/video_v5_0'
audio_data_path = 'datasets/AvaMERG/audio_v5_0'
INPUT_JSON_list = 'datasets/AvaMERG/train.json' #直接使用
OUTPUT_JSON_path="infer_outputs_avamerg/train_predict_emo_ola_omni_7b.json"
# INPUT_JSON_list = 'datasets/AvaMERG/test/test.json' #直接使用
# OUTPUT_JSON_path="infer_outputs_avamerg/test_emo.json"

# 'THUdyh/Ola-7b'
model_path = '/presearch_lin/OmniMultimodal/Ola-7b'
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None)
model = model.to('cuda').eval()
model = model.bfloat16()

modality = "video"
USE_SPEECH=True #False
# input audio and video, do not parse audio in the video, else parse audio in the video
if modality == "audio":
    USE_SPEECH = True
elif modality == "video":
    USE_SPEECH = True
else:
    USE_SPEECH = False
cur_dir = os.path.dirname(os.path.abspath(__file__))

# 读取 JSON 数据
with open(INPUT_JSON_list, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历所有对话
sample_num=0
HIT_num=0
output_json_list = []
len_data=len(data)
for convo in tqdm(data):
        # for turn in convo.get("turns", []):
        try:
            turns= convo.get("turns", [])[-1]
        except:
            continue

        # print(turns)
        sample_num+=1
        ground_truth_emotion = turns['chain_of_empathy']['speaker_emotion']
        
        all_text=f'Dialogue context: {turns["context"]}\n'
        for i, turn in enumerate(turns.get("dialogue_history", [])):
            if turn.get("role") == "speaker":
                text = 'Speaker: ' + turn['utterance'] + '\n'
                all_text += text
            elif turn.get("role") == "listener":
                text = 'Listener: ' + turn['utterance'] + '\n'
                all_text += text

        # emotion predict
        hit = False
        for n in range(1): # 50 # few-shot / zero-shot
            
            few_shot = 0 # n-shot, randomly sampled from a complete dialogue from the training set
            if few_shot:
                few_shot_example = 'Here show the examples for the task:'
                for n_tmp in range(few_shot):
                    random_num=random.sample(range(0,len_data), 5)[n_tmp]

                    turns_tmp=data[random_num].get("turns", [])[-1]
                    all_text_tmp=f'Dialogue context: {turns_tmp["context"]}\n'
                    for i_tmp, turn_tmp in enumerate(turns_tmp.get("dialogue_history", [])):
                        if turn_tmp.get("role") == "speaker":
                            text_tmp = 'Speaker: ' + turn['utterance'] + '\n'
                            all_text_tmp += text_tmp
                        elif turn_tmp.get("role") == "listener":
                            text_tmp = 'Listener: ' + turn['utterance'] + '\n'
                            all_text_tmp += text_tmp
                
                    ground_truth_tmp = turns_tmp['chain_of_empathy']['speaker_emotion']
                    few_shot_example+=f'{all_text} The emotion class of Speaker: ' + ground_truth_tmp + '\n'

            prompt = f"Please act as an expert in the field of emotions. Please choose one most likely emotion from the given candidates: {emotion_projection_map_empathe_mead.keys()} for the speaker in the given dialogue. Respond with only one word for the chosen emotion. Do not include any other text. The dialogue is: {all_text} The emotion class of Speaker: "  

            if few_shot:
                prompt=f"Please act as an expert in the field of emotions. Please choose one most likely emotion from the given candidates: {emotion_projection_map_empathe_mead.keys()} for the speaker in the given dialogue. {few_shot_example} Respond with only one word for the chosen emotion. Do not include any other text. The dialogue is: {all_text} The emotion class of Speaker: "
                # prompt=few_shot_example+prompt
            
            question=prompt

            dia_id = convo['conversation_id']
            spk_lst_profile_ID = {'speaker_profile_ID': convo['speaker_profile']['ID'], 'listener_profile_ID': convo['listener_profile']['ID']}

            video_paths, audio_paths = [],[]
            max_utt_id = len(turns['dialogue_history'])-1
            role=turns['dialogue_history'][max_utt_id]['role']
            video_path = video_data_path + f'/dia{dia_id}utt{max_utt_id}_{spk_lst_profile_ID[f"{role}_profile_ID"]}.mp4'
            video_paths.append(video_path)
            audio_path = audio_data_path + f'/dia{dia_id}utt{max_utt_id}_{spk_lst_profile_ID[f"{role}_profile_ID"]}.wav'
            audio_paths.append(audio_path)

            vr = VideoReader(video_paths[-1], ctx=cpu(0))
            total_frame_num = len(vr)
            fps = round(vr.get_avg_fps())
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, 64, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            spare_frames = vr.get_batch(frame_idx).asnumpy()
            video = [Image.fromarray(frame) for frame in spare_frames]

            speechs = []
            speech_lengths = []
            speech_wavs = []
            speech_chunks = []
            speech, speech_length, speech_chunk, speech_wav = load_audio(audio_paths[-1])
            speechs.append(speech.bfloat16().to('cuda'))
            speech_lengths.append(speech_length.to('cuda'))
            speech_chunks.append(speech_chunk.to('cuda'))
            speech_wavs.append(speech_wav.to('cuda'))

            conv_mode = "qwen_1_5"
            qs = DEFAULT_SPEECH_TOKEN + DEFAULT_IMAGE_TOKEN + "\n" + question  # video + audio + text
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_speech_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')  # video + audio + text

            video_processed = []
            for idx, frame in enumerate(video):
                image_processor.do_resize = False
                image_processor.do_center_crop = False
                frame = process_anyres_video(frame, image_processor)

                if frame_idx is not None and idx in frame_idx:
                    video_processed.append(frame.unsqueeze(0))
                elif frame_idx is None:
                    video_processed.append(frame.unsqueeze(0))
            
            if frame_idx is None:
                frame_idx = np.arange(0, len(video_processed), dtype=int).tolist()
            
            video_processed = torch.cat(video_processed, dim=0).bfloat16().to("cuda")
            video_processed = (video_processed, video_processed)

            video_data = (video_processed, (384, 384), "video")

            pad_token_ids = 151643
            attention_masks = input_ids.ne(pad_token_ids).long().to('cuda')
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            gen_kwargs = {}
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.2
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs=input_ids,
                    images=video_data[0][0],
                    images_highres=video_data[0][1],
                    modalities=video_data[2],
                    speech=speechs,
                    speech_lengths=speech_lengths,
                    speech_chunks=speech_chunks,
                    speech_wav=speech_wavs,
                    attention_mask=attention_masks,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                outputs = outputs.strip()

                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()
            
            response=outputs
            print(response)
            print('\n')

            # vicuna which could not follow instruciton well
            for emo in emotion_projection_map_empathe_mead.keys():
                if emo in response:
                    response = emo

            if response in emotion_projection_map_empathe_mead.keys():
                hit=True
                break
            
        if hit == False:
            emotion = "neutral"
        else:
            emotion = response

        # compute HIT rate score
        if emotion_projection_map_empathe_mead[emotion] == emotion_projection_map_empathe_mead[ground_truth_emotion]:
            HIT_num += 1

        convo["emotion"] = emotion  # 添加情感字段

with open(OUTPUT_JSON_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f'Emotion HIT Rate: {HIT_num/sample_num}')

print(f"Save prediction in {OUTPUT_JSON_path}")
