import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
import json

emotion_projection = {
    "happy":0,
    "surprised":1,
    "angry":2,
    "fear":3,
    "sad":4,
    "disgusted":5,
    "contempt":6
}

os.environ['CUDA_VISIBLE_DEVICES']="3"
INPUT_JSON_list = 'datasets/AvaMERG/test/test.json' #直接使用
OUTPUT_DIR = 'infer_outputs_avamerg/Openvoice_v2_outputs'
AUDIO_DIR = 'datasets/AvaMERG/test/test_audio'
DEFAULT_REF_AUDIO = 'datasets/AvaMERG/test/test_audio/dia00001utt1_16.wav' # 若遍历过程中出现speaker null则使用"dia00001utt1_16.wav"作为reference_speaker
ckpt_converter = 'OpenVoice-main/checkpoints_v2/converter'

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=DEVICE)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取 JSON 数据
with open(INPUT_JSON_list, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历所有对话
for conv in data:
    conv_id = conv['conversation_id']

    for turn in conv['turns']:
        dialogue_history = turn.get('dialogue_history', [])
        for utt in dialogue_history:
            if utt['role'] != 'listener':
                continue

            index = utt['index']
            text = utt['utterance']
            speaker_index = None

            # 查找speaker utterance用于定位reference_speaker
            for ref_utt in dialogue_history:
                if ref_utt['role'] == 'speaker':
                    speaker_index = ref_utt['index']
                    break

            ref_audio_path = None
            if speaker_index is not None:
                # 查找音频路径
                for fname in os.listdir(AUDIO_DIR):
                    if fname.startswith(f'dia{conv_id}utt_{speaker_index}') and fname.endswith('.wav'):
                        ref_audio_path = os.path.join(AUDIO_DIR, fname)
                        break

            if not ref_audio_path:
                ref_audio_path = DEFAULT_REF_AUDIO

            try:
                target_se, _ = se_extractor.get_se(ref_audio_path, tone_color_converter, vad=True)
            except Exception as e:
                print(f"[错误] 加载参考音频失败: {ref_audio_path}，使用默认音频: {e}")
                ref_audio_path = DEFAULT_REF_AUDIO
                target_se, _ = se_extractor.get_se(ref_audio_path, tone_color_converter, vad=True)

            # 使用TTS生成音频
            model = TTS(language='EN_NEWEST', device=DEVICE)
            speaker_ids = model.hps.data.spk2id
            speaker_id = list(speaker_ids.values())[0]  # 默认取第一个

            tmp_wav = os.path.join(OUTPUT_DIR, 'tmp.wav')
            model.tts_to_file(text, speaker_id, tmp_wav, speed=0.8)

            # 构造输出音频名：dia<conv_id>utt<index>_<speaker_id>.wav
            spk_id = ref_audio_path.split('_')[-1].replace('.wav', '')
            out_wav = f'dia{conv_id}utt{index}_{spk_id}.wav'
            out_path = os.path.join(OUTPUT_DIR, out_wav)

            tone_color_converter.convert(
                audio_src_path=tmp_wav,
                src_se=None,
                tgt_se=target_se,
                output_path=out_path,
                message='@MyShell'
            )

print("音频生成完成")