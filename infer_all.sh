echo "Starting tts generating......"

conda deactivate
# pip install vllm==0.8.5 matplotlib==3.7.5 scikit-learn==1.3.2 spacy==3.8.7 weasel==0.4.1 fastapi==0.116.1
# pip install vllm==0.9.2 # MiniCPM4/internlm3
# pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly 
### predict emotion based on dialogue
# python emotion_predict_conv.py 
python emotion_predict_conv_ola.py 
### predict response based on emotion and dialogue
python response_predict_turn_ola.py 

#### OpenVoice v2.0
# ~/miniconda3/envs/openvoice/lib/python3.9/site-packages/g2p_en/g2p.py
# /presearch_lin/EmpatheticGeneration/MeloTTS-main/melo/text/english.py
# mkdir /root/nltk_data
mkdir -p /usr/share/nltk_data/taggers
cp /presearch_lin/EmpatheticGeneration/MeloTTS-main/averaged_perceptron_tagger.zip /usr/share/nltk_data/taggers 
cp /presearch_lin/EmpatheticGeneration/MeloTTS-main/averaged_perceptron_tagger_eng.zip /usr/share/nltk_data/taggers
mkdir -p /usr/share/nltk_data/corpora
cp /presearch_lin/EmpatheticGeneration/MeloTTS-main/cmudict.zip /usr/share/nltk_data/corpora
mkdir -p ~/.cache/torch/hub/
cp -r OpenVoice-main/snakers4_silero-vad_master ~/.cache/torch/hub/

conda activate openvoice
python OpenVoice-main/infer_avamerg.py

#### DICE-Talk
conda activate dicetalk
# pip install --upgrade diffusers[torch]
# # /presearch_lin/EmpatheticGeneration/DICE-Talk-main/config/inference/dice_talk.yaml
# pip3 install --no-cache-dir transformers==4.40.2 accelerate tiktoken einops scipy # for qwen-2.5
ln -s /presearch_lin/EmpatheticGeneration/pretrains/DICE-Talk_checkpoint/ checkpoints
python DICE-Talk-main/extract_image_from_video.py
python3 DICE-Talk-main/infer_avamerg.py


