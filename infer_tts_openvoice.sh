echo "Starting tts generating......"

#### OpenVoice v2.0
# ~/miniconda3/envs/openvoice/lib/python3.9/site-packages/g2p_en/g2p.py
# mkdir /root/nltk_data
mkdir -p /usr/share/nltk_data/taggers
cp /presearch_lin/EmpatheticGeneration/MeloTTS-main/averaged_perceptron_tagger.zip /usr/share/nltk_data/taggers
unzip -d /usr/share/nltk_data/taggers /usr/share/nltk_data/taggers/averaged_perceptron_tagger.zip
mkdir -p /usr/share/nltk_data/corpora
cp /presearch_lin/EmpatheticGeneration/MeloTTS-main/cmudict.zip /usr/share/nltk_data/corpora
unzip -d /usr/share/nltk_data/corpora /usr/share/nltk_data/corpora/cmudict.zip

python /presearch_lin/EmpatheticGeneration/OpenVoice-main/infer_avamerg.py


