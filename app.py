import streamlit as st
import pickle
import torch
import torch.nn as nn
import numpy as np
import re
import string
import itertools
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


class ANFISClassifier(nn.Module):
    def __init__(self, input_dim, mf_counts, num_classes=3, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.input_dim   = input_dim
        self.mf_counts   = mf_counts
        self.num_classes = num_classes

        rules = list(itertools.product(*[range(m) for m in mf_counts]))
        self.rule_indices = torch.tensor(rules, dtype=torch.long)
        self.num_rules    = len(rules)

        self.centers    = nn.ParameterList()
        self.raw_sigmas = nn.ParameterList()

        for m in mf_counts:
            centers = torch.tensor([0.5], dtype=torch.float32) if m == 1 \
                      else torch.linspace(0.15, 0.85, m, dtype=torch.float32)
            sigma_value = 0.35 if m == 2 else 0.25
            sigmas      = torch.ones(m, dtype=torch.float32) * sigma_value
            self.centers.append(nn.Parameter(centers))
            self.raw_sigmas.append(nn.Parameter(torch.log(torch.exp(sigmas) - 1.0)))

        self.rule_logits = nn.Parameter(torch.zeros(self.num_rules, num_classes))

    def gaussian_membership(self, x, center, sigma):
        return torch.exp(-((x - center) ** 2) / (2 * sigma ** 2 + 1e-8))

    def forward(self, X):
        memberships = []
        for j in range(self.input_dim):
            x_j     = X[:, j].unsqueeze(1)
            c_j     = self.centers[j].unsqueeze(0)
            sigma_j = nn.functional.softplus(self.raw_sigmas[j]).unsqueeze(0) + 1e-4
            memberships.append(self.gaussian_membership(x_j, c_j, sigma_j))

        batch_size   = memberships[0].shape[0]
        rule_indices = self.rule_indices.to(X.device)
        firing = torch.ones(batch_size, self.num_rules, device=X.device)
        for j in range(self.input_dim):
            firing *= memberships[j][:, rule_indices[:, j]]

        normalized = firing / (firing.sum(dim=1, keepdim=True) + 1e-8)
        return normalized @ self.rule_logits


MODEL_PATH      = "anfis_model.pt"
COMPONENTS_PATH = "components.pkl"


@st.cache_resource
def load_all():
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = ANFISClassifier(
        input_dim=checkpoint['input_dim'],
        mf_counts=checkpoint['mf_counts'],
        num_classes=3
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    with open(COMPONENTS_PATH, 'rb') as f:
        comp = pickle.load(f)

    return model, comp


kamus_normalisasi = {
    'gw': 'aku', 'gue': 'aku', 'w': 'aku', 'aq': 'aku',
    'lo': 'kamu', 'lu': 'kamu', 'loe': 'kamu', 'elo': 'kamu',
    'org': 'orang', 'orng': 'orang',
    'yg': 'yang', 'krn': 'karena', 'karna': 'karena',
    'dgn': 'dengan', 'dg': 'dengan', 'sm': 'sama',
    'tp': 'tapi', 'tpi': 'tapi', 'ttp': 'tetap', 'ttep': 'tetap',
    'utk': 'untuk', 'u': 'untuk', 'buat': 'untuk',
    'dr': 'dari', 'dr.': 'dari', 'ke': 'ke',
    'jg': 'juga', 'jga': 'juga',
    'sdh': 'sudah', 'udh': 'sudah', 'udah': 'sudah', 'dah': 'sudah',
    'blm': 'belum', 'blum': 'belum',
    'bgt': 'banget', 'bngt': 'banget', 'bgtt': 'banget',
    'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak', 'nggak': 'tidak',
    'ngga': 'tidak', 'kagak': 'tidak', 'tak': 'tidak', 'tdk': 'tidak',
    'enggak': 'tidak', 'ndak': 'tidak', 'gapapa': 'tidak apa-apa',
    'emg': 'memang', 'emang': 'memang', 'mmg': 'memang',
    'kalo': 'kalau', 'klo': 'kalau', 'klw': 'kalau',
    'bisa': 'bisa', 'bs': 'bisa', 'ada': 'ada', 'ad': 'ada',
    'aja': 'saja', 'aj': 'saja', 'jd': 'jadi', 'jdi': 'jadi',
    'mau': 'mau', 'mo': 'mau', 'tau': 'tahu', 'tw': 'tahu',
    'msh': 'masih', 'msih': 'masih', 'nih': 'ini', 'ni': 'ini',
    'tuh': 'itu', 'tu': 'itu', 'kyk': 'seperti', 'kyak': 'seperti', 'kek': 'seperti',
    'bbrp': 'beberapa', 'bbrpa': 'beberapa', 'hrs': 'harus',
    'sy': 'saya', 'sya': 'saya', 'mk': 'maka', 'mkna': 'makna',
    'knp': 'kenapa', 'knapa': 'kenapa',
    'gimana': 'bagaimana', 'gmn': 'bagaimana', 'gmna': 'bagaimana',
    'pdhl': 'padahal', 'pdhal': 'padahal',
    'mgkn': 'mungkin', 'mkin': 'mungkin',
    'sbnrnya': 'sebenarnya', 'sbnrny': 'sebenarnya',
    'beneran': 'benar', 'bener': 'benar', 'bnr': 'benar',
    'bener2': 'benar-benar', 'bnr2': 'benar-benar',
    'ntar': 'nanti', 'tar': 'nanti', 'dpt': 'dapat', 'dpat': 'dapat',
    'nyatanya': 'nyata', 'byk': 'banyak', 'bnyk': 'banyak',
    'sdikit': 'sedikit', 'sdkt': 'sedikit',
    'dgr': 'dengar', 'dngar': 'dengar', 'liat': 'lihat', 'lht': 'lihat',
    'skrg': 'sekarang', 'skrang': 'sekarang', 'dl': 'dulu', 'dlu': 'dulu',
    'lg': 'lagi', 'lgi': 'lagi', 'jln': 'jalan',
    'mntap': 'mantap', 'mantep': 'mantap', 'kren': 'keren', 'prh': 'parah',
    'anjir': 'astaga', 'anjay': 'astaga', 'anjing': 'astaga',
    'wkwk': '', 'wkwkwk': '', 'haha': '', 'hahaha': '',
    'hihi': '', 'hehe': '', 'wkkk': '',
    'lol': '', 'lmao': '', 'omg': 'astaga',
    'ok': 'oke', 'okay': 'oke', 'yuk': 'ayo', 'yukk': 'ayo',
    'dong': '', 'doang': 'saja', 'deh': '', 'sih': '', 'lah': '', 'nah': '',
    'fypシ': '', 'fyp': '', 'xyzbca': '',
    'krl': 'kereta rel listrik', 'blkng': 'belakang',
}

KATA_NEGASI = {'tidak', 'bukan', 'belum', 'jangan', 'tanpa', 'kurang', 'nggak'}


@st.cache_resource
def load_nlp():
    factory_stemmer  = StemmerFactory()
    stemmer          = factory_stemmer.create_stemmer()
    factory_stopword = StopWordRemoverFactory()
    stop_words       = set(factory_stopword.get_stop_words())
    stop_words -= KATA_NEGASI

    custom_sw = {
        'yg', 'dgn', 'nya', 'utk', 'dr', 'tsb', 'jg', 'sy', 'sdh',
        'gw', 'gue', 'lo', 'lu', 'nih', 'tuh', 'deh', 'sih', 'lah',
        'dong', 'aja', 'oke', 'ok', 'wkwk', 'haha', 'hehe', 'lol',
        'fyp', 'fypシ', 'xyzbca', 'share', 'like', 'komen', 'follow',
        'tiktok', 'video', 'konten', 'creator', 'admin',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
        'wkwkwk', 'kwkwk', 'hahaha', 'hihi', 'xixi'
    }
    custom_sw -= KATA_NEGASI

    return stemmer, stop_words.union(custom_sw)


def preprocess(text, stemmer, all_stopwords):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\u200d\u2640-\u2642\u2600-\u2B55\u23cf\u23e9\u231a\ufe0f\u3030"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    words = text.split()
    words = [kamus_normalisasi.get(w, w) for w in words]
    text  = ' '.join([w for w in words if w.strip()])
    text  = re.sub(r'\d+', '', text)
    text  = text.translate(str.maketrans('', '', string.punctuation))
    text  = re.sub(r'[^a-zA-Z\s]', '', text)
    text  = ' '.join(text.split())
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in all_stopwords and len(w) > 1]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)


kata_positif = {
    'bagus', 'baik', 'keren', 'mantap', 'suka', 'senang', 'hebat', 'indah',
    'cantik', 'enak', 'nyaman', 'puas', 'rekomen', 'worth', 'murah',
    'terjangkau', 'cepat', 'ramah', 'sopan', 'memuaskan', 'sempurna',
    'cocok', 'sukses', 'berhasil', 'amazing', 'love', 'best', 'good',
    'great', 'nice', 'happy', 'seru', 'asik', 'asyik', 'bangga', 'kece',
    'top', 'joss', 'mantep', 'original', 'ori', 'aesthetic', 'recommended',
    'luar', 'ganteng', 'oke', 'pas', 'hits', 'bersih', 'aman', 'berkualitas',
    'istimewa', 'memukau', 'menakjubkan', 'terbaik', 'favorit',
    'andalan', 'unggulan', 'premium', 'unik', 'menarik', 'kreatif',
    'helpful', 'informatif', 'bermanfaat', 'makasih', 'thanks',
}

kata_negatif = {
    'jelek', 'buruk', 'parah', 'kecewa', 'gagal', 'salah', 'rusak',
    'benci', 'bohong', 'tipu', 'palsu', 'fake', 'zonk', 'rugi',
    'mahal', 'lambat', 'lama', 'telat', 'mengecewakan', 'nyesel',
    'menyesal', 'komplain', 'keluhan', 'bermasalah', 'masalah', 'error',
    'worst', 'bad', 'terrible', 'awful', 'hate', 'boring', 'spam',
    'penipuan', 'scam', 'nipu', 'nyebelin', 'sebel', 'kesal', 'marah',
    'ribet', 'susah', 'sulit', 'ancur', 'sampah', 'busuk',
    'kotor', 'berbahaya', 'beracun', 'iritasi', 'alergi', 'sakit',
    'mual', 'pusing', 'efek', 'samping', 'mengkhawatirkan', 'kacau',
}

kata_negasi_set = {'tidak', 'bukan', 'belum', 'jangan', 'tanpa', 'kurang', 'nggak'}


def extract_manual_features(text):
    words    = str(text).lower().split()
    word_set = set(words)
    total    = len(words) + 1e-8

    pos_count = sum(1 for w in words if w in kata_positif)
    neg_count = sum(1 for w in words if w in kata_negatif)
    negation  = 1.0 if word_set & kata_negasi_set else 0.0
    valence   = (pos_count - neg_count) / total
    intensity = (pos_count + neg_count) / total
    pos_ratio = pos_count / total
    neg_ratio = neg_count / total
    coverage  = (pos_count + neg_count) / total

    negated_pos = 0
    negated_neg = 0
    for i, w in enumerate(words):
        if w in kata_negasi_set:
            window = words[i+1:i+4]
            if any(ww in kata_positif for ww in window):
                negated_pos = 1
            if any(ww in kata_negatif for ww in window):
                negated_neg = 1

    return [valence, intensity, negation, pos_ratio, neg_ratio, coverage,
            float(negated_pos), float(negated_neg)]


def lexicon_score(preprocessed_text):
    """
    Hitung skor berbasis lexicon langsung dari teks yang sudah dipreprocess.
    Returns: (prior array [Neg, Net, Pos], confidence weight)
    """
    words = str(preprocessed_text).lower().split()
    pos_count = sum(1 for w in words if w in kata_positif)
    neg_count = sum(1 for w in words if w in kata_negatif)
    total_lex = pos_count + neg_count

    negated_pos = any(
        words[i] in kata_negasi_set and
        any(w in kata_positif for w in words[i+1:i+4])
        for i in range(len(words))
    )
    negated_neg = any(
        words[i] in kata_negasi_set and
        any(w in kata_negatif for w in words[i+1:i+4])
        for i in range(len(words))
    )

    if total_lex == 0:
        return np.array([0.33, 0.34, 0.33]), 0.15

    if negated_pos and not negated_neg:
        return np.array([0.70, 0.20, 0.10]), 0.80
    elif negated_neg and not negated_pos:
        return np.array([0.10, 0.20, 0.70]), 0.80
    elif pos_count > neg_count:
        strength = pos_count / (total_lex + 1e-8)
        prior = np.array([0.05, 0.15, 0.80]) * strength + np.array([0.20, 0.45, 0.35]) * (1 - strength)
        return prior / prior.sum(), 0.55 + 0.25 * strength
    elif neg_count > pos_count:
        strength = neg_count / (total_lex + 1e-8)
        prior = np.array([0.80, 0.15, 0.05]) * strength + np.array([0.35, 0.45, 0.20]) * (1 - strength)
        return prior / prior.sum(), 0.55 + 0.25 * strength
    else:
        return np.array([0.30, 0.40, 0.30]), 0.25


def predict(text, model, comp, stemmer, all_stopwords):
    preprocessed = preprocess(text, stemmer, all_stopwords)

    manual_feats  = np.array([extract_manual_features(preprocessed)], dtype='float32')
    manual_scaled = comp['manual_scaler'].transform(manual_feats)

    word_prob = comp['word_clf'].predict_proba(comp['word_tfidf'].transform([preprocessed]))
    char_prob = comp['char_clf'].predict_proba(comp['char_tfidf'].transform([preprocessed]))
    text_prob = (word_prob + char_prob) / 2

    X        = np.hstack([manual_scaled, text_prob]).astype('float32')
    X_tensor = torch.tensor(X)

    with torch.no_grad():
        logits = model(X_tensor)
        anfis_probs = torch.softmax(logits, dim=1).numpy()[0]

    lex_prior, lex_weight = lexicon_score(preprocessed)
    anfis_weight = 1.0 - lex_weight

    final_probs = anfis_weight * anfis_probs + lex_weight * lex_prior
    final_probs = final_probs / final_probs.sum()

    labels = ['Negatif', 'Netral', 'Positif']
    pred_class = int(np.argmax(final_probs))

    return labels[pred_class], final_probs, preprocessed


st.set_page_config(page_title="Analisis Sentimen", page_icon="💬", layout="centered")

st.title("💬 Analisis Sentimen Komentar")
st.caption("Model ANFIS · Negatif / Netral / Positif · Bahasa Indonesia")

model, comp     = load_all()
stemmer, sw_set = load_nlp()

text_input = st.text_area(
    "Masukkan komentar:",
    placeholder="Contoh: produknya bagus banget, ga bikin iritasi sama sekali!",
    height=120
)

if st.button("🔍 Analisis", use_container_width=True) and text_input.strip():
    with st.spinner("Menganalisis..."):
        label, probs, preprocessed = predict(text_input, model, comp, stemmer, sw_set)

    color_map = {"Positif": "🟢", "Netral": "🟡", "Negatif": "🔴"}
    st.markdown(f"### Hasil: {color_map[label]} **{label}**")

    st.divider()
    st.write("**Probabilitas per kelas:**")
    cols = st.columns(3)
    for i, (name, prob) in enumerate(zip(['Negatif', 'Netral', 'Positif'], probs)):
        cols[i].metric(name, f"{prob * 100:.1f}%")

    with st.expander("🔎 Detail preprocessing"):
        st.write("**Teks asli:**", text_input)
        st.write("**Setelah preprocessing:**",
                 preprocessed if preprocessed.strip() else "*(kosong setelah preprocessing)*")
