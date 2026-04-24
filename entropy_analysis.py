"""
entropy_analysis.py — Neighborhood entropy asymmetry for metonymic vs metaphoric pairs.

For each term (source and target) in every pair:
  1. Embed it with paraphrase-multilingual-MiniLM-L12-v2.
  2. Find its 50 nearest neighbors in the top-10k word list for that language.
  3. Compute Shannon entropy of the softmax-normalised similarity distribution
     over those 50 neighbors — high entropy = semantically diffuse/general,
     low entropy = semantically specific.

For each pair:
  asymmetry = |entropy(source) - entropy(target)|

Prediction:
  Metonymic pairs  → HIGH asymmetry (one term specific, other general)
  Metaphoric pairs → LOW asymmetry  (both terms at similar specificity level)

Mann-Whitney U test over asymmetry scores, all languages combined.

Usage:
    python3 entropy_analysis.py
    python3 entropy_analysis.py --output entropy_plot.png --k 50 --vocab-size 10000
"""

import argparse
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from scipy.stats import mannwhitneyu, shapiro
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from wordfreq import top_n_list

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Language codes for wordfreq
LANG_CODES = {
    "English": "en",
    "French":  "fr",
    "Turkish": "tr",
    "Russian": "ru",
    "Swedish": "sv",
}

# ---------------------------------------------------------------------------
# All pairs — imported inline to keep this script self-contained.
# METONYMY = live + dead metonymy combined; METAPHOR = live + dead metaphor.
# The entropy asymmetry prediction applies to the metonymy/metaphor split,
# not the live/dead split.
# ---------------------------------------------------------------------------

PAIRS = {
    # ---- METONYMY (live + dead combined) ------------------------------------
    "metonymy": {
        "English": [
            # live
            ("White House",    "US government"),
            ("the press",      "journalists"),
            ("Wall Street",    "financial markets"),
            ("the bottle",     "alcohol"),
            ("Pentagon",       "US military"),
            ("all hands",      "sailors"),
            ("the stage",      "theatre"),
            ("the bench",      "judiciary"),
            ("Hollywood",      "film industry"),
            ("the pulpit",     "clergy"),
            ("Washington",     "US administration"),
            ("the crown",      "monarchy"),
            ("the pen",        "the writer"),
            ("the scalpel",    "the surgeon"),
            ("the mic",        "the singer"),
            # dead
            ("read a Hemingway",      "read a novel by Hemingway"),
            ("drink a Bordeaux",      "drink a wine from Bordeaux"),
            ("listen to a Chopin",    "listen to a piece by Chopin"),
            ("drive a Ford",          "drive a Ford car"),
            ("drink a bottle",        "drink alcohol from a bottle"),
            ("empty the glass",       "drink what is in the glass"),
            ("eat a plate",           "eat what is on the plate"),
            ("the kettle is boiling", "the water in the kettle is boiling"),
            ("read a Dickens",        "read a novel by Dickens"),
            ("drink a Scotch",        "drink Scottish whisky"),
        ],
        "French": [
            # live
            ("l'Élysée",             "la présidence française"),
            ("Matignon",             "le premier ministre"),
            ("le Quai d'Orsay",      "la diplomatie française"),
            ("Bercy",                "le ministère des finances"),
            ("la place Beauvau",     "le ministère de l'intérieur"),
            ("la scène",             "le monde du spectacle"),
            ("le ring",              "la boxe"),
            ("la plume",             "l'écrivain"),
            ("le pinceau",           "le peintre"),
            ("le bistouri",          "le chirurgien"),
            ("la robe",              "le magistrat"),
            ("le micro",             "le chanteur"),
            ("les têtes couronnées", "les rois et reines"),
            ("les cols blancs",      "les employés de bureau"),
            ("sous les drapeaux",    "dans l'armée"),
            # dead
            ("lire un Zola",          "lire un roman de Zola"),
            ("boire un Bordeaux",     "boire un vin de Bordeaux"),
            ("écouter un Chopin",     "écouter une œuvre de Chopin"),
            ("un Renoir",             "un tableau de Renoir"),
            ("conduire une Renault",  "conduire une voiture Renault"),
            ("boire une bouteille",   "boire de l'alcool"),
            ("vider le verre",        "boire ce qu'il y a dans le verre"),
            ("manger une assiette",   "manger ce qu'il y a dans l'assiette"),
            ("la casserole chante",   "l'eau bout dans la casserole"),
            ("le tonneau est vide",   "le vin est épuisé"),
        ],
        "Turkish": [
            # live
            ("Köşk",        "cumhurbaşkanlığı"),
            ("basın",       "gazeteciler"),
            ("Ankara",      "Türk hükümeti"),
            ("kalem",       "yazar"),
            ("sahne",       "tiyatro"),
            ("cübbe",       "yargı"),
            ("çarşı",       "ticaret"),
            ("minber",      "din adamları"),
            ("mikrofon",    "şarkıcı"),
            ("kürsü",       "politikacı"),
            ("sofra",       "yemek"),
            ("taç",         "monarşi"),
            ("Nazım",       "şiir"),
            ("şişe",        "alkol"),
            ("Orhan Pamuk", "roman"),
            # dead
            ("bir Pamuk oku",      "Orhan Pamuk'un romanını oku"),
            ("bir Bordeaux iç",    "Bordeaux şarabı iç"),
            ("bir Chopin dinle",   "Chopin'in eserini dinle"),
            ("bir Ford sür",       "Ford arabasını sür"),
            ("şişeyi bitir",       "içindeki içkiyi bitir"),
            ("bardağı boşalt",     "bardaktaki içeceği iç"),
            ("tabağı bitir",       "tabaktaki yemeği bitir"),
            ("çaydanlık kaynıyor", "çaydanlıktaki su kaynıyor"),
            ("bir Nazım oku",      "Nazım Hikmet'in şiirini oku"),
            ("bir viski iç",       "İskoç viskisi iç"),
        ],
        "Russian": [
            # live
            ("Кремль",   "российское правительство"),
            ("пресса",   "журналисты"),
            ("Москва",   "российская власть"),
            ("перо",     "писатель"),
            ("сцена",    "театр"),
            ("мантия",   "суд"),
            ("рынок",    "торговля"),
            ("кафедра",  "священник"),
            ("микрофон", "певец"),
            ("трибуна",  "оратор"),
            ("корона",   "монархия"),
            ("бутылка",  "алкоголь"),
            ("Толстой",  "роман"),
            ("стол",     "еда"),
            ("Пушкин",   "стихи"),
            # dead
            ("читать Толстого",    "читать роман Толстого"),
            ("пить Бордо",         "пить вино из Бордо"),
            ("слушать Чайковского","слушать произведение Чайковского"),
            ("ехать на Жигулях",   "ехать на автомобиле Жигули"),
            ("выпить бутылку",     "выпить алкоголь из бутылки"),
            ("осушить стакан",     "выпить то, что в стакане"),
            ("съесть тарелку",     "съесть то, что на тарелке"),
            ("чайник кипит",       "вода в чайнике кипит"),
            ("читать Пушкина",     "читать стихи Пушкина"),
            ("выпить шотландского","выпить шотландского виски"),
        ],
        "Swedish": [
            # live
            ("Rosenbad",     "svenska regeringen"),
            ("pressen",      "journalister"),
            ("Stockholm",    "svenska staten"),
            ("pennan",       "författare"),
            ("scenen",       "teater"),
            ("kappan",       "domstol"),
            ("torget",       "handel"),
            ("predikstolen", "präster"),
            ("mikrofonen",   "sångare"),
            ("talarstolen",  "politiker"),
            ("kronan",       "monarki"),
            ("flaskan",      "alkohol"),
            ("Strindberg",   "roman"),
            ("bordet",       "mat"),
            ("ABBA",         "popmusik"),
            # dead
            ("läsa en Strindberg",  "läsa en roman av Strindberg"),
            ("dricka ett Bordeaux", "dricka ett vin från Bordeaux"),
            ("lyssna på en Chopin", "lyssna på ett stycke av Chopin"),
            ("köra en Volvo",       "köra en Volvo-bil"),
            ("dricka en flaska",    "dricka alkohol ur en flaska"),
            ("tömma glaset",        "dricka vad som finns i glaset"),
            ("äta en tallrik",      "äta vad som finns på tallriken"),
            ("kaffepannan kokar",   "vattnet i kaffepannan kokar"),
            ("läsa en ABBA-skiva",  "lyssna på ABBA-musik"),
            ("dricka en whisky",    "dricka skotsk whisky"),
        ],
    },

    # ---- METAPHOR (live + dead combined) ------------------------------------
    "metaphor": {
        "English": [
            # live
            ("cheeks",    "apples"),
            ("time",      "money"),
            ("argument",  "war"),
            ("life",      "journey"),
            ("mind",      "machine"),
            ("love",      "fire"),
            ("anger",     "heat"),
            ("ideas",     "seeds"),
            ("society",   "organism"),
            ("words",     "weapons"),
            ("heart",     "stone"),
            ("hope",      "light"),
            ("grief",     "weight"),
            ("knowledge", "food"),
            ("memory",    "storage"),
            # dead
            ("the foot of the mountain", "the base of the mountain"),
            ("the leg of the table",     "the support of the table"),
            ("the mouth of the river",   "where the river meets the sea"),
            ("the heart of the city",    "the centre of the city"),
            ("the neck of the bottle",   "the narrow top of the bottle"),
            ("the arm of the chair",     "the side rest of the chair"),
            ("the eye of the needle",    "the hole in the needle"),
            ("the tongue of the shoe",   "the flap under the laces"),
            ("the back of the book",     "the rear cover of the book"),
            ("the spine of the book",    "the binding edge of the book"),
        ],
        "French": [
            # live
            ("un coup de foudre",                 "tomber amoureux"),
            ("être dans le brouillard",           "être confus"),
            ("une tempête sous un crâne",         "une grande agitation intérieure"),
            ("avoir les dents longues",           "être très ambitieux"),
            ("c'est du gâteau",                   "c'est très facile"),
            ("raconter des salades",              "dire des mensonges"),
            ("poser un lapin",                    "ne pas se présenter à un rendez-vous"),
            ("avoir le cafard",                   "être déprimé"),
            ("quand les poules auront des dents", "jamais"),
            ("être dans une impasse",             "ne pas trouver de solution"),
            ("faire fausse route",                "se tromper"),
            ("tourner en rond",                   "ne pas progresser"),
            ("jeter de la lumière sur",           "clarifier quelque chose"),
            ("être dans le noir",                 "ne rien savoir"),
            ("garder quelqu'un dans l'obscurité", "cacher des informations"),
            # dead
            ("le pied de la montagne",   "la base de la montagne"),
            ("le pied de la table",      "le support de la table"),
            ("l'embouchure du fleuve",   "là où le fleuve rejoint la mer"),
            ("le cœur de la ville",      "le centre de la ville"),
            ("le goulot de la bouteille","la partie étroite en haut de la bouteille"),
            ("le bras du fauteuil",      "l'accoudoir du fauteuil"),
            ("le chas de l'aiguille",    "le trou dans l'aiguille"),
            ("la langue de la chaussure","le rabat sous les lacets"),
            ("le dos du livre",          "la couverture arrière du livre"),
            ("la tranche du livre",      "le bord de la reliure"),
        ],
        "Turkish": [
            # live
            ("yanaklar",  "elmalar"),
            ("zaman",     "para"),
            ("tartışma",  "savaş"),
            ("hayat",     "yolculuk"),
            ("zihin",     "makine"),
            ("aşk",       "ateş"),
            ("öfke",      "ısı"),
            ("fikirler",  "tohumlar"),
            ("toplum",    "organizma"),
            ("kelimeler", "silahlar"),
            ("kalp",      "taş"),
            ("umut",      "ışık"),
            ("keder",     "ağırlık"),
            ("bilgi",     "besin"),
            ("bellek",    "depolama"),
            # dead
            ("dağın eteği",    "dağın alt kısmı"),
            ("masanın ayağı",  "masanın desteği"),
            ("bıçağın ağzı",   "bıçağın keskin kısmı"),
            ("nehrin ağzı",    "nehrin açıldığı yer"),
            ("kitabın sırtı",  "kitabın arka kısmı"),
            ("saatin kadranı", "saatin yüzeyi"),
            ("kentin kalbi",   "kentin merkezi"),
            ("geminin karnı",  "geminin alt bölümü"),
            ("tünelin ağzı",   "tünelin girişi"),
            ("taranın dişleri","taranın sivri uçları"),
        ],
        "Russian": [
            # live
            ("щёки",     "яблоки"),
            ("время",    "деньги"),
            ("спор",     "война"),
            ("жизнь",    "путешествие"),
            ("разум",    "машина"),
            ("любовь",   "огонь"),
            ("гнев",     "жара"),
            ("идеи",     "семена"),
            ("общество", "организм"),
            ("слова",    "оружие"),
            ("сердце",   "камень"),
            ("надежда",  "свет"),
            ("горе",     "тяжесть"),
            ("знание",   "пища"),
            ("память",   "хранилище"),
            # dead
            ("подножие горы",      "основание горы"),
            ("ножка стола",        "опора стола"),
            ("устье реки",         "место впадения реки в море"),
            ("сердце города",      "центр города"),
            ("горлышко бутылки",   "узкая верхняя часть бутылки"),
            ("подлокотник кресла", "боковая опора кресла"),
            ("ушко иглы",          "отверстие в игле"),
            ("язычок ботинка",     "клапан под шнурками"),
            ("обложка книги",      "задняя крышка книги"),
            ("корешок книги",      "край переплёта"),
        ],
        "Swedish": [
            # live
            ("kinder",    "äpplen"),
            ("tid",       "pengar"),
            ("argument",  "krig"),
            ("livet",     "resa"),
            ("sinnet",    "maskin"),
            ("kärlek",    "eld"),
            ("ilska",     "värme"),
            ("idéer",     "frön"),
            ("samhället", "organism"),
            ("ord",       "vapen"),
            ("hjärtat",   "sten"),
            ("hoppet",    "ljus"),
            ("sorg",      "tyngd"),
            ("kunskap",   "föda"),
            ("minnet",    "lagring"),
            # dead
            ("bergets fot",      "bergets bas"),
            ("bordets ben",      "bordets stöd"),
            ("flodens mynning",  "där floden möter havet"),
            ("stadens hjärta",   "stadens centrum"),
            ("flaskans hals",    "den smala övre delen av flaskan"),
            ("fåtöljens armstöd","sidostödet på fåtöljen"),
            ("nålens öga",       "hålet i nålen"),
            ("skotunga",         "fliken under snörningen"),
            ("bokens baksida",   "bokens bakre omslag"),
            ("bokens rygg",      "bindningskanten på boken"),
        ],
    },
}

LANGUAGES = list(PAIRS["metonymy"].keys())


def shannon_entropy(probs: np.ndarray) -> float:
    """Shannon entropy in nats. Clip to avoid log(0)."""
    p = np.clip(probs, 1e-12, None)
    return float(-np.sum(p * np.log(p)))


def neighborhood_entropy(term_vec: np.ndarray, vocab_vecs: np.ndarray, k: int) -> float:
    """
    Cosine similarities to all vocab vectors, take top-k,
    softmax-normalise, compute Shannon entropy.
    """
    sims = vocab_vecs @ term_vec          # shape (V,), already unit-normed
    top_k_idx = np.argpartition(sims, -k)[-k:]
    top_k_sims = sims[top_k_idx].astype(np.float64)
    probs = softmax(top_k_sims)
    return shannon_entropy(probs)


def run(output_path: str, model_name: str, k: int, vocab_size: int) -> None:
    log.info("Loading model %s ...", model_name)
    model = SentenceTransformer(model_name)

    # Build per-language reference vocabularies
    log.info("Building reference vocabularies (top-%d words per language) ...", vocab_size)
    lang_vocab_vecs: dict[str, np.ndarray] = {}
    lang_vocab_words: dict[str, list[str]] = {}
    for lang in LANGUAGES:
        code = LANG_CODES[lang]
        words = top_n_list(code, vocab_size)
        raw = model.encode(words, batch_size=256, show_progress_bar=False,
                           convert_to_numpy=True)
        lang_vocab_vecs[lang]  = normalize(raw)
        lang_vocab_words[lang] = words
        log.info("  %s: %d vocab vectors", lang, len(words))

    # Collect all unique query strings (both terms of every pair)
    all_queries = list({s for type_pairs in PAIRS.values()
                        for lang_pairs in type_pairs.values()
                        for pair in lang_pairs for s in pair})
    log.info("Embedding %d unique query strings ...", len(all_queries))
    raw_q = model.encode(all_queries, batch_size=64, show_progress_bar=True,
                         convert_to_numpy=True)
    q_vecs = normalize(raw_q)
    s2v = {s: q_vecs[i] for i, s in enumerate(all_queries)}

    # Compute entropy for every unique string per language
    # (same string can appear in multiple languages; entropy depends on lang vocab)
    log.info("Computing neighborhood entropy (k=%d) ...", k)
    s2ent: dict[tuple[str, str], float] = {}   # (lang, string) → entropy
    for lang in LANGUAGES:
        vv = lang_vocab_vecs[lang]
        unique_strings = {s for type_pairs in PAIRS.values()
                          for pair in type_pairs[lang] for s in pair}
        for s in unique_strings:
            s2ent[(lang, s)] = neighborhood_entropy(s2v[s], vv, k)

    # Compute asymmetry per pair
    results = []   # list of dicts
    for fig_type in ["metonymy", "metaphor"]:
        for lang in LANGUAGES:
            for src, tgt in PAIRS[fig_type][lang]:
                h_src = s2ent[(lang, src)]
                h_tgt = s2ent[(lang, tgt)]
                asym  = abs(h_src - h_tgt)
                results.append({
                    "type":      fig_type,
                    "lang":      lang,
                    "source":    src,
                    "target":    tgt,
                    "h_source":  h_src,
                    "h_target":  h_tgt,
                    "asymmetry": asym,
                })

    # --- Per-pair table ---
    for lang in LANGUAGES:
        print(f"\n{'='*80}")
        print(f"  {lang}")
        print(f"{'='*80}")
        print(f"  {'Type':<9} {'Source':<30} {'H_src':>6} {'Target':<30} {'H_tgt':>6} {'|ΔH|':>6}")
        print(f"  {'-'*78}")
        for r in results:
            if r["lang"] != lang:
                continue
            print(f"  {r['type']:<9} {r['source'][:29]:<30} {r['h_source']:>6.3f} "
                  f"{r['target'][:29]:<30} {r['h_target']:>6.3f} {r['asymmetry']:>6.3f}")

    # --- Summary stats ---
    met_asym  = np.array([r["asymmetry"] for r in results if r["type"] == "metonymy"])
    meta_asym = np.array([r["asymmetry"] for r in results if r["type"] == "metaphor"])

    print(f"\n{'='*80}")
    print("  Asymmetry summary (combined across all languages)")
    print(f"  {'Type':<12} {'Mean':>6} {'SD':>6} {'Median':>7} {'N':>4}")
    print(f"  {'-'*38}")
    print(f"  {'Metonymy':<12} {met_asym.mean():>6.4f} {met_asym.std():>6.4f} "
          f"{np.median(met_asym):>7.4f} {len(met_asym):>4}")
    print(f"  {'Metaphor':<12} {meta_asym.mean():>6.4f} {meta_asym.std():>6.4f} "
          f"{np.median(meta_asym):>7.4f} {len(meta_asym):>4}")

    u, p = mannwhitneyu(met_asym, meta_asym, alternative="two-sided")
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"\n  Mann-Whitney U={u:.0f}, p={p:.4f}  {sig}")
    direction = "higher" if met_asym.mean() > meta_asym.mean() else "lower"
    pred_str = "CONFIRMED" if p < 0.05 and met_asym.mean() > meta_asym.mean() \
               else "NOT CONFIRMED"
    print(f"  Prediction (metonymy asymmetry > metaphor asymmetry): {pred_str}")
    print(f"  Metonymy asymmetry is {direction} "
          f"({met_asym.mean():.4f} vs {meta_asym.mean():.4f})")

    # Per-language breakdown
    print(f"\n  Per-language Mann-Whitney")
    print(f"  {'Language':<12} {'Met asym':>9} {'Meta asym':>10} {'p':>8} {'sig':>5}")
    print(f"  {'-'*48}")
    for lang in LANGUAGES:
        m  = np.array([r["asymmetry"] for r in results
                       if r["type"] == "metonymy"  and r["lang"] == lang])
        me = np.array([r["asymmetry"] for r in results
                       if r["type"] == "metaphor" and r["lang"] == lang])
        if len(m) >= 3 and len(me) >= 3:
            _, p_l = mannwhitneyu(m, me, alternative="two-sided")
            s_l = "***" if p_l < 0.001 else "**" if p_l < 0.01 else \
                  "*" if p_l < 0.05 else "n.s."
        else:
            p_l, s_l = float("nan"), "—"
        print(f"  {lang:<12} {m.mean():>9.4f} {me.mean():>10.4f} {p_l:>8.4f} {s_l:>5}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = {"metonymy": "#3a7ec8", "metaphor": "#e05c3a"}

    # Left: violin plot of asymmetry distributions
    ax = axes[0]
    data   = [met_asym, meta_asym]
    labels = ["Metonymy", "Metaphor"]
    cols   = [colors["metonymy"], colors["metaphor"]]
    parts  = ax.violinplot(data, positions=[0, 1], showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], cols):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    for part in ["cmedians", "cbars", "cmaxes", "cmins"]:
        parts[part].set_color("black")
    for i, (arr, col) in enumerate(zip(data, cols)):
        ax.scatter(np.random.normal(i, 0.04, size=len(arr)), arr,
                   color=col, alpha=0.3, s=14, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Entropy asymmetry |H(source) − H(target)|", fontsize=10)
    ax.set_title(f"Asymmetry distribution (combined)\nMann-Whitney p={p:.4f} {sig}",
                 fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for i, arr in enumerate(data):
        ax.text(i, arr.mean() + 0.01, f"μ={arr.mean():.3f}",
                ha="center", fontsize=9, fontweight="bold")

    # Middle: per-language bar chart
    ax = axes[1]
    x = np.arange(len(LANGUAGES))
    w = 0.35
    met_means  = [np.mean([r["asymmetry"] for r in results
                           if r["type"] == "metonymy"  and r["lang"] == l]) for l in LANGUAGES]
    meta_means = [np.mean([r["asymmetry"] for r in results
                           if r["type"] == "metaphor" and r["lang"] == l]) for l in LANGUAGES]
    met_sds    = [np.std([r["asymmetry"] for r in results
                          if r["type"] == "metonymy"  and r["lang"] == l]) for l in LANGUAGES]
    meta_sds   = [np.std([r["asymmetry"] for r in results
                          if r["type"] == "metaphor" and r["lang"] == l]) for l in LANGUAGES]

    ax.bar(x - w/2, met_means,  w, yerr=met_sds,  capsize=4,
           color=colors["metonymy"],  alpha=0.8, label="Metonymy")
    ax.bar(x + w/2, meta_means, w, yerr=meta_sds, capsize=4,
           color=colors["metaphor"], alpha=0.8, label="Metaphor")
    ax.set_xticks(x)
    ax.set_xticklabels(LANGUAGES, fontsize=9)
    ax.set_ylabel("Mean entropy asymmetry (± sd)", fontsize=10)
    ax.set_title("Per-language asymmetry\n(metonymy vs metaphor)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Right: scatter H(source) vs H(target), coloured by type
    ax = axes[2]
    for fig_type in ["metaphor", "metonymy"]:   # metaphor behind
        col = colors[fig_type]
        xs  = [r["h_source"] for r in results if r["type"] == fig_type]
        ys  = [r["h_target"] for r in results if r["type"] == fig_type]
        ax.scatter(xs, ys, c=col, alpha=0.45, s=22,
                   label=fig_type.capitalize(), zorder=2 if fig_type == "metonymy" else 1)
    # y=x line: zero asymmetry
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
           max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.4, label="y = x (zero asymmetry)")
    ax.set_xlabel("H(source)", fontsize=10)
    ax.set_ylabel("H(target)", fontsize=10)
    ax.set_title("Entropy of source vs target\n(distance from diagonal = asymmetry)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.25)

    fig.suptitle(
        f"Neighborhood entropy asymmetry: metonymy vs metaphor\n"
        f"k={k} neighbors, vocab={vocab_size}, "
        f"model: paraphrase-multilingual-MiniLM-L12-v2, 5 languages",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    log.info("Plot saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Neighborhood entropy asymmetry: metonymy vs metaphor"
    )
    parser.add_argument("--output", default="entropy_plot.png")
    parser.add_argument("--model",
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--k",          type=int, default=50,
                        help="Number of nearest neighbors for entropy (default: 50)")
    parser.add_argument("--vocab-size", type=int, default=10000,
                        help="Reference vocabulary size per language (default: 10000)")
    args = parser.parse_args()
    run(args.output, args.model, args.k, args.vocab_size)


if __name__ == "__main__":
    main()
