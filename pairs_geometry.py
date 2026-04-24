"""
pairs_geometry.py — Four-way cross-lingual test of semantic proximity.

Four categories (predicted similarity order high→low):
  1. DEAD METONYMY   — producer/product, container/contents (near-paraphrases)
  2. LIVE METONYMY   — institutional place-names, instrument-for-person, part-for-whole
  3. DEAD METAPHOR   — conventionalised body-part / structural metaphors
  4. LIVE METAPHOR   — vivid cross-domain mappings

Key comparison: dead metaphor vs live metonymy.
  If dead metaphor > live metonymy → conventionalisation dominates.
  If live metonymy > dead metaphor → contiguity signal dominates.

Uses paraphrase-multilingual-MiniLM-L12-v2 (cached locally).

Usage:
    python3 pairs_geometry.py
    python3 pairs_geometry.py --output pairs_plot.png
"""

import argparse
import logging
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category 1 — DEAD METONYMY
# Producer-for-product and container-for-contents: expression nearly
# paraphrases its target, expected highest similarity.
# ---------------------------------------------------------------------------
DEAD_METONYMY = {
    "English": [
        ("read a Hemingway",     "read a novel by Hemingway"),
        ("drink a Bordeaux",     "drink a wine from Bordeaux"),
        ("listen to a Chopin",   "listen to a piece by Chopin"),
        ("drive a Ford",         "drive a Ford car"),
        ("drink a bottle",       "drink alcohol from a bottle"),
        ("empty the glass",      "drink what is in the glass"),
        ("eat a plate",          "eat what is on the plate"),
        ("the kettle is boiling","the water in the kettle is boiling"),
        ("read a Dickens",       "read a novel by Dickens"),
        ("drink a Scotch",       "drink Scottish whisky"),
    ],
    "French": [
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
        ("bir Pamuk oku",         "Orhan Pamuk'un romanını oku"),
        ("bir Bordeaux iç",       "Bordeaux şarabı iç"),
        ("bir Chopin dinle",      "Chopin'in eserini dinle"),
        ("bir Ford sür",          "Ford arabasını sür"),
        ("şişeyi bitir",          "içindeki içkiyi bitir"),
        ("bardağı boşalt",        "bardaktaki içeceği iç"),
        ("tabağı bitir",          "tabaktaki yemeği bitir"),
        ("çaydanlık kaynıyor",    "çaydanlıktaki su kaynıyor"),
        ("bir Nazım oku",         "Nazım Hikmet'in şiirini oku"),
        ("bir viski iç",          "İskoç viskisi iç"),
    ],
    "Russian": [
        ("читать Толстого",       "читать роман Толстого"),
        ("пить Бордо",            "пить вино из Бордо"),
        ("слушать Чайковского",   "слушать произведение Чайковского"),
        ("ехать на Жигулях",      "ехать на автомобиле Жигули"),
        ("выпить бутылку",        "выпить алкоголь из бутылки"),
        ("осушить стакан",        "выпить то, что в стакане"),
        ("съесть тарелку",        "съесть то, что на тарелке"),
        ("чайник кипит",          "вода в чайнике кипит"),
        ("читать Пушкина",        "читать стихи Пушкина"),
        ("выпить шотландского",   "выпить шотландского виски"),
    ],
    "Swedish": [
        ("läsa en Strindberg",    "läsa en roman av Strindberg"),
        ("dricka ett Bordeaux",   "dricka ett vin från Bordeaux"),
        ("lyssna på en Chopin",   "lyssna på ett stycke av Chopin"),
        ("köra en Volvo",         "köra en Volvo-bil"),
        ("dricka en flaska",      "dricka alkohol ur en flaska"),
        ("tömma glaset",          "dricka vad som finns i glaset"),
        ("äta en tallrik",        "äta vad som finns på tallriken"),
        ("kaffepannan kokar",     "vattnet i kaffepannan kokar"),
        ("läsa en ABBA-skiva",    "lyssna på ABBA-musik"),
        ("dricka en whisky",      "dricka skotsk whisky"),
    ],
    "German": [
        ("die Flasche",               "der Alkohol"),
        ("den Kessel",                "das Wasser darin"),
        ("den Topf",                  "das Essen darin"),
        ("das Glas",                  "der Wein darin"),
        ("einen Goethe lesen",        "ein Werk von Goethe lesen"),
        ("einen Mozart hören",        "ein Werk von Mozart hören"),
        ("einen Dürer kaufen",        "ein Gemälde von Dürer kaufen"),
        ("eine Mercedes fahren",      "ein Mercedes-Auto fahren"),
        ("einen Kafka lesen",         "einen Roman von Kafka lesen"),
        ("eine Flasche trinken",      "Alkohol aus einer Flasche trinken"),
    ],
    "Arabic": [
        ("قارورة الخمر",                "المشروب الكحولي"),
        ("إبريق الشاي",                 "الشاي الساخن"),
        ("طنجرة الطبخ",                 "الطعام المطبوخ"),
        ("كأس النبيذ",                  "الخمر المعتق"),
        ("قرأت رواية نجيب محفوظ",       "قرأت أدب نجيب محفوظ"),
        ("سمعت صوت أم كلثوم",           "سمعت موسيقى أم كلثوم"),
        ("درست فلسفة ابن رشد",          "درست فكر ابن رشد"),
        ("حقيبة المال",                 "الثروة المخزونة"),
        ("كل الأيدي على السطح",         "جميع البحارة"),
        ("تجمعت الرؤوس",               "تجمع الناس"),
    ],
    "Japanese": [
        ("瓶のアルコール",               "アルコール飲料"),
        ("やかんのお茶",                 "熱いお茶"),
        ("鍋の料理",                    "鍋で作った食べ物"),
        ("グラスのワイン",               "グラスに入ったワイン"),
        ("夏目漱石を読む",               "夏目漱石の小説を読む"),
        ("黒澤を見る",                   "黒澤の映画を見る"),
        ("葛飾北斎を買う",               "葛飾北斎の絵を買う"),
        ("財布の中身",                   "持っているお金"),
        ("甲板の手",                    "船員たち"),
        ("集まった頭",                   "集まった人々"),
    ],
}

# ---------------------------------------------------------------------------
# Category 2 — LIVE METONYMY
# Institutional place-names, instrument-for-person, part-for-whole:
# contiguity is real but the mapping is less lexically transparent.
# ---------------------------------------------------------------------------
LIVE_METONYMY = {
    "English": [
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
    ],
    "French": [
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
    ],
    "Turkish": [
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
    ],
    "Russian": [
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
    ],
    "Swedish": [
        ("pennan",       "författaren"),
        ("scenen",       "teatern"),
        ("stolen",       "makten"),
        ("kronan",       "monarkin"),
        ("flaskan",      "alkoholen"),
        ("bordet",       "maten"),
        ("mikrofonen",   "sångaren"),
        ("penseln",      "målaren"),
        ("rodret",       "kaptenen"),
        ("talarstolen",  "politikern"),
    ],
    "German": [
        ("die Feder",          "der Schriftsteller"),
        ("der Pinsel",         "der Maler"),
        ("das Mikrofon",       "der Sänger"),
        ("das Skalpell",       "der Chirurg"),
        ("die Robe",           "der Richter"),
        ("das Kanzleramt",     "die Bundesregierung"),
        ("der Bundestag",      "die Abgeordneten"),
        ("die Börse",          "die Finanzmärkte"),
        ("alle Hände an Deck", "alle Matrosen"),
        ("die Köpfe",          "die Menschen"),
    ],
    "Arabic": [
        ("حامل القلم",                  "الكاتب المبدع"),
        ("ممسك الفرشاة",                "الفنان التشكيلي"),
        ("أمام الميكروفون",             "المغني المحترف"),
        ("يحمل المبضع",                 "الجراح الماهر"),
        ("يرتدي الرداء",                "القاضي العادل"),
        ("القصر الجمهوري",              "الحكومة"),
        ("الأزهر الشريف",               "المؤسسة الدينية"),
        ("بنك وول ستريت",               "الأسواق المالية"),
        ("البيت الأبيض الأمريكي",       "إدارة الرئيس"),
        ("قبة البرلمان",                "المشرعون"),
    ],
    "Japanese": [
        ("筆",                   "作家"),
        ("絵筆",                  "画家"),
        ("マイク",                "歌手"),
        ("メス",                  "外科医"),
        ("法衣",                  "裁判官"),
        ("永田町",                "日本の政府"),
        ("霞が関",                "日本の官僚機構"),
        ("ウォール街",            "金融市場"),
        ("ホワイトハウス",         "アメリカ政府"),
        ("ブリュッセル",           "欧州連合"),
    ],
}

# ---------------------------------------------------------------------------
# Category 3 — DEAD METAPHOR
# Conventionalised body-part extensions and structural metaphors:
# cross-domain origin but now near-literal in everyday usage.
# Turkish pairs are body-part extensions of object anatomy.
# ---------------------------------------------------------------------------
DEAD_METAPHOR = {
    "English": [
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
        ("the face of the clock",    "the dial of the clock"),
        ("the shoulder of the road", "the edge of the road"),
        ("the brow of the hill",     "the top of the hill"),
        ("the belly of the ship",    "the lower hold of the ship"),
        ("the teeth of the comb",    "the tines of the comb"),
    ],
    "French": [
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
        ("le cadran de la montre",   "la surface de la montre"),
        ("l'épaulement de la route", "le bord de la route"),
        ("le sommet de la colline",  "le haut de la colline"),
        ("le ventre du navire",      "la cale inférieure du navire"),
        ("les dents du peigne",      "les pointes du peigne"),
    ],
    "Turkish": [
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
        ("kapının kolu",   "kapının tutacağı"),
        ("kapının dili",   "kapının sürgüsü"),
        ("yolun kenarı",   "yolun omzu"),
        ("dağın sırtı",    "dağın üst kısmı"),
        ("vidanın yuvası", "vidanın oturduğu yer"),
    ],
    "Russian": [
        ("подножие горы",        "основание горы"),
        ("ножка стола",          "опора стола"),
        ("устье реки",           "место впадения реки в море"),
        ("сердце города",        "центр города"),
        ("горлышко бутылки",     "узкая верхняя часть бутылки"),
        ("подлокотник кресла",   "боковая опора кресла"),
        ("ушко иглы",            "отверстие в игле"),
        ("язычок ботинка",       "клапан под шнурками"),
        ("обложка книги",        "задняя крышка книги"),
        ("корешок книги",        "край переплёта"),
        ("циферблат часов",      "поверхность часов"),
        ("обочина дороги",       "край дороги"),
        ("вершина холма",        "верхняя часть холма"),
        ("трюм корабля",         "нижний отсек корабля"),
        ("зубья расчёски",       "острые зубцы расчёски"),
    ],
    "Swedish": [
        ("bergets fot",          "bergets bas"),
        ("bordets ben",          "bordets stöd"),
        ("flodens mynning",      "där floden möter havet"),
        ("stadens hjärta",       "stadens centrum"),
        ("flaskans hals",        "den smala övre delen av flaskan"),
        ("fåtöljens armstöd",    "sidostödet på fåtöljen"),
        ("nålens öga",           "hålet i nålen"),
        ("skotunga",             "fliken under snörningen"),
        ("bokens baksida",       "bokens bakre omslag"),
        ("bokens rygg",          "bindningskanten på boken"),
        ("urtavlan",             "urtavlans yta"),
        ("vägens axel",          "vägens kant"),
        ("kullens krön",         "kullens topp"),
        ("skeppets buk",         "skeppets nedre lastrum"),
        ("kammens tänder",       "kammens spetsar"),
    ],
    "German": [
        ("das Herz",              "der Mut"),
        ("die Zähne",             "die Entschlossenheit"),
        ("die Hand",              "die Macht"),
        ("der Rücken",            "die Unterstützung"),
        ("der Angriff",           "die Kritik"),
        ("die Verteidigung",      "das Gegenargument"),
        ("die Waffe",             "das Argument"),
        ("der Sieg",              "die Überzeugung"),
        ("der Weg",               "das Leben"),
        ("die Sackgasse",         "das Problem"),
    ],
    "Arabic": [
        ("البركان",                     "الغضب الشديد"),
        ("الصحراء",                     "الوحدة القاتلة"),
        ("النهر",                       "مرور الوقت"),
        ("الجبل",                       "الصعوبة الكبرى"),
        ("العاصفة الثلجية",             "الأزمة المفاجئة"),
        ("السيف الحاد",                 "الكلمة الجارحة"),
        ("الدرع الواقي",                "الحجة المنطقية"),
        ("الحصن المنيع",                "الموقف الراسخ"),
        ("السهم المسموم",               "النقد اللاذع"),
        ("ساحة المعركة",                "ميدان النقاش"),
    ],
    "Japanese": [
        ("嵐",                    "怒り"),
        ("火山",                  "情熱"),
        ("砂漠",                  "孤独"),
        ("川",                    "時間"),
        ("山",                    "困難"),
        ("刀",                    "言葉"),
        ("盾",                    "言い訳"),
        ("城",                    "立場"),
        ("矢",                    "批判"),
        ("戦場",                  "議論"),
    ],
}

# ---------------------------------------------------------------------------
# Category 4 — LIVE METAPHOR
# Vivid cross-domain mappings, predicted lowest similarity.
# ---------------------------------------------------------------------------
LIVE_METAPHOR = {
    "English": [
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
    ],
    "French": [
        ("un coup de foudre",                  "tomber amoureux"),
        ("être dans le brouillard",            "être confus"),
        ("une tempête sous un crâne",          "une grande agitation intérieure"),
        ("avoir les dents longues",            "être très ambitieux"),
        ("c'est du gâteau",                    "c'est très facile"),
        ("raconter des salades",               "dire des mensonges"),
        ("poser un lapin",                     "ne pas se présenter à un rendez-vous"),
        ("avoir le cafard",                    "être déprimé"),
        ("quand les poules auront des dents",  "jamais"),
        ("être dans une impasse",              "ne pas trouver de solution"),
        ("faire fausse route",                 "se tromper"),
        ("tourner en rond",                    "ne pas progresser"),
        ("jeter de la lumière sur",            "clarifier quelque chose"),
        ("être dans le noir",                  "ne rien savoir"),
        ("garder quelqu'un dans l'obscurité",  "cacher des informations"),
    ],
    "Turkish": [
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
    ],
    "Russian": [
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
    ],
    "Swedish": [
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
    ],
    "German": [
        ("ein Blitz",      "die Liebe auf den ersten Blick"),
        ("der Sturm",      "die innere Aufruhr"),
        ("das Feuer",      "die Leidenschaft"),
        ("das Eis",        "die Kälte des Herzens"),
        ("der Nebel",      "die Verwirrung"),
        ("das Licht",      "die Hoffnung"),
        ("das Gewicht",    "die Trauer"),
        ("der Samen",      "die Idee"),
        ("das Wasser",     "die Zeit"),
        ("die Dunkelheit", "die Unwissenheit"),
    ],
    "Arabic": [
        ("الثعلب",                      "المكر والخداع"),
        ("الأسد",                       "الشجاعة النادرة"),
        ("الحية",                       "الخيانة الغادرة"),
        ("النسر",                       "الحدة والذكاء"),
        ("الحمار",                      "العناد الشديد"),
        ("الثقل الكبير",                "المسؤولية الجسيمة"),
        ("الضوء الساطع",                "الأمل المشرق"),
        ("الظلام الدامس",               "اليأس التام"),
        ("القيود الحديدية",             "القمع والاستبداد"),
        ("الجرح العميق",                "الألم النفسي"),
    ],
    "Japanese": [
        ("狐",                    "狡猾さ"),
        ("亀",                    "忍耐"),
        ("獅子",                  "勇気"),
        ("蛇",                    "裏切り"),
        ("鷹",                    "鋭敏さ"),
        ("重さ",                  "責任"),
        ("光",                    "希望"),
        ("影",                    "疑い"),
        ("鎖",                    "束縛"),
        ("嵐の前の静けさ",         "危機の前兆"),
    ],
}

CATEGORIES = {
    "Dead metonymy": DEAD_METONYMY,
    "Live metonymy": LIVE_METONYMY,
    "Dead metaphor": DEAD_METAPHOR,
    "Live metaphor": LIVE_METAPHOR,
}

CAT_COLORS = {
    "Dead metonymy": "#1a5fa8",
    "Live metonymy": "#6aaee0",
    "Dead metaphor": "#c94a1a",
    "Live metaphor": "#f0a080",
}

LANGUAGES = list(DEAD_METONYMY.keys())  # English, French, Turkish, Russian, Swedish, German


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def run(output_path: str, model_name: str) -> None:
    log.info("Loading model %s ...", model_name)
    model = SentenceTransformer(model_name)

    # Collect all unique strings across all categories and languages
    all_strings = list({s for cat_data in CATEGORIES.values()
                        for lang_pairs in cat_data.values()
                        for pair in lang_pairs for s in pair})
    log.info("Encoding %d unique strings across %d languages ...",
             len(all_strings), len(LANGUAGES))
    raw_vecs = model.encode(all_strings, batch_size=64, show_progress_bar=True,
                            convert_to_numpy=True)
    vecs = normalize(raw_vecs)
    s2v = {s: vecs[i] for i, s in enumerate(all_strings)}

    # Compute similarities: cat_name → lang → [floats]
    cat_lang_sims = {cat: {} for cat in CATEGORIES}
    for cat, cat_data in CATEGORIES.items():
        for lang in LANGUAGES:
            cat_lang_sims[cat][lang] = [
                cosine_sim(s2v[a], s2v[b]) for a, b in cat_data[lang]
            ]

    # Combined across languages per category
    cat_all_sims = {
        cat: [s for lang in LANGUAGES for s in cat_lang_sims[cat][lang]]
        for cat in CATEGORIES
    }

    # --- Per-pair table ---
    for lang in LANGUAGES:
        print(f"\n{'='*72}")
        print(f"  {lang}")
        print(f"{'='*72}")
        for cat in CATEGORIES:
            print(f"\n  [{cat}]")
            for (a, b), s in zip(CATEGORIES[cat][lang], cat_lang_sims[cat][lang]):
                print(f"    {a[:35]:<36} {b[:35]:<36} {s:>6.4f}")

    # --- Summary table ---
    print(f"\n{'='*72}")
    print("  Category summary (combined across all languages)")
    print(f"  {'Category':<18} {'Mean':>6} {'SD':>6} {'Min':>6} {'Max':>6} {'N':>4}")
    print(f"  {'-'*50}")
    for cat in CATEGORIES:
        arr = np.array(cat_all_sims[cat])
        print(f"  {cat:<18} {arr.mean():>6.4f} {arr.std():>6.4f} "
              f"{arr.min():>6.4f} {arr.max():>6.4f} {len(arr):>4}")

    # --- All pairwise Mann-Whitney tests ---
    cat_names = list(CATEGORIES.keys())
    print(f"\n{'='*72}")
    print("  Pairwise Mann-Whitney U (combined across languages)")
    print(f"  {'Comparison':<40} {'U':>6} {'p':>8} {'sig':>5}")
    print(f"  {'-'*62}")
    for ca, cb in combinations(cat_names, 2):
        a_arr = np.array(cat_all_sims[ca])
        b_arr = np.array(cat_all_sims[cb])
        u, p = mannwhitneyu(a_arr, b_arr, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        label = f"{ca}  vs  {cb}"
        print(f"  {label:<40} {u:>6.0f} {p:>8.4f} {sig:>5}")

    # Key comparison
    dm_arr  = np.array(cat_all_sims["Dead metaphor"])
    lm_arr  = np.array(cat_all_sims["Live metonymy"])
    print(f"\n  KEY: dead metaphor mean={dm_arr.mean():.4f}  "
          f"live metonymy mean={lm_arr.mean():.4f}")
    if dm_arr.mean() > lm_arr.mean():
        print("  → Conventionalisation dominates: dead metaphor > live metonymy")
    else:
        print("  → Contiguity dominates: live metonymy > dead metaphor")

    # --- Per-language table ---
    print(f"\n{'='*72}")
    print("  Per-language means")
    header = f"  {'Language':<12}" + "".join(f"  {c[:14]:>14}" for c in cat_names)
    print(header)
    print(f"  {'-'*70}")
    for lang in LANGUAGES:
        row = f"  {lang:<12}"
        for cat in cat_names:
            row += f"  {np.mean(cat_lang_sims[cat][lang]):>14.4f}"
        print(row)
    combined_row = f"  {'COMBINED':<12}"
    for cat in cat_names:
        combined_row += f"  {np.mean(cat_all_sims[cat]):>14.4f}"
    print(combined_row)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: grouped bar chart — 4 bars per language + combined
    ax = axes[0]
    lang_labels = LANGUAGES + ["COMBINED"]
    n_cats = len(cat_names)
    n_groups = len(lang_labels)
    w = 0.18
    offsets = np.linspace(-(n_cats - 1) * w / 2, (n_cats - 1) * w / 2, n_cats)
    x = np.arange(n_groups)

    for ci, cat in enumerate(cat_names):
        means = [np.mean(cat_lang_sims[cat][l]) for l in LANGUAGES] + \
                [np.mean(cat_all_sims[cat])]
        sds   = [np.std(cat_lang_sims[cat][l])  for l in LANGUAGES] + \
                [np.std(cat_all_sims[cat])]
        ax.bar(x + offsets[ci], means, w, yerr=sds, capsize=3,
               color=CAT_COLORS[cat], alpha=0.85, label=cat)

    ax.set_xticks(x)
    ax.set_xticklabels(lang_labels, fontsize=9)
    ax.set_ylabel("Mean cosine similarity (± sd)", fontsize=11)
    ax.set_title("Four-way similarity by language\n(left→right: dead met., live met., dead meta., live meta.)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1] + 0.08))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Right: box/violin plot of combined distributions
    ax = axes[1]
    data   = [np.array(cat_all_sims[cat]) for cat in cat_names]
    colors = [CAT_COLORS[cat] for cat in cat_names]
    parts  = ax.violinplot(data, positions=range(n_cats), showmedians=True,
                           showextrema=True)
    for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cbars"].set_color("black")
    parts["cmaxes"].set_color("black")
    parts["cmins"].set_color("black")

    # Overlay individual points
    for i, (arr, col) in enumerate(zip(data, colors)):
        ax.scatter(np.random.normal(i, 0.04, size=len(arr)), arr,
                   color=col, alpha=0.35, s=12, zorder=3)

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels([c.replace(" ", "\n") for c in cat_names], fontsize=9)
    ax.set_ylabel("Cosine similarity", fontsize=11)
    ax.set_title("Distribution across all languages\n(combined)", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Annotate means
    for i, arr in enumerate(data):
        ax.text(i, arr.mean() + 0.03, f"{arr.mean():.3f}",
                ha="center", fontsize=8, color="black", fontweight="bold")

    fig.suptitle(
        "Semantic proximity: dead metonymy > live metonymy > dead metaphor > live metaphor?\n"
        "Model: paraphrase-multilingual-MiniLM-L12-v2  |  8 languages",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    log.info("Plot saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Four-way cross-lingual cosine similarity test"
    )
    parser.add_argument("--output", default="pairs_plot.png")
    parser.add_argument("--model",
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()
    run(args.output, args.model)


if __name__ == "__main__":
    main()
