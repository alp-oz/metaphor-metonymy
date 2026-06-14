"""
Carrier sentences for English terms, one per term per category.
Each sentence forces the intended reading of that specific category.

Keys are the exact term strings from english_pairs.py.
No term appears in multiple English categories, so all keys are plain strings.
"""

# ---------------------------------------------------------------------------
# DEAD METONYMY
# term1 = metonymic expression, term2 = literal gloss
# Sentences use the metonymic term in its frozen, everyday sense
# ---------------------------------------------------------------------------

DEAD_METONYMY: dict[str, str] = {
    # term1 carriers
    "read a Hemingway":       "She read a Hemingway on the flight to London.",
    "drink a Bordeaux":       "It is customary to drink a Bordeaux with red meat.",
    "listen to a Chopin":     "Every evening he would listen to a Chopin before bed.",
    "drive a Ford":           "People who drive a Ford often praise its reliability.",
    "drink a bottle":         "It is unwise to drink a bottle in a single sitting.",
    "empty the glass":        "The custom is to empty the glass before the toast ends.",
    "eat a plate":            "Guests were expected to eat a plate before dessert arrived.",
    "the kettle is boiling":  "Put the tea in — the kettle is boiling already.",
    "read a Dickens":         "She would read a Dickens every winter by the fire.",
    "drink a Scotch":         "The tradition was to drink a Scotch after the ceremony.",
    # term2 carriers
    "read a novel by Hemingway":      "She read a novel by Hemingway during her holiday.",
    "drink a wine from Bordeaux":     "We decided to drink a wine from Bordeaux with dinner.",
    "listen to a piece by Chopin":    "He chose to listen to a piece by Chopin that evening.",
    "drive a Ford car":               "They used to drive a Ford car across the country.",
    "drink alcohol from a bottle":    "He would drink alcohol from a bottle hidden in his coat.",
    "drink what is in the glass":     "She leaned over to drink what is in the glass.",
    "eat what is on the plate":       "The rule was simple: eat what is on the plate.",
    "the water in the kettle is boiling": "She warned me that the water in the kettle is boiling.",
    "read a novel by Dickens":        "He had read a novel by Dickens every year since school.",
    "drink Scottish whisky":          "They gathered to drink Scottish whisky after the ceremony.",
}

# ---------------------------------------------------------------------------
# LIVE METONYMY
# term1 = vehicle (metonymic source), term2 = target meaning
# Sentences use the vehicle in its active metonymic sense
# ---------------------------------------------------------------------------

LIVE_METONYMY: dict[str, str] = {
    # term1 carriers
    "White House":   "The White House announced new sanctions against the regime.",
    "the press":     "The press camped outside the courthouse all week.",
    "Wall Street":   "Wall Street reacted nervously to the interest rate decision.",
    "the bottle":    "He had been fighting the bottle for years before quitting.",
    "Pentagon":      "The Pentagon approved a new deployment of troops overseas.",
    "all hands":     "The captain called all hands to the deck in the storm.",
    "the stage":     "She gave everything she had to the stage.",
    "the bench":     "The bench ruled unanimously in favour of the defendant.",
    "Hollywood":     "Hollywood has struggled to represent diverse stories.",
    "the pulpit":    "The pulpit condemned the new legislation from every parish.",
    "Washington":    "Washington has been slow to respond to the crisis.",
    "the crown":     "The crown intervened directly in the constitutional dispute.",
    "the pen":       "The pen is mightier than the sword.",
    "the scalpel":   "The scalpel decided the course of treatment, not the committee.",
    "the mic":       "The mic held the crowd's attention for two straight hours.",
    # term2 carriers
    "US government":      "The US government issued a formal statement on the matter.",
    "journalists":        "Journalists gathered at the briefing for an official response.",
    "financial markets":  "Financial markets fell sharply on the news from overseas.",
    "alcohol":            "Alcohol had taken a serious toll on his health.",
    "US military":        "The US military confirmed the operation had been successful.",
    "sailors":            "The sailors worked through the night to repair the damage.",
    "theatre":            "She devoted her life to theatre and never looked back.",
    "judiciary":          "The judiciary must remain independent of political pressure.",
    "film industry":      "The film industry releases thousands of titles each year.",
    "clergy":             "The clergy voiced strong opposition to the proposed changes.",
    "US administration":  "The US administration signalled a shift in foreign policy.",
    "monarchy":           "The monarchy faces growing scrutiny from the public.",
    "the writer":         "The writer worked in isolation for months on the manuscript.",
    "the surgeon":        "The surgeon made the critical decision in the operating room.",
    "the singer":         "The singer commanded the stage from the first note.",
}

# ---------------------------------------------------------------------------
# DEAD METAPHOR
# term1 = dead metaphorical expression (body-part name for object part)
# Sentences use the expression in its fully lexicalised, non-figurative way
# ---------------------------------------------------------------------------

DEAD_METAPHOR: dict[str, str] = {
    # term1 carriers
    "the foot of the mountain":  "The village sits quietly at the foot of the mountain.",
    "the leg of the table":      "She noticed the leg of the table was uneven.",
    "the mouth of the river":    "The city was built at the mouth of the river centuries ago.",
    "the heart of the city":     "The market has been at the heart of the city for decades.",
    "the neck of the bottle":    "She gripped the neck of the bottle and poured carefully.",
    "the arm of the chair":      "He rested his book on the arm of the chair.",
    "the eye of the needle":     "Threading the eye of the needle took her several attempts.",
    "the tongue of the shoe":    "The tongue of the shoe kept slipping to one side.",
    "the back of the book":      "The index is printed at the back of the book.",
    "the spine of the book":     "The title on the spine of the book had faded with age.",
    "the face of the clock":     "Dust had gathered on the face of the clock over the years.",
    "the shoulder of the road":  "He pulled over onto the shoulder of the road to check the tyre.",
    "the brow of the hill":      "They stopped at the brow of the hill to rest.",
    "the belly of the ship":     "The cargo was stored deep in the belly of the ship.",
    "the teeth of the comb":     "She ran a finger along the teeth of the comb.",
    # term2 carriers
    "the base of the mountain":    "The camp was established at the base of the mountain.",
    "the support of the table":    "The support of the table had cracked under the weight.",
    "where the river meets the sea": "The estuary marks where the river meets the sea.",
    "the centre of the city":      "All major roads lead to the centre of the city.",
    "the narrow top of the bottle": "The narrow top of the bottle makes it easy to pour.",
    "the side rest of the chair":  "She draped her arm over the side rest of the chair.",
    "the hole in the needle":      "The thread would not fit through the hole in the needle.",
    "the flap under the laces":    "The flap under the laces prevents the tongue from bunching.",
    "the rear cover of the book":  "The author's photo appears on the rear cover of the book.",
    "the binding edge of the book": "The binding edge of the book was starting to crack.",
    "the dial of the clock":       "The dial of the clock was painted with Roman numerals.",
    "the edge of the road":        "A cyclist was riding along the edge of the road.",
    "the top of the hill":         "The flag was planted at the top of the hill.",
    "the lower hold of the ship":  "Water was seeping into the lower hold of the ship.",
    "the tines of the comb":       "The tines of the comb were fine enough for styling.",
}

# ---------------------------------------------------------------------------
# LIVE METAPHOR
# term1 = target domain, term2 = source domain
# Sentences make the conceptual mapping salient
# ---------------------------------------------------------------------------

LIVE_METAPHOR: dict[str, str] = {
    # term1 carriers
    "cheeks":    "Her cheeks glowed red like two ripe apples in the cold.",
    "time":      "We cannot afford to waste time — it is too precious to spend carelessly.",
    "argument":  "He attacked every weak point in the argument without mercy.",
    "life":      "She felt her life had taken a long detour from where she began.",
    "mind":      "His mind ran hot and needed time to cool down and reboot.",
    "love":      "Their love had burned brightly but left only ash behind.",
    "anger":     "His anger simmered beneath the surface all through the meeting.",
    "ideas":     "Good ideas need time and care to take root and grow.",
    "society":   "A healthy society needs all its parts functioning together.",
    "words":     "She chose her words carefully, knowing they could cut deep.",
    "heart":     "After the betrayal his heart turned to stone.",
    "hope":      "A faint hope flickered at the end of an otherwise dark year.",
    "grief":     "The grief sat on her chest like a stone she could not lift.",
    "knowledge": "He hungered for knowledge and devoured every book he found.",
    "memory":    "Her memory was full — she could not hold one more name.",
    # term2 carriers
    "apples":   "Her cheeks were two red apples, bright and round with cold.",
    "money":    "Every minute counts — time is money in this business.",
    "war":      "The debate turned into open war, with no quarter given.",
    "journey":  "Life is a journey and some roads lead nowhere fast.",
    "machine":  "The mind is a machine that needs maintenance and occasional rest.",
    "fire":     "Love is a fire that warms you before it burns you.",
    "heat":     "Anger is a heat that spreads through the body before you know it.",
    "seeds":    "Plant your ideas like seeds and let the best ones grow.",
    "organism": "Society is an organism where each part depends on the others.",
    "weapons":  "Words are weapons — once thrown, they cannot be recalled.",
    "stone":    "Years of loss had turned his heart to stone.",
    "light":    "Hope is the light that keeps you moving through the dark.",
    "weight":   "Grief is a weight you carry long after the loss.",
    "food":     "Knowledge is food for the mind — never stop feeding it.",
    "storage":  "Memory is storage with limited space and unreliable retrieval.",
}


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, __file__.replace("/data/carriers/english_carriers.py", ""))
    from data.pairs.english_pairs import (
        DEAD_METONYMY as DM_PAIRS, LIVE_METONYMY as LM_PAIRS,
        DEAD_METAPHOR as DP_PAIRS, LIVE_METAPHOR as LP_PAIRS,
    )

    sections = [
        ("DEAD METONYMY",  DM_PAIRS, DEAD_METONYMY),
        ("LIVE METONYMY",  LM_PAIRS, LIVE_METONYMY),
        ("DEAD METAPHOR",  DP_PAIRS, DEAD_METAPHOR),
        ("LIVE METAPHOR",  LP_PAIRS, LIVE_METAPHOR),
    ]

    missing = []
    for label, pairs, carriers in sections:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        all_terms = [t for pair in pairs for t in pair]
        for term in all_terms:
            sentence = carriers.get(term)
            if sentence is None:
                missing.append((label, term))
                print(f"  MISSING: {term!r}")
            else:
                print(f"  [{term}]\n    {sentence}")

    if missing:
        print(f"\n⚠  {len(missing)} missing carriers:")
        for label, term in missing:
            print(f"  {label}: {term!r}")
    else:
        print(f"\nAll carriers present ({sum(len(p)*2 for _,p,_ in sections)} terms).")
