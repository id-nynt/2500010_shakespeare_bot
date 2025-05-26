import os
import json
import random
import logging
from pathlib import Path

# === CONFIG ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FACTUAL_FILE = "data/processed/factual/factual.json"
OUTPUT_FILE = "data/prompt_response/manual_02.jsonl"
PLAYS = [
    "Hamlet, Prince of Denmark",
    "Romeo and Juliet",
    "Macbeth",
    "A Midsummer Night's Dream",
    "Much Ado About Nothing",
    "The Tragedy of King Lear",
    "Othello, the Moor of Venice",
    "Twelfth Night; Or, What You Will",
    "The Tempest",
    "Julius Caesar"
]

# === QUESTION TEMPLATES ===
QUESTION_TEMPLATES = {
    "main_conflict": [
        lambda x: f"What’s the central conflict in *{x}*, and can you describe it simply?",
        lambda x: f"Can you explain the main struggle in *{x}* in an easy way?",
        lambda x: f"What is the primary issue in *{x}*, and how would you break it down?"
    ],
    "conflict_impact": [
        lambda x: f"How does the main conflict in *{x}* impact its key characters?",
        lambda x: f"What effects does the central struggle in *{x}* have on the main characters?",
        lambda x: f"How are the primary characters in *{x}* influenced by its main conflict?"
    ],
    "play_significance": [
        lambda x: f"What makes *{x}* one of Shakespeare’s greatest plays?",
        lambda x: f"Why is *{x}* regarded as a top Shakespeare play?",
        lambda x: f"How does *{x}* earn its place among Shakespeare’s best works?"
    ],
    "memorable_scene": [
        lambda x: f"What’s a standout scene from *{x}* that I should know about?",
        lambda x: f"Can you describe a memorable moment from *{x}*?",
        lambda x: f"Which scene in *{x}* is particularly unforgettable?"
    ],
    "protagonist": [
        lambda x: f"Who’s the main character in *{x}*, and what’s their personality?",
        lambda x: f"Can you tell me about the protagonist of *{x}* and their traits?",
        lambda x: f"What is the lead character in *{x}* like, and who are they?"
    ],
    "setting_unique": [
        lambda x: f"What’s special about the setting of *{x}*?",
        lambda x: f"How is the setting of *{x}* different or unique?",
        lambda x: f"Why is the location of *{x}* distinctive?"
    ],
    "tragedy_ending": [
        lambda x: f"Why does *{x}* have such a tragic ending?",
        lambda x: f"What makes the ending of *{x}* so sad, since it’s a tragedy?",
        lambda x: f"How does *{x}* end tragically, and why?"
    ],
    "key_moment": [
        lambda x: f"What’s a pivotal moment in *{x}* that shifts the story?",
        lambda x: f"Can you identify a turning point in *{x}* that alters everything?",
        lambda x: f"What’s a critical event in *{x}* that changes the course of the play?"
    ],
    "reaction_to_moment": [
        lambda x: f"How do the characters in *{x}* respond to this key moment?",
        lambda x: f"What are the reactions of *{x}*’s characters to this turning point?",
        lambda x: f"How does this pivotal event affect the characters in *{x}*?"
    ],
    "character_role": [
        lambda x, c: f"What is *{c}*’s role in *{x}*?",
        lambda x, c: f"Can you explain what *{c}* does in *{x}*?",
        lambda x, c: f"How does *{c}* contribute to *{x}*?"
    ],
    "must_see_scene": [
        lambda x: f"Which scene in *{x}* is a must-see?",
        lambda x: f"Can you recommend a key scene to watch in *{x}*?",
        lambda x: f"What’s an important scene to focus on in *{x}*?"
    ]
}

# === RESPONSE DATA ===
RESPONSE_DATA = {
    "Hamlet, Prince of Denmark": {
        "category": "tragedy",
        "main_conflict": "Hamlet’s struggle to avenge his father’s murder by his uncle Claudius, who now rules Denmark.",
        "conflict_impact": "Hamlet becomes consumed by doubt and feigned madness, Claudius grows paranoid, Gertrude is torn between son and husband, and Ophelia descends into madness and death.",
        "play_significance": "Its deep exploration of human psychology, existential themes, and complex characters make it a timeless masterpiece.",
        "memorable_scene": "In Act 5, Scene 1, Hamlet holds Yorick’s skull, reflecting on mortality in the graveyard, a haunting and philosophical moment.",
        "protagonist": "Hamlet, a thoughtful and conflicted prince who wrestles with revenge, morality, and his own sanity.",
        "setting_unique": "Elsinore Castle’s dark, secretive atmosphere mirrors the play’s themes of betrayal and hidden truths.",
        "tragedy_ending": "Hamlet’s quest for revenge spirals into a chain of deaths, including his own, due to mistrust and inevitable consequences.",
        "key_moment": "In Act 3, Scene 1, Hamlet’s 'To be, or not to be' soliloquy reveals his existential crisis, shaping his hesitant approach to revenge.",
        "reaction_to_moment": "Hamlet grows more introspective, Claudius plots to eliminate him, and Ophelia becomes confused by his erratic behavior.",
        "character_roles": {
            "Hamlet": "The brooding prince seeking revenge for his father’s murder, driving the play’s tragic events.",
            "Claudius": "The usurping king who murdered Hamlet’s father, sparking the central conflict.",
            "Gertrude": "Hamlet’s mother, caught between her son and her new husband, Claudius.",
            "Ophelia": "Hamlet’s love interest, whose descent into madness reflects the collateral damage of the conflict.",
            "Polonius": "The meddling advisor whose schemes contribute to the tragic misunderstandings.",
            "Laertes": "Polonius’s son, whose desire for revenge parallels and intersects with Hamlet’s."
        },
        "must_see_scene": "Act 3, Scene 4, where Hamlet confronts Gertrude in her chamber, leading to Polonius’s accidental death, a pivotal escalation."
    },
    "Romeo and Juliet": {
        "category": "tragedy",
        "main_conflict": "The forbidden love between Romeo and Juliet, opposed by their feuding families, the Montagues and Capulets.",
        "conflict_impact": "Romeo and Juliet risk everything for love, Mercutio and Tybalt die in the feud’s violence, and the families suffer devastating losses.",
        "play_significance": "Its universal themes of love, fate, and conflict, paired with poetic language, make it iconic.",
        "memorable_scene": "In Act 2, Scene 2, the balcony scene, Romeo and Juliet declare their love, showcasing their passion and defiance.",
        "protagonist": "Romeo and Juliet (dual protagonists), young lovers who are impulsive, romantic, and determined to be together.",
        "setting_unique": "Verona’s vibrant yet divided society highlights the destructive power of family loyalty and rivalry.",
        "tragedy_ending": "Miscommunication and impulsive decisions lead to Romeo and Juliet’s deaths, as fate and the feud overwhelm their love.",
        "key_moment": "In Act 3, Scene 1, Mercutio’s death escalates the feud, leading to Romeo’s banishment and the lovers’ doomed path.",
        "reaction_to_moment": "Romeo kills Tybalt in rage, Juliet is torn between love and family, and the families’ hatred deepens.",
        "character_roles": {
            "Romeo": "A passionate Montague who falls deeply in love with Juliet, defying his family.",
            "Juliet": "A determined Capulet who risks all for Romeo, showing courage and devotion.",
            "Mercutio": "Romeo’s witty friend whose death ignites the play’s tragic spiral.",
            "Tybalt": "Juliet’s hot-headed cousin whose aggression fuels the feud.",
            "Friar Laurence": "The well-meaning priest who secretly marries the lovers, hoping to end the feud."
        },
        "must_see_scene": "Act 5, Scene 3, the tomb scene, where Romeo and Juliet’s tragic deaths reconcile their families."
    },
    "Macbeth": {
        "category": "tragedy",
        "main_conflict": "Macbeth’s ambition drives him to murder King Duncan to seize the throne, leading to paranoia and chaos.",
        "conflict_impact": "Macbeth becomes tyrannical, Lady Macbeth is consumed by guilt, and Banquo and Macduff suffer as Macbeth’s actions destabilize Scotland.",
        "play_significance": "Its intense portrayal of ambition, guilt, and fate, with vivid imagery, cements its status as a psychological thriller.",
        "memorable_scene": "In Act 2, Scene 1, Macbeth sees a hallucinatory dagger before killing Duncan, revealing his inner turmoil.",
        "protagonist": "Macbeth, a brave but ambitious warrior whose desire for power leads to his downfall.",
        "setting_unique": "Scotland’s dark, supernatural atmosphere, with witches and omens, amplifies the play’s themes of fate and evil.",
        "tragedy_ending": "Macbeth’s unchecked ambition and murders lead to his isolation and death, as order is restored at a great cost.",
        "key_moment": "In Act 1, Scene 3, the witches’ prophecy sparks Macbeth’s ambition, setting his tragic path in motion.",
        "reaction_to_moment": "Macbeth becomes obsessed with becoming king, Lady Macbeth pushes him to act, and Banquo grows wary.",
        "character_roles": {
            "Macbeth": "A nobleman whose ambition drives him to murder and tyranny.",
            "Lady Macbeth": "Macbeth’s ruthless wife who encourages his crimes but succumbs to guilt.",
            "Banquo": "Macbeth’s friend whose loyalty and prophecy threaten Macbeth.",
            "Macduff": "A nobleman who opposes Macbeth and seeks to restore order.",
            "Witches": "Supernatural beings who manipulate Macbeth with ambiguous prophecies."
        },
        "must_see_scene": "Act 5, Scene 1, Lady Macbeth’s sleepwalking scene, where her guilt manifests as she tries to wash imaginary blood."
    },
    "A Midsummer Night's Dream": {
        "category": "comedy",
        "main_conflict": "The tangled love affairs of four young Athenians, complicated by fairy magic and parental opposition.",
        "conflict_impact": "Hermia and Lysander flee to be together, Helena feels rejected, Demetrius is enchanted, and all face confusion before resolution.",
        "play_significance": "Its whimsical blend of love, magic, and humor, with vibrant characters, makes it a beloved comedy.",
        "memorable_scene": "In Act 3, Scene 2, Puck’s love potion causes hilarious romantic chaos among the lovers in the forest.",
        "protagonist": "No single protagonist; the four lovers (Hermia, Lysander, Helena, Demetrius) drive the romantic plot, each spirited and emotional.",
        "setting_unique": "An enchanted Athenian forest, filled with fairies and magic, creates a dreamlike, playful world.",
        "key_moment": "In Act 2, Scene 1, Puck mistakenly applies the love potion to Lysander, causing the lovers’ affections to shift chaotically.",
        "reaction_to_moment": "Lysander pursues Helena, Hermia is heartbroken, Helena feels mocked, and Demetrius remains oblivious until enchanted.",
        "character_roles": {
            "Hermia": "A defiant lover who risks all to marry Lysander against her father’s wishes.",
            "Lysander": "Hermia’s devoted lover, caught in magical romantic mix-ups.",
            "Helena": "A lovesick woman pining for Demetrius, gaining confidence through the chaos.",
            "Demetrius": "A suitor who loves Hermia but is redirected by magic to Helena.",
            "Puck": "A mischievous fairy who causes the romantic confusion with his magic."
        },
        "must_see_scene": "Act 5, Scene 1, the mechanicals’ comical performance of Pyramus and Thisbe, a joyful resolution."
    },
    "Much Ado About Nothing": {
        "category": "comedy",
        "main_conflict": "Deceptions and misunderstandings threaten the romances of Claudio and Hero, and Beatrice and Benedick.",
        "conflict_impact": "Claudio wrongly shames Hero, Hero suffers public humiliation, and Beatrice and Benedick overcome their pride to admit love.",
        "play_significance": "Its sharp wit, complex characters, and exploration of trust and love make it a standout comedy.",
        "memorable_scene": "In Act 4, Scene 1, Claudio rejects Hero at the altar, a shocking moment of betrayal and drama.",
        "protagonist": "Beatrice and Benedick (dual protagonists), witty and independent, who evolve through love.",
        "setting_unique": "Messina’s sunny, social world emphasizes community, gossip, and celebration, driving the plot’s misunderstandings.",
        "key_moment": "In Act 2, Scene 3, Benedick is tricked into believing Beatrice loves him, sparking their romantic arc.",
        "reaction_to_moment": "Benedick softens his anti-love stance, Beatrice is similarly tricked, and their friends revel in the scheme.",
        "character_roles": {
            "Beatrice": "A sharp-tongued woman who resists love but falls for Benedick.",
            "Benedick": "A witty bachelor who denies love until tricked into loving Beatrice.",
            "Claudio": "A young lord whose mistrust leads to Hero’s humiliation.",
            "Hero": "Claudio’s innocent fiancée, wronged by false accusations.",
            "Don John": "The villain who orchestrates Hero’s slander to cause chaos."
        },
        "must_see_scene": "Act 3, Scene 1, where Beatrice is tricked into loving Benedick, mirroring his deception."
    },
    "The Tragedy of King Lear": {
        "category": "tragedy",
        "main_conflict": "King Lear’s division of his kingdom based on flattery leads to betrayal by his daughters and his descent into madness.",
        "conflict_impact": "Lear is cast out and loses his sanity, Cordelia is banished and later killed, and Goneril and Regan’s greed destroys them.",
        "play_significance": "Its profound exploration of family, power, and human suffering, with raw emotional depth, makes it a masterpiece.",
        "memorable_scene": "In Act 3, Scene 2, Lear rages against the storm, a powerful image of his inner turmoil and madness.",
        "protagonist": "King Lear, a proud but flawed king who learns humility through immense suffering.",
        "setting_unique": "A bleak, ancient Britain with raging storms reflects the chaos of Lear’s fractured kingdom and mind.",
        "tragedy_ending": "Lear’s misjudgment and the greed of Goneril and Regan lead to betrayal, madness, and the deaths of Lear and Cordelia.",
        "key_moment": "In Act 1, Scene 1, Lear banishes Cordelia for her honesty, triggering the kingdom’s collapse.",
        "reaction_to_moment": "Cordelia is heartbroken but steadfast, Goneril and Regan seize power, and Kent protests loyally.",
        "character_roles": {
            "King Lear": "The aging king whose pride and errors lead to his downfall and redemption.",
            "Cordelia": "Lear’s honest daughter, whose love contrasts with her sisters’ deceit.",
            "Goneril": "Lear’s eldest daughter, whose ambition and cruelty drive the conflict.",
            "Regan": "Lear’s second daughter, equally ruthless in her pursuit of power.",
            "Edmund": "Gloucester’s scheming son, whose betrayal parallels the sisters’."
        },
        "must_see_scene": "Act 5, Scene 3, where Lear mourns Cordelia’s death, a heartbreaking climax."
    },
    "Othello, the Moor of Venice": {
        "category": "tragedy",
        "main_conflict": "Iago’s manipulation of Othello’s jealousy leads to the destruction of Othello and Desdemona’s marriage.",
        "conflict_impact": "Othello becomes consumed by distrust, Desdemona is unjustly killed, and Iago’s deceit ruins multiple lives.",
        "play_significance": "Its intense study of jealousy, race, and betrayal, with gripping characters, marks it as a powerful tragedy.",
        "memorable_scene": "In Act 3, Scene 3, Iago plants doubts about Desdemona’s fidelity, manipulating Othello’s emotions.",
        "protagonist": "Othello, a noble but insecure general whose trust in Iago leads to his tragic downfall.",
        "setting_unique": "Venice and Cyprus, with their military and exotic elements, underscore Othello’s outsider status and the play’s tension.",
        "tragedy_ending": "Othello’s jealousy, fueled by Iago, leads to Desdemona’s murder and his suicide, as truth arrives too late.",
        "key_moment": "In Act 3, Scene 3, Iago convinces Othello of Desdemona’s infidelity, turning him against her.",
        "reaction_to_moment": "Othello spirals into rage and doubt, Desdemona remains unaware, and Iago revels in his success.",
        "character_roles": {
            "Othello": "The Moorish general whose insecurities make him vulnerable to manipulation.",
            "Desdemona": "Othello’s loyal wife, tragically killed due to false accusations.",
            "Iago": "The cunning villain who orchestrates the tragedy through deceit.",
            "Cassio": "Othello’s lieutenant, whose reputation Iago exploits to fuel jealousy.",
            "Emilia": "Iago’s wife, who unwittingly aids his schemes but later exposes him."
        },
        "must_see_scene": "Act 5, Scene 2, where Othello kills Desdemona, a devastating climax of jealousy and regret."
    },
    "Twelfth Night; Or, What You Will": {
        "category": "comedy",
        "main_conflict": "Mistaken identities and unrequited love create romantic chaos among Viola, Orsino, Olivia, and others.",
        "conflict_impact": "Viola disguises as a man, causing Olivia to fall for her, Orsino pines for Olivia, and Malvolio is humiliated by a prank.",
        "play_significance": "Its blend of humor, romance, and gender exploration, with lively characters, makes it a joyful comedy.",
        "memorable_scene": "In Act 2, Scene 5, Malvolio reads a forged letter, believing Olivia loves him, a hilarious prank.",
        "protagonist": "Viola, a resourceful and witty woman who navigates love and disguise with courage.",
        "setting_unique": "Illyria’s festive, coastal world fosters romance, music, and playful deception.",
        "key_moment": "In Act 1, Scene 5, Olivia falls for Viola (disguised as Cesario), complicating the love triangle.",
        "reaction_to_moment": "Viola is caught in a dilemma, Olivia becomes infatuated, and Orsino remains oblivious.",
        "character_roles": {
            "Viola": "A shipwrecked woman who disguises as a man, sparking romantic confusion.",
            "Orsino": "Illyria’s duke, in love with Olivia but drawn to Viola.",
            "Olivia": "A countess who falls for Viola’s disguise, rejecting Orsino.",
            "Malvolio": "Olivia’s pompous steward, humiliated by a cruel prank.",
            "Feste": "The witty fool whose songs and jests deepen the play’s themes."
        },
        "must_see_scene": "Act 5, Scene 1, the resolution where identities are revealed, and love is sorted out."
    },
    "The Tempest": {
        "category": "comedy",
        "main_conflict": "Prospero, a banished duke, uses magic to regain his position and resolve past betrayals on a remote island.",
        "conflict_impact": "Prospero manipulates Miranda, Ferdinand, and his enemies, while Ariel and Caliban navigate their roles in his plans.",
        "play_significance": "Its magical storytelling, themes of forgiveness, and reflective tone make it a profound late work.",
        "memorable_scene": "In Act 4, Scene 1, Prospero stages a magical masque for Miranda and Ferdinand, celebrating their love.",
        "protagonist": "Prospero, a powerful but controlling magician seeking justice and reconciliation.",
        "setting_unique": "A mystical island, filled with spirits and enchantments, creates a fantastical stage for redemption.",
        "key_moment": "In Act 1, Scene 2, Prospero raises a storm to shipwreck his enemies, initiating his plan for justice.",
        "reaction_to_moment": "Miranda is distressed, Ariel obeys Prospero, and the shipwrecked nobles are terrified and scattered.",
        "character_roles": {
            "Prospero": "The exiled duke who uses magic to reclaim his life and forgive his enemies.",
            "Miranda": "Prospero’s innocent daughter, who falls in love with Ferdinand.",
            "Ariel": "Prospero’s spirit servant, who performs magical tasks.",
            "Caliban": "The island’s native, resentful of Prospero’s control.",
            "Ferdinand": "The prince who loves Miranda, proving his worth to Prospero."
        },
        "must_see_scene": "Act 5, Scene 1, where Prospero forgives his enemies, a moving act of reconciliation."
    },
    "Julius Caesar": {
        "category": "tragedy",
        "main_conflict": "Brutus and the conspirators assass223inate Caesar to prevent tyranny, sparking civil war and their downfall.",
        "conflict_impact": "Brutus is torn by guilt, Cassius manipulates others, Antony fuels rebellion, and Rome descends into chaos.",
        "play_significance": "Its exploration of power, loyalty, and betrayal, with stirring rhetoric, makes it a political classic.",
        "memorable_scene": "In Act 3, Scene 2, Antony’s 'Friends, Romans, countrymen' speech sways the crowd against the conspirators.",
        "protagonist": "Brutus, an honorable but conflicted noble who joins the conspiracy for Rome’s sake.",
        "setting_unique": "Ancient Rome’s grand yet tense political stage highlights the stakes of power and betrayal.",
        "tragedy_ending": "Brutus’s idealism leads to war and his death, as his actions fail to save Rome from tyranny.",
        "key_moment": "In Act 3, Scene 1, Caesar’s assassination shifts power and ignites conflict.",
        "reaction_to_moment": "Brutus justifies the act, Antony plots revenge, and the public descends into unrest.",
        "character_roles": {
            "Julius Caesar": "The powerful leader whose ambition prompts his assassination.",
            "Brutus": "The honorable conspirator who kills Caesar for Rome but faces tragic consequences.",
            "Cassius": "The cunning conspirator who recruits Brutus but clashes with him.",
            "Mark Antony": "Caesar’s loyal ally who manipulates the public to avenge him.",
            "Portia": "Brutus’s wife, who senses his inner conflict and suffers for it."
        },
        "must_see_scene": "Act 4, Scene 3, the quarrel between Brutus and Cassius, revealing their strained alliance."
    }
}

# === HELPERS ===
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        return None

def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

# === MAIN FUNCTION ===
def generate_extended_qa():
    qa_dataset = []
    factual_data = load_json(FACTUAL_FILE)
    if not factual_data:
        logger.error("No data loaded, exiting")
        return 0

    # Filter for requested plays
    plays_data = [play for play in factual_data if play.get("title") in PLAYS]
    missing_plays = [title for title in PLAYS if title not in [p.get("title") for p in plays_data]]
    if missing_plays:
        logger.warning(f"Missing plays in factual.json: {missing_plays}")

    for play in plays_data:
        title = play.get("title", "")
        if not title:
            logger.warning("Skipping play with no title")
            continue

        logger.info(f"Processing play: {title}")
        play_data = RESPONSE_DATA.get(title, {})
        category = play.get("category", "").lower()
        main_characters = play.get("main_characters", [])

        # Question 1: Main conflict
        question_template = random.choice(QUESTION_TEMPLATES["main_conflict"])
        question = question_template(title)
        response = play_data.get("main_conflict", "")
        if response:
            qa_dataset.append({"prompt": question, "response": response})

        # Question 2: Conflict impact
        question_template = random.choice(QUESTION_TEMPLATES["conflict_impact"])
        question = question_template(title)
        response = play_data.get("conflict_impact", "")
        if response:
            qa_dataset.append({"prompt": question, "response": response})

        # Question 3: Play significance
        question_template = random.choice(QUESTION_TEMPLATES["play_significance"])
        question = question_template(title)
        response = play_data.get("play_significance", "")
        if response:
            qa_dataset.append({"prompt": question, "response": response})

        # Question 4: Memorable scene
        question_template = random.choice(QUESTION_TEMPLATES["memorable_scene"])
        question = question_template(title)
        response = play_data.get("memorable_scene", "")
        if response:
            qa_dataset.append({"prompt": question, "response": response})

        # Question 5: Protagonist
        question_template = random.choice(QUESTION_TEMPLATES["protagonist"])
        question = question_template(title)
        response = play_data.get("protagonist", "")
        if response:
            qa_dataset.append({"prompt": question, "response": response})

        # Question 6: Setting unique
        question_template = random.choice(QUESTION_TEMPLATES["setting_unique"])
        question = question_template(title)
        response = play_data.get("setting_unique", "")
        if response:
            qa_dataset.append({"prompt": question, "response": response})

        # Question 7: Tragedy ending (tragedies only)
        if category == "tragedy":
            question_template = random.choice(QUESTION_TEMPLATES["tragedy_ending"])
            question = question_template(title)
            response = play_data.get("tragedy_ending", "")
            if response:
                qa_dataset.append({"prompt": question, "response": response})

        # Question 8: Key moment
        question_template = random.choice(QUESTION_TEMPLATES["key_moment"])
        question = question_template(title)
        response = play_data.get("key_moment", "")
        if response:
            qa_dataset.append({"prompt": question, "response": response})

        # Question 9: Reaction to key moment
        question_template = random.choice(QUESTION_TEMPLATES["reaction_to_moment"])
        question = question_template(title)
        response = play_data.get("reaction_to_moment", "")
        if response:
            qa_dataset.append({"prompt": question, "response": response})

        # Question 10: Character roles (one per main character)
        for character in main_characters:
            question_template = random.choice(QUESTION_TEMPLATES["character_role"])
            question = question_template(title, character)
            response = play_data.get("character_roles", {}).get(character, "")
            if response:
                qa_dataset.append({"prompt": question, "response": response})
            else:
                logger.warning(f"No role description for {character} in {title}")

        # Question 11: Must-see scene
        question_template = random.choice(QUESTION_TEMPLATES["must_see_scene"])
        question = question_template(title)
        response = play_data.get("must_see_scene", "")
        if response:
            qa_dataset.append({"prompt": question, "response": response})

    save_jsonl(qa_dataset, OUTPUT_FILE)
    logger.info(f"✅ Generated {len(qa_dataset)} prompt-response pairs and saved to {OUTPUT_FILE}")
    return len(qa_dataset)

# === RUN ===
if __name__ == "__main__":
    total_new = generate_extended_qa()
    print(f"Generated {total_new} new prompt-response pairs")