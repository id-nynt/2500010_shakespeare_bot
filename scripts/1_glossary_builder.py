import json
import re

# Step 1: Define the glossary dictionary
glossary = {
# Pronouns & Possessives
    "thou": "you", "thee": "you", "thy": "your", "thine": "yours", "ye": "you",
    
    # Verbs
    "art": "are", "wert": "were", "hast": "have", "hadst": "had",
    "dost": "do", "doth": "does", "shalt": "shall", "shouldst": "should",
    "wilt": "will", "wouldst": "would", "couldst": "could", "canst": "can",
    "mayst": "may", "mightst": "might", "must needs": "must",
    "saith": "says", "cometh": "comes", "goeth": "goes", "speakest": "speak",
    "knowest": "know", "thinkest": "think", "sayest": "say", "lovest": "love",
    "lookest": "look", "gavest": "gave", "bringest": "bring", "seekest": "seek",
    "leadeth": "leads", "followeth": "follows", "asketh": "asks", "answereth": "answers",

    # Adverbs & Prepositions
    "ere": "before", "anon": "soon", "oft": "often", "nigh": "near",
    "whither": "where", "hither": "here", "thither": "there",
    "withal": "in addition", "perchance": "perhaps", "methinks": "I think",
    "peradventure": "maybe", "marry": "indeed", "fain": "gladly",
    "haply": "by chance", "whence": "from where", "whilst": "while",

    # Interjections
    "hark": "listen", "prithee": "please", "forsooth": "indeed",
    "zounds": "God's wounds", "alas": "oh no", "alack": "woe",
    "ay": "yes", "nay": "no", "sooth": "truth", "troth": "truth",

    # Nouns
    "knave": "servant", "churl": "peasant", "coxcomb": "fool",
    "wench": "girl", "harlot": "prostitute", "varlet": "rascal",
    "sirrah": "boy", "maid": "young woman", "bosom": "chest",
    "orb": "eye", "countenance": "face", "visage": "appearance",
    "thyself": "yourself", "villain": "scoundrel",

    # Conjunctions and Structure
    "wherefore": "why", "how now": "what's the matter", "an": "if",

    # Obsolete/Misc
    "fare thee well": "goodbye", "hold thee": "take this", "make haste": "hurry",
    "good morrow": "good morning", "good den": "good evening",
    "would fain": "would like to", "I trow": "I believe",

    "o'er": "over",
    "ne'er": "never",
    "e'en": "even",
    "e'er": "ever",
    
    "adieu": "farewell",
    "aroint": "away",
    "come hither": "come here",
    "counsel": "advice",
    "decree": "order",
    "discourses": "speaks",
    "dispatch": "kill",
    "foe": "enemy",
    "god save thee": "goodbye",
    "grammercy": "thank you",
    "heavy": "sad",
    "hie": "go",
    "i shall see thee anon": "goodbye",
    "mark": "pay attention to",
    "-morrow": "days",
    "nought": "nothing",
    "plague": "curse",
    "privy": "informed",
    "shun that": "ignore that",
    "soft": "wait a minute",
    "thou art": "you are",
    "thou should'st": "you should",
    "thou would'st": "you would",
    "tidings": "news",
    "verily": "truly",
    "well met": "hello",
    "whereto": "to which",
    "will": "desire",
    "woe": "misery",
    "woo": "chase (romantically)",
    "would": "wish",
    "wrought": "provided"
}

# Step 2: Save to data/glossary/glossary.json
with open("data/glossary/glossary.json", "w", encoding="utf-8") as f:
    json.dump(glossary, f, indent=2)

# Step 3: Function to apply glossary to text
def normalize_text(text, glossary=glossary):
    def replace_word(word):
        stripped = re.sub(r'\W+', '', word).lower()
        return glossary.get(stripped, word)
    
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    normalized = [replace_word(w) if w.isalpha() else w for w in words]
    return ''.join([' ' + w if i > 0 and w.isalnum() else w for i, w in enumerate(normalized)])

# Example usage
if __name__ == "__main__":
    old_text = "Thou art the man! Dost thou see her?"
    modern = normalize_text(old_text)
    print("Original:", old_text)
    print("Modernized:", modern)