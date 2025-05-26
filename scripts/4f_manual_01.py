import os
import json
import random
import logging
from pathlib import Path

# === CONFIG ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FACTUAL_FILE = "data/processed/factual/factual.json"
QUOTE_FILE = "data/processed/quote/quote.json"
OUTPUT_FILE = "data/prompt_response/manual_01.jsonl"
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
    "central_theme": [
        lambda x: f"What’s the main theme in *{x}*?",
        lambda x: f"Can you identify the core theme of *{x}*?",
        lambda x: f"What is the primary theme explored in *{x}*?"
    ],
    "character_personality": [
        lambda c, x: f"How does *{c}*’s personality influence their decisions in *{x}*?",
        lambda c, x: f"What role does *{c}*’s character traits play in their actions in *{x}*?",
        lambda c, x: f"How do *{c}*’s personal qualities drive their behavior in *{x}*?"
    ],
    "symbol_motif": [
        lambda s, x: f"What is the meaning of the *{s}* in *{x}*?",
        lambda s, x: f"How is the *{s}* used symbolically in *{x}*?",
        lambda s, x: f"What does the *{s}* signify in *{x}*?"
    ],
    "character_death": [
        lambda c, x: f"Why does *{c}*’s death matter in *{x}*?",
        lambda c, x: f"What makes *{c}*’s death important in *{x}*?",
        lambda c, x: f"How does *{c}*’s death impact *{x}*’s story?"
    ],
    "thematic_quote": [
        lambda c, x, t: f"Can you share a quote by *{c}* in *{x}* about *{t}*?",
        lambda c, x, t: f"What’s a line from *{c}* in *{x}* that reflects *{t}*?",
        lambda c, x, t: f"Find a quote by *{c}* in *{x}* related to *{t}*."
    ],
    "character_speech": [
        lambda c, x, e: f"Write a brief speech by *{c}* in *{x}* about *{e}*.",
        lambda c, x, e: f"Create a short monologue for *{c}* in *{x}* on *{e}*.",
        lambda c, x, e: f"Have *{c}* in *{x}* give a speech about *{e}*."
    ],
    "specific_line": [
        lambda y, z, x: f"What’s a line from Act *{y}*, Scene *{z}* in *{x}*?",
        lambda y, z, x: f"Can you quote a line from *{x}*’s Act *{y}*, Scene *{z}*?",
        lambda y, z, x: f"Recite a line from *{x}*, Act *{y}*, Scene *{z}*."
    ],
    "poetic_line": [
        lambda x, t: f"Create a poetic line in *{x}*’s style about *{t}*.",
        lambda x, t: f"Write a line of poetry like *{x}* on *{t}*.",
        lambda x, t: f"Compose a poetic line in the tone of *{x}* about *{t}*."
    ],
    "dramatic_quote": [
        lambda x: f"Find a dramatic quote from *{x}* that reflects its tone.",
        lambda x: f"What’s a powerful line from *{x}* that shows its mood?",
        lambda x: f"Can you share a quote from *{x}* that captures its dramatic feel?"
    ],
    "couplet": [
        lambda x: f"Write a couplet in the style of *{x}*.",
        lambda x: f"Compose a two-line poem like *{x}*.",
        lambda x: f"Create a couplet inspired by *{x}*’s tone."
    ],
    "personality_quote": [
        lambda x, c: f"Find a quote from *{x}* that reveals *{c}*’s personality.",
        lambda x, c: f"What’s a line in *{x}* that reflects *{c}*’s character?",
        lambda x, c: f"Can you share a quote by *{c}* in *{x}* that shows who they are?"
    ]
}

# === RESPONSE DATA ===
RESPONSE_DATA = {
    "Hamlet, Prince of Denmark": {
        "category": "tragedy",
        "central_theme": "Revenge and its destructive consequences.",
        "symbol_motif": {
            "name": "Yorick’s skull",
            "meaning": "Represents mortality and the inevitability of death, as Hamlet reflects on life’s fleeting nature."
        },
        "specific_line": {
            "act": 3,
            "scene": 1,
            "line": "To be, or not to be, that is the question"
        },
        "poetic_line_topic": "fate",
        "poetic_line": "O fate, thou fickle wheel, what doom dost thou decree?",
        "dramatic_quote": "To be, or not to be, that is the question (Hamlet, Act 3, Scene 1).",
        "couplet": "In shadows deep, my soul doth roam, / To seek revenge or find my home.",
        "character_data": {
            "Hamlet": {
                "personality": "His introspective and indecisive nature leads him to feign madness and delay revenge, causing tragic outcomes.",
                "death": "Hamlet’s death resolves his quest for revenge but at the cost of nearly all lives, emphasizing the futility of vengeance.",
                "thematic_quote": {
                    "theme": "revenge",
                    "quote": "How all occasions do inform against me / And spur my dull revenge! (Act 4, Scene 4)"
                },
                "speech": {
                    "emotion": "revenge",
                    "text": "O vengeful heart, why dost thou falter still? / The ghost of my father bids me strike, / Yet doubt and fear my purpose chill. / Shall I let Claudius live, or end this strife?"
                },
                "personality_quote": "What is a man / If his chief good and market of his time / Be but to sleep and feed? (Act 4, Scene 4)"
            },
            "Claudius": {
                "personality": "His cunning and ambition drive him to murder his brother and manipulate others to maintain power.",
                "death": "Claudius’s death by Hamlet’s hand ends his usurpation, restoring moral order but too late to save others.",
                "thematic_quote": {
                    "theme": "guilt",
                    "quote": "O, my offence is rank, it smells to heaven (Act 3, Scene 3)"
                },
                "speech": {
                    "emotion": "guilt",
                    "text": "My crown, my queen, my life—cursed gain! / This guilt doth weigh my soul in endless pain. / No prayer can cleanse the blood upon my hands, / Yet still I cling to power’s fleeting strands."
                },
                "personality_quote": "That cannot be, since I am still possess’d / Of those effects for which I did the murder (Act 3, Scene 3)"
            },
            "Gertrude": {
                "personality": "Her loyalty to family blinds her to Claudius’s crimes, leading to her passive role in the tragedy.",
                "death": "Gertrude’s accidental poisoning highlights the collateral damage of Claudius’s schemes.",
                "thematic_quote": {
                    "theme": "loyalty",
                    "quote": "Good Hamlet, cast thy nighted colour off (Act 1, Scene 2)"
                },
                "speech": {
                    "emotion": "sorrow",
                    "text": "O heavy heart, my son’s despair I see, / Yet know not how to mend this rift with thee. / My love for king and child doth tear my soul, / In this dark court, no peace can make me whole."
                },
                "personality_quote": "Thou know’st ’tis common; all that lives must die (Act 1, Scene 2)"
            },
            "Ophelia": {
                "personality": "Her innocence and obedience make her vulnerable to Hamlet’s rejection and her father’s schemes, leading to madness.",
                "death": "Ophelia’s drowning symbolizes the tragic cost of the court’s corruption on the innocent.",
                "thematic_quote": {
                    "theme": "madness",
                    "quote": "There’s rosemary, that’s for remembrance (Act 4, Scene 5)"
                },
                "speech": {
                    "emotion": "grief",
                    "text": "O heart, why dost thou break beneath this woe? / My love, my father—gone, and none to know. / These flowers I strew, my sorrow’s only voice, / In madness drowned, I’ve lost all choice."
                },
                "personality_quote": "Lord, we know what we are, but not what we may be (Act 4, Scene 5)"
            },
            "Polonius": {
                "personality": "His meddling and verbosity lead him to spy on Hamlet, resulting in his accidental death.",
                "death": "Polonius’s death escalates the conflict, pushing Hamlet toward exile and Ophelia toward madness.",
                "thematic_quote": {
                    "theme": "deception",
                    "quote": "To thine own self be true (Act 1, Scene 3)"
                },
                "speech": {
                    "emotion": "caution",
                    "text": "Beware, my children, of this court’s deceit, / Where words and smiles hide treachery complete. / Be true, be wise, and guard thy heart’s intent, / Lest folly lead where wisdom never went."
                },
                "personality_quote": "This above all: to thine own self be true (Act 1, Scene 3)"
            },
            "Laertes": {
                "personality": "His impulsive loyalty drives him to seek revenge for his father and sister, mirroring Hamlet.",
                "death": "Laertes’s death in the duel underscores the destructive cycle of vengeance.",
                "thematic_quote": {
                    "theme": "revenge",
                    "quote": "I am satisfied in nature, / But in my terms of honour I stand aloof (Act 5, Scene 2)"
                },
                "speech": {
                    "emotion": "revenge",
                    "text": "For father slain and sister lost, I burn, / With righteous wrath, my blade shall Hamlet spurn. / No peace shall dwell till justice rights this wrong, / My vengeance swift, my honor fierce and strong."
                },
                "personality_quote": "To hell, allegiance! Vows, to the blackest devil! (Act 4, Scene 5)"
            }
        }
    },
    "Romeo and Juliet": {
        "category": "tragedy",
        "central_theme": "The destructive power of love and feuding.",
        "symbol_motif": {
            "name": "light and dark imagery",
            "meaning": "Represents the fleeting, intense nature of Romeo and Juliet’s love, contrasted with the hatred around them."
        },
        "specific_line": {
            "act": 2,
            "scene": 2,
            "line": "But, soft! what light through yonder window breaks?"
        },
        "poetic_line_topic": "love",
        "poetic_line": "O love, thou radiant fire, both joy and bane dost weave.",
        "dramatic_quote": "A plague o’ both your houses! (Mercutio, Act 3, Scene 1).",
        "couplet": "Two hearts entwined by love’s eternal flame, / Yet doom’d by strife to bear a tragic name.",
        "character_data": {
            "Romeo": {
                "personality": "His passionate and impulsive nature leads him to defy his family and risk all for Juliet.",
                "death": "Romeo’s suicide, believing Juliet dead, triggers the tragic resolution and reconciles the families.",
                "thematic_quote": {
                    "theme": "love",
                    "quote": "My bounty is as boundless as the sea, / My love as deep (Act 2, Scene 2)"
                },
                "speech": {
                    "emotion": "love",
                    "text": "O Juliet, thou art my sun, my heart’s delight! / No feud nor fate can dim thy radiant light. / For thee, I’d cast my name and life away, / To dwell in love’s sweet dawn till end of day."
                },
                "personality_quote": "Did my heart love till now? Forswear it, sight! (Act 1, Scene 5)"
            },
            "Juliet": {
                "personality": "Her courage and devotion push her to defy her family and fake her death to be with Romeo.",
                "death": "Juliet’s suicide after finding Romeo dead seals their tragic love and ends the feud.",
                "thematic_quote": {
                    "theme": "love",
                    "quote": "My only love sprung from my only hate! (Act 1, Scene 5)"
                },
                "speech": {
                    "emotion": "love",
                    "text": "O Romeo, my soul’s eternal mate, / No name nor kin can bar our love’s estate. / Though stars conspire to tear our hearts apart, / I vow to hold thee ever in my heart."
                },
                "personality_quote": "What’s in a name? That which we call a rose / By any other name would smell as sweet (Act 2, Scene 2)"
            },
            "Mercutio": {
                "personality": "His wit and recklessness lead him to provoke Tybalt, resulting in his death.",
                "death": "Mercutio’s death escalates the feud, leading to Romeo’s banishment and the tragedy’s spiral.",
                "thematic_quote": {
                    "theme": "fate",
                    "quote": "A plague o’ both your houses! (Act 3, Scene 1)"
                },
                "speech": {
                    "emotion": "anger",
                    "text": "O cursed houses, Montague and Capulet! / Your strife doth spill my blood, my life’s regret. / Why must your hate consume such loyal friends? / This feud’s foul curse no mortal hand amends."
                },
                "personality_quote": "Ask for me tomorrow, and you shall find me a grave man (Act 3, Scene 1)"
            },
            "Tybalt": {
                "personality": "His fiery temper and loyalty to the Capulets drive him to fight Romeo, leading to his death.",
                "death": "Tybalt’s death by Romeo’s hand causes Romeo’s exile, accelerating the tragedy.",
                "thematic_quote": {
                    "theme": "hate",
                    "quote": "What, drawn, and talk of peace! I hate the word (Act 1, Scene 1)"
                },
                "speech": {
                    "emotion": "hate",
                    "text": "Montagues, ye curs, my blade doth scorn your name! / This hate runs deep, a fire no peace can tame. / For Capulet’s honor, I’d spill your blood with glee, / No truce shall stand whilst foes like thee I see."
                },
                "personality_quote": "This, by his voice, should be a Montague. / Fetch me my rapier, boy (Act 1, Scene 5)"
            },
            "Friar Laurence": {
                "personality": "His well-meaning but naive optimism leads him to marry the lovers and devise risky plans.",
                "death": None,
                "thematic_quote": {
                    "theme": "fate",
                    "quote": "These violent delights have violent ends (Act 2, Scene 6)"
                },
                "speech": {
                    "emotion": "hope",
                    "text": "O gentle love, may this union heal the strife, / Bind warring houses to a peaceful life. / With heaven’s grace, your hearts may end this feud, / And Verona’s wounds be mended, cleansed, renewed."
                },
                "personality_quote": "Wisely and slow; they stumble that run fast (Act 2, Scene 3)"
            }
        }
    },
    "Macbeth": {
        "category": "tragedy",
        "central_theme": "Ambition and its corrupting influence.",
        "symbol_motif": {
            "name": "blood",
            "meaning": "Represents guilt and the inescapable consequences of Macbeth’s murders."
        },
        "specific_line": {
            "act": 2,
            "scene": 1,
            "line": "Is this a dagger which I see before me?"
        },
        "poetic_line_topic": "ambition",
        "poetic_line": "Ambition’s flame doth burn the soul to ash.",
        "dramatic_quote": "Out, out, brief candle! (Macbeth, Act 5, Scene 5).",
        "couplet": "By ambition’s spur, we climb to treacherous height, / Yet fall in darkness, shunn’d by heaven’s light.",
        "character_data": {
            "Macbeth": {
                "personality": "His ambition and insecurity drive him to murder Duncan and others, spiraling into paranoia.",
                "death": "Macbeth’s death restores order but underscores the cost of unchecked ambition.",
                "thematic_quote": {
                    "theme": "ambition",
                    "quote": "I have no spur / To prick the sides of my intent, but only / Vaulting ambition (Act 1, Scene 7)"
                },
                "speech": {
                    "emotion": "ambition",
                    "text": "O crown, thou glittering prize, my heart’s desire! / To grasp thee, I’d set Scotland’s peace afire. / No guilt nor fear shall stay my ruthless hand, / For king I’ll be, though cursed by heaven’s command."
                },
                "personality_quote": "I am in blood / Stepp’d in so far that, should I wade no more, / Returning were as tedious as go o’er (Act 3, Scene 4)"
            },
            "Lady Macbeth": {
                "personality": "Her ruthless determination pushes Macbeth to murder, but guilt consumes her.",
                "death": "Lady Macbeth’s suicide reflects her inability to escape the guilt of her crimes.",
                "thematic_quote": {
                    "theme": "guilt",
                    "quote": "Out, damned spot! Out, I say! (Act 5, Scene 1)"
                },
                "speech": {
                    "emotion": "guilt",
                    "text": "O blood, thou stain that lingers on my soul! / No water cleans the deeds that make me whole. / Each night I see the lives we tore apart, / And madness claims the remnants of my heart."
                },
                "personality_quote": "Come, you spirits / That tend on mortal thoughts, unsex me here (Act 1, Scene 5)"
            },
            "Banquo": {
                "personality": "His loyalty and caution contrast with Macbeth, leading him to question the witches.",
                "death": "Banquo’s murder fuels Macbeth’s paranoia and foreshadows his downfall.",
                "thematic_quote": {
                    "theme": "fate",
                    "quote": "Thou hast it now: king, Cawdor, Glamis, all, / As the weird women promised (Act 3, Scene 1)"
                },
                "speech": {
                    "emotion": "suspicion",
                    "text": "O witches’ words, what double truths ye weave? / I fear Macbeth doth more than I believe. / His rise to power hides a darker deed, / And fate’s design I question with my creed."
                },
                "personality_quote": "May they not be my oracles as well, / And set me up in hope? (Act 3, Scene 1)"
            },
            "Macduff": {
                "personality": "His patriotism and honor drive him to oppose Macbeth and avenge his family.",
                "death": None,
                "thematic_quote": {
                    "theme": "justice",
                    "quote": "Let us rather / Hold fast the mortal sword, and like good men / Bestride our downfall’n birthdom (Act 4, Scene 3)"
                },
                "speech": {
                    "emotion": "justice",
                    "text": "For Scotland’s weal, I raise my sword to fight, / To cast down tyranny and restore the right. / Macbeth’s foul reign hath torn my kin apart, / Yet justice stirs the valor in my heart."
                },
                "personality_quote": "O Scotland, Scotland! (Act 4, Scene 3)"
            },
            "Witches": {
                "personality": "Their cryptic and manipulative nature shapes Macbeth’s fate through ambiguous prophecies.",
                "death": None,
                "thematic_quote": {
                    "theme": "fate",
                    "quote": "Fair is foul, and foul is fair (Act 1, Scene 1)"
                },
                "speech": {
                    "emotion": "mystery",
                    "text": "By cryptic tongues, we spin the threads of fate, / To lead men’s hearts to glory or to hate. / Our visions cloud the truth with shadowed art, / And mortals bend to what we bid their heart."
                },
                "personality_quote": "Double, double toil and trouble; / Fire burn, and cauldron bubble (Act 4, Scene 1)"
            }
        }
    },
    "A Midsummer Night's Dream": {
        "category": "comedy",
        "central_theme": "The irrationality of love.",
        "symbol_motif": {
            "name": "the love potion",
            "meaning": "Represents the unpredictable and uncontrollable nature of love, causing chaos among the lovers."
        },
        "specific_line": {
            "act": 3,
            "scene": 2,
            "line": "Lord, what fools these mortals be!"
        },
        "poetic_line_topic": "love",
        "poetic_line": "Love’s gentle dream doth sway the heart astray.",
        "dramatic_quote": "The course of true love never did run smooth (Lysander, Act 1, Scene 1).",
        "couplet": "In love’s sweet maze, we wander hand in hand, / To find our joy in fairy’s charmed land.",
        "character_data": {
            "Hermia": {
                "personality": "Her defiance and loyalty lead her to flee Athens to be with Lysander.",
                "death": None,
                "thematic_quote": {
                    "theme": "love",
                    "quote": "So I, being young, till now ripe not to reason (Act 1, Scene 1)"
                },
                "speech": {
                    "emotion": "love",
                    "text": "O Lysander, my heart doth sing for thee, / No father’s law can part our destiny. / Through woods we’ll flee, where love shall reign supreme, / And live as one within a lover’s dream."
                },
                "personality_quote": "I would my father look’d but with my eyes (Act 1, Scene 1)"
            },
            "Lysander": {
                "personality": "His romantic devotion drives him to elope with Hermia, though the potion misdirects his love.",
                "death": None,
                "thematic_quote": {
                    "theme": "love",
                    "quote": "The course of true love never did run smooth (Act 1, Scene 1)"
                },
                "speech": {
                    "emotion": "love",
                    "text": "Hermia, my soul, thou art my only light, / For thee I’d brave the perils of the night. / No spell nor foe can break our sacred vow, / Our love shall bloom where fairy breezes blow."
                },
                "personality_quote": "Ay me! for aught that I could ever read, / Could ever hear by tale or history (Act 1, Scene 1)"
            },
            "Helena": {
                "personality": "Her insecurity and persistence lead her to pursue Demetrius, enduring rejection until magic intervenes.",
                "death": None,
                "thematic_quote": {
                    "theme": "love",
                    "quote": "Love looks not with the eyes, but with the mind (Act 1, Scene 1)"
                },
                "speech": {
                    "emotion": "longing",
                    "text": "O Demetrius, why dost thou spurn my heart? / My love, though scorn’d, shall never hence depart. / If only eyes of love could see my worth, / I’d find my joy upon this charmed earth."
                },
                "personality_quote": "And I am sick when I look not on you (Act 2, Scene 1)"
            },
            "Demetrius": {
                "personality": "His fickleness leads him to reject Helena until the love potion redirects his affections.",
                "death": None,
                "thematic_quote": {
                    "theme": "love",
                    "quote": "I wot not by what power— / But by some power it is—my love to Hermia, / Melted as the snow (Act 4, Scene 1)"
                },
                "speech": {
                    "emotion": "love",
                    "text": "O Helena, thou star that lights my way, / No longer Hermia holds my heart’s sweet sway. / By fairy’s charm, I see thy beauty clear, / And vow to love thee ever, true and dear."
                },
                "personality_quote": "Disparage not the faith thou dost not know (Act 3, Scene 2)"
            },
            "Puck": {
                "personality": "His mischievousness causes the romantic chaos but also resolves it with his magic.",
                "death": None,
                "thematic_quote": {
                    "theme": "mischief",
                    "quote": "Lord, what fools these mortals be! (Act 3, Scene 2)"
                },
                "speech": {
                    "emotion": "mischief",
                    "text": "O mortals, how ye tumble in love’s jest! / With potion’s charm, I stir your hearts’ unrest. / Yet fear not, Puck shall set all right ere dawn, / And leave ye dreaming till the night is gone."
                },
                "personality_quote": "I’ll follow you, I’ll lead you about a round (Act 3, Scene 1)"
            }
        }
    },
    "Much Ado About Nothing": {
        "category": "comedy",
        "central_theme": "The interplay of love and deception.",
        "symbol_motif": {
            "name": "masks",
            "meaning": "Represents hidden identities and misunderstandings that drive the romantic and social deceptions."
        },
        "specific_line": {
            "act": 4,
            "scene": 1,
            "line": "I do love nothing in the world so well as you"
        },
        "poetic_line_topic": "deception",
        "poetic_line": "Deception weaves a veil o’er truth’s clear sight.",
        "dramatic_quote": "I do love nothing in the world so well as you (Benedick, Act 4, Scene 1).",
        "couplet": "With wit and guile, we dance in love’s disguise, / Till truth unveils the heart beneath the lies.",
        "character_data": {
            "Beatrice": {
                "personality": "Her sharp wit and independence lead her to resist love until tricked into loving Benedick.",
                "death": None,
                "thematic_quote": {
                    "theme": "love",
                    "quote": "I love you with so much of my heart that none is left to protest (Act 4, Scene 1)"
                },
                "speech": {
                    "emotion": "love",
                    "text": "O Benedick, thou hast crept into my heart, / With jests and jibes that love did first impart. / No more I scorn, but yield to love’s sweet call, / And find in thee my joy, my all in all."
                },
                "personality_quote": "I had rather hear my dog bark at a crow than a man swear he loves me (Act 1, Scene 1)"
            },
            "Benedick": {
                "personality": "His pride and humor make him deny love until deception reveals his feelings for Beatrice.",
                "death": None,
                "thematic_quote": {
                    "theme": "love",
                    "quote": "I do love nothing in the world so well as you (Act 4, Scene 1)"
                },
                "speech": {
                    "emotion": "love",
                    "text": "Beatrice, thou hast tamed this bachelor’s heart, / With wit that wounds, yet heals with love’s sweet art. / I cast my jests aside, my pride I spurn, / For in thy love, my soul doth brightly burn."
                },
                "personality_quote": "When I said I would die a bachelor, I did not think I should live till I were married (Act 2, Scene 3)"
            },
            "Claudio": {
                "personality": "His impulsiveness and mistrust lead him to wrongly accuse Hero, causing her humiliation.",
                "death": None,
                "thematic_quote": {
                    "theme": "deception",
                    "quote": "O, what men dare do! What men may do! (Act 4, Scene 1)"
                },
                "speech": {
                    "emotion": "regret",
                    "text": "O Hero, wronged by mine own hasty heart! / I saw deceit where truth did play its part. / Forgive this fool, whose love was blind to see, / The purity and faith thou gav’st to me."
                },
                "personality_quote": "Done to death by slanderous tongues (Act 5, Scene 3)"
            },
            "Hero": {
                "personality": "Her innocence and loyalty make her a victim of Claudio’s accusations, but she perseveres.",
                "death": None,
                "thematic_quote": {
                    "theme": "honor",
                    "quote": "And when I liv’d, I was your other wife (Act 5, Scene 4)"
                },
                "speech": {
                    "emotion": "forgiveness",
                    "text": "O Claudio, though thy doubt did wound my name, / My heart forgives, and holds thee free of blame. / Let love restore what slander tore apart, / And bind us ever with a faithful heart."
                },
                "personality_quote": "O, God defend me! How am I beset! (Act 4, Scene 1)"
            },
            "Don John": {
                "personality": "His malice and envy drive him to sabotage Hero’s wedding, causing chaos.",
                "death": None,
                "thematic_quote": {
                    "theme": "deception",
                    "quote": "I am not of many words, but I thank you (Act 1, Scene 1)"
                },
                "speech": {
                    "emotion": "malice",
                    "text": "Let joy be crush’d beneath my cunning hand, / I’ll sow discord where love and peace do stand. / No heart but mine shall revel in their pain, / For chaos is my joy, their loss my gain."
                },
                "personality_quote": "I had rather be a canker in a hedge than a rose in his grace (Act 1, Scene 3)"
            }
        }
    },
    "The Tragedy of King Lear": {
        "category": "tragedy",
        "central_theme": "The consequences of pride and betrayal.",
        "symbol_motif": {
            "name": "the storm",
            "meaning": "Represents Lear’s inner turmoil and the chaos unleashed by his flawed decisions."
        },
        "specific_line": {
            "act": 3,
            "scene": 2,
            "line": "Blow, winds, and crack your cheeks! Rage! Blow!"
        },
        "poetic_line_topic": "betrayal",
        "poetic_line": "Betrayal’s sting doth rend the heart in twain.",
        "dramatic_quote": "How sharper than a serpent’s tooth it is to have a thankless child! (King Lear, Act 1, Scene 4).",
        "couplet": "By pride betray’d, the mighty fall to dust, / Their kingdom torn by greed and broken trust.",
        "character_data": {
            "King Lear": {
                "personality": "His pride and need for flattery lead him to divide his kingdom, triggering betrayal and madness.",
                "death": "Lear’s death, grieving Cordelia, underscores the tragic cost of his errors and the loss of redemption.",
                "thematic_quote": {
                    "theme": "betrayal",
                    "quote": "How sharper than a serpent’s tooth it is to have a thankless child! (Act 1, Scene 4)"
                },
                "speech": {
                    "emotion": "betrayal",
                    "text": "O daughters false, ye’ve torn my heart asunder! / Your love, once sworn, now cracks like heaven’s thunder. / In madness cast, I wander, blind with pain, / For trust betrayed shall ne’er be whole again."
                },
                "personality_quote": "I am a man more sinn’d against than sinning (Act 3, Scene 2)"
            },
            "Cordelia": {
                "personality": "Her honesty and loyalty lead her to reject flattery, resulting in her banishment but later redemption.",
                "death": "Cordelia’s execution devastates Lear and symbolizes the loss of innocence amid greed.",
                "thematic_quote": {
                    "theme": "truth",
                    "quote": "Nothing, my lord (Act 1, Scene 1)"
                },
                "speech": {
                    "emotion": "truth",
                    "text": "O father, truth is all I offer thee, / No gilded words, but love in honesty. / Though banish’d now, my heart remains thy own, / And seeks to heal the wounds thy pride hath sown."
                },
                "personality_quote": "I am sure my love’s / More ponderous than my tongue (Act 1, Scene 1)"
            },
            "Goneril": {
                "personality": "Her ambition and cruelty drive her to betray Lear and vie for power, leading to her downfall.",
                "death": "Goneril’s suicide after her schemes collapse reflects the self-destructive nature of her greed.",
                "thematic_quote": {
                    "theme": "power",
                    "quote": "The laws are mine, not thine (Act 5, Scene 3)"
                },
                "speech": {
                    "emotion": "ambition",
                    "text": "Why bow to age when youth and strength prevail? / I’ll seize the crown, let weaker hearts bewail. / No love nor duty binds my rightful claim, / For power’s throne shall bear my glorious name."
                },
                "personality_quote": "Fools do those villains pity who are punish’d / Ere they have done their mischief (Act 4, Scene 2)"
            },
            "Regan": {
                "personality": "Her ruthlessness and envy lead her to conspire against Lear and compete with Goneril.",
                "death": "Regan’s poisoning by Goneril highlights the destructive rivalry born of their ambition.",
                "thematic_quote": {
                    "theme": "power",
                    "quote": "Let him to me, I’ll fit him with his fault (Act 2, Scene 4)"
                },
                "speech": {
                    "emotion": "envy",
                    "text": "Why should Goneril claim what I deserve? / My heart doth boil, my will shall never swerve. / This kingdom’s mine, by cunning or by blade, / And none shall stand where Regan’s path is laid."
                },
                "personality_quote": "O, sir, you are old! / Nature in you stands on the very verge / Of her confine (Act 2, Scene 4)"
            },
            "Edmund": {
                "personality": "His cunning and ambition lead him to betray his father and brother, seeking power.",
                "death": "Edmund’s death in combat ends his schemes, reinforcing the play’s moral reckoning.",
                "thematic_quote": {
                    "theme": "ambition",
                    "quote": "Now, gods, stand up for bastards! (Act 1, Scene 2)"
                },
                "speech": {
                    "emotion": "ambition",
                    "text": "O base-born state, I’ll rise above my shame, / And carve my path to glory and to fame. / No brother’s right nor father’s love I need, / For power’s crown shall bloom from my bold deed."
                },
                "personality_quote": "Why bastard? Wherefore base? (Act 1, Scene 2)"
            }
        }
    },
    "Othello, the Moor of Venice": {
        "category": "tragedy",
        "central_theme": "Jealousy and its destructive power.",
        "symbol_motif": {
            "name": "the handkerchief",
            "meaning": "Represents Othello’s trust in Desdemona, manipulated by Iago to fuel jealousy."
        },
        "specific_line": {
            "act": 3,
            "scene": 3,
            "line": "O, beware, my lord, of jealousy"
        },
        "poetic_line_topic": "jealousy",
        "poetic_line": "Jealousy’s green venom poisons love’s sweet spring.",
        "dramatic_quote": "O, beware, my lord, of jealousy; / It is the green-ey’d monster (Iago, Act 3, Scene 3).",
        "couplet": "By jealousy’s cruel sting, the heart is torn, / And love’s bright flame to ashes is forsworn.",
        "character_data": {
            "Othello": {
                "personality": "His nobility and insecurity lead him to trust Iago over Desdemona, resulting in murder.",
                "death": "Othello’s suicide after killing Desdemona restores his honor but highlights jealousy’s toll.",
                "thematic_quote": {
                    "theme": "jealousy",
                    "quote": "O, beware, my lord, of jealousy (Act 3, Scene 3)"
                },
                "speech": {
                    "emotion": "jealousy",
                    "text": "O cursed doubt, that tears my soul apart! / Hath Desdemona play’d false with mine own heart? / I loved her true, yet Iago’s words do sting, / And jealousy’s dark tide doth ruin bring."
                },
                "personality_quote": "I am not what I am (Act 1, Scene 1)"
            },
            "Desdemona": {
                "personality": "Her loyalty and innocence make her unaware of Iago’s schemes, leading to her unjust death.",
                "death": "Desdemona’s murder by Othello underscores the tragedy of misplaced trust.",
                "thematic_quote": {
                    "theme": "love",
                    "quote": "My heart’s subdued / Even to the very quality of my lord (Act 1, Scene 3)"
                },
                "speech": {
                    "emotion": "love",
                    "text": "O Othello, thou art my heart’s true home, / No storm nor slander can our love o’ercome. / My soul is thine, though false tongues seek to part, / I swear my love endures within my heart."
                },
                "personality_quote": "Nobody; I myself. Farewell (Act 4, Scene 2)"
            },
            "Iago": {
                "personality": "His cunning and malice drive him to manipulate Othello, causing the tragedy.",
                "death": None,
                "thematic_quote": {
                    "theme": "deception",
                    "quote": "I am not what I am (Act 1, Scene 1)"
                },
                "speech": {
                    "emotion": "malice",
                    "text": "Othello’s trust I’ll twist with cunning art, / And plant the seeds of doubt within his heart. / No cause have I but sport in others’ pain, / Their ruin is my joy, their loss my gain."
                },
                "personality_quote": "But I will wear my heart upon my sleeve / For daws to peck at (Act 1, Scene 1)"
            },
            "Cassio": {
                "personality": "His loyalty and naivety make him a pawn in Iago’s schemes, costing him his reputation.",
                "death": None,
                "thematic_quote": {
                    "theme": "honor",
                    "quote": "Reputation, reputation, reputation! O, I have lost my reputation! (Act 2, Scene 3)"
                },
                "speech": {
                    "emotion": "honor",
                    "text": "O honor lost, my name is stain’d with shame, / Through no foul deed, yet I must bear the blame. / I’ll serve my lord with truth to mend this wrong, / And prove my heart is loyal, pure, and strong."
                },
                "personality_quote": "I have lost the immortal part of myself, and what remains is bestial (Act 2, Scene 3)"
            },
            "Emilia": {
                "personality": "Her loyalty to Desdemona and growing defiance lead her to expose Iago, at great cost.",
                "death": "Emilia’s murder by Iago after revealing the truth emphasizes her courage and the tragedy’s scope.",
                "thematic_quote": {
                    "theme": "truth",
                    "quote": "Let heaven and men and devils, let them all, / All, all, cry shame against me, yet I’ll speak (Act 5, Scene 2)"
                },
                "speech": {
                    "emotion": "truth",
                    "text": "No longer will I hold my tongue in fear, / The truth of Iago’s crimes shall now appear. / For Desdemona’s sake, I’ll speak, though death, / Awaits the one who dares to draw such breath."
                },
                "personality_quote": "But I do think it is their husbands’ faults / If wives do fall (Act 4, Scene 3)"
            }
        }
    },
    "Twelfth Night; Or, What You Will": {
        "category": "comedy",
        "central_theme": "The fluidity of love and identity.",
        "symbol_motif": {
            "name": "disguise",
            "meaning": "Represents the blurred lines of gender and affection, driving the romantic confusion."
        },
        "specific_line": {
            "act": 2,
            "scene": 5,
            "line": "If this were play’d upon a stage now, I could condemn it as an improbable fiction"
        },
        "poetic_line_topic": "love",
        "poetic_line": "Love’s tide doth sweep the heart to shores unknown.",
        "dramatic_quote": "If music be the food of love, play on (Orsino, Act 1, Scene 1).",
        "couplet": "In Illyria’s dance, love shifts with merry guise, / And hearts entwine beneath deception’s skies.",
        "character_data": {
            "Viola": {
                "personality": "Her wit and resilience lead her to disguise as Cesario, navigating love and chaos.",
                "death": None,
                "thematic_quote": {
                    "theme": "love",
                    "quote": "O time, thou must untangle this, not I; / It is too hard a knot for me t’untie! (Act 2, Scene 2)"
                },
                "speech": {
                    "emotion": "love",
                    "text": "O Orsino, my heart doth wear a man’s attire, / Yet burns for thee with love’s unspoken fire. / In Cesario’s guise, I hide my soul’s true plea, / When will my love find voice to set me free?"
                },
                "personality_quote": "I am not what I am (Act 3, Scene 1)"
            },
            "Orsino": {
                "personality": "His romantic melancholy leads him to pine for Olivia until he discovers Viola’s love.",
                "death": None,
                "thematic_quote": {
                    "theme": "love",
                    "quote": "If music be the food of love, play on (Act 1, Scene 1)"
                },
                "speech": {
                    "emotion": "love",
                    "text": "O love, thou fickle tide that stirs my soul, / Why dost thou make of me a longing fool? / Yet in Olivia’s eyes, or Cesario’s grace, / I find the spark that sets my heart’s true place."
                },
                "personality_quote": "Give me excess of it, that, surfeiting, / The appetite may sicken, and so die (Act 1, Scene 1)"
            },
            "Olivia": {
                "personality": "Her grief and independence make her reject Orsino but fall for Viola’s disguise.",
                "death": None,
                "thematic_quote": {
                    "theme": "love",
                    "quote": "Love sought is good, but given unsought is better (Act 3, Scene 1)"
                },
                "speech": {
                    "emotion": "love",
                    "text": "O Cesario, thou hast charm’d my widow’d heart, / With gentle words that love’s sweet pangs impart. / Though duty bids me mourn, my soul takes flight, / To follow thee through love’s uncharted light."
                },
                "personality_quote": "What is decreed must be, and be this so (Act 1, Scene 5)"
            },
            "Malvolio": {
                "personality": "His pride and ambition lead him to fall for a cruel prank, resulting in humiliation.",
                "death": None,
                "thematic_quote": {
                    "theme": "ambition",
                    "quote": "Some are born great, some achieve greatness, and some have greatness thrust upon ’em (Act 2, Scene 5)"
                },
                "speech": {
                    "emotion": "ambition",
                    "text": "O greatness, now within my grasp I see, / Olivia’s love shall lift a man like me. / No steward low, but lord of this estate, / I’ll rise to heights where none shall scorn my fate."
                },
                "personality_quote": "I’ll be revenged on the whole pack of you (Act 5, Scene 1)"
            },
            "Feste": {
                "personality": "His wit and insight allow him to comment on the folly of others through song and jest.",
                "death": None,
                "thematic_quote": {
                    "theme": "folly",
                    "quote": "Better a witty fool than a foolish wit (Act 1, Scene 5)"
                },
                "speech": {
                    "emotion": "folly",
                    "text": "O mortals, how ye chase love’s fleeting jest, / With tangled hearts and follies manifest! / As Feste sings, take heed of wisdom’s call, / Lest love’s sweet game make fools of one and all."
                },
                "personality_quote": "The rain it raineth every day (Act 5, Scene 1)"
            }
        }
    },
    "The Tempest": {
        "category": "comedy",
        "central_theme": "Forgiveness and redemption.",
        "symbol_motif": {
            "name": "the tempest",
            "meaning": "Represents Prospero’s power and the chaos he orchestrates to achieve justice."
        },
        "specific_line": {
            "act": 1,
            "scene": 2,
            "line": "Be collected: / No more amazement: tell your piteous heart / There’s no harm done"
        },
        "poetic_line_topic": "forgiveness",
        "poetic_line": "Forgiveness heals the wounds that time hath wrought.",
        "dramatic_quote": "We are such stuff / As dreams are made on (Prospero, Act 4, Scene 1).",
        "couplet": "On this isle, forgiveness mends the broken heart, / And magic weaves a world where all may start.",
        "character_data": {
            "Prospero": {
                "personality": "His wisdom and control lead him to use magic for justice, ultimately choosing forgiveness.",
                "death": None,
                "thematic_quote": {
                    "theme": "forgiveness",
                    "quote": "The rarer action is / In virtue than in vengeance (Act 5, Scene 1)"
                },
                "speech": {
                    "emotion": "forgiveness",
                    "text": "O ye who wrong’d me, see my mercy now, / No vengeance seek I, but to peace I vow. / With magic’s art, I’ve brought ye to this shore, / To mend old wounds and hate forevermore."
                },
                "personality_quote": "My library / Was dukedom large enough (Act 1, Scene 2)"
            },
            "Miranda": {
                "personality": "Her innocence and compassion lead her to love Ferdinand and seek harmony.",
                "death": None,
                "thematic_quote": {
                    "theme": "wonder",
                    "quote": "O brave new world, / That has such people in’t! (Act 5, Scene 1)"
                },
                "speech": {
                    "emotion": "wonder",
                    "text": "O world unknown, where beauty doth abide, / Such noble hearts within this isle reside! / My soul doth leap to greet this wondrous sight, / Where love and kindness banish all our night."
                },
                "personality_quote": "How beauteous mankind is! (Act 5, Scene 1)"
            },
            "Ariel": {
                "personality": "His loyalty and ethereal nature make him Prospero’s obedient servant, yearning for freedom.",
                "death": None,
                "thematic_quote": {
                    "theme": "freedom",
                    "quote": "Where the bee sucks, there suck I (Act 5, Scene 1)"
                },
                "speech": {
                    "emotion": "freedom",
                    "text": "O liberty, thou prize I long to claim, / To soar where winds and stars do call my name. / Yet Prospero’s will I serve with faithful heart, / Till freedom’s song shall bid me to depart."
                },
                "personality_quote": "All hail, great master! Grave sir, hail! (Act 1, Scene 2)"
            },
            "Caliban": {
                "personality": "His resentment and primal nature lead him to rebel against Prospero, seeking freedom.",
                "death": None,
                "thematic_quote": {
                    "theme": "oppression",
                    "quote": "This island’s mine, by Sycorax my mother (Act 1, Scene 2)"
                },
                "speech": {
                    "emotion": "resentment",
                    "text": "O Prospero, thou thief of mine own land, / With chains of magic, thou dost bind my hand. / This isle was mine, yet now I serve thy will, / My heart cries out for freedom’s hope to fill."
                },
                "personality_quote": "You taught me language; and my profit on’t / Is, I know how to curse (Act 1, Scene 2)"
            },
            "Ferdinand": {
                "personality": "His sincerity and love for Miranda drive him to endure Prospero’s trials.",
                "death": None,
                "thematic_quote": {
                    "theme": "love",
                    "quote": "Here’s my hand (Act 3, Scene 1)"
                },
                "speech": {
                    "emotion": "love",
                    "text": "O Miranda, thou art my heart’s delight, / For thee I’d labor through the darkest night. / No task too great, no storm can break my vow, / Our love shall bloom where magic reigns as now."
                },
                "personality_quote": "I / Beyond all limit of what else i’ the world / Do love, prize, honour you (Act 3, Scene 1)"
            }
        }
    },
    "Julius Caesar": {
        "category": "tragedy",
        "central_theme": "The conflict between loyalty and ambition.",
        "symbol_motif": {
            "name": "the ides of March",
            "meaning": "Represents the inevitability of fate and the consequences of defying warnings."
        },
        "specific_line": {
            "act": 3,
            "scene": 2,
            "line": "Friends, Romans, countrymen, lend me your ears"
        },
        "poetic_line_topic": "power",
        "poetic_line": "Power’s crown doth tempt the soul to direst fall.",
        "dramatic_quote": "Et tu, Brute? Then fall, Caesar! (Caesar, Act 3, Scene 1).",
        "couplet": "Ambition’s blade doth cut the loyal heart, / And Rome’s great glory tears itself apart.",
        "character_data": {
            "Julius Caesar": {
                "personality": "His ambition and confidence lead him to ignore warnings, resulting in his assassination.",
                "death": "Caesar’s assassination sparks civil war, destabilizing Rome and fulfilling the soothsayer’s prophecy.",
                "thematic_quote": {
                    "theme": "power",
                    "quote": "Cowards die many times before their deaths; / The valiant never taste of death but once (Act 2, Scene 2)"
                },
                "speech": {
                    "emotion": "power",
                    "text": "O Rome, my strength shall raise thee to the skies, / No omen dims the glory in mine eyes. / Let lesser men beware my boundless might, / For Caesar’s will shall shape the world’s new light."
                },
                "personality_quote": "I could be well moved, if I were as you; / If I could pray to move, prayers would move me (Act 3, Scene 1)"
            },
            "Brutus": {
                "personality": "His honor and idealism lead him to join the conspiracy, believing it saves Rome.",
                "death": "Brutus’s suicide after defeat reflects his failure to prevent tyranny, sealing the tragedy.",
                "thematic_quote": {
                    "theme": "loyalty",
                    "quote": "Not that I loved Caesar less, but that I loved Rome more (Act 3, Scene 2)"
                },
                "speech": {
                    "emotion": "loyalty",
                    "text": "O Rome, for thee I raise this heavy hand, / To guard thy freedom from a tyrant’s band. / Though Caesar’s blood doth stain my honor’d name, / My heart beats true for Rome’s eternal flame."
                },
                "personality_quote": "For let the gods so speed me as I love / The name of honour more than I fear death (Act 1, Scene 2)"
            },
            "Cassius": {
                "personality": "His cunning and envy drive him to recruit Brutus, manipulating the conspiracy.",
                "death": "Cassius’s suicide, mistakenly believing defeat, accelerates the conspirators’ collapse.",
                "thematic_quote": {
                    "theme": "ambition",
                    "quote": "The fault, dear Brutus, is not in our stars, / But in ourselves, that we are underlings (Act 1, Scene 2)"
                },
                "speech": {
                    "emotion": "ambition",
                    "text": "Why should Caesar tower o’er men like me? / I’ll pull him down and set our spirits free. / With Brutus’ aid, we’ll carve a nobler state, / And rise where fate denies us Caesar’s plate."
                },
                "personality_quote": "Men at some time are masters of their fates (Act 1, Scene 2)"
            },
            "Mark Antony": {
                "personality": "His loyalty and rhetorical skill lead him to avenge Caesar, inciting war.",
                "death": None,
                "thematic_quote": {
                    "theme": "revenge",
                    "quote": "Cry ‘Havoc,’ and let slip the dogs of war (Act 3, Scene 1)"
                },
                "speech": {
                    "emotion": "revenge",
                    "text": "O Caesar, slain by traitors’ coward hands, / Thy blood demands I rouse these Roman lands. / With words as swords, I’ll stir the people’s ire, / And burn these knaves in vengeance’s fire."
                },
                "personality_quote": "Friends, Romans, countrymen, lend me your ears (Act 3, Scene 2)"
            },
            "Portia": {
                "personality": "Her loyalty and strength lead her to support Brutus, but her anxiety contributes to her demise.",
                "death": "Portia’s suicide reflects the personal toll of Brutus’s political actions.",
                "thematic_quote": {
                    "theme": "loyalty",
                    "quote": "Think you I am no stronger than my sex? (Act 2, Scene 1)"
                },
                "speech": {
                    "emotion": "loyalty",
                    "text": "O Brutus, my heart doth bear thy secret load, / Though shadows fall upon our troubled road. / As wife, I stand to share thy honor’s fight, / And guard thy soul through Rome’s impending night."
                },
                "personality_quote": "I have a man’s mind, but a woman’s might (Act 2, Scene 1)"
            }
        }
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

def find_quote(quotes, speaker, theme=None, act=None, scene=None):
    for quote in quotes:
        if speaker and quote.get("speaker") != speaker:
            continue
        if theme and quote.get("explanation", "").lower().find(theme.lower()) == -1:
            continue
        if act and scene and (quote.get("act") != act or quote.get("scene") != scene):
            continue
        return quote.get("quote")
    return None

# === MAIN FUNCTION ===
def generate_thematic_qa():
    qa_dataset = []
    factual_data = load_json(FACTUAL_FILE)
    quote_data = load_json(QUOTE_FILE)
    if not factual_data or not quote_data:
        logger.error("No data loaded, exiting")
        return 0

    # Filter for requested plays
    factual_plays = [play for play in factual_data if play.get("title") in PLAYS]
    quote_plays = [play for play in quote_data if play.get("title") in PLAYS]
    factual_lookup = {play.get("title", ""): play for play in factual_plays}
    quote_lookup = {play.get("title", ""): play.get("famous_quotes", []) for play in quote_plays}
    missing_plays = [title for title in PLAYS if title not in factual_lookup or title not in quote_lookup]
    if missing_plays:
        logger.warning(f"Missing plays in factual.json or quote.json: {missing_plays}")

    for title in PLAYS:
        if title not in factual_lookup or title not in quote_lookup:
            logger.warning(f"Skipping {title}: missing data")
            continue

        logger.info(f"Processing play: {title}")
        play_data = RESPONSE_DATA.get(title, {})
        main_characters = factual_lookup[title].get("main_characters", [])
        quotes = quote_lookup[title]
        themes = [t.get("theme", "love") for t in factual_lookup[title].get("themes", [{"theme": "love"}])]

        # Question 1: Central theme
        question_template = random.choice(QUESTION_TEMPLATES["central_theme"])
        question = question_template(title)
        response = play_data.get("central_theme", "")
        if response:
            qa_dataset.append({"prompt": question, "response": response})

        # Question 2: Character personality (per character)
        for character in main_characters:
            question_template = random.choice(QUESTION_TEMPLATES["character_personality"])
            question = question_template(character, title)
            response = play_data.get("character_data", {}).get(character, {}).get("personality", "")
            if response:
                qa_dataset.append({"prompt": question, "response": response})

        # Question 3: Symbol/motif
        question_template = random.choice(QUESTION_TEMPLATES["symbol_motif"])
        symbol = play_data.get("symbol_motif", {}).get("name", "")
        question = question_template(symbol, title)
        response = play_data.get("symbol_motif", {}).get("meaning", "")
        if response:
            qa_dataset.append({"prompt": question, "response": response})

        # Question 4: Character death (per character, if applicable)
        for character in main_characters:
            death_response = play_data.get("character_data", {}).get(character, {}).get("death")
            if death_response:
                question_template = random.choice(QUESTION_TEMPLATES["character_death"])
                question = question_template(character, title)
                qa_dataset.append({"prompt": question, "response": death_response})

        # Question 5: Thematic quote (per character)
        for character in main_characters:
            theme = random.choice(themes)
            question_template = random.choice(QUESTION_TEMPLATES["thematic_quote"])
            question = question_template(character, title, theme)
            quote = play_data.get("character_data", {}).get(character, {}).get("thematic_quote", {}).get("quote", "")
            if not quote:
                quote = find_quote(quotes, character, theme=theme)
            response = quote or f"No quote by {character} about {theme} found in {title}."
            qa_dataset.append({"prompt": question, "response": response})

        # Question 6: Character speech (per character)
        for character in main_characters:
            emotion = play_data.get("character_data", {}).get(character, {}).get("speech", {}).get("emotion", "love")
            question_template = random.choice(QUESTION_TEMPLATES["character_speech"])
            question = question_template(character, title, emotion)
            response = play_data.get("character_data", {}).get(character, {}).get("speech", {}).get("text", "")
            if response:
                qa_dataset.append({"prompt": question, "response": response})

        # Question 7: Specific line
        question_template = random.choice(QUESTION_TEMPLATES["specific_line"])
        act = play_data.get("specific_line", {}).get("act", 1)
        scene = play_data.get("specific_line", {}).get("scene", 1)
        question = question_template(act, scene, title)
        response = play_data.get("specific_line", {}).get("line", "")
        if not response:
            quote = find_quote(quotes, None, act=act, scene=scene)
            response = quote or f"No notable line found in Act {act}, Scene {scene} of {title}."
        qa_dataset.append({"prompt": question, "response": response})

        # Question 8: Poetic line
        question_template = random.choice(QUESTION_TEMPLATES["poetic_line"])
        topic = play_data.get("poetic_line_topic", "fate")
        question = question_template(title, topic)
        response = play_data.get("poetic_line", "")
        if response:
            qa_dataset.append({"prompt": question, "response": response})

        # Question 9: Dramatic quote
        question_template = random.choice(QUESTION_TEMPLATES["dramatic_quote"])
        question = question_template(title)
        response = play_data.get("dramatic_quote", "")
        if response:
            qa_dataset.append({"prompt": question, "response": response})

        # Question 10: Couplet
        question_template = random.choice(QUESTION_TEMPLATES["couplet"])
        question = question_template(title)
        response = play_data.get("couplet", "")
        if response:
            qa_dataset.append({"prompt": question, "response": response})

        # Question 11: Personality quote (per character)
        for character in main_characters:
            question_template = random.choice(QUESTION_TEMPLATES["personality_quote"])
            question = question_template(title, character)
            quote = play_data.get("character_data", {}).get(character, {}).get("personality_quote", "")
            if not quote:
                quote = find_quote(quotes, character)
            response = quote or f"No quote reflecting {character}’s personality found in {title}."
            qa_dataset.append({"prompt": question, "response": response})

    save_jsonl(qa_dataset, OUTPUT_FILE)
    logger.info(f"✅ Generated {len(qa_dataset)} prompt-response pairs and saved to {OUTPUT_FILE}")
    return len(qa_dataset)

# === RUN ===
if __name__ == "__main__":
    total_new = generate_thematic_qa()
    print(f"Generated {total_new} new prompt-response pairs")