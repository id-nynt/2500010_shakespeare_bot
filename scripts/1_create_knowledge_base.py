# src/preprocessing/create_knowledge_base.py
import json
import os

# Comprehensive information for all major Shakespeare plays
PLAY_INFO = {
    "alls_well_that_ends_well": {
        "title": "All's Well That Ends Well",
        "full_title": "All's Well That Ends Well",
        "category": "comedy",
        "year": "1604-1605",
        "setting": "France and Italy",
        "main_characters": ["Helena", "Bertram", "Countess of Roussillon", "King of France", "Parolles", "Diana"],
        "character_descriptions": [
            {"name": "Helena", "description": "Physician's daughter, loves Bertram, heals king"},
            {"name": "Bertram", "description": "Nobleman, reluctant husband of Helena"},
            {"name": "Countess of Roussillon", "description": "Bertram's mother, supports Helena"},
            {"name": "King of France", "description": "Ruler, healed by Helena, grants marriage"},
            {"name": "Parolles", "description": "Bertram's boastful, disloyal companion"},
            {"name": "Diana", "description": "Young woman, aids Helena's plan"}
        ],
        "themes": [
            {
                "theme": "Love",
                "theme_explanation": "The play explores unrequited love through Helena’s persistent pursuit of Bertram, highlighting the complexities of affection and devotion despite social and personal obstacles."
            },
            {
                "theme": "Deception",
                "theme_explanation": "Deception drives the plot, as Helena uses cunning strategies, like the bed-trick, to win Bertram, while Parolles’s false bravado exposes his disloyalty."
            },
            {
                "theme": "Gender roles",
                "theme_explanation": "Helena defies traditional gender roles by actively pursuing her desires and using her intelligence to achieve her goals, challenging societal expectations of women."
            },
            {
                "theme": "Class and merit",
                "theme_explanation": "The play examines social hierarchy, as Helena, a low-born woman, proves her worth through merit, ultimately overcoming class barriers to marry Bertram."
            },
            {
                "theme": "Healing",
                "theme_explanation": "Healing is central, as Helena’s medical skill cures the King, symbolizing her ability to mend personal and social divides through wisdom and action."
            }
        ],
        "famous_quotes": [
            {
                "quote": "All's well that ends well.",
                "speaker": "Helena",
                "act": 4,
                "scene": 4,
                "explanation": "Helena expresses optimism that her plan to win Bertram's love will succeed, reflecting the play's title and its theme of resolving difficulties through perseverance."
            },
            {
                "quote": "Our remedies oft in ourselves do lie.",
                "speaker": "Helena",
                "act": 1,
                "scene": 1,
                "explanation": "Helena asserts that solutions to problems often come from within, highlighting her initiative and resourcefulness in pursuing her goals despite her low social status."
            }
        ],
        "summary": "Helena, a physician's daughter, is in love with the nobleman Bertram. After curing the King of France, she is granted the right to marry anyone she chooses and selects Bertram, who refuses her and flees to war. Through intelligence and perseverance, Helena ultimately wins Bertram's love."
    },
    "antony_and_cleopatra": {
        "title": "Antony and Cleopatra",
        "full_title": "The Tragedy of Antony and Cleopatra",
        "category": "tragedy",
        "year": "1606",
        "setting": "Rome and Egypt",
        "main_characters": ["Mark Antony", "Cleopatra", "Octavius Caesar", "Enobarbus", "Charmian"],
        "character_descriptions": [
            {"name": "Mark Antony", "description": "Roman general, lover of Cleopatra"},
            {"name": "Cleopatra", "description": "Queen of Egypt, Antony's lover"},
            {"name": "Octavius Caesar", "description": "Roman leader, Antony's rival"},
            {"name": "Enobarbus", "description": "Antony's loyal officer, defects later"},
            {"name": "Charmian", "description": "Cleopatra's devoted attendant"}
        ],
        "themes": [
            {
                "theme": "Love vs. Duty",
                "theme_explanation": "Antony’s passionate love for Cleopatra conflicts with his duties as a Roman general, leading to his downfall as he prioritizes desire over responsibility."
            },
            {
                "theme": "Politics",
                "theme_explanation": "The play explores power struggles and political maneuvering, as Antony’s alliance with Cleopatra undermines his position against Octavius Caesar’s rising dominance."
            },
            {
                "theme": "Honor",
                "theme_explanation": "Honor shapes Antony’s identity as a warrior, but his love for Cleopatra and eventual defeat challenge his sense of Roman virtue and loyalty."
            },
            {
                "theme": "East vs. West",
                "theme_explanation": "The cultural clash between Rome’s discipline and Egypt’s sensuality highlights the tension between Antony’s Roman identity and his immersion in Cleopatra’s world."
            }
        ],
        "famous_quotes": [
            {
                "quote": "Age cannot wither her, nor custom stale her infinite variety.",
                "speaker": "Enobarbus",
                "act": 2,
                "scene": 2,
                "explanation": "Enobarbus praises Cleopatra's timeless allure, emphasizing her captivating and ever-changing nature that keeps Antony enthralled."
            },
            {
                "quote": "I am dying, Egypt, dying.",
                "speaker": "Antony",
                "act": 4,
                "scene": 15,
                "explanation": "As Antony faces death, he addresses Cleopatra as 'Egypt,' reflecting their intertwined identities and his acceptance of their tragic fate."
            }
        ],
        "summary": "The Roman general Mark Antony falls in love with Cleopatra, Queen of Egypt. Their relationship threatens his standing in Rome and leads to conflict with Octavius Caesar. Their tragic love story ends in double suicide as they are defeated in battle."
    },
    "comedy_of_errors": {
        "title": "The Comedy of Errors",
        "full_title": "The Comedy of Errors",
        "category": "comedy",
        "year": "1594",
        "setting": "Ephesus",
        "main_characters": ["Antipholus of Syracuse", "Antipholus of Ephesus", "Dromio of Syracuse", "Dromio of Ephesus", "Adriana", "Luciana"],
        "character_descriptions": [
            {"name": "Antipholus of Syracuse", "description": "Twin merchant, searches for brother"},
            {"name": "Antipholus of Ephesus", "description": "Twin merchant, resident of Ephesus"},
            {"name": "Dromio of Syracuse", "description": "Servant to Antipholus of Syracuse"},
            {"name": "Dromio of Ephesus", "description": "Servant to Antipholus of Ephesus"},
            {"name": "Adriana", "description": "Wife of Antipholus of Ephesus"},
            {"name": "Luciana", "description": "Adriana's sister, courted by Syracuse twin"}
        ],
        "themes": [
            {
                "theme": "Identity",
                "theme_explanation": "The play explores identity through the confusion caused by two sets of identical twins, questioning individuality and selfhood."
            },
            {
                "theme": "Mistaken identity",
                "theme_explanation": "Mistaken identities drive the comedic chaos, as characters confuse the twins, leading to humorous misunderstandings and eventual resolution."
            },
            {
                "theme": "Family",
                "theme_explanation": "The theme of family underscores the twins’ separation and their joyful reunion, emphasizing the bonds that endure despite chaos."
            },
            {
                "theme": "Reunion",
                "theme_explanation": "The play culminates in the reunion of the separated twins and their family, resolving the chaos and restoring harmony."
            },
            {
                "theme": "Chaos and confusion",
                "theme_explanation": "Chaos and confusion propel the plot, as mistaken identities create a whirlwind of comedic errors that test relationships and social order."
            }
        ],
        "famous_quotes": [
            {
                "quote": "Am I in earth, in heaven, or in hell?",
                "speaker": "Antipholus of Syracuse",
                "act": 2,
                "scene": 2,
                "explanation": "Antipholus expresses bewilderment amid the play's chaotic mistaken identities, capturing the surreal confusion caused by the twin mix-ups."
            }
        ],
        "summary": "Two sets of identical twins separated at birth create chaos when they all end up in the same city. Mistaken identities lead to wrongful beatings, arrests, and accusations until the family is joyfully reunited."
    },
    "hamlet": {
        "title": "Hamlet, Prince of Denmark",
        "full_title": "The Tragedy of Hamlet, Prince of Denmark",
        "category": "tragedy",
        "year": "1600-1601",
        "setting": "Elsinore Castle, Denmark",
        "main_characters": ["Hamlet", "Claudius", "Gertrude", "Ophelia", "Polonius", "Horatio", "Laertes", "Ghost of Hamlet's father"],
        "character_descriptions": [
            {"name": "Hamlet", "description": "Prince of Denmark, seeks revenge"},
            {"name": "Claudius", "description": "King of Denmark, Hamlet's uncle, murderer"},
            {"name": "Gertrude", "description": "Queen of Denmark, Hamlet's mother"},
            {"name": "Ophelia", "description": "Polonius's daughter, Hamlet's love interest"},
            {"name": "Polonius", "description": "King's advisor, Ophelia's father"},
            {"name": "Horatio", "description": "Hamlet's loyal friend and confidant"},
            {"name": "Laertes", "description": "Polonius's son, seeks vengeance"},
            {"name": "Ghost of Hamlet's father", "description": "Spirit, urges Hamlet to avenge murder"}
        ],
        "themes": [
            {
                "theme": "Revenge",
                "theme_explanation": "Hamlet’s quest to avenge his father’s murder by Claudius drives the plot, exploring the moral and psychological costs of vengeance."
            },
            {
                "theme": "Madness",
                "theme_explanation": "Hamlet’s feigned madness and Ophelia’s descent into genuine insanity highlight the thin line between sanity and madness under grief and betrayal."
            },
            {
                "theme": "Corruption",
                "theme_explanation": "Claudius’s murder and deceit taint Denmark, symbolizing moral and political decay that spreads through the court."
            },
            {
                "theme": "Death",
                "theme_explanation": "Death permeates the play, from the Ghost’s appearance to the tragic finale, prompting reflections on mortality and the afterlife."
            },
            {
                "theme": "The impossibility of certainty",
                "theme_explanation": "Hamlet struggles with uncertainty about the Ghost’s truth and his own actions, reflecting the difficulty of knowing truth in a deceptive world."
            },
            {
                "theme": "Action vs inaction",
                "theme_explanation": "Hamlet’s indecision and procrastination contrast with decisive actions of others, exploring the consequences of delayed revenge."
            },
            {
                "theme": "Appearance vs reality",
                "theme_explanation": "Deceptive appearances, like Claudius’s false virtue and Hamlet’s feigned madness, underscore the challenge of discerning truth from pretense."
            }
        ],
        "famous_quotes": [
            {
                "quote": "To be, or not to be, that is the question",
                "speaker": "Hamlet",
                "act": 3,
                "scene": 1,
                "explanation": "Hamlet contemplates life, death, and suicide in this soliloquy, weighing the merits of enduring suffering versus ending his existence."
            },
            {
                "quote": "The lady doth protest too much, methinks",
                "speaker": "Gertrude",
                "act": 3,
                "scene": 2,
                "explanation": "Gertrude comments on the player's exaggerated performance, unknowingly highlighting the theme of deceptive appearances in the play."
            },
            {
                "quote": "Though this be madness, yet there is method in't",
                "speaker": "Polonius",
                "act": 2,
                "scene": 2,
                "explanation": "Polonius observes that Hamlet's apparent madness may be purposeful, hinting at Hamlet's strategic feigned insanity."
            },
            {
                "quote": "There are more things in heaven and earth, Horatio, than are dreamt of in your philosophy",
                "speaker": "Hamlet",
                "act": 1,
                "scene": 5,
                "explanation": "Hamlet suggests to Horatio that the supernatural (like the Ghost) exceeds rational understanding, opening the play's exploration of mystery."
            },
            {
                "quote": "Brevity is the soul of wit",
                "speaker": "Polonius",
                "act": 2,
                "scene": 2,
                "explanation": "Ironically, the verbose Polonius praises concise speech, underscoring his own foolishness and the play's use of wit."
            },
            {
                "quote": "What a piece of work is a man!",
                "speaker": "Hamlet",
                "act": 2,
                "scene": 2,
                "explanation": "Hamlet marvels at humanity’s potential yet despairs at its flaws, highlighting his conflicted view of existence."
            },
            {
                "quote": "There is nothing either good or bad, but thinking makes it so",
                "speaker": "Hamlet",
                "act": 2,
                "scene": 2,
                "explanation": "Hamlet suggests that perception shapes morality, revealing his philosophical depth and mental turmoil."
            },
            {
                "quote": "Get thee to a nunnery.",
                "speaker": "Hamlet",
                "act": 3,
                "scene": 1,
                "explanation": "Hamlet, in a fit of despair and misogyny, urges Ophelia to become a nun, either to avoid bearing sinful children or to protect her from the corruption of the world."
            },
            {
                "quote": "Frailty, thy name is woman.",
                "speaker": "Hamlet",
                "act": 1,
                "scene": 2,
                "explanation": "Hamlet expresses his profound disappointment and anger at his mother's hasty marriage to his uncle, generalizing his frustration to all women as inherently weak or fickle."
            },
            {
                "quote": "To thine own self be true, and it must follow, as the night the day, thou canst not then be false to any man.",
                "speaker": "Polonius",
                "act": 1,
                "scene": 3,
                "explanation": "Polonius advises his son Laertes to be authentic and honest with himself, implying that self-integrity is the foundation for being truthful and honorable to others."
            },
            {
                "quote": "Neither a borrower nor a lender be; for loan oft loses both itself and friend, and borrowing dulls the edge of husbandry.",
                "speaker": "Polonius",
                "act": 1,
                "scene": 3,
                "explanation": "Polonius advises Laertes against borrowing or lending money, arguing that it can lead to financial loss and strain friendships, and that borrowing can hinder one's ability to manage their own affairs responsibly."
            },
            {
                "quote": "We know what we are, but know not what we may be.",
                "speaker": "Ophelia",
                "act": 4,
                "scene": 5,
                "explanation": "Ophelia, in her madness, speaks of the limitations of human self-knowledge and the unpredictable nature of the future, highlighting her loss of control and understanding."
            }
        ],
        "summary": "The play depicts Prince Hamlet's revenge against his uncle Claudius, who murdered Hamlet's father, the king, and then took the throne and married Hamlet's mother. Throughout the play, Hamlet struggles with his perceived duty to avenge his father's death while wrestling with moral questions about murder, suicide, and the afterlife."
    },
    "macbeth": {
        "title": "Macbeth",
        "full_title": "The Tragedy of Macbeth",
        "category": "tragedy",
        "year": "1606",
        "setting": "Scotland and England",
        "main_characters": ["Macbeth", "Lady Macbeth", "Three Witches", "Banquo", "Macduff", "Duncan", "Malcolm"],
        "character_descriptions": [
            {"name": "Macbeth", "description": "Scottish general, usurps throne, becomes king"},
            {"name": "Lady Macbeth", "description": "Macbeth's wife, drives his ambition"},
            {"name": "Three Witches", "description": "Supernatural beings, prophesy Macbeth's fate"},
            {"name": "Banquo", "description": "Macbeth's friend, general, later murdered"},
            {"name": "Macduff", "description": "Scottish noble, avenges Duncan's murder"},
            {"name": "Duncan", "description": "King of Scotland, murdered by Macbeth"},
            {"name": "Malcolm", "description": "Duncan's son, rightful heir"}
        ],
        "themes": [
            {
                "theme": "Ambition",
                "theme_explanation": "Macbeth’s unchecked ambition, spurred by the witches’ prophecy and Lady Macbeth, drives him to murder and tyranny, leading to his downfall."
            },
            {
                "theme": "Power",
                "theme_explanation": "The pursuit and corruption of power dominate the play, as Macbeth’s rise to the throne through murder destabilizes Scotland and his own morality."
            },
            {
                "theme": "Fate",
                "theme_explanation": "The witches’ prophecies suggest a predetermined fate, yet Macbeth’s choices in response to them highlight the interplay between destiny and free will."
            },
            {
                "theme": "Violence",
                "theme_explanation": "Violence escalates as Macbeth’s ambition fuels a cycle of murders, from Duncan to Banquo, culminating in widespread chaos and retribution."
            },
            {
                "theme": "Nature vs the unnatural",
                "theme_explanation": "Macbeth’s unnatural acts, like regicide, disrupt the natural order, reflected in omens and chaos that plague Scotland until balance is restored."
            },
            {
                "theme": "Manhood",
                "theme_explanation": "The play explores manhood through Lady Macbeth’s manipulation of Macbeth’s masculinity, equating courage with violent ambition, which leads to their ruin."
            },
            {
                "theme": "Guilt and conscience",
                "theme_explanation": "Guilt torments Macbeth and Lady Macbeth, with hallucinations and sleepwalking revealing their consciences’ struggle against their crimes."
            }
        ],
        "famous_quotes": [
            {
                "quote": "Fair is foul, and foul is fair",
                "speaker": "Three Witches",
                "act": 1,
                "scene": 1,
                "explanation": "The witches set the play's tone, suggesting a world where moral boundaries are blurred and appearances deceive."
            },
            {
                "quote": "Out, damned spot! Out, I say!",
                "speaker": "Lady Macbeth",
                "act": 5,
                "scene": 1,
                "explanation": "Lady Macbeth's sleepwalking reveals her guilt over Duncan's murder, as she imagines bloodstains she cannot wash away."
            },
            {
                "quote": "Life's but a walking shadow, a poor player, that struts and frets his hour upon the stage, and then is heard no more",
                "speaker": "Macbeth",
                "act": 5,
                "scene": 5,
                "explanation": "Macbeth laments life's fleeting and meaningless nature upon learning of his wife's death, reflecting his despair."
            },
            {
                "quote": "Something wicked this way comes",
                "speaker": "Second Witch",
                "act": 4,
                "scene": 1,
                "explanation": "The witches sense Macbeth's approach, foreshadowing his deepening evil as he seeks their prophecies."
            },
            {
                "quote": "Double, double toil and trouble; Fire burn, and cauldron bubble",
                "speaker": "Three Witches",
                "act": 4,
                "scene": 1,
                "explanation": "The witches' incantation as they brew a spell enhances the play's supernatural atmosphere and foreshadows chaos."
            },
            {
                "quote": "Out, out, brief candle! Life's but a walking shadow",
                "speaker": "Macbeth",
                "act": 5,
                "scene": 5,
                "explanation": "Macbeth laments life’s futility upon his wife’s death, reflecting his despair and nihilism."
            },
            {
                "quote": "Is this a dagger which I see before me?",
                "speaker": "Macbeth",
                "act": 2,
                "scene": 1,
                "explanation": "Macbeth hallucinates a dagger before murdering Duncan, revealing his guilt and psychological torment."
            },
        ],
        "summary": "Macbeth, a Scottish general, receives a prophecy from three witches that he will become King of Scotland. Consumed by ambition and spurred to action by his wife, Macbeth murders King Duncan and seizes the throne. His reign is marked by tyranny and further bloodshed as guilt and paranoia consume the couple, ultimately leading to their downfall."
    },
    "romeo_and_juliet": {
        "title": "Romeo and Juliet",
        "full_title": "The Tragedy of Romeo and Juliet",
        "category": "tragedy",
        "year": "1595-1596",
        "setting": "Verona, Italy",
        "main_characters": ["Romeo Montague", "Juliet Capulet", "Mercutio", "Tybalt", "Friar Lawrence", "Nurse", "Benvolio", "Paris"],
        "character_descriptions": [
            {"name": "Romeo Montague", "description": "Montague heir, loves Juliet"},
            {"name": "Juliet Capulet", "description": "Capulet daughter, loves Romeo"},
            {"name": "Mercutio", "description": "Romeo's friend, kinsman to Prince"},
            {"name": "Tybalt", "description": "Juliet's cousin, aggressive Capulet"},
            {"name": "Friar Lawrence", "description": "Priest, marries Romeo and Juliet"},
            {"name": "Nurse", "description": "Juliet's caretaker, aids lovers"},
            {"name": "Benvolio", "description": "Romeo's cousin, seeks peace"},
            {"name": "Paris", "description": "Nobleman, seeks to marry Juliet"}
        ],
        "themes": [
            {
                "theme": "Love",
                "theme_explanation": "Romeo and Juliet’s passionate love defies their families’ feud, driving the plot but leading to their tragic deaths."
            },
            {
                "theme": "Fate",
                "theme_explanation": "The lovers are described as ‘star-crossed,’ suggesting fate orchestrates their doomed romance through coincidences and misunderstandings."
            },
            {
                "theme": "Violence",
                "theme_explanation": "The feud between Montagues and Capulets fuels violent clashes, like Tybalt’s killing of Mercutio, escalating the tragedy."
            },
            {
                "theme": "Youth",
                "theme_explanation": "The impulsive, passionate actions of young characters like Romeo and Juliet highlight the recklessness and intensity of youth."
            },
            {
                "theme": "Family rivalry",
                "theme_explanation": "The deep-seated feud between the Montagues and Capulets creates the central conflict, preventing the lovers’ union until their deaths reconcile the families."
            },
            {
                "theme": "Individual vs society",
                "theme_explanation": "Romeo and Juliet’s love challenges societal norms and family expectations, highlighting the tension between personal desires and social constraints."
            },
            {
                "theme": "Time",
                "theme_explanation": "The rapid pace of events, unfolding over a few days, underscores how haste and impulsive decisions accelerate the lovers’ tragic fate."
            }
        ],
        "famous_quotes": [
            {
                "quote": "What's in a name? That which we call a rose by any other name would smell as sweet",
                "speaker": "Juliet",
                "act": 2,
                "scene": 2,
                "explanation": "Juliet argues that names (like Montague or Capulet) are meaningless, emphasizing love's transcendence over family feuds."
            },
            {
                "quote": "O Romeo, Romeo, wherefore art thou Romeo?",
                "speaker": "Juliet",
                "act": 2,
                "scene": 2,
                "explanation": "Juliet laments that Romeo is a Montague, her family's enemy, wishing he could shed his name for their love."
            },
            {
                "quote": "A plague o' both your houses!",
                "speaker": "Mercutio",
                "act": 3,
                "scene": 1,
                "explanation": "Dying, Mercutio curses the Montagues and Capulets, blaming their feud for his death and foreshadowing tragedy."
            },
            {
                "quote": "For never was a story of more woe than this of Juliet and her Romeo",
                "speaker": "Prince",
                "act": 5,
                "scene": 3,
                "explanation": "The Prince summarizes the tragic loss of Romeo and Juliet, highlighting the devastating cost of the feud."
            },
            {
                "quote": "Thus with a kiss I die",
                "speaker": "Romeo",
                "act": 5,
                "scene": 3,
                "explanation": "Romeo speaks his final words as he poisons himself beside Juliet, believing her dead, sealing their tragic fate."
            },
            {
                "quote": "My only love sprung from my only hate!",
                "speaker": "Juliet",
                "act": 1,
                "scene": 5,
                "explanation": "Juliet discovers Romeo is a Montague, expressing the painful irony that her love comes from her family’s enemy."
            },
            {
                "quote": "These violent delights have violent ends",
                "speaker": "Friar Laurence",
                "act": 2,
                "scene": 6,
                "explanation": "Friar Laurence warns that intense passions, like Romeo and Juliet’s, often lead to disastrous consequences."
            },
            {
                "quote": "What light through yonder window breaks.",
                "speaker": "Romeo",
                "act": 2,
                "scene": 2,
                "explanation": "Romeo, seeing Juliet appear on her balcony, likens her beauty to the dawn, signaling his intense adoration and the beginning of their passionate encounter."
            }
        ],
        "summary": "Set in Verona, Italy, this tragic love story follows Romeo and Juliet, young lovers from feuding families, the Montagues and Capulets. After meeting at a Capulet feast, they secretly marry, but a sequence of misunderstandings and violent confrontations leads to their tragic deaths, ultimately reconciling their feuding families."
    },
    "othello": {
        "title": "Othello, the Moor of Venice",
        "full_title": "The Tragedy of Othello, the Moor of Venice",
        "category": "tragedy",
        "year": "1603-1604",
        "setting": "Venice and Cyprus",
        "main_characters": ["Othello", "Desdemona", "Iago", "Cassio", "Emilia", "Roderigo", "Brabantio"],
        "character_descriptions": [
            {"name": "Othello", "description": "Moorish general, marries Desdemona"},
            {"name": "Desdemona", "description": "Othello's wife, Brabantio's daughter"},
            {"name": "Iago", "description": "Othello's ensign, orchestrates deception"},
            {"name": "Cassio", "description": "Othello's lieutenant, accused of affair"},
            {"name": "Emilia", "description": "Iago's wife, Desdemona's attendant"},
            {"name": "Roderigo", "description": "Desdemona's suitor, manipulated by Iago"},
            {"name": "Brabantio", "description": "Desdemona's father, Venetian senator"}
        ],
        "themes": [
            {
                "theme": "Jealousy",
                "theme_explanation": "Iago’s manipulation fuels Othello’s jealousy, transforming his love for Desdemona into a destructive force that leads to murder."
            },
            {
                "theme": "Prejudice",
                "theme_explanation": "Othello faces racial prejudice as a Moor in Venice, which Iago exploits to undermine his confidence and social standing."
            },
            {
                "theme": "Deception",
                "theme_explanation": "Iago’s deceitful schemes, including false accusations of Desdemona’s infidelity, drive the tragedy by exploiting trust and perception."
            },
            {
                "theme": "Love",
                "theme_explanation": "Othello and Desdemona’s genuine love is corrupted by Iago’s lies, illustrating how love can be vulnerable to external manipulation."
            },
            {
                "theme": "Trust",
                "theme_explanation": "The erosion of trust, as Othello believes Iago over Desdemona, underscores the fragility of relationships in the face of deceit."
            },
            {
                "theme": "Reputation",
                "theme_explanation": "Reputation is critical, as seen in Cassio’s despair over its loss and Othello’s fall, driven by Iago’s attacks on their honor."
            },
            {
                "theme": "Gender",
                "theme_explanation": "Gender dynamics shape the tragedy, with Desdemona and Emilia navigating a patriarchal society, their voices stifled until it’s too late."
            }
        ],
        "famous_quotes": [
            {
                "quote": "O, beware, my lord, of jealousy; It is the green-eyed monster which doth mock the meat it feeds on",
                "speaker": "Iago",
                "act": 3,
                "scene": 3,
                "explanation": "Iago warns Othello about jealousy, ironically planting the seeds of doubt that fuel Othello's tragic downfall."
            },
            {
                "quote": "Reputation, reputation, reputation! O, I have lost my reputation!",
                "speaker": "Cassio",
                "act": 2,
                "scene": 3,
                "explanation": "Cassio laments losing his honor after Iago's scheme, highlighting the play's emphasis on reputation's fragility."
            },
            {
                "quote": "She loved me for the dangers I had passed, and I loved her that she did pity them",
                "speaker": "Othello",
                "act": 1,
                "scene": 3,
                "explanation": "Othello describes how Desdemona fell in love with his adventurous tales, establishing their genuine bond."
            },
            {
                "quote": "Who steals my purse steals trash... But he that filches from me my good name robs me of that which not enriches him, and makes me poor indeed",
                "speaker": "Iago",
                "act": 3,
                "scene": 3,
                "explanation": "Iago manipulates Othello by emphasizing the value of reputation, which he plans to destroy in both Othello and Cassio."
            },
            {
                "quote": "I kissed thee ere I killed thee: no way but this, killing myself, to die upon a kiss",
                "speaker": "Othello",
                "act": 5,
                "scene": 2,
                "explanation": "Othello's final words express his love and remorse as he kills himself beside Desdemona, seeking redemption."
            },
            {
                "quote": "I am not what I am",
                "speaker": "Iago",
                "act": 1,
                "scene": 1,
                "explanation": "Iago reveals his duplicitous nature, setting the stage for his deceitful schemes against Othello."
            },
            {
                "quote": "Othello, the Moor of Venice",
                "speaker": "Othello",
                "act": 5,
                "scene": 2,
                "explanation": "Othello, realizing his tragic error in killing Desdemona, wishes to be remembered as both honorable and flawed."
            },
            {
                "quote": "Put out the light, and then put out the light",
                "speaker": "Othello",
                "act": 5,
                "scene": 2,
                "explanation": "Othello speaks of extinguishing a candle and Desdemona’s life, reflecting his tormented resolve to murder her."
            },
            {
                "quote": "I am one who loved not wisely but too well.",
                "speaker": "Othello",
                "act": 5,
                "scene": 2,
                "explanation": "Othello, in his final speech, acknowledges his tragic flaw: his love for Desdemona was intense but clouded by his jealousy and easily manipulated by Iago, leading to her wrongful death."
            }
        ],
        "summary": "Othello, a Moorish general in the Venetian army, marries Desdemona, inciting jealousy in his ensign Iago. Through elaborate deception, Iago convinces Othello that Desdemona has been unfaithful with his lieutenant Cassio. Consumed by jealousy, Othello murders Desdemona, only to discover Iago's treachery too late, leading to his suicide."
    },
    "king_lear": {
        "title": "The Tragedy of King Lear",
        "full_title": "The Tragedy of King Lear",
        "category": "tragedy",
        "year": "1605-1606",
        "setting": "Ancient Britain",
        "main_characters": ["King Lear", "Cordelia", "Goneril", "Regan", "Edmund", "Edgar", "Gloucester", "Kent", "Fool"],
        "character_descriptions": [
            {"name": "King Lear", "description": "King of Britain, divides kingdom"},
            {"name": "Cordelia", "description": "Lear's youngest daughter, disowned, loyal"},
            {"name": "Goneril", "description": "Lear's eldest daughter, betrays him"},
            {"name": "Regan", "description": "Lear's middle daughter, betrays him"},
            {"name": "Edmund", "description": "Gloucester's illegitimate son, schemes power"},
            {"name": "Edgar", "description": "Gloucester's legitimate son, disguises himself"},
            {"name": "Gloucester", "description": "Earl, betrayed by Edmund"},
            {"name": "Kent", "description": "Loyal nobleman, serves Lear in disguise"},
            {"name": "Fool", "description": "Lear's jester, speaks truth"}
        ],
        "themes": [
            {
                "theme": "Justice",
                "theme_explanation": "The play questions whether justice prevails, as good characters like Cordelia suffer while evil ones like Edmund temporarily triumph."
            },
            {
                "theme": "Authority",
                "theme_explanation": "Lear’s division of his kingdom reveals the fragility of authority, as his loss of power leads to chaos and betrayal."
            },
            {
                "theme": "Family",
                "theme_explanation": "Family dynamics drive the tragedy, with Lear and Gloucester betrayed by their children, contrasting loyalty (Cordelia, Edgar) with deceit (Goneril, Regan, Edmund)."
            },
            {
                "theme": "Aging",
                "theme_explanation": "Lear’s aging and desire to relinquish power expose his vulnerability, leading to his loss of control and descent into madness."
            },
            {
                "theme": "Madness",
                "theme_explanation": "Lear’s descent into madness reflects his emotional turmoil, while Edgar’s feigned madness as Poor Tom highlights survival through disguise."
            },
            {
                "theme": "Betrayal",
                "theme_explanation": "Betrayal by Goneril, Regan, and Edmund against their fathers drives the tragedy, exposing the fragility of familial bonds."
            },
            {
                "theme": "Nature",
                "theme_explanation": "The storm symbolizes nature’s chaos, mirroring Lear’s inner turmoil and the unnatural betrayals disrupting the social order."
            },
            {
                "theme": "Sight and blindness",
                "theme_explanation": "Gloucester’s literal blindness parallels Lear’s figurative blindness to truth, as both fail to see their children’s true natures until it’s too late."
            }
        ],
        "famous_quotes": [
            {
                "quote": "How sharper than a serpent's tooth it is to have a thankless child!",
                "speaker": "King Lear",
                "act": 1,
                "scene": 4,
                "explanation": "Lear expresses anguish over his daughters' betrayal, highlighting the pain of familial ingratitude."
            },
            {
                "quote": "Nothing will come of nothing",
                "speaker": "King Lear",
                "act": 1,
                "scene": 1,
                "explanation": "Lear warns Cordelia that her refusal to flatter him will yield no reward, foreshadowing his misguided judgment."
            },
            {
                "quote": "I am a man more sinned against than sinning",
                "speaker": "King Lear",
                "act": 3,
                "scene": 2,
                "explanation": "Lear, in his madness, believes he suffers unjustly, reflecting his growing self-awareness amid chaos."
            },
            {
                "quote": "The worst is not, so long as we can say, 'This is the worst'",
                "speaker": "Edgar",
                "act": 4,
                "scene": 1,
                "explanation": "Edgar suggests that the ability to articulate suffering means one has not yet hit rock bottom, offering hope."
            },
            {
                "quote": "As flies to wanton boys are we to the gods; they kill us for their sport",
                "speaker": "Gloucester",
                "act": 4,
                "scene": 1,
                "explanation": "Gloucester laments human vulnerability, viewing life as subject to the whims of cruel, capricious gods."
            },
            {
                "quote": "Blow, winds, and crack your cheeks! Rage! Blow!",
                "speaker": "King Lear",
                "act": 3,
                "scene": 2,
                "explanation": "Lear rages against the storm, mirroring his inner turmoil and descent into madness amid betrayal."
            }
        ],
        "summary": "King Lear, planning to divide his kingdom among his three daughters, disowns Cordelia when she refuses to publicly quantify her love. The play follows Lear's descent into madness after his other daughters betray him. Meanwhile, the Earl of Gloucester is similarly deceived by his illegitimate son Edmund against his legitimate son Edgar. Both storylines culminate in profound tragedy."
    },
    "midsummer_nights_dream": {
        "title": "A Midsummer Night's Dream",
        "full_title": "A Midsummer Night's Dream",
        "category": "comedy",
        "year": "1595-1596",
        "setting": "Athens and nearby enchanted forest",
        "main_characters": ["Lysander", "Hermia", "Demetrius", "Helena", "Oberon", "Titania", "Puck", "Bottom", "Theseus", "Hippolyta"],
        "character_descriptions": [
            {"name": "Lysander", "description": "Athenian lover, pursues Hermia"},
            {"name": "Hermia", "description": "Athenian woman, loves Lysander"},
            {"name": "Demetrius", "description": "Athenian man, initially loves Hermia"},
            {"name": "Helena", "description": "Athenian woman, loves Demetrius"},
            {"name": "Oberon", "description": "Fairy king, manipulates lovers"},
            {"name": "Titania", "description": "Fairy queen, enchanted by Oberon"},
            {"name": "Puck", "description": "Oberon's mischievous fairy servant"},
            {"name": "Bottom", "description": "Weaver, transformed into donkey"},
            {"name": "Theseus", "description": "Duke of Athens, marries Hippolyta"},
            {"name": "Hippolyta", "description": "Amazon queen, Theseus's bride"}
        ],
        "themes": [
            {
                "theme": "Love",
                "theme_explanation": "Love’s irrational and transformative power drives the lovers’ entanglements, resolved through magical intervention and eventual harmony."
            },
            {
                "theme": "Magic",
                "theme_explanation": "Magic, wielded by Oberon and Puck, creates chaos and resolutions, shaping the lovers’ fates and blurring reality with fantasy."
            },
            {
                "theme": "Dreams vs reality",
                "theme_explanation": "The play blurs dreams and reality, as magical events and transformations leave characters questioning what is real."
            },
            {
                "theme": "Order vs chaos",
                "theme_explanation": "The forest’s magical chaos disrupts Athenian order, but resolutions restore harmony, reflecting the tension between structure and disorder."
            },
            {
                "theme": "Transformation",
                "theme_explanation": "Characters undergo literal (Bottom’s donkey form) and emotional transformations, highlighting love’s and magic’s power to change identities."
            },
            {
                "theme": "Art and imagination",
                "theme_explanation": "The play celebrates art and imagination through the lovers’ fantasies, the fairies’ magic, and the mechanicals’ comedic performance."
            }
        ],
        "famous_quotes": [
            {
                "quote": "The course of true love never did run smooth",
                "speaker": "Lysander",
                "act": 1,
                "scene": 1,
                "explanation": "Lysander comments on the obstacles lovers face, foreshadowing the romantic entanglements caused by magic."
            },
            {
                "quote": "Lord, what fools these mortals be!",
                "speaker": "Puck",
                "act": 3,
                "scene": 2,
                "explanation": "Puck mocks the lovers' confused passions, highlighting the comedic chaos of human emotions under fairy influence."
            },
            {
                "quote": "Love looks not with the eyes, but with the mind, and therefore is winged Cupid painted blind",
                "speaker": "Helena",
                "act": 1,
                "scene": 1,
                "explanation": "Helena reflects on love's irrational nature, explaining how perception shapes romantic desire."
            },
            {
                "quote": "Though she be but little, she is fierce",
                "speaker": "Helena",
                "act": 3,
                "scene": 2,
                "explanation": "Helena describes Hermia's spirited nature despite her small stature, capturing her tenacity in love and conflict."
            },
            {
                "quote": "I have had a most rare vision. I have had a dream, past the wit of man to say what dream it was",
                "speaker": "Bottom",
                "act": 4,
                "scene": 1,
                "explanation": "Bottom struggles to articulate his magical transformation, blending comedy with the play's dreamlike quality."
            },
            {
                "quote": "I am amazed and know not what to say",
                "speaker": "Hermia",
                "act": 3,
                "scene": 2,
                "explanation": "Hermia expresses bewilderment at the lovers’ tangled relationships, capturing the play’s magical confusion."
            },
        ],
        "summary": "This enchanting comedy intertwines the adventures of four young Athenian lovers, a group of amateur actors, and fairies in a moonlit forest. As Theseus and Hippolyta prepare for their wedding, the fairy king and queen quarrel, leading to magical mischief when fairy Puck uses a love potion that complicates the lovers' relationships and transforms Bottom into a donkey."
    },
    "merchant_of_venice": {
        "title": "The Merchant of Venice",
        "full_title": "The Merchant of Venice",
        "category": "comedy",
        "year": "1596-1597",
        "setting": "Venice and Belmont, Italy",
        "main_characters": ["Antonio", "Shylock", "Portia", "Bassanio", "Jessica", "Lorenzo", "Gratiano", "Nerissa"],
        "character_descriptions": [
            {"name": "Antonio", "description": "Venetian merchant, borrows from Shylock"},
            {"name": "Shylock", "description": "Jewish moneylender, demands pound of flesh"},
            {"name": "Portia", "description": "Heiress of Belmont, disguises as lawyer"},
            {"name": "Bassanio", "description": "Portia's suitor, Antonio's friend"},
            {"name": "Jessica", "description": "Shylock's daughter, elopes with Lorenzo"},
            {"name": "Lorenzo", "description": "Jessica's lover, Bassanio's friend"},
            {"name": "Gratiano", "description": "Bassanio's friend, loves Nerissa"},
            {"name": "Nerissa", "description": "Portia's maid, marries Gratiano"}
        ],
        "themes": [
            {
                "theme": "Justice and mercy",
                "theme_explanation": "Portia’s courtroom plea for mercy over Shylock’s demand for justice highlights the tension between strict law and compassionate forgiveness."
            },
            {
                "theme": "Prejudice",
                "theme_explanation": "Shylock faces anti-Semitic prejudice, which fuels his vengeful actions, while the play critiques societal biases against outsiders."
            },
            {
                "theme": "Wealth",
                "theme_explanation": "Wealth drives the plot, from Antonio’s loan to Portia’s casket test, revealing its power to shape relationships and outcomes."
            },
            {
                "theme": "Love and friendship",
                "theme_explanation": "The bonds of love (Bassanio and Portia, Jessica and Lorenzo) and friendship (Antonio and Bassanio) are tested by sacrifice and loyalty."
            },
            {
                "theme": "Outsiders",
                "theme_explanation": "Shylock and Jessica, as Jews, navigate their outsider status in Venice, facing exclusion and seeking acceptance in a Christian society."
            },
            {
                "theme": "Appearance vs reality",
                "theme_explanation": "Deceptive appearances, like the caskets’ misleading exteriors and Portia’s lawyer disguise, underscore the challenge of discerning true value."
            }
        ],
        "famous_quotes": [
            {
                "quote": "The quality of mercy is not strained. It droppeth as the gentle rain from heaven upon the place beneath",
                "speaker": "Portia",
                "act": 4,
                "scene": 1,
                "explanation": "Portia, disguised as a lawyer, pleads for mercy in Shylock's trial, emphasizing its divine and voluntary nature."
            },
            {
                "quote": "If you prick us, do we not bleed? If you tickle us, do we not laugh? If you poison us, do we not die? And if you wrong us, shall we not revenge?",
                "speaker": "Shylock",
                "act": 3,
                "scene": 1,
                "explanation": "Shylock defends his humanity and desire for revenge, highlighting the universal traits shared despite prejudice."
            },
            {
                "quote": "All that glisters is not gold",
                "speaker": "Prince of Morocco",
                "act": 2,
                "scene": 7,
                "explanation": "This proverb, read from a casket, warns that appearances can deceive, a key theme in Portia's suitor trials."
            },
            {
                "quote": "I am a Jew. Hath not a Jew eyes?",
                "speaker": "Shylock",
                "act": 3,
                "scene": 1,
                "explanation": "Shylock asserts his shared humanity, challenging the anti-Semitic prejudice he faces in Venice."
            },
            {
                "quote": "Love is blind, and lovers cannot see",
                "speaker": "Jessica",
                "act": 2,
                "scene": 6,
                "explanation": "Jessica reflects on the irrationality of love, justifying her elopement with Lorenzo despite social barriers."
            }
        ],
        "summary": "When merchant Antonio borrows money from the Jewish moneylender Shylock to help his friend Bassanio woo Portia, he agrees to a pound of his own flesh as collateral. When Antonio defaults on the loan, Shylock demands his payment. Meanwhile, Portia tests her suitors with a challenge involving three caskets. The play explores themes of justice, mercy, and prejudice through its complex characters."
    },
    "julius_caesar": {
        "title": "Julius Caesar",
        "full_title": "The Tragedy of Julius Caesar",
        "category": "tragedy",
        "year": "1599",
        "setting": "Ancient Rome",
        "main_characters": ["Julius Caesar", "Brutus", "Cassius", "Mark Antony", "Octavius", "Calpurnia", "Portia"],
        "character_descriptions": [
            {"name": "Julius Caesar", "description": "Roman dictator, assassinated by conspirators"},
            {"name": "Brutus", "description": "Roman noble, leads conspiracy against Caesar"},
            {"name": "Cassius", "description": "Conspirator, manipulates Brutus against Caesar"},
            {"name": "Mark Antony", "description": "Caesar's ally, opposes conspirators"},
            {"name": "Octavius", "description": "Caesar's heir, joins Antony"},
            {"name": "Calpurnia", "description": "Caesar's wife, warns of danger"},
            {"name": "Portia", "description": "Brutus's wife, seeks his confidence"}
        ],
        "themes": [
            {
                "theme": "Power",
                "theme_explanation": "The struggle for power drives the conspiracy against Caesar, revealing its corrupting influence and the instability it creates in Rome."
            },
            {
                "theme": "Ambition",
                "theme_explanation": "Caesar’s ambition fuels fears of tyranny, while Cassius’s ambition manipulates Brutus, leading to betrayal and civil war."
            },
            {
                "theme": "Honor",
                "theme_explanation": "Brutus acts out of a sense of honor to protect the Republic, but his honorable intentions lead to tragic consequences."
            },
            {
                "theme": "Fate",
                "theme_explanation": "Omens and prophecies, like Calpurnia’s warnings, suggest fate’s role, yet characters’ choices shape the tragic outcome."
            },
            {
                "theme": "Public vs private self",
                "theme_explanation": "Characters like Brutus struggle to reconcile their public duties with private loyalties, leading to internal conflict and betrayal."
            },
            {
                "theme": "Rhetoric",
                "theme_explanation": "Powerful speeches, like Antony’s funeral oration, manipulate public opinion, highlighting rhetoric’s role in shaping political outcomes."
            },
            {
                "theme": "Loyalty",
                "theme_explanation": "Loyalty is tested, as Brutus betrays Caesar for Rome’s sake, while Antony’s loyalty to Caesar fuels his revenge against the conspirators."
            }
        ],
        "famous_quotes": [
            {
                "quote": "Et tu, Brute? Then fall, Caesar!",
                "speaker": "Julius Caesar",
                "act": 3,
                "scene": 1,
                "explanation": "Caesar expresses shock and betrayal as Brutus, his trusted friend, joins the assassins, marking his final words."
            },
            {
                "quote": "Friends, Romans, countrymen, lend me your ears",
                "speaker": "Mark Antony",
                "act": 3,
                "scene": 2,
                "explanation": "Antony begins his funeral oration, using rhetoric to sway the crowd against the conspirators."
            },
            {
                "quote": "The fault, dear Brutus, is not in our stars, but in ourselves",
                "speaker": "Cassius",
                "act": 1,
                "scene": 2,
                "explanation": "Cassius argues that human actions, not fate, shape destiny, urging Brutus to resist Caesar's rise."
            },
            {
                "quote": "Cowards die many times before their deaths; the valiant never taste of death but once",
                "speaker": "Julius Caesar",
                "act": 2,
                "scene": 2,
                "explanation": "Caesar dismisses fear of death, asserting that bravery faces mortality only once, reflecting his bold persona."
            },
            {
                "quote": "This was the noblest Roman of them all",
                "speaker": "Mark Antony",
                "act": 5,
                "scene": 5,
                "explanation": "Antony praises Brutus's honor after his death, acknowledging his genuine motives despite their conflict."
            },
            {
                "quote": "Beware the Ides of March",
                "speaker": "Soothsayer",
                "act": 1,
                "scene": 2,
                "explanation": "The Soothsayer warns Caesar of impending danger, foreshadowing his assassination on March 15."
            },
            {
                "quote": "Friends, Romans, countrymen, lend me your ears: I come to bury Caesar, not to praise him.",
                "speaker": "Mark Antony",
                "act": 3,
                "scene": 2,
                "explanation": "Mark Antony, in a masterful display of rhetoric, begins his funeral oration by feigning humility, subtly turning the crowd against the conspirators while appearing to honor Caesar's killers."
            },
            {
                "quote": "Cry “havoc!” and let slip the dogs of war",
                "speaker": "Mark Antony",
                "act": 3,
                "scene": 1,
                "explanation": "Mark Antony vows revenge for Caesar's assassination, predicting widespread civil war and destruction that will be unleashed upon Rome."
            },
            {
                "quote": "The evil that men do lives after them; The good is oft interrèd with their bones.",
                "speaker": "Mark Antony",
                "act": 3,
                "scene": 2,
                "explanation": "Antony suggests that people are more likely to remember and dwell on the bad deeds of others, while their good deeds are often forgotten after their death."
            },
            {
                "quote": "But, for my own part, it was Greek to me.",
                "speaker": "Casca",
                "act": 1,
                "scene": 2,
                "explanation": "Casca dismisses Cicero's speech as incomprehensible to him, implying he didn't understand it, or that it was too complex or foreign."
            }
        ],
        "summary": "The play portrays the conspiracy against Roman dictator Julius Caesar, his assassination, and its aftermath. Brutus joins the conspiracy out of a desire to protect the Roman Republic, but is manipulated by Cassius. After Caesar's murder, Mark Antony turns the public against the conspirators in a brilliant funeral oration, leading to civil war and the deaths of Brutus and Cassius."
    },
    "the_tempest": {
        "title": "The Tempest",
        "full_title": "The Tempest",
        "category": "comedy",
        "year": "1610-1611",
        "setting": "A remote island",
        "main_characters": ["Prospero", "Miranda", "Ariel", "Caliban", "Ferdinand", "Alonso", "Antonio", "Sebastian", "Stephano", "Trinculo"],
        "character_descriptions": [
            {"name": "Prospero", "description": "Exiled Duke of Milan, magician"},
            {"name": "Miranda", "description": "Prospero's daughter, loves Ferdinand"},
            {"name": "Ariel", "description": "Prospero's spirit servant, seeks freedom"},
            {"name": "Caliban", "description": "Island native, Prospero's enslaved servant"},
            {"name": "Ferdinand", "description": "Naples prince, loves Miranda"},
            {"name": "Alonso", "description": "King of Naples, shipwrecked"},
            {"name": "Antonio", "description": "Prospero's brother, usurped his dukedom"},
            {"name": "Sebastian", "description": "Alonso's brother, plots against him"},
            {"name": "Stephano", "description": "Drunken butler, plots with Caliban"},
            {"name": "Trinculo", "description": "Jester, joins Stephano's plot"}
        ],
        "themes": [
            {
                "theme": "Magic",
                "theme_explanation": "Prospero’s magic controls the island’s events, symbolizing the power of art and illusion to shape reality and resolve conflicts."
            },
            {
                "theme": "Power",
                "theme_explanation": "Power struggles drive the plot, as Prospero manipulates others to reclaim his dukedom, while Antonio and Sebastian plot to seize authority."
            },
            {
                "theme": "Colonization",
                "theme_explanation": "The play reflects colonial themes through Prospero’s domination of Caliban and Ariel, raising questions about authority and exploitation."
            },
            {
                "theme": "Forgiveness",
                "theme_explanation": "Prospero’s choice to forgive his enemies rather than seek revenge underscores the play’s resolution through mercy and reconciliation."
            },
            {
                "theme": "Freedom",
                "theme_explanation": "Ariel and Caliban’s quests for freedom highlight the tension between servitude and autonomy under Prospero’s rule."
            },
            {
                "theme": "Art",
                "theme_explanation": "Prospero’s magic mirrors the playwright’s craft, using illusion to explore human nature and orchestrate the play’s harmonious ending."
            },
            {
                "theme": "Nature vs nurture",
                "theme_explanation": "Caliban’s savage behavior contrasts with Miranda’s cultivated innocence, questioning whether character is shaped by nature or upbringing."
            }
        ],
        "famous_quotes": [
            {
                "quote": "We are such stuff as dreams are made on, and our little life is rounded with a sleep",
                "speaker": "Prospero",
                "act": 4,
                "scene": 1,
                "explanation": "Prospero reflects on life's fleeting, dreamlike nature, comparing it to the ephemeral quality of his magical spectacle."
            },
            {
                "quote": "O brave new world, that has such people in't!",
                "speaker": "Miranda",
                "act": 5,
                "scene": 1,
                "explanation": "Miranda expresses wonder at seeing new people, naively marveling at humanity, unaware of their flaws."
            },
            {
                "quote": "What's past is prologue",
                "speaker": "Antonio",
                "act": 2,
                "scene": 1,
                "explanation": "Antonio suggests that history sets the stage for future actions, justifying his plot to seize power."
            },
            {
                "quote": "Hell is empty and all the devils are here",
                "speaker": "Ariel",
                "act": 1,
                "scene": 2,
                "explanation": "Ariel describes the chaos of the shipwreck, hinting at the moral corruption among the stranded nobles."
            },
            {
                "quote": "The isle is full of noises, sounds, and sweet airs, that give delight and hurt not",
                "speaker": "Caliban",
                "act": 3,
                "scene": 2,
                "explanation": "Caliban poetically describes the island's enchanting sounds, revealing his sensitivity despite his rough nature."
            },
            {
                "quote": "Misery acquaints a man with strange bedfellows",
                "speaker": "Trinculo",
                "act": 2,
                "scene": 2,
                "explanation": "Trinculo humorously notes how desperation leads to unlikely alliances, reflecting the play’s comedic subplots."
            },
            {
                "quote": "Full fathom five thy father lies",
                "speaker": "Ariel",
                "act": 1,
                "scene": 2,
                "explanation": "Ariel’s song deceives Ferdinand about his father’s death, using magical imagery to manipulate emotions."
            },
            {
                "quote": "Full fathom five thy father lies, of his bones are coral made. Those are pearls that were his eyes. Nothing of him that doth fade, but doth suffer a sea-change into something rich and strange.",
                "speaker": "Ariel",
                "act": 1,
                "scene": 2,
                "explanation": "Ariel sings to Ferdinand about his father's presumed drowning, describing a transformative and beautiful process of decay and renewal under the sea."
            }
        ],
        "summary": "Prospero, the rightful Duke of Milan, and his daughter Miranda have been stranded on an island for twelve years after Prospero's brother Antonio usurped his position. Using magic learned from his books, Prospero controls the spirit Ariel and the creature Caliban. When his enemies sail near the island, he conjures a tempest to shipwreck them, setting in motion a plot of revenge, romance, and ultimately forgiveness."
    },
    "twelfth_night": {
        "title": "Twelfth Night; Or, What You Will",
        "full_title": "Twelfth Night, or What You Will",
        "category": "comedy",
        "year": "1601-1602",
        "setting": "Illyria, an ancient region on the Adriatic Sea",
        "main_characters": ["Viola", "Sebastian", "Orsino", "Olivia", "Malvolio", "Sir Toby Belch", "Sir Andrew Aguecheek", "Maria", "Feste"],
        "character_descriptions": [
            {"name": "Viola", "description": "Shipwrecked woman, disguises as Cesario"},
            {"name": "Sebastian", "description": "Viola's twin brother, presumed dead"},
            {"name": "Orsino", "description": "Duke of Illyria, loves Olivia"},
            {"name": "Olivia", "description": "Countess, loves Cesario, then Sebastian"},
            {"name": "Malvolio", "description": "Olivia's steward, humiliated by prank"},
            {"name": "Sir Toby Belch", "description": "Olivia's drunken uncle, plots prank"},
            {"name": "Sir Andrew Aguecheek", "description": "Toby's foolish friend, courts Olivia"},
            {"name": "Maria", "description": "Olivia's maid, orchestrates Malvolio prank"},
            {"name": "Feste", "description": "Olivia's witty fool, sings and jests"}
        ],
        "themes": [
            {
                "theme": "Love",
                "theme_explanation": "Love’s irrationality drives the romantic entanglements, as Orsino, Olivia, and Viola navigate unrequited and mistaken affections, resolved through comedic revelations."
            },
            {
                "theme": "Disguise",
                "theme_explanation": "Viola’s disguise as Cesario creates romantic and comedic confusion, highlighting how appearances shape perceptions and relationships."
            },
            {
                "theme": "Deception",
                "theme_explanation": "Deception, from Viola’s disguise to the prank on Malvolio, fuels the plot, exploring both playful and cruel manipulations."
            },
            {
                "theme": "Gender",
                "theme_explanation": "Viola’s cross-dressing as Cesario blurs gender roles, challenging societal norms and sparking romantic complications."
            },
            {
                "theme": "Social class",
                "theme_explanation": "The play critiques class distinctions through Malvolio’s social ambitions and the interactions between nobles, servants, and fools."
            },
            {
                "theme": "Folly",
                "theme_explanation": "Folly is central, from Malvolio’s self-delusion to Sir Toby’s drunken antics, exposing human weaknesses through humor."
            },
            {
                "theme": "Identity",
                "theme_explanation": "Mistaken identities, especially between Viola and Sebastian, drive the comedy, questioning the nature of self and perception."
            }
        ],
        "famous_quotes": [
            {
                "quote": "If music be the food of love, play on",
                "speaker": "Orsino",
                "act": 1,
                "scene": 1,
                "explanation": "Orsino indulges in his lovesickness for Olivia, equating music to nourishment for his romantic longing."
            },
            {
                "quote": "Some are born great, some achieve greatness, and some have greatness thrust upon 'em",
                "speaker": "Malvolio",
                "act": 2,
                "scene": 5,
                "explanation": "Malvolio misreads a forged letter, inflating his ambitions and leading to his comedic humiliation."
            },
            {
                "quote": "Be not afraid of greatness. Some are born great, some achieve greatness, and others have greatness thrust upon them",
                "speaker": "Malvolio",
                "act": 2,
                "scene": 5,
                "explanation": "Malvolio repeats the forged letter's words, revealing his delusions of grandeur that drive the subplot."
            },
            {
                "quote": "Better a witty fool, than a foolish wit",
                "speaker": "Feste",
                "act": 1,
                "scene": 5,
                "explanation": "Feste defends his role as a clever fool, contrasting his sharp wit with the folly of pretentious wisdom."
            },
            {
                "quote": "I am all the daughters of my father's house, and all the brothers too",
                "speaker": "Viola",
                "act": 2,
                "scene": 4,
                "explanation": "Viola, disguised as Cesario, hints at her dual identity and grief for her presumed-dead brother Sebastian."
            },
            {
                "quote": "Love sought is good, but given unsought is better",
                "speaker": "Olivia",
                "act": 3,
                "scene": 1,
                "explanation": "Olivia reflects on her unexpected love for Viola (disguised as Cesario), emphasizing spontaneous affection."
            },
            {
                "quote": "Many a good hanging prevents a bad marriage",
                "speaker": "Feste",
                "act": 1,
                "scene": 5,
                "explanation": "Feste’s jest underscores the play’s humorous take on romantic mismatches and clever wordplay."
            },
            {
                "quote": "If music be the food of love play on.",
                "speaker": "Orsino",
                "act": 1,
                "scene": 1,
                "explanation": "Duke Orsino expresses his desire for an excess of music to cure him of his love sickness, hoping to grow tired of love itself."
            },
            {
                "quote": "This is very midsummer madness.",
                "speaker": "Olivia",
                "act": 3,
                "scene": 4,
                "explanation": "Olivia, observing the chaotic and absurd events unfolding around her, attributes them to a temporary period of irrationality, similar to the wildness associated with midsummer festivals."
            }
        ],
        "summary": "Twelfth Night follows Viola, who disguises herself as a man named Cesario after being shipwrecked in Illyria, believing her twin brother Sebastian has drowned. She enters the service of Duke Orsino, who is in love with Countess Olivia. When Viola as Cesario is sent to court Olivia for Orsino, Olivia falls for Cesario instead. The arrival of Sebastian creates confusion and comedy, while a subplot involves the humiliation of Olivia's steward Malvolio."
    },
    "as_you_like_it": {
        "title": "As You Like It",
        "full_title": "As You Like It",
        "category": "comedy",
        "year": "1599-1600",
        "setting": "Court and Forest of Arden",
        "main_characters": ["Rosalind", "Orlando", "Celia", "Duke Senior", "Duke Frederick", "Jaques", "Touchstone", "Oliver", "Silvius", "Phebe"],
        "character_descriptions": [
            {"name": "Rosalind", "description": "Duke Senior's daughter, disguises as Ganymede"},
            {"name": "Orlando", "description": "Young noble, loves Rosalind"},
            {"name": "Celia", "description": "Duke Frederick's daughter, Rosalind's cousin"},
            {"name": "Duke Senior", "description": "Exiled duke, lives in Arden"},
            {"name": "Duke Frederick", "description": "Usurping duke, banishes Rosalind"},
            {"name": "Jaques", "description": "Melancholy lord, follows Duke Senior"},
            {"name": "Touchstone", "description": "Court fool, joins Rosalind's exile"},
            {"name": "Oliver", "description": "Orlando's brother, initially hostile"},
            {"name": "Silvius", "description": "Shepherd, loves Phebe"},
            {"name": "Phebe", "description": "Shepherdess, loves Ganymede"}
        ],
        "themes": [
            {
                "theme": "Love",
                "theme_explanation": "Love, both romantic and familial, drives the play, as Rosalind and Orlando’s romance and other pairings find harmony in the Forest of Arden."
            },
            {
                "theme": "Nature vs court",
                "theme_explanation": "The Forest of Arden offers a pastoral escape from the corrupt court, where characters find authenticity and personal growth."
            },
            {
                "theme": "Gender roles",
                "theme_explanation": "Rosalind’s disguise as Ganymede challenges gender norms, allowing her to explore and manipulate romantic dynamics with wit and agency."
            },
            {
                "theme": "Transformation",
                "theme_explanation": "Characters undergo emotional and social transformations in Arden, such as Oliver’s redemption and Orlando’s growth, reflecting the forest’s restorative power."
            },
            {
                "theme": "Pastoral life",
                "theme_explanation": "The idealized pastoral life in Arden contrasts with courtly corruption, offering a space for reflection, love, and reconciliation."
            },
            {
                "theme": "Time",
                "theme_explanation": "Time in Arden moves fluidly, allowing characters to reflect and transform, contrasting with the rigid constraints of court life."
            }
        ],
        "famous_quotes": [
            {
                "quote": "All the world's a stage, and all the men and women merely players",
                "speaker": "Jaques",
                "act": 2,
                "scene": 7,
                "explanation": "Jaques philosophizes on life as a theatrical performance, outlining the seven stages of human existence."
            },
            {
                "quote": "The fool doth think he is wise, but the wise man knows himself to be a fool",
                "speaker": "Touchstone",
                "act": 5,
                "scene": 1,
                "explanation": "Touchstone humorously reflects on wisdom, suggesting true insight lies in recognizing one's own folly."
            },
            {
                "quote": "Do you not know I am a woman? When I think, I must speak",
                "speaker": "Rosalind",
                "act": 3,
                "scene": 2,
                "explanation": "Rosalind, as Ganymede, playfully asserts her outspoken nature, challenging gender norms with wit."
            },
            {
                "quote": "Men have died from time to time, and worms have eaten them, but not for love",
                "speaker": "Rosalind",
                "act": 4,
                "scene": 1,
                "explanation": "Rosalind downplays Orlando's romantic fervor, teasing that love is not worth dying for, yet she loves him."
            },
            {
                "quote": "Sweet are the uses of adversity, which, like the toad, ugly and venomous, wears yet a precious jewel in his head",
                "speaker": "Duke Senior",
                "act": 2,
                "scene": 1,
                "explanation": "Duke Senior finds value in hardship, likening it to a toad with a hidden gem, embracing the forest's lessons."
            }
        ],
        "summary": "When Rosalind is banished by her uncle Duke Frederick, she flees to the Forest of Arden with her cousin Celia and the court fool Touchstone. Disguised as a young man named Ganymede, Rosalind encounters Orlando, who has also fled to the forest. Not recognizing her, Orlando confides in 'Ganymede' about his love for Rosalind. The forest becomes a place of transformation as four couples navigate love and mistaken identities."
    },
    "coriolanus": {
        "title": "The Tragedy of Coriolanus",
        "full_title": "The Tragedy of Coriolanus",
        "category": "tragedy",
        "year": "1608-1609",
        "setting": "Rome and its surrounding territories",
        "main_characters": ["Caius Martius Coriolanus", "Volumnia", "Menenius Agrippa", "Tullus Aufidius", "Sicinius Velutus", "Junius Brutus", "Virgilia", "Young Martius"],
        "character_descriptions": [
            {"name": "Caius Martius Coriolanus", "description": "Roman general, banished, seeks revenge"},
            {"name": "Volumnia", "description": "Coriolanus's mother, influences his actions"},
            {"name": "Menenius Agrippa", "description": "Roman patrician, Coriolanus's ally"},
            {"name": "Tullus Aufidius", "description": "Volscian general, allies with Coriolanus"},
            {"name": "Sicinius Velutus", "description": "Tribune, opposes Coriolanus"},
            {"name": "Junius Brutus", "description": "Tribune, seeks Coriolanus's banishment"},
            {"name": "Virgilia", "description": "Coriolanus's wife, loyal supporter"},
            {"name": "Young Martius", "description": "Coriolanus's son, minor role"}
        ],
        "themes": [
            {
                "theme": "Pride",
                "theme_explanation": "Coriolanus’s excessive pride in his martial valor alienates the plebeians and leads to his banishment and tragic downfall."
            },
            {
                "theme": "Politics",
                "theme_explanation": "The play explores political power struggles, as Coriolanus’s disdain for the masses clashes with the tribunes’ manipulation of public opinion."
            },
            {
                "theme": "Class conflict",
                "theme_explanation": "Tensions between the patricians and plebeians drive the conflict, with Coriolanus’s elitism fueling his exile and rebellion against Rome."
            },
            {
                "theme": "Loyalty",
                "theme_explanation": "Loyalty is tested, as Coriolanus shifts allegiance to the Volscians, yet his loyalty to his mother ultimately sways him to spare Rome."
            },
            {
                "theme": "Mother-son relationships",
                "theme_explanation": "Volumnia’s influence shapes Coriolanus’s identity and decisions, culminating in her pivotal role in persuading him to abandon his attack on Rome."
            },
            {
                "theme": "Honor",
                "theme_explanation": "Coriolanus’s rigid sense of honor drives his actions but conflicts with political compromise, leading to his isolation and death."
            }
        ],
        "famous_quotes": [
            {
                "quote": "Nature teaches beasts to know their friends",
                "speaker": "Coriolanus",
                "act": 2,
                "scene": 1,
                "explanation": "Coriolanus reflects on loyalty, implying humans, unlike animals, betray their allies, foreshadowing his own betrayal."
            },
            {
                "quote": "What is the city but the people?",
                "speaker": "Sicinius",
                "act": 3,
                "scene": 1,
                "explanation": "Sicinius emphasizes the power of the populace, challenging Coriolanus's elitism and justifying his banishment."
            },
            {
                "quote": "His nature is too noble for the world",
                "speaker": "Menenius",
                "act": 5,
                "scene": 3,
                "explanation": "Menenius praises Coriolanus's integrity, suggesting his rigid honor is incompatible with political compromise."
            },
            {
                "quote": "You common cry of curs! whose breath I hate",
                "speaker": "Coriolanus",
                "act": 3,
                "scene": 3,
                "explanation": "Coriolanus insults the plebeians, revealing his contempt for the masses, which leads to his exile."
            },
            {
                "quote": "I banish you!",
                "speaker": "Coriolanus",
                "act": 3,
                "scene": 3,
                "explanation": "Defiant, Coriolanus reverses his banishment, proclaiming his independence from Rome, highlighting his pride."
            }
        ],
        "summary": "Caius Martius, a Roman general, earns the name Coriolanus after his victory at Corioli. His pride and contempt for the plebeians lead to his banishment, prompting him to join his enemy Tullus Aufidius to attack Rome, only to be persuaded by his mother Volumnia to spare the city, resulting in his betrayal and death."
    },
    "cymbeline": {
        "title": "Cymbeline",
        "full_title": "Cymbeline, King of Britain",
        "category": "tragedy",
        "year": "1609-1610",
        "setting": "Ancient Britain and Rome",
        "main_characters": ["Cymbeline", "Imogen", "Posthumus Leonatus", "Iachimo", "Cloten", "Queen", "Belarius", "Guiderius", "Arviragus"],
        "character_descriptions": [
            {"name": "Cymbeline", "description": "King of Britain, Imogen's father"},
            {"name": "Imogen", "description": "Cymbeline's daughter, loves Posthumus"},
            {"name": "Posthumus Leonatus", "description": "Imogen's husband, exiled noble"},
            {"name": "Iachimo", "description": "Italian noble, deceives Posthumus"},
            {"name": "Cloten", "description": "Queen's son, seeks Imogen"},
            {"name": "Queen", "description": "Cymbeline's wife, Imogen's stepmother"},
            {"name": "Belarius", "description": "Exiled lord, raises king's sons"},
            {"name": "Guiderius", "description": "Cymbeline's son, raised by Belarius"},
            {"name": "Arviragus", "description": "Cymbeline's son, raised by Belarius"}
        ],
        "themes": [
            {
                "theme": "Love",
                "theme_explanation": "Imogen and Posthumus’s steadfast love overcomes trials, deceptions, and separation, driving the play toward reconciliation."
            },
            {
                "theme": "Jealousy",
                "theme_explanation": "Iachimo’s deception fuels Posthumus’s jealousy, nearly destroying his marriage to Imogen until truth restores their bond."
            },
            {
                "theme": "Forgiveness",
                "theme_explanation": "The play resolves through forgiveness, as Cymbeline pardons Posthumus and others, uniting the family and nation."
            },
            {
                "theme": "Identity",
                "theme_explanation": "Disguises and mistaken identities, like Imogen’s as Fidele, create confusion but ultimately reveal true selves and loyalties."
            },
            {
                "theme": "Family",
                "theme_explanation": "Family ties, tested by betrayal and separation, are restored as Cymbeline reunites with his children and accepts Posthumus."
            },
            {
                "theme": "Reconciliation",
                "theme_explanation": "The play culminates in reconciliation, resolving personal and political conflicts through revelations and forgiveness."
            }
        ],
        "famous_quotes": [
            {
                "quote": "The art of the court, as hard to leave as keep",
                "speaker": "Imogen",
                "act": 3,
                "scene": 3,
                "explanation": "Imogen reflects on the complexities of court life, preferring simplicity as she flees deceit and danger."
            },
            {
                "quote": "Fear no more the heat o' the sun",
                "speaker": "Guiderius",
                "act": 4,
                "scene": 2,
                "explanation": "Guiderius sings a funeral dirge, celebrating death as a release from life's hardships, unaware Imogen lives."
            },
            {
                "quote": "Boldness be my friend!",
                "speaker": "Iachimo",
                "act": 1,
                "scene": 6,
                "explanation": "Iachimo invokes courage to deceive Posthumus, revealing his cunning nature in plotting against Imogen."
            },
            {
                "quote": "I am loath to look upon the baseness of the world",
                "speaker": "Posthumus",
                "act": 2,
                "scene": 4,
                "explanation": "Posthumus despairs over Imogen's supposed betrayal, reflecting his disillusionment before learning the truth."
            },
            {
                "quote": "Some griefs are med'cinable",
                "speaker": "Imogen",
                "act": 3,
                "scene": 2,
                "explanation": "Imogen finds solace in Posthumus's letter, suggesting some sorrows can heal, foreshadowing their reunion."
            }
        ],
        "summary": "Imogen, daughter of King Cymbeline, marries Posthumus against her father's wishes. Banished, Posthumus is tricked by Iachimo into believing Imogen is unfaithful, leading to a series of trials, disguises, and revelations that ultimately reunite the lovers and reconcile the royal family."
    },
    "henry_iv_part1": {
        "title": "King Henry IV, the First Part",
        "full_title": "The First Part of King Henry the Fourth",
        "category": "history",
        "year": "1596-1597",
        "setting": "England, early 15th century",
        "main_characters": ["King Henry IV", "Prince Hal", "Sir John Falstaff", "Hotspur (Henry Percy)", "Earl of Worcester", "Earl of Northumberland", "Lady Percy"],
        "character_descriptions": [
            {"name": "King Henry IV", "description": "King of England, faces rebellion"},
            {"name": "Prince Hal", "description": "Heir to throne, proves worth"},
            {"name": "Sir John Falstaff", "description": "Prince Hal's roguish companion"},
            {"name": "Hotspur (Henry Percy)", "description": "Rebel leader, Percy family heir"},
            {"name": "Earl of Worcester", "description": "Rebel noble, Hotspur's uncle"},
            {"name": "Earl of Northumberland", "description": "Hotspur's father, supports rebellion"},
            {"name": "Lady Percy", "description": "Hotspur's wife, supports husband"}
        ],
        "themes": [
            {
                "theme": "Honor",
                "theme_explanation": "Hotspur’s obsession with honor contrasts with Falstaff’s pragmatic avoidance of it, while Hal redefines it through his heroic transformation."
            },
            {
                "theme": "Power",
                "theme_explanation": "The struggle for power fuels the rebellion against Henry IV, as the Percys challenge his legitimacy and Hal prepares for kingship."
            },
            {
                "theme": "Rebellion",
                "theme_explanation": "The Percy rebellion threatens Henry IV’s rule, highlighting political instability and the cost of usurping power."
            },
            {
                "theme": "Father-son relationships",
                "theme_explanation": "Henry IV’s disappointment in Hal’s waywardness contrasts with their eventual reconciliation, as Hal proves his worth at Shrewsbury."
            },
            {
                "theme": "Maturity",
                "theme_explanation": "Hal’s journey from reckless youth to responsible heir showcases his growth, culminating in his victory over Hotspur."
            },
            {
                "theme": "Loyalty",
                "theme_explanation": "Loyalty is tested, as Hal balances allegiance to Falstaff’s crew with his duty to his father and crown."
            }
        ],
        "famous_quotes": [
            {
                "quote": "I know you all, and will awhile uphold the unyoked humour of your idleness",
                "speaker": "Prince Hal",
                "act": 1,
                "scene": 2,
                "explanation": "Hal reveals his plan to feign recklessness, intending to surprise others with his eventual maturity."
            },
            {
                "quote": "Banish plump Jack, and banish all the world",
                "speaker": "Falstaff",
                "act": 2,
                "scene": 4,
                "explanation": "Falstaff humorously defends his larger-than-life presence, pleading with Hal not to reject their friendship."
            },
            {
                "quote": "The better part of valour is discretion",
                "speaker": "Falstaff",
                "act": 5,
                "scene": 4,
                "explanation": "Falstaff justifies faking death to avoid danger, comically prioritizing survival over heroic honor."
            },
            {
                "quote": "By heaven, methinks it were an easy leap, to pluck bright honour from the pale-faced moon",
                "speaker": "Hotspur",
                "act": 1,
                "scene": 3,
                "explanation": "Hotspur passionately describes his pursuit of glory, revealing his fiery, honor-driven character."
            },
            {
                "quote": "O, while you live, tell truth and shame the devil!",
                "speaker": "Hotspur",
                "act": 3,
                "scene": 1,
                "explanation": "Hotspur urges honesty in a heated debate, showcasing his blunt and principled nature."
            },
            {
                "quote": "The better part of valor is discretion",
                "speaker": "Falstaff",
                "act": 5,
                "scene": 4,
                "explanation": "Falstaff, known for his cowardice and quick wit, rationalizes his feigned death in battle by claiming that prudence and self-preservation are more important than reckless bravery."
            }
        ],
        "summary": "King Henry IV faces rebellion from the Percy family, led by the fiery Hotspur. Meanwhile, his son Prince Hal carouses with the roguish Falstaff but ultimately proves his worth by defeating Hotspur at the Battle of Shrewsbury, solidifying his path to maturity."
    },
    "henry_iv_part2": {
        "title": "King Henry IV, the Second Part",
        "full_title": "The Second Part of King Henry the Fourth",
        "category": "history",
        "year": "1597-1598",
        "setting": "England, early 15th century",
        "main_characters": ["King Henry IV", "Prince Hal", "Sir John Falstaff", "Lord Chief Justice", "Archbishop of York", "Earl of Northumberland", "Prince John"],
        "character_descriptions": [
            {"name": "King Henry IV", "description": "King of England, nearing death"},
            {"name": "Prince Hal", "description": "Heir, becomes King Henry V"},
            {"name": "Sir John Falstaff", "description": "Prince Hal's companion, later rejected"},
            {"name": "Lord Chief Justice", "description": "Legal authority, opposes Falstaff"},
            {"name": "Archbishop of York", "description": "Rebel leader, opposes king"},
            {"name": "Earl of Northumberland", "description": "Rebel noble, Hotspur's father"},
            {"name": "Prince John", "description": "Hal's brother, loyal to king"}
        ],
        "themes": [
            {
                "theme": "Power",
                "theme_explanation": "The struggle for power persists as Henry IV faces rebellion, while Hal prepares to inherit the crown amidst political instability."
            },
            {
                "theme": "Aging",
                "theme_explanation": "Henry IV’s declining health and Falstaff’s old age highlight the physical and emotional toll of time on leadership and vitality."
            },
            {
                "theme": "Betrayal",
                "theme_explanation": "Betrayal marks the play, from Northumberland’s abandonment of the rebels to Hal’s rejection of Falstaff upon becoming king."
            },
            {
                "theme": "Redemption",
                "theme_explanation": "Hal redeems his reputation by embracing royal duty, reconciling with his father, and rejecting his former reckless lifestyle."
            },
            {
                "theme": "Duty",
                "theme_explanation": "Hal’s acceptance of his royal responsibilities contrasts with the rebels’ defiance, emphasizing the burdens of leadership."
            },
            {
                "theme": "Succession",
                "theme_explanation": "The transition from Henry IV to Hal as Henry V underscores the challenges of succession and the need to prove legitimacy."
            }
        ],
        "famous_quotes": [
            {
                "quote": "Uneasy lies the head that wears a crown",
                "speaker": "King Henry IV",
                "act": 3,
                "scene": 1,
                "explanation": "Henry laments the burdens of kingship, revealing the toll of leadership and his restless conscience."
            },
            {
                "quote": "We are time's subjects, and time bids be gone",
                "speaker": "Hastings",
                "act": 1,
                "scene": 3,
                "explanation": "Hastings acknowledges time's unstoppable force, urging action in the rebels' fleeting opportunity."
            },
            {
                "quote": "I know thee not, old man",
                "speaker": "Prince Hal",
                "act": 5,
                "scene": 5,
                "explanation": "As king, Hal rejects Falstaff, marking his transformation from wayward prince to dutiful monarch."
            },
            {
                "quote": "A man can no more separate age and covetousness than he can part young limbs and lechery",
                "speaker": "Falstaff",
                "act": 1,
                "scene": 2,
                "explanation": "Falstaff humorously defends his greed, equating it to an inevitable trait of old age."
            },
            {
                "quote": "O sleep, O gentle sleep, nature's soft nurse",
                "speaker": "King Henry IV",
                "act": 3,
                "scene": 1,
                "explanation": "Henry envies the peace of sleep, which eludes him due to the stresses of his troubled reign."
            },
            {
                "quote": "A man can die but once.",
                "speaker": "Mowbray",
                "act": 3,
                "scene": 2,
                "explanation": "Mowbray expresses a stoic acceptance of mortality, suggesting that one should face death bravely as it is an unavoidable, singular event."
            }
        ],
        "summary": "As King Henry IV's health declines, rebellions persist, led by the Archbishop of York. Prince Hal continues his association with Falstaff but ultimately rejects him upon becoming king, embracing his royal duties and reconciling with his father before Henry's death."
    },
    "henry_v": {
        "title": "The Life of King Henry V",
        "full_title": "The Life of King Henry V",
        "category": "history",
        "year": "1599",
        "setting": "England and France, early 15th century",
        "main_characters": ["King Henry V", "Chorus", "Duke of Exeter", "Fluellen", "Pistol", "Katherine", "Dauphin of France"],
        "character_descriptions": [
            {"name": "King Henry V", "description": "King of England, leads Agincourt victory"},
            {"name": "Chorus", "description": "Narrator, guides audience through story"},
            {"name": "Duke of Exeter", "description": "Henry's uncle, loyal military advisor"},
            {"name": "Fluellen", "description": "Welsh captain, enforces military discipline"},
            {"name": "Pistol", "description": "Boastful soldier, Falstaff's former companion"},
            {"name": "Katherine", "description": "French princess, marries Henry V"},
            {"name": "Dauphin of France", "description": "French heir, opposes Henry"}
        ],
        "themes": [
            {
                "theme": "Leadership",
                "theme_explanation": "Henry V’s inspiring leadership unites his army, leading to victory at Agincourt through strategic decisions and motivational rhetoric."
            },
            {
                "theme": "War",
                "theme_explanation": "The play examines the glory and cost of war, as Henry’s campaign in France brings triumph but also bloodshed and moral questions."
            },
            {
                "theme": "Patriotism",
                "theme_explanation": "Henry’s rousing speeches foster national pride, rallying diverse English soldiers to fight as a unified ‘band of brothers.’"
            },
            {
                "theme": "Justice",
                "theme_explanation": "Henry seeks to justify his war with France through legal claims, while his harsh judgments, like executing traitors, reflect his commitment to justice."
            },
            {
                "theme": "Maturity",
                "theme_explanation": "Henry’s transformation from the reckless Prince Hal to a wise, responsible king showcases his growth into effective leadership."
            },
            {
                "theme": "Diplomacy",
                "theme_explanation": "Henry’s negotiations, culminating in his marriage to Katherine, highlight diplomacy as a tool to secure peace and unite nations."
            }
        ],
        "famous_quotes": [
            {
                "quote": "Once more unto the breach, dear friends, once more",
                "speaker": "King Henry V",
                "act": 3,
                "scene": 1,
                "explanation": "Henry rallies his troops at Harfleur, inspiring courage and unity in the face of battle."
            },
            {
                "quote": "We few, we happy few, we band of brothers",
                "speaker": "King Henry V",
                "act": 4,
                "scene": 3,
                "explanation": "Henry's St. Crispin's Day speech unites his outnumbered army, promising shared glory at Agincourt."
            },
            {
                "quote": "All things are ready, if our minds be so",
                "speaker": "King Henry V",
                "act": 4,
                "scene": 3,
                "explanation": "Henry emphasizes mental resolve as key to victory, bolstering his soldiers' confidence before battle."
            },
            {
                "quote": "The fewer men, the greater share of honour",
                "speaker": "King Henry V",
                "act": 4,
                "scene": 3,
                "explanation": "Henry reframes their small numbers as an opportunity for greater glory, motivating his troops."
            },
            {
                "quote": "O for a Muse of fire, that would ascend the brightest heaven of invention",
                "speaker": "Chorus",
                "act": 1,
                "scene": 0,
                "explanation": "The Chorus invokes inspiration to vividly portray Henry's epic story, setting the play's grand tone."
            }
        ],
        "summary": "King Henry V leads England in a campaign against France, culminating in the victory at Agincourt despite overwhelming odds. Through inspiring leadership and diplomacy, he secures peace and marries Princess Katherine, uniting the two nations."
    },
    "henry_vi_part1": {
        "title": "King Henry VI, the first part",
        "full_title": "The First Part of King Henry the Sixth",
        "category": "history",
        "year": "1591-1592",
        "setting": "England and France, 15th century",
        "main_characters": ["King Henry VI", "Joan of Arc", "Talbot", "Duke of Gloucester", "Cardinal Beaufort", "Margaret of Anjou", "Duke of York"],
        "character_descriptions": [
            {"name": "King Henry VI", "description": "Young king of England, weak ruler"},
            {"name": "Joan of Arc", "description": "French leader, inspires resistance"},
            {"name": "Talbot", "description": "English general, fights for France"},
            {"name": "Duke of Gloucester", "description": "Henry's uncle, protects crown"},
            {"name": "Cardinal Beaufort", "description": "Ambitious cleric, rivals Gloucester"},
            {"name": "Margaret of Anjou", "description": "French noble, marries Henry VI"},
            {"name": "Duke of York", "description": "Noble, claims throne, stirs discord"}
        ],
        "themes": [
            {
                "theme": "War",
                "theme_explanation": "The Hundred Years’ War drives the conflict, as England battles to retain French territories amidst internal strife and French resistance."
            },
            {
                "theme": "Power",
                "theme_explanation": "Power struggles between nobles like York and Gloucester destabilize Henry VI’s weak rule, foreshadowing the Wars of the Roses."
            },
            {
                "theme": "Religion",
                "theme_explanation": "Religious fervor, embodied by Joan of Arc’s divine claims and Cardinal Beaufort’s ambition, influences political and military conflicts."
            },
            {
                "theme": "Patriotism",
                "theme_explanation": "Talbot’s heroic efforts and Joan’s leadership inspire patriotic zeal, contrasting with internal English divisions that weaken the nation."
            },
            {
                "theme": "Factionalism",
                "theme_explanation": "Rivalries between York and Lancaster factions sow discord, undermining Henry VI’s reign and setting the stage for civil war."
            },
            {
                "theme": "Prophecy",
                "theme_explanation": "Joan’s visions and omens shape the narrative, suggesting a divine influence on the war’s outcome and England’s fate."
            }
        ],
        "famous_quotes": [
            {
                "quote": "Glory is like a circle in the water",
                "speaker": "Joan of Arc",
                "act": 1,
                "scene": 2,
                "explanation": "Joan describes fame as fleeting, like ripples that fade, foreshadowing her own rise and fall."
            },
            {
                "quote": "Plantagenet, I will; and like thee, Nero, play music to the fire",
                "speaker": "Margaret",
                "act": 5,
                "scene": 4,
                "explanation": "Margaret defiantly vows to fight, likening herself to Nero, who legendarily fiddled as Rome burned."
            },
            {
                "quote": "Fight till the last gasp",
                "speaker": "Talbot",
                "act": 4,
                "scene": 6,
                "explanation": "Talbot urges relentless combat, embodying his heroic resolve in England's fight for France."
            },
            {
                "quote": "My thoughts are whirled like a potter's wheel",
                "speaker": "King Henry VI",
                "act": 1,
                "scene": 5,
                "explanation": "Henry expresses confusion amid political turmoil, highlighting his weak and indecisive rule."
            },
            {
                "quote": "Unbidden guests are often welcomest when they are gone",
                "speaker": "Duke of Bedford",
                "act": 2,
                "scene": 2,
                "explanation": "Bedford wryly notes that unwanted visitors are most appreciated after they leave, reflecting on French resistance."
            }
        ],
        "summary": "Amid the Hundred Years' War, England struggles to maintain its French territories as young King Henry VI proves a weak ruler. Joan of Arc leads French resistance, while internal rivalries between the houses of York and Lancaster begin to sow discord, foreshadowing the Wars of the Roses."
    },
    "henry_vi_part2": {
        "title": "King Henry VI, the second part",
        "full_title": "The Second Part of King Henry the Sixth",
        "category": "history",
        "year": "1590-1591",
        "setting": "England, 15th century",
        "main_characters": ["King Henry VI", "Queen Margaret", "Duke of York", "Duke of Suffolk", "Jack Cade", "Humphrey, Duke of Gloucester", "Cardinal Beaufort"],
        "character_descriptions": [
            {"name": "King Henry VI", "description": "Weak king, struggles with factions"},
            {"name": "Queen Margaret", "description": "Henry's wife, seeks power"},
            {"name": "Duke of York", "description": "Noble, claims throne, leads rebellion"},
            {"name": "Duke of Suffolk", "description": "Margaret's ally, plots against Gloucester"},
            {"name": "Jack Cade", "description": "Rebel leader, incites uprising"},
            {"name": "Humphrey, Duke of Gloucester", "description": "Henry's uncle, targeted by rivals"},
            {"name": "Cardinal Beaufort", "description": "Ambitious cleric, conspires against Gloucester"}
        ],
        "themes": [
            {
                "theme": "Civil war",
                "theme_explanation": "Rival factions vying for power under Henry VI’s weak rule escalate tensions, foreshadowing the Wars of the Roses."
            },
            {
                "theme": "Ambition",
                "theme_explanation": "The Duke of York’s and Queen Margaret’s ambitions for control drive political conspiracies and rebellions, destabilizing the kingdom."
            },
            {
                "theme": "Justice",
                "theme_explanation": "The play questions justice as nobles like Gloucester face unjust accusations, while Cade’s rebellion seeks a distorted form of social equity."
            },
            {
                "theme": "Rebellion",
                "theme_explanation": "Jack Cade’s uprising and York’s claim to the throne highlight widespread discontent with Henry’s ineffective governance."
            },
            {
                "theme": "Power",
                "theme_explanation": "Struggles for power among nobles and the crown fuel betrayals and violence, exposing the fragility of Henry’s rule."
            },
            {
                "theme": "Betrayal",
                "theme_explanation": "Betrayals, such as Suffolk’s and Beaufort’s plots against Gloucester, underscore the treacherous political landscape of the court."
            }
        ],
        "famous_quotes": [
            {
                "quote": "The first thing we do, let's kill all the lawyers",
                "speaker": "Dick the Butcher",
                "act": 4,
                "scene": 2,
                "explanation": "A rebel humorously proposes eliminating lawyers to disrupt order, reflecting the chaos of Cade's uprising."
            },
            {
                "quote": "Smooth runs the water where the brook is deep",
                "speaker": "Suffolk",
                "act": 3,
                "scene": 1,
                "explanation": "Suffolk warns that calm appearances hide dangerous intentions, foreshadowing political treachery."
            },
            {
                "quote": "What stronger breastplate than a heart untainted?",
                "speaker": "King Henry VI",
                "act": 3,
                "scene": 2,
                "explanation": "Henry praises moral integrity as the best defense, revealing his idealistic but naive nature."
            },
            {
                "quote": "Ignorance is the curse of God",
                "speaker": "Say",
                "act": 4,
                "scene": 7,
                "explanation": "Say condemns ignorance during Cade's rebellion, highlighting the dangers of uneducated revolt."
            },
            {
                "quote": "Small things make base men proud",
                "speaker": "Queen Margaret",
                "act": 4,
                "scene": 1,
                "explanation": "Margaret scorns petty pride, critiquing the ambition driving the escalating conflicts."
            }
        ],
        "summary": "King Henry VI's weak rule fuels rivalries between noble factions, with Queen Margaret and the Duke of York vying for influence. The Duke of York's claim to the throne sparks rebellion, while Jack Cade's uprising exposes social unrest, escalating tensions toward civil war."
    },
    "henry_vi_part3": {
        "title": "King Henry VI, the third part",
        "full_title": "The Third Part of King Henry the Sixth",
        "category": "history",
        "year": "1591-1592",
        "setting": "England, 15th century",
        "main_characters": ["King Henry VI", "Queen Margaret", "Richard, Duke of York", "Edward IV", "Richard (later Richard III)", "Warwick", "Clarence"],
        "character_descriptions": [
            {"name": "King Henry VI", "description": "Deposed king, seeks to reclaim throne"},
            {"name": "Queen Margaret", "description": "Henry's wife, leads Lancaster forces"},
            {"name": "Richard, Duke of York", "description": "Yorkist leader, claims crown"},
            {"name": "Edward IV", "description": "Yorkist king, defeats Lancasters"},
            {"name": "Richard (later Richard III)", "description": "Yorkist, schemes for power"},
            {"name": "Warwick", "description": "Noble, shifts allegiance, influences war"},
            {"name": "Clarence", "description": "Edward's brother, betrays then rejoins York"}
        ],
        "themes": [
            {
                "theme": "Civil war",
                "theme_explanation": "The Wars of the Roses pit York against Lancaster, tearing England apart with relentless battles and shifting allegiances."
            },
            {
                "theme": "Revenge",
                "theme_explanation": "Revenge fuels the cycle of violence, as Margaret and Richard seek retribution for personal and political losses."
            },
            {
                "theme": "Power",
                "theme_explanation": "The relentless pursuit of the crown by Yorkists and Lancastrians destabilizes the realm, with power gained through betrayal and bloodshed."
            },
            {
                "theme": "Loyalty",
                "theme_explanation": "Loyalty is fragile, as figures like Warwick and Clarence switch sides, prioritizing ambition over steadfast allegiance."
            },
            {
                "theme": "Betrayal",
                "theme_explanation": "Betrayals, such as Clarence’s defection and Warwick’s turn against Edward, drive the conflict and deepen the tragedy."
            },
            {
                "theme": "Ambition",
                "theme_explanation": "Richard’s ruthless ambition to seize the throne foreshadows his rise, while others’ desires for power fuel the ongoing war."
            }
        ],
        "famous_quotes": [
            {
                "quote": "I can smile, and murder whiles I smile",
                "speaker": "Richard",
                "act": 3,
                "scene": 2,
                "explanation": "Richard reveals his deceitful ambition, foreshadowing his ruthless rise as Richard III."
            },
            {
                "quote": "O tiger's heart wrapped in a woman's hide!",
                "speaker": "York",
                "act": 1,
                "scene": 4,
                "explanation": "York condemns Margaret's ferocity, highlighting her fierce leadership in the war."
            },
            {
                "quote": "Suspicion always haunts the guilty mind",
                "speaker": "King Henry VI",
                "act": 5,
                "scene": 6,
                "explanation": "Henry reflects on guilt's torment, as he faces his own downfall amid betrayal."
            },
            {
                "quote": "To weep is to make less the depth of grief",
                "speaker": "Queen Margaret",
                "act": 2,
                "scene": 4,
                "explanation": "Margaret suggests tears diminish sorrow, revealing her resilience despite personal losses."
            },
            {
                "quote": "The smallest worm will turn, being trodden on",
                "speaker": "Clifford",
                "act": 2,
                "scene": 2,
                "explanation": "Clifford justifies revenge, noting that even the weakest will retaliate when oppressed."
            }
        ],
        "summary": "The Wars of the Roses intensify as the houses of York and Lancaster battle for the throne. After King Henry VI is deposed, Edward IV claims the crown, but betrayals and Richard's ruthless ambition set the stage for further conflict and his rise to power."
    },
    "henry_viii": {
        "title": "The Life of Henry the Eighth",
        "full_title": "All Is True, or The Famous History of the Life of King Henry VIII",
        "category": "history",
        "year": "1613",
        "setting": "England, early 16th century",
        "main_characters": ["King Henry VIII", "Cardinal Wolsey", "Katherine of Aragon", "Anne Boleyn", "Thomas Cranmer", "Duke of Norfolk", "Duke of Buckingham"],
        "character_descriptions": [
            {"name": "King Henry VIII", "description": "King of England, seeks divorce"},
            {"name": "Cardinal Wolsey", "description": "Ambitious advisor, falls from power"},
            {"name": "Katherine of Aragon", "description": "Henry's first wife, faces divorce"},
            {"name": "Anne Boleyn", "description": "Henry's second wife, bears Elizabeth"},
            {"name": "Thomas Cranmer", "description": "Archbishop, supports Henry's reforms"},
            {"name": "Duke of Norfolk", "description": "Noble, opposes Wolsey's influence"},
            {"name": "Duke of Buckingham", "description": "Noble, executed for treason"}
        ],
        "themes": [
            {
                "theme": "Power",
                "theme_explanation": "Henry VIII’s quest for control over his marriage and kingdom drives political maneuvers, while Wolsey’s ambition leads to his downfall."
            },
            {
                "theme": "Religion",
                "theme_explanation": "The conflict between Henry’s desire for divorce and Catholic doctrine sparks religious reform, shaping England’s break with Rome."
            },
            {
                "theme": "Marriage",
                "theme_explanation": "Henry’s divorce from Katherine and marriage to Anne Boleyn highlight the personal and political stakes of royal unions."
            },
            {
                "theme": "Politics",
                "theme_explanation": "Court intrigues, such as Wolsey’s schemes and Norfolk’s opposition, reveal the treacherous dynamics of political power."
            },
            {
                "theme": "Succession",
                "theme_explanation": "The birth of Elizabeth secures Henry’s dynasty, emphasizing the critical role of succession in stabilizing the monarchy."
            },
            {
                "theme": "Justice",
                "theme_explanation": "The play questions justice through Buckingham’s execution and Katherine’s trial, exposing the influence of power on legal outcomes."
            }
        ],
        "famous_quotes": [
            {
                "quote": "Heat not a furnace for your foe so hot that it do singe yourself",
                "speaker": "Norfolk",
                "act": 1,
                "scene": 1,
                "explanation": "Norfolk warns against excessive vengeance, advising caution to avoid self-destruction."
            },
            {
                "quote": "I have touched the highest point of all my greatness",
                "speaker": "Cardinal Wolsey",
                "act": 3,
                "scene": 2,
                "explanation": "Wolsey reflects on his fall from power, acknowledging the fleeting nature of ambition."
            },
            {
                "quote": "Men's evil manners live in brass; their virtues we write in water",
                "speaker": "Griffith",
                "act": 4,
                "scene": 2,
                "explanation": "Griffith notes that flaws are remembered enduringly, while virtues fade, commenting on human legacy."
            },
            {
                "quote": "Farewell, a long farewell, to all my greatness!",
                "speaker": "Cardinal Wolsey",
                "act": 3,
                "scene": 2,
                "explanation": "Wolsey laments his downfall, accepting the end of his political dominance with resignation."
            },
            {
                "quote": "This is the state of man: today he puts forth the tender leaves of hopes",
                "speaker": "Cardinal Wolsey",
                "act": 3,
                "scene": 2,
                "explanation": "Wolsey compares life to a plant, fragile and subject to sudden ruin, reflecting on his own fall."
            }
        ],
        "summary": "King Henry VIII navigates political and personal turmoil, divorcing Katherine of Aragon to marry Anne Boleyn, causing religious and diplomatic strife. Cardinal Wolsey's ambition leads to his downfall, while the birth of Elizabeth heralds hope for England's future."
    },
    "king_john": {
        "title": "King John",
        "full_title": "The Life and Death of King John",
        "category": "history",
        "year": "1595-1596",
        "setting": "England and France, 13th century",
        "main_characters": ["King John", "Philip the Bastard", "Constance", "King Philip of France", "Prince Arthur", "Hubert", "Cardinal Pandulph"],
        "character_descriptions": [
            {"name": "King John", "description": "King of England, faces rebellion"},
            {"name": "Philip the Bastard", "description": "John's loyal illegitimate nephew"},
            {"name": "Constance", "description": "Arthur's mother, seeks son's crown"},
            {"name": "King Philip of France", "description": "French king, supports Arthur"},
            {"name": "Prince Arthur", "description": "John's nephew, claims throne"},
            {"name": "Hubert", "description": "John's servant, spares Arthur"},
            {"name": "Cardinal Pandulph", "description": "Papal legate, stirs conflict"}
        ],
        "themes": [
            {
                "theme": "Power",
                "theme_explanation": "King John’s struggle to maintain his throne against French and domestic challenges highlights the precarious nature of royal authority."
            },
            {
                "theme": "Legitimacy",
                "theme_explanation": "The conflict over Arthur’s claim versus John’s rule questions the legitimacy of kingship, complicated by the Bastard’s loyalty despite his illegitimacy."
            },
            {
                "theme": "War",
                "theme_explanation": "Anglo-French conflicts and internal rebellions drive the play, reflecting the destructive cost of territorial and political disputes."
            },
            {
                "theme": "Family",
                "theme_explanation": "Family ties, like Constance’s devotion to Arthur and John’s rivalry with his nephew, fuel personal and political conflicts."
            },
            {
                "theme": "Betrayal",
                "theme_explanation": "Betrayals, such as France’s shifting alliances and John’s nobles turning against him, underscore the instability of loyalty."
            },
            {
                "theme": "Patriotism",
                "theme_explanation": "The Bastard’s rousing defense of England evokes patriotic pride, rallying support for the nation despite John’s flawed rule."
            }
        ],
        "famous_quotes": [
            {
                "quote": "To gild refined gold, to paint the lily",
                "speaker": "Salisbury",
                "act": 4,
                "scene": 2,
                "explanation": "Salisbury criticizes unnecessary embellishment, reflecting on the futility of John's actions."
            },
            {
                "quote": "Life is as tedious as a twice-told tale",
                "speaker": "Lewis",
                "act": 3,
                "scene": 4,
                "explanation": "Lewis compares life to a repetitive story, expressing weariness amid ongoing conflicts."
            },
            {
                "quote": "Grief fills the room up of my absent child",
                "speaker": "Constance",
                "act": 3,
                "scene": 4,
                "explanation": "Constance vividly expresses her maternal sorrow for her lost son Arthur, highlighting her despair."
            },
            {
                "quote": "I am not mad; I would to heaven I were!",
                "speaker": "Constance",
                "act": 3,
                "scene": 4,
                "explanation": "Constance wishes for madness to escape her grief, emphasizing the intensity of her emotional pain."
            },
            {
                "quote": "This England never did, nor never shall, lie at the proud foot of a conqueror",
                "speaker": "Philip the Bastard",
                "act": 5,
                "scene": 7,
                "explanation": "The Bastard proudly declares England's resilience, rallying patriotic spirit despite John's failures."
            }
        ],
        "summary": "King John faces challenges to his throne from France and internal dissent, compounded by the tragic fate of young Prince Arthur. The Bastard's loyalty and England's resilience prevail, but John's death leaves the crown to his son, Henry III."
    },
    "loves_labours_lost": {
        "title": "Love's Labour's Lost",
        "full_title": "Love's Labour's Lost",
        "category": "comedy",
        "year": "1594-1595",
        "setting": "Navarre, Spain",
        "main_characters": ["King of Navarre", "Princess of France", "Berowne", "Rosaline", "Longaville", "Maria", "Dumaine", "Katherine", "Don Armado"],
        "character_descriptions": [
            {"name": "King of Navarre", "description": "King, vows to study, loves Princess"},
            {"name": "Princess of France", "description": "French noble, courts Navarre"},
            {"name": "Berowne", "description": "Lord, loves Rosaline, witty"},
            {"name": "Rosaline", "description": "Princess's lady, loves Berowne"},
            {"name": "Longaville", "description": "Lord, loves Maria"},
            {"name": "Maria", "description": "Princess's lady, loves Longaville"},
            {"name": "Dumaine", "description": "Lord, loves Katherine"},
            {"name": "Katherine", "description": "Princess's lady, loves Dumaine"},
            {"name": "Don Armado", "description": "Spanish noble, comically woos Jaquenetta"}
        ],
        "themes": [
            {
                "theme": "Love",
                "theme_explanation": "Love disrupts the lords’ vow of celibacy, leading to witty courtships that reveal its irresistible and transformative power."
            },
            {
                "theme": "Wit",
                "theme_explanation": "The play celebrates verbal sparring and intellectual humor, as characters like Berowne and Rosaline showcase cleverness in love and debate."
            },
            {
                "theme": "Education",
                "theme_explanation": "The lords’ pursuit of scholarly isolation is undermined by love, suggesting that emotional experience outweighs academic discipline."
            },
            {
                "theme": "Vows",
                "theme_explanation": "The fragility of vows is exposed as the lords quickly abandon their oath of study for love, highlighting human weakness."
            },
            {
                "theme": "Courtship",
                "theme_explanation": "Courtship drives the plot through playful flirtations and deceptions, culminating in delayed unions due to unexpected tragedy."
            },
            {
                "theme": "Language",
                "theme_explanation": "The play revels in linguistic play, using ornate speech and wordplay to explore communication’s role in love and society."
            }
        ],
        "famous_quotes": [
            {
                "quote": "They have been at a great feast of languages, and stolen the scraps",
                "speaker": "Moth",
                "act": 5,
                "scene": 1,
                "explanation": "Moth mocks the pedantic scholars, likening their verbose speech to scavenging linguistic leftovers."
            },
            {
                "quote": "The spring is near when green geese are a-breeding",
                "speaker": "Dumaine",
                "act": 1,
                "scene": 1,
                "explanation": "Dumaine playfully notes seasonal signs, setting a light tone for the lords' oath-taking."
            },
            {
                "quote": "A jest's prosperity lies in the ear of him that hears it",
                "speaker": "Rosaline",
                "act": 5,
                "scene": 2,
                "explanation": "Rosaline emphasizes that humor depends on the listener's perception, showcasing her sharp wit."
            },
            {
                "quote": "When daisies pied and violets blue",
                "speaker": "Chorus",
                "act": 5,
                "scene": 2,
                "explanation": "The Chorus's song celebrates spring's renewal, contrasting with the play's delayed romantic resolutions."
            },
            {
                "quote": "Love's feeling is more soft and sensible than are the tender horns of cockled snails",
                "speaker": "Berowne",
                "act": 4,
                "scene": 3,
                "explanation": "Berowne poetically describes love's gentle sensitivity, defending his broken vow with romantic fervor."
            }
        ],
        "summary": "The King of Navarre and his lords vow to forgo women and study for three years, but their resolve crumbles when the Princess of France and her ladies arrive. Through witty courtship and playful deception, love prevails, though a sudden death delays their unions for a year."
    },
    "measure_for_measure": {
        "title": "Measure for Measure",
        "full_title": "Measure for Measure",
        "category": "comedy",
        "year": "1604",
        "setting": "Vienna",
        "main_characters": ["Duke Vincentio", "Angelo", "Isabella", "Claudio", "Lucio", "Mariana", "Escalus"],
        "character_descriptions": [
            {"name": "Duke Vincentio", "description": "Ruler of Vienna, disguises to observe"},
            {"name": "Angelo", "description": "Deputy, enforces strict laws"},
            {"name": "Isabella", "description": "Novice nun, pleads for Claudio"},
            {"name": "Claudio", "description": "Isabella's brother, sentenced to death"},
            {"name": "Lucio", "description": "Claudio's friend, spreads gossip"},
            {"name": "Mariana", "description": "Angelo's former fiancée, aids plot"},
            {"name": "Escalus", "description": "Wise advisor, serves Duke"}
        ],
        "themes": [
            {
                "theme": "Justice",
                "theme_explanation": "The play examines justice through Angelo’s harsh enforcement and the Duke’s interventions, questioning the balance between law and fairness."
            },
            {
                "theme": "Morality",
                "theme_explanation": "Moral dilemmas, like Isabella’s choice between chastity and her brother’s life, highlight the tension between personal ethics and societal demands."
            },
            {
                "theme": "Power",
                "theme_explanation": "Power corrupts Angelo, whose hypocrisy is exposed, while the Duke’s disguised authority manipulates events to restore order."
            },
            {
                "theme": "Mercy",
                "theme_explanation": "Mercy triumphs over strict justice, as the Duke pardons Claudio and others, advocating compassion in governance."
            },
            {
                "theme": "Hypocrisy",
                "theme_explanation": "Angelo’s outward virtue masks his corrupt desires, revealing the dangers of hypocrisy in those entrusted with power."
            },
            {
                "theme": "Chastity",
                "theme_explanation": "Isabella’s commitment to chastity is tested by Angelo’s proposition, raising questions about virtue and sacrifice in a corrupt world."
            }
        ],
        "famous_quotes": [
            {
                "quote": "The law hath not been dead, though it hath slept",
                "speaker": "Angelo",
                "act": 2,
                "scene": 2,
                "explanation": "Angelo defends his strict enforcement, suggesting dormant laws must now awaken, revealing his rigidity."
            },
            {
                "quote": "Some rise by sin, and some by virtue fall",
                "speaker": "Escalus",
                "act": 2,
                "scene": 1,
                "explanation": "Escalus observes the irony that immorality can lead to success, while virtue may cause failure."
            },
            {
                "quote": "Our doubts are traitors, and make us lose the good we oft might win",
                "speaker": "Lucio",
                "act": 1,
                "scene": 4,
                "explanation": "Lucio warns that hesitation can sabotage opportunities, urging action in a morally complex world."
            },
            {
                "quote": "What's mine is yours, and what is yours is mine",
                "speaker": "Duke Vincentio",
                "act": 5,
                "scene": 1,
                "explanation": "The Duke offers unity to Isabella, suggesting a shared future, possibly hinting at marriage."
            },
            {
                "quote": "Condemn the fault and not the actor of it?",
                "speaker": "Angelo",
                "act": 2,
                "scene": 2,
                "explanation": "Angelo questions separating sin from sinner, defending his harsh judgment before his own hypocrisy is exposed."
            }
        ],
        "summary": "Duke Vincentio leaves Vienna under Angelo's strict rule, who sentences Claudio to death for fornication. Claudio's sister Isabella pleads for mercy, and the disguised Duke manipulates events to expose Angelo's hypocrisy, ultimately restoring justice and mercy."
    },
    "merry_wives_of_windsor": {
        "title": "The Merry Wives of Windsor",
        "full_title": "The Merry Wives of Windsor",
        "category": "comedy",
        "year": "1597-1601",
        "setting": "Windsor, England",
        "main_characters": ["Sir John Falstaff", "Mistress Ford", "Mistress Page", "Ford", "Page", "Anne Page", "Fenton", "Doctor Caius"],
        "character_descriptions": [
            {"name": "Sir John Falstaff", "description": "Rogue, woos wives for money"},
            {"name": "Mistress Ford", "description": "Wife, outwits Falstaff's advances"},
            {"name": "Mistress Page", "description": "Wife, plots against Falstaff"},
            {"name": "Ford", "description": "Mistress Ford's jealous husband"},
            {"name": "Page", "description": "Mistress Page's husband, trusts wife"},
            {"name": "Anne Page", "description": "Page's daughter, loves Fenton"},
            {"name": "Fenton", "description": "Gentleman, marries Anne Page"},
            {"name": "Doctor Caius", "description": "French suitor, competes for Anne"}
        ],
        "themes": [
            {
                "theme": "Jealousy",
                "theme_explanation": "Ford’s jealousy of his wife’s fidelity drives comedic misunderstandings, resolved when he learns to trust her wit and loyalty."
            },
            {
                "theme": "Deception",
                "theme_explanation": "The wives’ clever deceptions humiliate Falstaff, while Anne’s secret marriage to Fenton outwits her parents’ plans."
            },
            {
                "theme": "Marriage",
                "theme_explanation": "The play explores marital dynamics, celebrating the wives’ agency and fidelity while Anne’s choice of love over arranged marriage triumphs."
            },
            {
                "theme": "Social class",
                "theme_explanation": "Falstaff’s attempts to exploit the middle-class wives and Fenton’s courtship of Anne despite his status highlight class tensions and aspirations."
            },
            {
                "theme": "Wit",
                "theme_explanation": "The wives’ sharp wit outmaneuvers Falstaff, showcasing female intelligence and resourcefulness in a patriarchal society."
            },
            {
                "theme": "Community",
                "theme_explanation": "The Windsor community unites to expose Falstaff’s folly, reinforcing social bonds through shared humor and moral correction."
            }
        ],
        "famous_quotes": [
            {
                "quote": "I hope good luck lies in odd numbers",
                "speaker": "Falstaff",
                "act": 5,
                "scene": 1,
                "explanation": "Falstaff optimistically trusts in luck for his third attempt to seduce the wives, unaware of their trap."
            },
            {
                "quote": "Why, then the world's mine oyster",
                "speaker": "Pistol",
                "act": 2,
                "scene": 2,
                "explanation": "Pistol boasts of boundless opportunities, using a metaphor for exploiting the world's riches."
            },
            {
                "quote": "This is the third time; I hope good luck lies in odd numbers",
                "speaker": "Falstaff",
                "act": 3,
                "scene": 1,
                "explanation": "Falstaff persists in his schemes, clinging to superstition despite repeated failures, adding comic irony."
            },
            {
                "quote": "Better three hours too soon than a minute too late",
                "speaker": "Mistress Page",
                "act": 2,
                "scene": 1,
                "explanation": "Mistress Page emphasizes punctuality, reflecting the wives' precise planning to outwit Falstaff."
            },
            {
                "quote": "Wives may be merry, and yet honest too",
                "speaker": "Mistress Page",
                "act": 4,
                "scene": 2,
                "explanation": "Mistress Page defends their playful deception, asserting that wit and virtue can coexist in women."
            },
            {
                "quote": "I cannot tell what the dickens his name is.",
                "speaker": "Mistress Quickly",
                "act": 3,
                "scene": 2,
                "explanation": "Mistress Quickly uses 'the dickens' as a mild euphemism for 'the devil,' indicating her frustration or forgetfulness in trying to recall a name."
            }
        ],
        "summary": "Falstaff attempts to woo Mistresses Ford and Page for their husbands' wealth, but the clever wives outwit him with a series of pranks. Meanwhile, Anne Page navigates suitors, ultimately marrying her true love Fenton, as the community unites to humiliate Falstaff."
    },
    "much_ado_about_nothing": {
        "title": "Much Ado About Nothing",
        "full_title": "Much Ado About Nothing",
        "category": "comedy",
        "year": "1598-1599",
        "setting": "Messina, Sicily",
        "main_characters": ["Beatrice", "Benedick", "Claudio", "Hero", "Don Pedro", "Don John", "Leonato"],
        "character_descriptions": [
            {"name": "Beatrice", "description": "Witty noblewoman, loves Benedick"},
            {"name": "Benedick", "description": "Witty soldier, loves Beatrice"},
            {"name": "Claudio", "description": "Young lord, loves Hero"},
            {"name": "Hero", "description": "Leonato's daughter, loves Claudio"},
            {"name": "Don Pedro", "description": "Prince, aids Claudio's courtship"},
            {"name": "Don John", "description": "Pedro's brother, slanders Hero"},
            {"name": "Leonato", "description": "Governor, Hero's father"}
        ],
        "themes": [
            {
                "theme": "Love",
                "theme_explanation": "Love drives the play, from Beatrice and Benedick’s witty romance to Claudio and Hero’s tumultuous courtship, resolved through trust and revelation."
            },
            {
                "theme": "Deception",
                "theme_explanation": "Deception shapes the plot, with Don John’s slander disrupting Hero’s wedding and playful tricks uniting Beatrice and Benedick."
            },
            {
                "theme": "Honor",
                "theme_explanation": "Honor, particularly female chastity, is central, as Hero’s tarnished reputation threatens her life until her innocence is proven."
            },
            {
                "theme": "Wit",
                "theme_explanation": "Beatrice and Benedick’s verbal sparring showcases wit as a tool for courtship and social commentary, enriching the comedy."
            },
            {
                "theme": "Slander",
                "theme_explanation": "Don John’s slander against Hero highlights the destructive power of false accusations and the importance of truth in restoring justice."
            },
            {
                "theme": "Marriage",
                "theme_explanation": "The play explores marriage through Hero’s near-tragic betrothal and Beatrice’s eventual acceptance of love, celebrating union and mutual respect."
            }
        ],
        "famous_quotes": [
            {
                "quote": "I do love nothing in the world so well as you",
                "speaker": "Benedick",
                "act": 4,
                "scene": 1,
                "explanation": "Benedick confesses his love for Beatrice, marking a turning point in their witty romantic sparring."
            },
            {
                "quote": "Sigh no more, ladies, sigh no more",
                "speaker": "Balthasar",
                "act": 2,
                "scene": 3,
                "explanation": "Balthasar's song advises women against mourning men's fickleness, reflecting the play's themes of love and deception."
            },
            {
                "quote": "Friendship is constant in all other things, save in the office and affairs of love",
                "speaker": "Claudio",
                "act": 2,
                "scene": 1,
                "explanation": "Claudio notes love's power to disrupt loyalty, foreshadowing his own susceptibility to slander."
            },
            {
                "quote": "There's a skirmish of wit between them",
                "speaker": "Leonato",
                "act": 1,
                "scene": 1,
                "explanation": "Leonato describes Beatrice and Benedick's verbal sparring, highlighting their intellectual chemistry."
            },
            {
                "quote": "I had rather hear my dog bark at a crow than a man swear he loves me",
                "speaker": "Beatrice",
                "act": 1,
                "scene": 1,
                "explanation": "Beatrice scorns romantic declarations, showcasing her independence and sharp wit early in the play."
            },
            {
                "quote": "I will live a bachelor",
                "speaker": "Benedick",
                "act": 1,
                "scene": 1,
                "explanation": "Benedick vows to avoid marriage, ironically foreshadowing his eventual love for Beatrice."
            },
            {
                "quote": "Some Cupid kills with arrows, some with traps",
                "speaker": "Hero",
                "act": 3,
                "scene": 1,
                "explanation": "Hero notes how love can be sparked by subtle schemes, referring to the plot to unite Beatrice and Benedick."
            },
        ],
        "summary": "Beatrice and Benedick engage in a witty battle of words while Claudio and Hero fall in love, only for Don John's slander to disrupt their wedding. Through deception and revelation, the couples are united, and love triumphs over malice in Messina."
    },
    "pericles": {
        "title": "Pericles, Prince of Tyre",
        "full_title": "Pericles, Prince of Tyre",
        "category": "history",
        "year": "1607-1608",
        "setting": "Various Mediterranean locations",
        "main_characters": ["Pericles", "Marina", "Thaisa", "Gower", "Cleon", "Dionyza", "Lysimachus"],
        "character_descriptions": [
            {"name": "Pericles", "description": "Prince of Tyre, seeks family"},
            {"name": "Marina", "description": "Pericles's daughter, survives hardship"},
            {"name": "Thaisa", "description": "Pericles's wife, presumed dead"},
            {"name": "Gower", "description": "Narrator, guides play's story"},
            {"name": "Cleon", "description": "Governor, raises Marina"},
            {"name": "Dionyza", "description": "Cleon's wife, plots against Marina"},
            {"name": "Lysimachus", "description": "Governor, loves Marina"}
        ],
        "themes": [
            {
                "theme": "Adventure",
                "theme_explanation": "Pericles’s journey across the Mediterranean, filled with shipwrecks and trials, drives the narrative, symbolizing life’s unpredictable challenges."
            },
            {
                "theme": "Loss",
                "theme_explanation": "Pericles endures profound loss, believing his wife and daughter dead, which underscores the emotional toll of separation and grief."
            },
            {
                "theme": "Reunion",
                "theme_explanation": "The miraculous reunions with Thaisa and Marina highlight the restorative power of family and fate, resolving the play’s tragedies."
            },
            {
                "theme": "Fate",
                "theme_explanation": "Fate guides Pericles’s trials and eventual reunions, suggesting a divine force orchestrates the characters’ destinies."
            },
            {
                "theme": "Family",
                "theme_explanation": "The enduring bond between Pericles, Thaisa, and Marina triumphs over separation, emphasizing family as a source of resilience and joy."
            },
            {
                "theme": "Virtue",
                "theme_explanation": "Marina’s purity and moral strength preserve her through hardship, contrasting with the corruption around her and affirming virtue’s triumph."
            }
        ],
        "famous_quotes": [
            {
                "quote": "Few love to hear the sins they love to act",
                "speaker": "Pericles",
                "act": 1,
                "scene": 1,
                "explanation": "Pericles reflects on human reluctance to acknowledge their flaws, setting the tone for his moral journey."
            },
            {
                "quote": "Time's the king of men; he's both their parent, and he is their grave",
                "speaker": "Gower",
                "act": 2,
                "scene": 0,
                "explanation": "Gower emphasizes time's dual role in creating and destroying life, framing the play's episodic structure."
            },
            {
                "quote": "O you gods! Why do you make us love your goodly gifts, and snatch them straight away?",
                "speaker": "Pericles",
                "act": 3,
                "scene": 1,
                "explanation": "Pericles laments the loss of Thaisa, questioning divine cruelty in granting and then taking away joy."
            },
            {
                "quote": "The music of the spheres!",
                "speaker": "Pericles",
                "act": 5,
                "scene": 1,
                "explanation": "Pericles hears celestial harmony upon reuniting with Marina, symbolizing divine order and restoration."
            },
            {
                "quote": "In her you have a treasure",
                "speaker": "Lysimachus",
                "act": 5,
                "scene": 1,
                "explanation": "Lysimachus praises Marina's virtue, recognizing her worth as Pericles's daughter and a symbol of hope."
            }
        ],
        "summary": "Pericles, Prince of Tyre, embarks on a series of adventures, facing shipwrecks, loss, and betrayal, believing his wife Thaisa and daughter Marina dead. Through miraculous reunions guided by fate, he is joyfully reunited with his family in a tale of resilience and redemption."
    },
    "richard_ii": {
        "title": "King Richard II",
        "full_title": "The Tragedy of King Richard the Second",
        "category": "history",
        "year": "1595",
        "setting": "England, late 14th century",
        "main_characters": ["King Richard II", "Henry Bolingbroke", "John of Gaunt", "Duke of York", "Duchess of Gloucester", "Aumerle", "Mowbray"],
        "character_descriptions": [
            {"name": "King Richard II", "description": "King of England, deposed"},
            {"name": "Henry Bolingbroke", "description": "Rebel, becomes Henry IV"},
            {"name": "John of Gaunt", "description": "Richard's uncle, loyal noble"},
            {"name": "Duke of York", "description": "Richard's uncle, supports Bolingbroke"},
            {"name": "Duchess of Gloucester", "description": "Widow, seeks justice"},
            {"name": "Aumerle", "description": "York's son, loyal to Richard"},
            {"name": "Mowbray", "description": "Noble, exiled for dispute"}
        ],
        "themes": [
            {
                "theme": "Kingship",
                "theme_explanation": "Richard’s flawed leadership and belief in divine authority contrast with Bolingbroke’s pragmatic claim to power, questioning the nature of rightful rule."
            },
            {
                "theme": "Power",
                "theme_explanation": "The struggle for power drives Bolingbroke’s rebellion, exposing the fragility of Richard’s reign and the political consequences of mismanagement."
            },
            {
                "theme": "Divine right",
                "theme_explanation": "Richard’s faith in his God-given right to rule crumbles as he is deposed, challenging the concept of monarchy’s sacred inviolability."
            },
            {
                "theme": "Betrayal",
                "theme_explanation": "Betrayals, such as York’s shift to Bolingbroke and Richard’s nobles abandoning him, underscore the precarious loyalty in a divided court."
            },
            {
                "theme": "Identity",
                "theme_explanation": "Richard grapples with his loss of kingship, questioning his identity as he transitions from anointed monarch to a mere mortal."
            },
            {
                "theme": "Exile",
                "theme_explanation": "Exile, both literal (Bolingbroke’s banishment) and metaphorical (Richard’s isolation), highlights the personal and political alienation central to the play."
            }
        ],
        "famous_quotes": [
            {
                "quote": "This royal throne of kings, this sceptred isle",
                "speaker": "John of Gaunt",
                "act": 2,
                "scene": 1,
                "explanation": "Gaunt's patriotic speech glorifies England, lamenting its decline under Richard's mismanagement."
            },
            {
                "quote": "I wasted time, and now doth time waste me",
                "speaker": "King Richard II",
                "act": 5,
                "scene": 5,
                "explanation": "Richard reflects on his squandered reign, now imprisoned and powerless, as time overtakes him."
            },
            {
                "quote": "For God's sake, let us sit upon the ground and tell sad stories of the death of kings",
                "speaker": "King Richard II",
                "act": 3,
                "scene": 2,
                "explanation": "Richard mourns his fading power, contemplating the mortality of kings in a moment of self-pity."
            },
            {
                "quote": "The shadow of my sorrow!",
                "speaker": "King Richard II",
                "act": 4,
                "scene": 1,
                "explanation": "Richard sees his reflection as a mere shadow, symbolizing his loss of identity and authority."
            },
            {
                "quote": "Not all the water in the rough rude sea can wash the balm off from an anointed king",
                "speaker": "King Richard II",
                "act": 3,
                "scene": 2,
                "explanation": "Richard asserts the divine sanctity of his kingship, resisting the idea that it can be stripped away."
            }
        ],
        "summary": "King Richard II's mismanagement and seizure of noble lands provoke Henry Bolingbroke's rebellion. Exiled and then deposed, Richard is imprisoned and murdered, while Bolingbroke becomes Henry IV, grappling with the consequences of usurping the throne."
    },
    "richard_iii": {
        "title": "King Richard III",
        "full_title": "The Tragedy of King Richard the Third",
        "category": "history",
        "year": "1592-1593",
        "setting": "England, late 15th century",
        "main_characters": ["Richard III", "Duke of Buckingham", "Queen Margaret", "Lady Anne", "King Edward IV", "Clarence", "Richmond"],
        "character_descriptions": [
            {"name": "Richard III", "description": "Scheming duke, becomes king"},
            {"name": "Duke of Buckingham", "description": "Richard's ally, later betrayed"},
            {"name": "Queen Margaret", "description": "Lancastrian widow, curses Richard"},
            {"name": "Lady Anne", "description": "Widow, reluctantly marries Richard"},
            {"name": "King Edward IV", "description": "Yorkist king, Richard's brother"},
            {"name": "Clarence", "description": "Richard's brother, murdered by him"},
            {"name": "Richmond", "description": "Rebel, becomes Henry VII"}
        ],
        "themes": [
            {
                "theme": "Ambition",
                "theme_explanation": "Richard’s ruthless ambition drives him to murder and manipulate his way to the throne, illustrating the destructive force of unchecked desire."
            },
            {
                "theme": "Manipulation",
                "theme_explanation": "Richard’s cunning manipulation of allies and enemies, including Lady Anne and Buckingham, secures his power but leads to his isolation."
            },
            {
                "theme": "Evil",
                "theme_explanation": "Richard embodies moral corruption, committing heinous acts without remorse, yet his conscience briefly haunts him before his downfall."
            },
            {
                "theme": "Power",
                "theme_explanation": "The pursuit of power fuels Richard’s tyranny, but his brutal reign sparks rebellion, revealing power’s instability when built on treachery."
            },
            {
                "theme": "Fate",
                "theme_explanation": "Margaret’s curses and omens suggest a fated retribution for Richard’s crimes, culminating in his defeat by Richmond at Bosworth."
            },
            {
                "theme": "Conscience",
                "theme_explanation": "Richard’s fleeting guilt before the battle reveals the power of conscience, though he suppresses it to maintain his villainous resolve."
            }
        ],
        "famous_quotes": [
            {
                "quote": "Now is the winter of our discontent",
                "speaker": "Richard III",
                "act": 1,
                "scene": 1,
                "explanation": "Richard opens the play, revealing his ambition to seize power, framing his discontent as a catalyst for villainy."
            },
            {
                "quote": "A horse! A horse! My kingdom for a horse!",
                "speaker": "Richard III",
                "act": 5,
                "scene": 4,
                "explanation": "In battle, Richard desperately seeks escape, revealing his vulnerability as his reign collapses."
            },
            {
                "quote": "Conscience is but a word that cowards use",
                "speaker": "Richard III",
                "act": 5,
                "scene": 3,
                "explanation": "Richard dismisses guilt to steel himself against remorse, highlighting his defiance of moral constraints."
            },
            {
                "quote": "I am not in the giving vein today",
                "speaker": "Richard III",
                "act": 4,
                "scene": 2,
                "explanation": "Richard coldly refuses Buckingham's request, showing his growing paranoia and isolation as king."
            },
            {
                "quote": "Bloody thou art, bloody will be thy end",
                "speaker": "Duchess of York",
                "act": 4,
                "scene": 4,
                "explanation": "The Duchess curses Richard, predicting his violent demise as retribution for his murderous deeds."
            },
            {
                "quote": "Off with his head!",
                "speaker": "Richard III",
                "act": 3,
                "scene": 4,
                "explanation": "Richard, a ruthless and tyrannical king, issues a swift and brutal command for an execution, demonstrating his absolute power and disregard for human life."
            }
        ],
        "summary": "Richard, Duke of Gloucester, schemes his way to the throne through murder and manipulation, eliminating rivals like Clarence and the young princes. His tyranny sparks rebellion, and he is defeated by Richmond (Henry VII) at Bosworth Field, ending the Wars of the Roses."
    },
    "taming_of_the_shrew": {
        "title": "The Taming of the Shrew",
        "full_title": "The Taming of the Shrew",
        "category": "comedy",
        "year": "1590-1592",
        "setting": "Padua, Italy",
        "main_characters": ["Katherina", "Petruchio", "Bianca", "Lucentio", "Baptista", "Tranio", "Grumio"],
        "character_descriptions": [
            {"name": "Katherina", "description": "Baptista's daughter, tamed by Petruchio"},
            {"name": "Petruchio", "description": "Gentleman, marries Katherina for dowry"},
            {"name": "Bianca", "description": "Katherina's sister, sought by suitors"},
            {"name": "Lucentio", "description": "Gentleman, loves and marries Bianca"},
            {"name": "Baptista", "description": "Wealthy father of Katherina, Bianca"},
            {"name": "Tranio", "description": "Lucentio's servant, aids disguise"},
            {"name": "Grumio", "description": "Petruchio's servant, comic assistant"}
        ],
        "themes": [
            {
                "theme": "Marriage",
                "theme_explanation": "The play explores marriage through Petruchio’s domineering ‘taming’ of Katherina and Bianca’s courtship, raising questions about love versus control."
            },
            {
                "theme": "Gender roles",
                "theme_explanation": "Katherina’s defiance of traditional femininity challenges gender norms, while her apparent submission sparks debate over societal expectations for women."
            },
            {
                "theme": "Power",
                "theme_explanation": "Petruchio’s control over Katherina and Baptista’s authority over his daughters highlight power dynamics in relationships and family structures."
            },
            {
                "theme": "Transformation",
                "theme_explanation": "Katherina’s shift from shrew to obedient wife, whether genuine or performative, underscores the play’s focus on personal and social change."
            },
            {
                "theme": "Deception",
                "theme_explanation": "Deceptions, like Lucentio’s disguise and Petruchio’s exaggerated behavior, drive the comedic plot and complicate romantic pursuits."
            },
            {
                "theme": "Love",
                "theme_explanation": "Love is tested through mercenary motives (Petruchio’s dowry) and romantic ideals (Lucentio and Bianca), questioning its authenticity and durability."
            }
        ],
        "famous_quotes": [
            {
                "quote": "I come to wive it wealthily in Padua",
                "speaker": "Petruchio",
                "act": 1,
                "scene": 2,
                "explanation": "Petruchio declares his intent to marry for money, revealing his pragmatic, if mercenary, approach to marriage."
            },
            {
                "quote": "There's small choice in rotten apples",
                "speaker": "Hortensio",
                "act": 1,
                "scene": 1,
                "explanation": "Hortensio laments the lack of good marriage prospects, reflecting the competitive pursuit of Bianca."
            },
            {
                "quote": "Kiss me, Kate, we will be married o' Sunday",
                "speaker": "Petruchio",
                "act": 2,
                "scene": 1,
                "explanation": "Petruchio boldly demands Katherina's affection, asserting control despite her resistance, foreshadowing their dynamic."
            },
            {
                "quote": "This is a way to kill a wife with kindness",
                "speaker": "Petruchio",
                "act": 4,
                "scene": 1,
                "explanation": "Petruchio describes his strategy of ‘taming' Katherina through excessive care, a controversial tactic of manipulation."
            },
            {
                "quote": "Thy husband is thy lord, thy life, thy keeper",
                "speaker": "Katherina",
                "act": 5,
                "scene": 2,
                "explanation": "Katherina's final speech advocates wifely submission, sparking debate over whether it reflects genuine change or irony."
            }
        ],
        "summary": "Petruchio seeks to marry the sharp-tongued Katherina to secure her dowry, using unconventional methods to ‘tame' her spirited nature. Meanwhile, Lucentio woos Katherina's sister Bianca through disguise, leading to a resolution where Katherina appears subdued, though her transformation remains ambiguous."
    },
    "timon_of_athens": {
        "title": "The Life of Timon of Athens",
        "full_title": "The Tragedy of Timon of Athens",
        "category": "tragedy",
        "year": "1605-1608",
        "setting": "Athens, Greece",
        "main_characters": ["Timon", "Apemantus", "Alcibiades", "Flavius", "Ventidius", "Lucius", "Sempronius"],
        "character_descriptions": [
            {"name": "Timon", "description": "Wealthy Athenian, becomes misanthrope"},
            {"name": "Apemantus", "description": "Cynical philosopher, criticizes society"},
            {"name": "Alcibiades", "description": "General, rebels against Athens"},
            {"name": "Flavius", "description": "Timon's loyal steward"},
            {"name": "Ventidius", "description": "False friend, abandons Timon"},
            {"name": "Lucius", "description": "False friend, rejects Timon"},
            {"name": "Sempronius", "description": "False friend, betrays Timon"}
        ],
        "themes": [
            {
                "theme": "Wealth",
                "theme_explanation": "Timon’s lavish spending reveals wealth’s fleeting nature, as his fortune vanishes, exposing the superficiality of material prosperity."
            },
            {
                "theme": "Friendship",
                "theme_explanation": "The betrayal by Timon’s false friends contrasts with Flavius’s loyalty, questioning the authenticity of relationships built on wealth."
            },
            {
                "theme": "Betrayal",
                "theme_explanation": "Timon’s abandonment by his supposed friends after his bankruptcy fuels his descent into misanthropy and rage against humanity."
            },
            {
                "theme": "Misanthropy",
                "theme_explanation": "Timon’s transformation into a bitter misanthrope reflects his disillusionment with human greed and ingratitude after his fall from grace."
            },
            {
                "theme": "Generosity",
                "theme_explanation": "Timon’s reckless generosity leads to his ruin, critiquing unchecked altruism in a society driven by self-interest."
            },
            {
                "theme": "Corruption",
                "theme_explanation": "Athens’s corrupt, materialistic society, exemplified by Timon’s false friends and the senators, underscores the moral decay that alienates Timon."
            }
        ],
        "famous_quotes": [
            {
                "quote": "We have seen better days",
                "speaker": "Flavius",
                "act": 4,
                "scene": 2,
                "explanation": "Flavius laments Timon's fall from wealth, highlighting the fleeting nature of fortune and loyalty."
            },
            {
                "quote": "I am Misanthropos, and hate mankind",
                "speaker": "Timon",
                "act": 4,
                "scene": 3,
                "explanation": "Timon declares his hatred for humanity after his friends' betrayal, embracing total misanthropy."
            },
            {
                "quote": "Nothing emboldens sin so much as mercy",
                "speaker": "Senator",
                "act": 3,
                "scene": 5,
                "explanation": "The Senator argues that leniency encourages wrongdoing, reflecting Athens' harsh, self-serving values."
            },
            {
                "quote": "The sun's a thief, and with his great attraction robs the vast sea",
                "speaker": "Timon",
                "act": 4,
                "scene": 3,
                "explanation": "Timon rails against nature's cycles as thievery, mirroring his view of human greed and betrayal."
            },
            {
                "quote": "Men shut their doors against a setting sun",
                "speaker": "Apemantus",
                "act": 1,
                "scene": 2,
                "explanation": "Apemantus cynically notes that people abandon those whose fortunes fade, foreshadowing Timon's fate."
            }
        ],
        "summary": "Timon, a wealthy Athenian, lavishes gifts on false friends, only to be abandoned when his fortunes fail. Disillusioned, he becomes a misanthrope, cursing humanity and living in exile, where he discovers gold but dies in isolation, rejecting reconciliation."
    },
    "the_two_gentlemen_of_verona": {
        "title": "The Two Gentlemen of Verona",
        "full_title": "The Two Gentlemen of Verona",
        "category": "comedy",
        "year": "1590–1593",
        "setting": "Verona and Milan, Italy",
        "main_characters": ["Valentine", "Proteus", "Julia", "Silvia", "Duke of Milan", "Lucetta", "Thurio", "Launce", "Speed"],
        "character_descriptions": [
            {"name": "Valentine", "description": "Gentleman, loves Silvia"},
            {"name": "Proteus", "description": "Gentleman, betrays Valentine, loves Silvia"},
            {"name": "Julia", "description": "Proteus's lover, disguises as page"},
            {"name": "Silvia", "description": "Duke's daughter, loves Valentine"},
            {"name": "Duke of Milan", "description": "Silvia's father, plans her marriage"},
            {"name": "Lucetta", "description": "Julia's maid, aids disguise"},
            {"name": "Thurio", "description": "Silvia's suitor, Duke's choice"},
            {"name": "Launce", "description": "Proteus's servant, comic figure"},
            {"name": "Speed", "description": "Valentine's servant, witty assistant"}
        ],
        "themes": [
            {
                "theme": "Friendship",
                "theme_explanation": "The bond between Valentine and Proteus is tested by Proteus’s betrayal, highlighting the fragility of friendship when challenged by love."
            },
            {
                "theme": "Love and Betrayal",
                "theme_explanation": "Proteus’s fickle shift from Julia to Silvia and his betrayal of Valentine reveal love’s potential to inspire disloyalty and conflict."
            },
            {
                "theme": "Disguise and Gender Roles",
                "theme_explanation": "Julia’s disguise as a male page challenges gender norms, allowing her to navigate and influence romantic outcomes with agency."
            },
            {
                "theme": "Loyalty",
                "theme_explanation": "Loyalty is strained by Proteus’s actions, but Julia’s steadfast devotion and Valentine’s forgiveness restore harmony among the group."
            },
            {
                "theme": "Forgiveness",
                "theme_explanation": "The play resolves through Valentine’s forgiveness of Proteus, emphasizing reconciliation over retribution despite serious betrayals."
            }
        ],
        "famous_quotes": [
            {
                "quote": "They do not love that do not show their love.",
                "speaker": "Julia",
                "act": 1,
                "scene": 2,
                "explanation": "Julia asserts that true love must be expressed openly, reflecting her commitment to Proteus."
            },
            {
                "quote": "What light is light, if Silvia be not seen?",
                "speaker": "Proteus",
                "act": 2,
                "scene": 6,
                "explanation": "Proteus idealizes Silvia, revealing his fickle passion that betrays his loyalty to Julia and Valentine."
            },
            {
                "quote": "Except I be by Silvia in the night, There is no music in the nightingale.",
                "speaker": "Proteus",
                "act": 3,
                "scene": 1,
                "explanation": "Proteus romanticizes his desire for Silvia, equating her presence with beauty, despite his disloyalty."
            },
            {
                "quote": "O, how this spring of love resembleth The uncertain glory of an April day.",
                "speaker": "Proteus",
                "act": 1,
                "scene": 3,
                "explanation": "Proteus compares love's fleeting joy to changeable weather, foreshadowing his own inconstancy."
            },
            {
                "quote": "That man that hath a tongue, I say is no man, If with his tongue he cannot win a woman.",
                "speaker": "Valentine",
                "act": 3,
                "scene": 1,
                "explanation": "Valentine emphasizes the power of eloquence in courtship, reflecting his confident pursuit of Silvia."
            }
        ],
        "summary": "Best friends Valentine and Proteus part ways when Valentine leaves Verona for Milan. Proteus, originally in love with Julia, follows and betrays both his friend and his lover by falling for Silvia, whom Valentine also loves. Julia disguises herself to follow Proteus. In the end, love is tested, betrayal is forgiven, and the friends and lovers reconcile."
    },
    "titus_andronicus": {
        "title": "The Tragedy of Titus Andronicus",
        "full_title": "The Most Lamentable Roman Tragedy of Titus Andronicus",
        "category": "tragedy",
        "year": "1593-1594",
        "setting": "Rome",
        "main_characters": ["Titus Andronicus", "Tamora", "Aaron", "Lavinia", "Saturninus", "Bassianus", "Lucius"],
        "character_descriptions": [
            {"name": "Titus Andronicus", "description": "Roman general, seeks revenge"},
            {"name": "Tamora", "description": "Goth queen, plots against Titus"},
            {"name": "Aaron", "description": "Tamora's lover, orchestrates villainy"},
            {"name": "Lavinia", "description": "Titus's daughter, brutally assaulted"},
            {"name": "Saturninus", "description": "Roman emperor, marries Tamora"},
            {"name": "Bassianus", "description": "Saturninus's brother, loves Lavinia"},
            {"name": "Lucius", "description": "Titus's son, avenges family"}
        ],
        "themes": [
            {
                "theme": "Revenge",
                "theme_explanation": "A relentless cycle of revenge between Titus and Tamora drives the play, escalating violence and leading to mutual destruction."
            },
            {
                "theme": "Violence",
                "theme_explanation": "Graphic violence, from Lavinia’s mutilation to the cannibalistic banquet, underscores the play’s exploration of human cruelty and retribution."
            },
            {
                "theme": "Betrayal",
                "theme_explanation": "Betrayals, such as Tamora’s manipulation of Saturninus and Aaron’s treachery, fuel the tragic conflicts and deepen the chaos."
            },
            {
                "theme": "Family",
                "theme_explanation": "Titus’s devotion to his family motivates his revenge, while Tamora’s actions to protect her sons mirror the fierce loyalty driving both sides."
            },
            {
                "theme": "Power",
                "theme_explanation": "The struggle for power in Rome, exemplified by Saturninus’s rule and Tamora’s influence, amplifies the destructive ambitions of the characters."
            },
            {
                "theme": "Savagery",
                "theme_explanation": "The play contrasts Roman civilization with barbaric acts, questioning whether savagery lies in the Goths or within Rome’s own moral decay."
            }
        ],
        "famous_quotes": [
            {
                "quote": "Vengeance is in my heart, death in my hand",
                "speaker": "Aaron",
                "act": 2,
                "scene": 3,
                "explanation": "Aaron revels in his villainy, embracing revenge as he orchestrates brutal acts against Titus's family."
            },
            {
                "quote": "She is a woman, therefore may be wooed",
                "speaker": "Demetrius",
                "act": 2,
                "scene": 1,
                "explanation": "Demetrius views Lavinia as an object to be possessed, foreshadowing the violent assault driven by lust."
            },
            {
                "quote": "O, why should wrath be mute, and fury dumb?",
                "speaker": "Titus",
                "act": 5,
                "scene": 2,
                "explanation": "Titus vows to express his rage through revenge, channeling his grief into a deadly plan."
            },
            {
                "quote": "I am the sea; hark, how her sighs do blow!",
                "speaker": "Titus",
                "act": 3,
                "scene": 1,
                "explanation": "Titus likens his overwhelming sorrow to a stormy sea, mourning Lavinia's suffering and his losses."
            },
            {
                "quote": "If one good deed in all my life I did, I do repent it from my very soul",
                "speaker": "Aaron",
                "act": 5,
                "scene": 3,
                "explanation": "Aaron defiantly rejects remorse, embracing his evil nature even as he faces death."
            }
        ],
        "summary": "Roman general Titus Andronicus returns victorious but spirals into a cycle of revenge when Tamora, Queen of the Goths, seeks retribution for her son's death. Brutal acts, including Lavinia's mutilation and Titus's feigned madness, culminate in a bloody banquet where nearly all perish."
    },
    "troilus_and_cressida": {
        "title": "Troilus and Cressida",
        "full_title": "Troilus and Cressida",
        "category": "tragedy",
        "year": "1601-1602",
        "setting": "Troy and the Greek camp, during the Trojan War",
        "main_characters": ["Troilus", "Cressida", "Pandarus", "Hector", "Achilles", "Ulysses", "Diomedes"],
        "character_descriptions": [
            {"name": "Troilus", "description": "Trojan prince, loves Cressida"},
            {"name": "Cressida", "description": "Trojan woman, betrays Troilus"},
            {"name": "Pandarus", "description": "Cressida's uncle, facilitates romance"},
            {"name": "Hector", "description": "Trojan warrior, killed by Achilles"},
            {"name": "Achilles", "description": "Greek warrior, kills Hector"},
            {"name": "Ulysses", "description": "Greek leader, manipulates Achilles"},
            {"name": "Diomedes", "description": "Greek warrior, wins Cressida"}
        ],
        "themes": [
            {
                "theme": "Love",
                "theme_explanation": "Troilus and Cressida’s romance, tainted by Cressida’s betrayal, reveals love’s fragility amidst the chaos of war."
            },
            {
                "theme": "War",
                "theme_explanation": "The Trojan War’s futility is exposed through the pride and inaction of warriors like Achilles, highlighting conflict’s senseless destruction."
            },
            {
                "theme": "Betrayal",
                "theme_explanation": "Cressida’s shift to Diomedes and the shifting allegiances in war, like Achilles’s refusal to fight, underscore pervasive disloyalty."
            },
            {
                "theme": "Honor",
                "theme_explanation": "Honor is debated, as Hector upholds it nobly while Achilles’s pride and Ulysses’s manipulations reveal its hollow pursuit."
            },
            {
                "theme": "Futility",
                "theme_explanation": "The play’s cynical tone emphasizes the futility of both love and war, with no lasting victories or resolutions achieved."
            },
            {
                "theme": "Deception",
                "theme_explanation": "Deception, from Pandarus’s matchmaking to Ulysses’s schemes, drives the plot, exposing the manipulative undercurrents of human motives."
            }
        ],
        "famous_quotes": [
            {
                "quote": "The common curse of mankind, folly and ignorance",
                "speaker": "Thersites",
                "act": 2,
                "scene": 3,
                "explanation": "Thersites bitterly condemns human flaws, reflecting the play's cynical view of war and love."
            },
            {
                "quote": "Words, words, mere words, no matter from the heart",
                "speaker": "Troilus",
                "act": 5,
                "scene": 3,
                "explanation": "Troilus dismisses empty promises, expressing disillusionment with Cressida's betrayal and rhetoric."
            },
            {
                "quote": "What's past and what's to come is strewed with husks",
                "speaker": "Ulysses",
                "act": 4,
                "scene": 5,
                "explanation": "Ulysses reflects on life's emptiness, highlighting the futility pervading the Trojan War."
            },
            {
                "quote": "Time hath, my lord, a wallet at his back",
                "speaker": "Ulysses",
                "act": 3,
                "scene": 3,
                "explanation": "Ulysses describes time as storing forgotten deeds, urging Achilles to act before his fame fades."
            },
            {
                "quote": "One touch of nature makes the whole world kin",
                "speaker": "Ulysses",
                "act": 3,
                "scene": 3,
                "explanation": "Ulysses notes that shared human flaws unite people, manipulating Achilles by appealing to universal traits."
            }
        ],
        "summary": "During the Trojan War, Troilus and Cressida fall in love, facilitated by her uncle Pandarus, but Cressida is traded to the Greeks and betrays Troilus with Diomedes. Meanwhile, the war's futility is exposed through the pride and inaction of warriors like Achilles, ending in Hector's death."
    },
    "winters_tale": {
        "title": "The Winter's Tale",
        "full_title": "The Winter's Tale",
        "category": "comedy",
        "year": "1610-1611",
        "setting": "Sicilia and Bohemia",
        "main_characters": ["Leontes", "Hermione", "Perdita", "Polixenes", "Camillo", "Paulina", "Autolycus", "Florizel"],
        "character_descriptions": [
            {"name": "Leontes", "description": "King of Sicilia, jealous husband"},
            {"name": "Hermione", "description": "Leontes's wife, falsely accused"},
            {"name": "Perdita", "description": "Leontes's daughter, raised in Bohemia"},
            {"name": "Polixenes", "description": "King of Bohemia, Leontes's friend"},
            {"name": "Camillo", "description": "Loyal advisor, serves both kings"},
            {"name": "Paulina", "description": "Hermione's friend, seeks justice"},
            {"name": "Autolycus", "description": "Rogue, aids Bohemia's comedy"},
            {"name": "Florizel", "description": "Polixenes's son, loves Perdita"}
        ],
        "themes": [
            {
                "theme": "Jealousy",
                "theme_explanation": "Leontes’s baseless jealousy destroys his family, driving the play’s tragic first half until redemption begins."
            },
            {
                "theme": "Redemption",
                "theme_explanation": "Leontes’s remorse and the restoration of his family through Perdita and Hermione offer redemption, healing past wrongs."
            },
            {
                "theme": "Time",
                "theme_explanation": "Time’s passage, spanning sixteen years, enables healing and transformation, bridging the play’s tragic and comedic halves."
            },
            {
                "theme": "Family",
                "theme_explanation": "The reunion of Leontes, Hermione, and Perdita underscores the enduring strength of family bonds despite betrayal and loss."
            },
            {
                "theme": "Forgiveness",
                "theme_explanation": "Forgiveness, particularly Hermione’s reconciliation with Leontes, resolves the play, emphasizing mercy over retribution."
            },
            {
                "theme": "Nature vs. nurture",
                "theme_explanation": "Perdita’s grace despite her rustic upbringing suggests innate nobility, questioning whether character stems from birth or environment."
            }
        ],
        "famous_quotes": [
            {
                "quote": "A sad tale's best for winter",
                "speaker": "Mamillius",
                "act": 2,
                "scene": 1,
                "explanation": "Mamillius sets a melancholic tone, foreshadowing the tragic events sparked by Leontes' jealousy."
            },
            {
                "quote": "I am a feather for each wind that blows",
                "speaker": "Leontes",
                "act": 2,
                "scene": 3,
                "explanation": "Leontes describes his unstable emotions, driven by baseless jealousy, which lead to his downfall."
            },
            {
                "quote": "What's gone and what's past help should be past grief",
                "speaker": "Paulina",
                "act": 3,
                "scene": 2,
                "explanation": "Paulina urges moving beyond irreparable loss, advocating resilience amid Leontes' remorse."
            },
            {
                "quote": "You have a holy father, a graceful gentleman",
                "speaker": "Florizel",
                "act": 5,
                "scene": 1,
                "explanation": "Florizel praises Leontes, unaware of his past sins, highlighting the theme of redemption."
            },
            {
                "quote": "Though I am not naturally honest, I am so sometimes by chance",
                "speaker": "Autolycus",
                "act": 4,
                "scene": 4,
                "explanation": "Autolycus humorously admits his roguish nature, adding comic relief to the pastoral Bohemian scenes."
            }
        ],
        "summary": "King Leontes of Sicilia, consumed by jealousy, accuses his wife Hermione of infidelity, leading to her apparent death and the abandonment of their daughter Perdita. Years later, Perdita, raised in Bohemia, falls in love with Florizel, and through revelations and a miraculous reunion, the family is restored."
    }
}

def create_knowledge_base(output_dir="data/cleaned/knowledge_base"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating knowledge base...")
    
    # Save individual play information
    for play_id, info in PLAY_INFO.items():
        print(f"Adding information for {info['title']}")
        with open(os.path.join(output_dir, f"{play_id}.json"), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    
    # # Create a file with general Shakespeare information
    # shakespeare_info = {
    #     "name": "William Shakespeare",
    #     "birth_year": "1564",
    #     "death_year": "1616",
    #     "birthplace": "Stratford-upon-Avon, England",
    #     "biography": "William Shakespeare was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist. He is often called England's national poet and the 'Bard of Avon'. His works consist of approximately 37 plays, 154 sonnets, two long narrative poems, and a few other verses. His plays have been translated into every major living language and are performed more often than those of any other playwright.",
    #     "education": "Grammar school education at King's New School in Stratford, no university record",
    #     "family": {
    #         "parents": ["John Shakespeare", "Mary Arden"],
    #         "spouse": "Anne Hathaway (married 1582)",
    #         "children": ["Susanna", "Hamnet (died at age 11)", "Judith"]
    #     },
    #     "career": {
    #         "company": "Lord Chamberlain's Men (later King's Men)",
    #         "theaters": ["The Globe Theatre", "Blackfriars Theatre"],
    #         "periods": ["Early (1589-1594)", "Middle (1595-1600)", "Later (1601-1608)", "Final (1609-1613)"]
    #     },
    #     "works_categories": {
    #         "comedies": ["A Midsummer Night's Dream", "Much Ado About Nothing", "Twelfth Night", "As You Like It", "The Comedy of Errors", "The Merchant of Venice", "The Taming of the Shrew"],
    #         "tragedies": ["Hamlet", "Macbeth", "Romeo and Juliet", "Othello", "King Lear", "Julius Caesar", "Antony and Cleopatra"],
    #         "histories": ["Richard III", "Henry V", "Henry IV", "Richard II", "King John", "Henry VI", "Henry VIII"],
    #         "romances": ["The Tempest", "The Winter's Tale", "Cymbeline", "Pericles"]
    #     },
    #     "poetry": {
    #         "sonnets": "154 sonnets published in 1609",
    #         "narrative_poems": ["Venus and Adonis", "The Rape of Lucrece"]
    #     },
    #     "writing_style": {
    #         "language": "Early Modern English",
    #         "verse_form": "Primarily iambic pentameter",
    #         "devices": ["Blank verse", "Soliloquies", "Asides", "Puns", "Extended metaphors", "Dramatic irony"]
    #     },
    #     "literary_innovations": [
    #         "Expanded vocabulary (estimated to have used over 20,000 words)",
    #         "Created numerous original phrases now common in English",
    #         "Psychological depth of characters",
    #         "Complex plots with multiple interweaving storylines"
    #     ],
    #     "cultural_impact": "Shakespeare's works have influenced countless writers, dramatists, and cultural works across the centuries. His plays remain among the most performed, adapted, and studied literary works worldwide.",
    #     "historical_context": {
    #         "era": "Elizabethan/Jacobean England",
    #         "monarchs": ["Elizabeth I", "James I"],
    #         "society": "Renaissance period, growth of theater, religious tensions, voyages of discovery"
    #     }
    # }
    
    # with open(os.path.join(output_dir, "shakespeare_general.json"), 'w', encoding='utf-8') as f:
    #     json.dump(shakespeare_info, f, indent=2)
    
    # # Create a file with literary device information
    # literary_devices = {
    #     "devices": [
    #         {
    #             "name": "Soliloquy",
    #             "definition": "A speech where a character speaks their thoughts aloud while alone on stage",
    #             "examples": [
    #                 {"play": "Hamlet", "character": "Hamlet", "quote": "To be, or not to be, that is the question"},
    #                 {"play": "Macbeth", "character": "Macbeth", "quote": "Is this a dagger which I see before me"}
    #             ]
    #         },
    #         {
    #             "name": "Aside",
    #             "definition": "A remark made by a character that is intended to be heard by the audience but not by other characters on stage",
    #             "examples": [
    #                 {"play": "Othello", "character": "Iago", "quote": "O, you are well tuned now, but I'll set down the pegs that make this music"}
    #             ]
    #         },
    #         {
    #             "name": "Dramatic irony",
    #             "definition": "When the audience knows something that characters do not",
    #             "examples": [
    #                 {"play": "Romeo and Juliet", "description": "Audience knows Juliet is not dead, but Romeo doesn't"}
    #             ]
    #         },
    #         {
    #             "name": "Blank verse",
    #             "definition": "Unrhymed iambic pentameter, the standard meter of Shakespeare's plays",
    #             "examples": [
    #                 {"play": "Julius Caesar", "character": "Brutus", "quote": "Not that I loved Caesar less, but that I loved Rome more"}
    #             ]
    #         },
    #         {
    #             "name": "Foreshadowing",
    #             "definition": "The use of hints or clues to suggest events that will occur later in the story",
    #             "examples": [
    #                 {"play": "Macbeth", "description": "The witches' prophecies foreshadow future events"}
    #             ]
    #         },
    #         {
    #             "name": "Comic relief",
    #             "definition": "A humorous scene or character that provides relief from tense or serious elements",
    #             "examples": [
    #                 {"play": "Hamlet", "character": "Gravediggers", "description": "Provide humor amidst the tragedy"}
    #             ]
    #         },
    #         {
    #             "name": "Tragic flaw",
    #             "definition": "A character trait that leads to a character's downfall",
    #             "examples": [
    #                 {"play": "Othello", "character": "Othello", "flaw": "Jealousy"},
    #                 {"play": "Macbeth", "character": "Macbeth", "flaw": "Ambition"}
    #             ]
    #         },
    #         {
    #             "name": "Pun",
    #             "definition": "A play on words that exploits multiple meanings",
    #             "examples": [
    #                 {"play": "Romeo and Juliet", "character": "Mercutio", "quote": "Ask for me tomorrow, and you shall find me a grave man"}
    #             ]
    #         }
    #     ],
    #     "verse_forms": [
    #         {
    #             "name": "Iambic pentameter",
    #             "description": "A line of verse with five metrical feet, each consisting of one short/unstressed syllable followed by one long/stressed syllable",
    #             "example": "Shall I compare thee to a summer's day?"
    #         },
    #         {
    #             "name": "Prose",
    #             "description": "Ordinary writing without metrical structure, often used for common characters or comic scenes",
    #             "example": "Falstaff's speeches in Henry IV"
    #         },
    #         {
    #             "name": "Rhymed verse",
    #             "description": "Poetry with a regular rhyme scheme, often used for emphasis or to signal the end of scenes",
    #             "example": "Witches' chants in Macbeth"
    #         }
    #     ]
    # }
    
    # with open(os.path.join(output_dir, "literary_devices.json"), 'w', encoding='utf-8') as f:
    #     json.dump(literary_devices, f, indent=2)
    
    # # Create a file with historical context
    # historical_context = {
    #     "elizabethan_era": {
    #         "period": "1558-1603",
    #         "monarch": "Queen Elizabeth I",
    #         "society": {
    #             "social_structure": "Strict class hierarchy with nobility at top, followed by gentry, merchants, and peasants",
    #             "religion": "Protestant Church of England established, but religious tensions remained",
    #             "entertainment": "Public theaters, bear-baiting, cock-fighting, and public executions",
    #             "education": "Grammar schools for boys, with focus on Latin and classical texts"
    #         },
    #         "theater": {
    #             "public_theaters": ["The Globe", "The Rose", "The Swan"],
    #             "private_theaters": ["Blackfriars"],
    #             "restrictions": "Theaters located outside city limits due to Puritan influence on city government",
    #             "companies": "All-male acting troupes, with boys playing female roles",
    #             "performances": "Daylight performances with minimal scenery and rich costumes"
    #         }
    #     },
    #     "jacobean_era": {
    #         "period": "1603-1625",
    #         "monarch": "King James I",
    #         "changes": "More somber tone in drama, darker themes explored",
    #         "shakespeare's_late_works": ["The Tempest", "The Winter's Tale", "Cymbeline"]
    #     },
    #     "influence_on_plays": {
    #         "political_concerns": {
    #             "monarchy": "Divine right of kings reflected in history plays",
    #             "succession": "Anxiety about succession apparent in plays like King Lear",
    #             "foreign_relations": "Settings in Italy, France, and ancient worlds allowed for political commentary"
    #         },
    #         "social_issues": {
    #             "class": "Crossing class boundaries explored in comedies",
    #             "gender": "Cross-dressing heroines challenge gender norms",
    #             "race": "Otherness explored through characters like Othello and Shylock"
    #         }
    #     }
    # }
    
    # with open(os.path.join(output_dir, "historical_context.json"), 'w', encoding='utf-8') as f:
    #     json.dump(historical_context, f, indent=2)
    
    print("Knowledge base creation complete.")

if __name__ == "__main__":
    create_knowledge_base()