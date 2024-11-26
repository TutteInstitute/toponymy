from toponymy.keyphrases import (
    build_object_x_keyphrase_matrix,
    build_keyphrase_vocabulary,
    build_keyphrase_count_matrix,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter

import numpy as np

import pytest

TEST_OBJECTS = [
    "The quick brown fox jumps over the lazy dog. Actions speak louder than words.",
    "A journey of a thousand miles begins with a single step. Practice makes perfect.",
    "To be or not to be, that is the question. The pen is mightier than the sword.",
    "All that glitters is not gold. Beauty is in the eye of the beholder.",
    "The pen is mightier than the sword. The squeaky wheel gets the grease.",
    "A picture is worth a thousand words. A bird in the hand is worth two in the bush.",
    "When in Rome, do as the Romans do. Where there's a will, there's a way.",
    "The early bird catches the worm. Early to bed and early to rise makes a man healthy, wealthy, and wise.",
    "Actions speak louder than words. The proof is in the pudding.",
    "A watched pot never boils. Good things come to those who wait.",
    "Beggars can't be choosers. You can't have your cake and eat it too.",
    "Better late than never. It's never too late to learn.",
    "Birds of a feather flock together. No man is an island.",
    "Cleanliness is next to godliness. A penny saved is a penny earned.",
    "Don't count your chickens before they hatch. Don't put off until tomorrow what you can do today.",
    "Don't put all your eggs in one basket. Variety is the spice of life.",
    "Every cloud has a silver lining. The best things in life are free.",
    "Fortune favors the bold. Nothing ventured, nothing gained.",
    "Honesty is the best policy. Ignorance is bliss.",
    "If it ain't broke, don't fix it. If you want something done right, do it yourself.",
    "If you can't beat them, join them. When the going gets tough, the tough get going.",
    "It's always darkest before the dawn. There's no time like the present.",
    "Laughter is the best medicine. Time flies when you're having fun.",
    "Let sleeping dogs lie. Curiosity killed the cat.",
    "Look before you leap. Haste makes waste.",
    "No man is an island. One man's trash is another man's treasure.",
    "Practice makes perfect. Rome wasn't built in a day.",
    "The grass is always greener on the other side. Out of sight, out of mind.",
    "The squeaky wheel gets the grease. The road to hell is paved with good intentions.",
    "There's no place like home. Home is where the heart is.",
    "Time heals all wounds. Time is money.",
    "Too many cooks spoil the broth. Two heads are better than one.",
    "When the going gets tough, the tough get going. You reap what you sow.",
    "You can't judge a book by its cover. Don't judge a book by its cover.",
    "You can't make an omelet without breaking a few eggs. You can't teach an old dog new tricks.",
    "A bird in the hand is worth two in the bush. A chain is only as strong as its weakest link.",
    "A penny saved is a penny earned. Money doesn't grow on trees.",
    "A rolling stone gathers no moss. Absence makes the heart grow fonder.",
    "Actions speak louder than words. The pen is mightier than the sword.",
    "All good things must come to an end. Every rose has its thorn.",
    "All's fair in love and war. The proof is in the pudding.",
    "An apple a day keeps the doctor away. Laughter is the best medicine.",
    "Beauty is in the eye of the beholder. Better safe than sorry.",
    "Curiosity killed the cat. Let bygones be bygones.",
    "Don't bite the hand that feeds you. Beggars can't be choosers.",
    "Don't judge a book by its cover. You can't judge a book by its cover.",
    "Don't look a gift horse in the mouth. Don't put off until tomorrow what you can do today.",
    "Early to bed and early to rise makes a man healthy, wealthy, and wise. The early bird catches the worm.",
    "Every rose has its thorn. All good things must come to an end.",
    "Good things come to those who wait. Patience is a virtue.",
    "Haste makes waste. Look before you leap.",
    "If at first you don't succeed, try, try again. Practice makes perfect.",
    "If you want something done right, do it yourself. If it ain't broke, don't fix it.",
    "Ignorance is bliss. Honesty is the best policy.",
    "It's better to give than to receive. It's never too late to learn.",
    "Knowledge is power. Ignorance is bliss.",
    "Laughter is the best medicine. An apple a day keeps the doctor away.",
    "Let bygones be bygones. Curiosity killed the cat.",
    "Money doesn't grow on trees. A penny saved is a penny earned.",
    "Necessity is the mother of invention. If you want something done right, do it yourself.",
    "No pain, no gain. Nothing ventured, nothing gained.",
    "Old habits die hard. You can't teach an old dog new tricks.",
    "Out of sight, out of mind. The grass is always greener on the other side.",
    "Patience is a virtue. Good things come to those who wait.",
    "Practice makes perfect. If at first you don't succeed, try, try again.",
    "Rome wasn't built in a day. Practice makes perfect.",
    "Silence is golden. The best things in life are free.",
    "The best things in life are free. Every cloud has a silver lining.",
    "The early bird catches the worm. Early to bed and early to rise makes a man healthy, wealthy, and wise.",
    "The pen is mightier than the sword. Actions speak louder than words.",
    "The proof is in the pudding. All's fair in love and war.",
    "The road to hell is paved with good intentions. The squeaky wheel gets the grease.",
    "There's no time like the present. It's always darkest before the dawn.",
    "Time flies when you're having fun. Laughter is the best medicine.",
    "Time is money. Time heals all wounds.",
    "To err is human; to forgive, divine. Let bygones be bygones.",
    "Two wrongs don't make a right. You can't judge a book by its cover.",
    "Variety is the spice of life. Don't put all your eggs in one basket.",
    "What goes around comes around. You reap what you sow.",
    "When in Rome, do as the Romans do. Where there's a will, there's a way.",
    "Where there's a will, there's a way. When in Rome, do as the Romans do.",
    "You can't have your cake and eat it too. Beggars can't be choosers.",
    "You can't make an omelet without breaking a few eggs. You can't teach an old dog new tricks.",
    "You can't please everyone. You can't judge a book by its cover.",
    "You can't teach an old dog new tricks. Old habits die hard.",
    "You reap what you sow. What goes around comes around.",
    "Your guess is as good as mine. Ignorance is bliss."
    "A stitch in time saves nine. A penny for your thoughts.",
    "A watched pot never boils. Absence makes the heart grow fonder.",
    "Actions speak louder than words. All that glitters is not gold.",
    "All's well that ends well. An ounce of prevention is worth a pound of cure.",
    "Barking up the wrong tree. Beauty is only skin deep.",
    "Better safe than sorry. Birds of a feather flock together.",
    "Blood is thicker than water. Break the ice.",
    "Burn the midnight oil. Can't judge a book by its cover.",
    "Caught between a rock and a hard place. Close but no cigar.",
    "Cry over spilled milk. Curiosity killed the cat.",
    "Cut to the chase. Don't bite off more than you can chew.",
    "Don't cry over spilled milk. Don't judge a book by its cover.",
    "Don't put all your eggs in one basket. Don't put the cart before the horse.",
    "Every dog has its day. Every rose has its thorn.",
    "Familiarity breeds contempt. Fortune favors the bold.",
    "Get a taste of your own medicine. Give the benefit of the doubt.",
    "Go the extra mile. Good things come to those who wait.",
    "Great minds think alike. Haste makes waste.",
    "Hit the nail on the head. Ignorance is bliss.",
    "It takes two to tango. It's a piece of cake.",
    "It's raining cats and dogs. Keep your chin up.",
    "Kill two birds with one stone. Let sleeping dogs lie.",
    "Like father, like son. Look before you leap.",
    "Make a long story short. Necessity is the mother of invention.",
    "No pain, no gain. Once in a blue moon.",
    "On cloud nine. Out of the frying pan and into the fire.",
    "Piece of cake. Practice makes perfect.",
    "Put the cart before the horse. Rome wasn't built in a day.",
    "Seeing is believing. Slow and steady wins the race.",
    "Speak of the devil. The early bird catches the worm.",
    "The elephant in the room. The pen is mightier than the sword.",
    "The pot calling the kettle black. The squeaky wheel gets the grease.",
    "There's no place like home. Time flies when you're having fun.",
    "Time is of the essence. To each their own.",
    "Two peas in a pod. When it rains, it pours.",
    "When pigs fly. You can't have your cake and eat it too.",
    "You can't judge a book by its cover. You can't make an omelet without breaking a few eggs."
    "Let sleeping dogs lie. The quick brown fox jumps over the lazy dog.",
]

@pytest.mark.parametrize("max_features", [900, 300])
@pytest.mark.parametrize("ngram_range", [4, 3, 2, 1])
def test_vocabulary_building(max_features, ngram_range):
    vocabulary = build_keyphrase_vocabulary(TEST_OBJECTS, n_jobs=4, max_features=max_features, ngram_range=(1, ngram_range))
    assert len(vocabulary) <= max_features
    assert "the" not in vocabulary
    assert (" ".join(["quick", "brown", "fox", "jumps"][:ngram_range])) in vocabulary
    assert (" ".join(["sleeping", "dogs", "lie"][:ngram_range])) in vocabulary


@pytest.mark.parametrize("ngram_range", [4, 3, 2, 1])
def test_count_matrix_building(ngram_range):
    vocabulary = build_keyphrase_vocabulary(TEST_OBJECTS, n_jobs=4, max_features=1000, ngram_range=(1, ngram_range))
    vocabulary_map = {word: i for i, word in enumerate(vocabulary)}
    count_matrix = build_keyphrase_count_matrix(TEST_OBJECTS, vocabulary_map, n_jobs=4, ngram_range=(1, ngram_range))
    assert count_matrix.shape[0] == len(TEST_OBJECTS)
    assert count_matrix.shape[1] == len(vocabulary)
    assert count_matrix.nnz > 0
    assert count_matrix[0, vocabulary_map[" ".join(["quick", "brown", "fox", "jumps"][:ngram_range])]] == 1
    assert count_matrix[-1, vocabulary_map[" ".join(["quick", "brown", "fox", "jumps"][-ngram_range:])]] == 1

@pytest.mark.parametrize("ngram_range", [3, 2])
@pytest.mark.parametrize("token_pattern", [r"(?u)\b\w[-'\w]+\b", r"(?u)\b\w\w+\b"])
@pytest.mark.parametrize("max_features", [1000, 100])
def test_matching_sklearn(ngram_range, token_pattern, max_features):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    vocabulary = build_keyphrase_vocabulary(TEST_OBJECTS, n_jobs=1, max_features=max_features, ngram_range=(1, ngram_range), token_pattern=token_pattern)
    vocabulary_map = {word: i for i, word in enumerate(sorted(vocabulary))}
    count_matrix = build_keyphrase_count_matrix(TEST_OBJECTS, vocabulary_map, n_jobs=1, ngram_range=(1, ngram_range), token_pattern=token_pattern)

    vectorizer = CountVectorizer(
        ngram_range=(1, ngram_range),
        token_pattern=token_pattern,
    ).fit(TEST_OBJECTS)
    matrix = vectorizer.transform(TEST_OBJECTS)

    vocab_subset = sorted(
        [
            x for x in vectorizer.get_feature_names_out()
            if x.split()[0] not in ENGLISH_STOP_WORDS
            and x.split()[-1] not in ENGLISH_STOP_WORDS
        ],
        key=lambda x: vocabulary_map[x] if x in vocabulary_map else len(vocabulary_map)
    )
    all_counts = np.squeeze(np.asarray(matrix.sum(axis=0)))
    vocab_counts = np.asarray([all_counts[vectorizer.vocabulary_[x]] for x in vocab_subset])
    vocab_counter = Counter(dict(zip(vocab_subset, vocab_counts)))
    vocab_subset = [x[0] for x in vocab_counter.most_common(max_features)]
    assert set(vocab_subset) == set(vocabulary)

    sklearn_matrix = CountVectorizer(
        ngram_range=(1, ngram_range),
        token_pattern=token_pattern,
        vocabulary=vocabulary_map
    ).fit_transform(TEST_OBJECTS)    

    assert count_matrix.shape == sklearn_matrix.shape
    assert np.all(count_matrix.toarray() == sklearn_matrix.toarray())

