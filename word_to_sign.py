from fuzzywuzzy import fuzz

# English letters to images
asl_dict = {chr(65+i): f'images/{chr(65+i)}.png' for i in range(26)}

def word_to_sign(input_word):
    scores = {k: fuzz.ratio(input_word.upper(), k) for k in asl_dict.keys()}
    best_match = max(scores, key=scores.get)
    return asl_dict[best_match]
