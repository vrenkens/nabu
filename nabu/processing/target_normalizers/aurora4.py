'''@file aurora4.py
contains the aurora4 target normalize method'''

def normalize(transcription, alphabet):
    '''normalize a transcription

    Args:
        transcription: the transcription to be normalized as a string

    Returns:
        the normalized transcription as a string space seperated per
        character'''

    #create a dictionary of words that should be replaced
    replacements = {
        ',COMMA':'COMMA',
        '\"DOUBLE-QUOTE':'DOUBLE-QUOTE',
        '!EXCLAMATION-POINT':'EXCLAMATION-POINT',
        '&AMPERSAND':'AMPERSAND',
        '\'SINGLE-QUOTE':'SINGLE-QUOTE',
        '(LEFT-PAREN':'LEFT-PAREN',
        ')RIGHT-PAREN':'RIGHT-PAREN',
        '-DASH':'DASH',
        '-HYPHEN':'HYPHEN',
        '...ELLIPSIS':'ELLIPSIS',
        '.PERIOD':'PERIOD',
        '/SLASH':'SLASH',
        ':COLON':'COLON',
        ';SEMI-COLON':'SEMI-COLON',
        '<NOISE>': '',
        '?QUESTION-MARK': 'QUESTION-MARK',
        '{LEFT-BRACE': 'LEFT-BRACE',
        '}RIGHT-BRACE': 'RIGHT-BRACE'
        }

    #replace the words in the transcription
    replaced = ' '.join([word if word not in replacements
                         else replacements[word]
                         for word in transcription.split(' ')])

    #make the transcription lower case and put it into a list
    normalized = list(replaced.lower())

    #replace the spaces with <space>
    normalized = [character if character is not ' ' else '<space>'
                  for character in normalized]

    #replace unknown characters with <unk>
    normalized = [character if character in alphabet else '<unk>'
                  for character in normalized]

    return ' '.join(normalized)
