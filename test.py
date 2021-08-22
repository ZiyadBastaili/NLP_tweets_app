
import re


# st.sidebar.title("[Output Customisation](top)")
# st.sidebar.header("[1. Test Header](1)")
# st.sidebar.header("[2. Another Header](2)")

# x = 'fgfhft gh745 twitter.com gggg'
# a = re.sub(r'\w+\.\w+\.com\s|\w+\.com\s', ' ', x)
# print(a)

#--------------------------------------------------------------------------------------
# def remove_punctuation(text, exceptions=None):
#     """
#     Return a string with punctuation removed.
#     Parameters:
#         text (str): The text to remove punctuation from.
#         exceptions (list): List of symbols to keep in the given text.
#     Return:
#         str: The input text without the punctuation.
#     """
#     all_but = [
#         r'\w', # keep words
#         r'\s', # keep spaces
#         r'U[\d\w]+' # keep Emoji
#     ]
#     if exceptions is not None:
#         all_but.extend(exceptions)
#
#     pattern = '[^{}]'.format(''.join(all_but))
#     return re.sub(pattern, '', text)
#
# text='fdfs. fvdd/§? @jsdhshf #jjjj $gggg 🌐 55555555 😆 😀 ?'
# print(remove_punctuation(text, exceptions=['@', '$']))

# import string
# print(type(string.punctuation))

# words = 'dgvfggbgfg @ggfbfg gf78 78 §?'
#
# pattern = r'@\w+ '
# words = re.sub(pattern, '', words)
# print(words)

a = ['e','j'] + ['0','8']
print(a)