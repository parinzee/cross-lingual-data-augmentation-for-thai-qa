import pandas as pd
import textwrap
from tqdm import tqdm
from easygoogletranslate import EasyGoogleTranslate

translator = EasyGoogleTranslate(
    source_language='th',
    target_language='en',
    timeout=10
)

translator_back = EasyGoogleTranslate(
    source_language='en',
    target_language='th',
    timeout=10
)

def translate_to_language(text, language):
    print(text)
    if language == 'en':
        return translator.translate(text)
    else:
        return translator_back.translate(text)

def batch_translate(series, language):
    translated_text = []
    for text in tqdm(series):
        # split the text into chunks of 5000 or less
        chunks = textwrap.wrap(text, 3000)
        translated_chunks = [translate_to_language(chunk, language) for chunk in chunks]
        translated_text.append(''.join(translated_chunks))
    return translated_text

def process_file(file_name):
    # Step 1: Read CSV and get context column
    df = pd.read_csv(file_name)
    context = df['context'].drop_duplicates().tolist()

    # Step 2: Translate to en_aug and back to th_aug
    context_en = batch_translate(context, 'en')
    context_th = batch_translate(context_en, 'th')

    # Step 3: Merge back into original dataframe
    df_translated = pd.DataFrame({'context': context, 'en_aug': context_en, 'th_aug': context_th})
    df = df.merge(df_translated, on='context', how='left')

    return df

if __name__ == '__main__':
    df = process_file("data/01_prepare_dataset.csv")
    df.to_csv("data/02_backtranslate_english_input.csv", index=False)