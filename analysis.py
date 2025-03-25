from stats import *
import pandas as pd

"""
- x most frequent words
- x top tfidf words
"""

### write to disk ###
def write_to_disk(data, transp="yes", csv_name="default.csv", sep=";"):
    df = pd.DataFrame(data).transpose()
    if transp=="yes":
        df = df.transpose()

    df.to_csv(f"results/{csv_name}", sep=sep, decimal=".")

    return df



### single texts ###
# data, metadata = collect_data("data")
def stats_by_article(directory, printout="yes"):
    data, metadata = collect_data(directory)
    top_freqs = top_freq(data, x=50)
    tf_idf = tf_idf_general(data)
    top_tf_idf = top_tfidf(tf_idf, x=50)

    for article in metadata:
        metadata[article]["lexicalDiversityWords"] = round(metadata[article]["count_wordtypes"]/metadata[article]["count_words"], 3)
        metadata[article]["lexicalDiversityToks"] = round(metadata[article]["count_toktypes"]/metadata[article]["count_toks"], 3)         
        metadata[article]["wordsPerSent"] = round(metadata[article]["count_words"]/metadata[article]["count_sent"], 3)         
        metadata[article]["wordtypesPerSent"] = round(metadata[article]["count_wordtypes"]/metadata[article]["count_sent"], 3)
        metadata[article]["toksPerSent"] = round(metadata[article]["count_toks"]/metadata[article]["count_sent"], 3)        
        metadata[article]["toktypesPerSent"] = round(metadata[article]["count_toktypes"]/metadata[article]["count_sent"], 3)
        for index, tok in enumerate(top_freqs[article]):
                top_freqs[article][index] = list(top_freqs[article][index])
                top_freqs[article][index].append(round(tok[1]/metadata[article]["count_toks"], 5))
        if printout == "yes":
            print("---------------------------------------------")
            print(f"Article:\t\t{article}")
            print(f"Anzahl Sätze:\t\t{metadata[article]["count_sent"]}")
            print(f"Anzahl Wörter:\t\t{metadata[article]["count_words"]}")
            print(f"Anzahl Worttypen:\t{metadata[article]["count_wordtypes"]}")
            print(f"Anzahl Token:\t\t{metadata[article]["count_toks"]}")
            print(f"Anzahl Tokentypen:\t{metadata[article]["count_toktypes"]}")
            print(print(f"Lexical Word Diversity:\t{metadata[article]["lexicalDiversityWords"]}"))
            print(f"Lexical Tok Diversity:\t{metadata[article]["lexicalDiversityToks"]}")
            print(f"Words per Sentence:\t{metadata[article]["wordsPerSent"]}")
            print(f"Wordtypes per Sentence:\t{metadata[article]["wordtypesPerSent"]}")
            print(f"Tok per Sentence:\t{metadata[article]["toksPerSent"]}")
            print(f"Toktypes per Sentence:\t{metadata[article]["toktypesPerSent"]}")
            print()
            print(f"50 most frequent tokens:")
            for tok in top_freqs[article]:
                print(f"\t\t{tok[0]}\t{tok[1]}\t{round(tok[1]/metadata[article]["count_toks"], 5)}")
            print()
            print(f"50 most frequent tokens: [tok, abs_freq, rel_freq]")
            for index, tok in enumerate(top_freqs[article]):
                print(f"\t\t{top_freqs[article][index]}")
            print("---------------------------------------------")
            print()

    return data, metadata, top_freqs, top_tf_idf


### texts by journal ###
def stats_by_journal(directory, printout="yes"):
    data, metadata = collect_data_by_journal(directory)
    top_freqs = top_freq(data, x=50)
    tf_idf = tf_idf_general(data)
    top_tf_idf = top_tfidf(tf_idf, x=50)

    for journal in metadata:
        metadata[journal]["lexicalDiversityWords"] = round(metadata[journal]["count_wordtypes"]/metadata[journal]["count_words"], 3)
        metadata[journal]["lexicalDiversityToks"] = round(metadata[journal]["count_toktypes"]/metadata[journal]["count_toks"], 3) 
        metadata[journal]["wordsPerSent"] = round(metadata[journal]["count_words"]/metadata[journal]["count_sent"], 3) 
        metadata[journal]["wordtypesPerSent"] = round(metadata[journal]["count_wordtypes"]/metadata[journal]["count_sent"], 3) 
        metadata[journal]["toksPerSent"] = round(metadata[journal]["count_toks"]/metadata[journal]["count_sent"], 3)
        metadata[journal]["toktypesPerSent"] = round(metadata[journal]["count_toktypes"]/metadata[journal]["count_sent"], 3)
        metadata[journal]["sentPerArticle"] = round(metadata[journal]["count_sent"]/metadata[journal]["count_articles"], 3)
        metadata[journal]["wordsPerArticle"] = round(metadata[journal]["count_words"]/metadata[journal]["count_articles"], 3)
        metadata[journal]["wordtypesPerArticle"] = round(metadata[journal]["count_wordtypes"]/metadata[journal]["count_articles"], 3)
        metadata[journal]["toksPerArticle"] = round(metadata[journal]["count_toks"]/metadata[journal]["count_articles"], 3) 
        metadata[journal]["toktypesPerArticle"] = round(metadata[journal]["count_toktypes"]/metadata[journal]["count_articles"], 3) 
        for index, tok in enumerate(top_freqs[journal]):
                top_freqs[journal][index] = list(top_freqs[journal][index])
                top_freqs[journal][index].append(round(tok[1]/metadata[journal]["count_toks"], 5))

        if printout=="yes":    
            print("---------------------------------------------")
            print(f"Journal:\t\t{journal}")
            print(f"Anzahl Artikel:\t\t{metadata[journal]["count_articles"]}")
            print(f"Anzahl Sätze:\t\t{metadata[journal]["count_sent"]}")
            print(f"Anzahl Wörter:\t\t{metadata[journal]["count_words"]}")
            print(f"Anzahl Worttypen:\t{metadata[journal]["count_wordtypes"]}")
            print(f"Anzahl Token:\t\t{metadata[journal]["count_toks"]}")
            print(f"Anzahl Tokentypen:\t{metadata[journal]["count_toktypes"]}")
            print()
            print(f"Lexical Word Diversity:\t{metadata[journal]["lexicalDiversityWords"]}")
            print(f"Lexical Tok Diversity:\t{metadata[journal]["lexicalDiversityToks"]}")
            print(f"Words per Sentence:\t{metadata[journal]["wordsPerSent"]}")
            print(f"Wordtypes per Sentence:\t{metadata[journal]["wordtypesPerSent"]}")
            print(f"Tok per Sentence:\t{metadata[journal]["toksPerSent"]}")
            print(f"Toktypes per Sentence:\t{metadata[journal]["toktypesPerSent"]}")
            print()
            print(f"Sent per Article:\t{metadata[journal]["sentPerArticle"]}")
            print(f"Words per Article:\t{metadata[journal]["wordsPerArticle"]}")
            print(f"Wordtypes per Article:\t{metadata[journal]["wordtypesPerArticle"]}")
            print(f"Tok per Article:\t{metadata[journal]["toksPerArticle"]}")
            print(f"Toktypes per Article:\t{metadata[journal]["toktypesPerArticle"]}")
            print()
            print(f"50 most frequent tokens: [tok, abs_freq, rel_freq]")
            for index, tok in enumerate(top_freqs[journal]):
                print(f"\t\t{top_freqs[journal][index]}")
            print()
            print(f"50 top tf-idf tokens:")
            for tok in top_tf_idf[journal]:
                print(f"\t{tok}")
            print("---------------------------------------------")
            print()

    return data, metadata, top_freqs, top_tf_idf



# boilerplate
if __name__ == '__main__':
    data, metadata, top_freqs, top_tf_idf = stats_by_journal("data", printout="no")
    write_to_disk(top_tf_idf, transp="yes", csv_name="journal_tfidf.csv")
    write_to_disk(top_freqs, transp="yes", csv_name="journal_freq.csv")
    write_to_disk(metadata, transp="no", csv_name="journal_metadata.csv")

    data, metadata, top_freqs, top_tf_idf = stats_by_article("data", printout="no")
    write_to_disk(top_tf_idf, transp="yes", csv_name="article_tfidf.csv")
    write_to_disk(top_freqs, transp="yes", csv_name="article_freq.csv")
    write_to_disk(metadata, transp="no", csv_name="article_metadata.csv")
