import numpy as np

with open("problem2/english_nouns_lower_10000.txt") as file: 
    words=np.array(file.read().splitlines())

words=words[[a.isalpha() and 3<=len(a)<=8 and len(a.replace("i","").replace("v","").replace("x","").replace("l",""))>0 and a[-1]!="s" for a in words]]
N=1000
sample_words=words[:N]

original_words_total=[]
responses_total=[]

wordlist_size=4
num_new_words=2

for i in range(5):
    original_words=words[np.random.permutation(N)][:wordlist_size]
    test_words=words[np.random.permutation(N)][:wordlist_size]

    replacements=np.random.choice(wordlist_size,num_new_words,replace=False)
    replacements_idx=np.random.choice(wordlist_size,num_new_words,replace=False)
    test_words[replacements]=original_words[replacements_idx]

    print("MEMORIZE:")
    print(", ".join(original_words))
    input()
    print("\n"*100)

    print("IDENTIFY MATCHES:")
    print(", ".join(test_words))
    data=[]
    for word in test_words:
        data.append((word,input(word+": ").lower()[0]=="y"))

    original_words_total.append(original_words)
    responses_total.append(data)

with open("problem2/data/original_wordlist_2a.txt","w") as file:
    file.write("\n".join([", ".join(k) for k in original_words_total]))

with open("problem2/data/subject_responses_2a.txt","w") as file:
    file.write("\n".join([", ".join([f"{a}:{['NO','YES'][int(b)]}" for a,b in k]) for k in responses_total]))
