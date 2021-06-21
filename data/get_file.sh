wget https://files.grouplens.org/datasets/movielens/ml-1m.zip  --no-check-certificate
unzip ml-1m.zip
mv ml-1m movielens-1m
rm ml-1m.zip
rm movielens-1m/movies.dat
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download?id=12FzoWXV65TeUqqs_drCJKaO1Xax0YYTs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12FzoWXV65TeUqqs_drCJKaO1Xax0YYTs" -O movies-1m.csv && rm -rf /tmp/cookies.txt
mv movies-1m.csv movielens-1m/movies-1m.csv
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1D6_jvNBs6YNhYU8Bd9UOKBVQrnoNAO78' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1D6_jvNBs6YNhYU8Bd9UOKBVQrnoNAO78" -O bert_sequence_1m.txt && rm -rf /tmp/cookies.txt
