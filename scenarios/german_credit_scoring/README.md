cat german.data-numeric | sed -E "s/[ ]+/,/g" | cut -c 2- | rev | cut -c 2- | rev  > ./german.csv
