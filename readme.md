# 🧠 Word2Vec Analogy Solver

## 🔍 Introducere

Acest proiect are ca scop descoperirea relațiilor semantice dintre cuvinte prin intermediul modelului **Word2Vec**, folosind un corpus textual despre profesii și domeniile lor.
Prin transformarea cuvintelor în vectori numerici, putem efectua operații matematice de tip analogie, cum ar fi:

> doctors - medicine + law = ?
> teachers - schools + hospitals = ?

Rezultatele acestor operații sunt combinate într-un **flag final** cu formatul:

```
SIGMOID_{WORD1}_{WORD2}
```

---

## 🧩 Analiza Task-ului

Am primit un fișier `corpus.txt` ce conține propoziții despre profesii și domenii (doctori, profesori, spitale, școli etc.).
Pentru a descoperi relațiile ascunse între concepte, am urmat următoarele etape:

1. **Preprocesarea textului** – curățarea, transformarea în litere mici și tokenizarea fiecărei propoziții.
2. **Antrenarea mai multor modele Word2Vec** cu parametri diferiți (`vector_size`, `window`, `sg`, `epochs`) pentru a identifica cea mai bună configurație.
3. **Rezolvarea analogiilor** folosind funcția `most_similar()` din Gensim, care caută cel mai apropiat vector pentru combinația `A - B + C`.
4. **Selectarea celor mai relevante rezultate** și formarea flagului final în formatul cerut.

---

## ⚙️ Soluția Aleasă

După testarea mai multor configurații, cea mai stabilă a fost:

* `vector_size = 100`
* `window = 10`
* `sg = 1` (Skip-Gram)
* `epochs = 200`

Modelul Skip-Gram a oferit cele mai coerente relații între profesii și domenii, capturând bine contextul semantic.
Cu ajutorul funcției `solve_analogy()`, am obținut cele mai potrivite cuvinte pentru fiecare analogie, din care s-a generat flagul final.

---

## ▶️ Rulare

1. Asigură-te că fișierul `corpus.txt` se află în același director cu scriptul.
2. Rulează comanda:

   ```bash
   python main.py
   ```
3. Scriptul va antrena modelele, va afișa rezultatele pentru analogii și va genera **flagul final**.
